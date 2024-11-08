import hashlib
import json
import re
from charset_normalizer import detect
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm
from redis import Redis
from typing import Set, Dict, List, Optional, Tuple
from transformers import MarianMTModel, MarianTokenizer
from elasticsearch import Elasticsearch
from urllib.parse import urljoin, urlparse
import time
import logging
from config_utils import CrawlConfig, PDFProcessor, setup_logging
from page_rank import PageRankAnalyzer
from semantic_search import SemanticSearchIndexer

ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

class RobustCrawler:
    def __init__(
        self,
        config: CrawlConfig,
        redis_client: Optional[Redis] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.redis = redis_client
        self.logger = logger or logging.getLogger(__name__)
        self.seen_urls = {}  # Domain -> URLs mapping
        self.last_request_time = 0
        self._setup_elasticsearch()
        self._setup_translation_models()
        self.pdf_processor = PDFProcessor(self.logger)
        self.semantic_indexer = self._setup_semantic_search()
        
        # Compile excluded patterns
        self.excluded_patterns = [re.compile(pattern) for pattern in self.config.excluded_patterns]

    def _setup_translation_models(self) -> None:
        """Initialize translation models for supported languages"""
        self.translation_models = {}
        for source in self.config.supported_languages:
            for target in self.config.supported_languages:
                if source != target:
                    model_name = f'Helsinki-NLP/opus-mt-{source}-{target}'
                    try:
                        self.translation_models[(source, target)] = {
                            'model': MarianMTModel.from_pretrained(model_name),
                            'tokenizer': MarianTokenizer.from_pretrained(model_name)
                        }
                    except Exception as e:
                        self.logger.error(f"Error loading translation model {model_name}: {str(e)}")

    def _setup_elasticsearch(self) -> None:
        """Initialize Elasticsearch client"""
        self.es = Elasticsearch(
            ELASTICSEARCH_URL,
            api_key=ELASTICSEARCH_API_KEY,
        )

    def _setup_semantic_search(self) -> SemanticSearchIndexer:
        """Initialize semantic search indexer"""
        indexer = SemanticSearchIndexer(
            es_client=self.es,
            index_name=self.config.elasticsearch_index
        )
        indexer.create_index()
        return indexer

    def _normalize_url(self, url: str, base_url: str) -> Optional[str]:
        """Normalize and validate a URL"""
        try:
            full_url = urljoin(base_url, url)
            parsed = urlparse(full_url)
            
            if not parsed.scheme or not parsed.netloc:
                return None
                
            if parsed.scheme not in ['http', 'https']:
                return None
                
            clean_url = parsed._replace(fragment='').geturl()
            
            for pattern in self.excluded_patterns:
                if pattern.search(clean_url):
                    return None
                    
            return clean_url
            
        except Exception as e:
            self.logger.debug(f"URL normalization error: {str(e)}")
            return None

    def _make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retries and error handling"""
        for attempt in range(self.config.max_retries):
            try:
                headers = {
                    'User-Agent': self.config.user_agents[attempt % len(self.config.user_agents)]
                }
                
                response = requests.get(
                    url,
                    headers=headers,
                    timeout=self.config.timeout,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return response
                    
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): "
                    f"Status {response.status_code} for {url}"
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Request error (attempt {attempt + 1}/{self.config.max_retries}): "
                    f"{str(e)} for {url}"
                )
                
            time.sleep(2 ** attempt)  # Exponential backoff
            
        return None

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Set[str]:
        """Extract and normalize all links from a page"""
        links = set()
        try:
            for anchor in soup.find_all('a', href=True):
                normalized = self._normalize_url(anchor['href'], base_url)
                if normalized:
                    links.add(normalized)
        except Exception as e:
            self.logger.error(f"Error extracting links from {base_url}: {str(e)}")
        return links

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content"""
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove any remaining special characters while preserving Arabic text
        content = re.sub(r'[^\w\s\.,!?\-\'\":،؟]', '', content)
        
        return content

    def _detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails

    def _translate_content(self, text: str, source_lang: str) -> Dict[str, str]:
        """Translate content to all supported languages"""
        translations = {source_lang: text}
        
        for target_lang in self.config.supported_languages:
            if target_lang != source_lang:
                try:
                    model_pair = self.translation_models.get((source_lang, target_lang))
                    if model_pair:
                        inputs = model_pair['tokenizer'](text, return_tensors="pt", padding=True, truncation=True)
                        outputs = model_pair['model'].generate(**inputs)
                        translated = model_pair['tokenizer'].decode(outputs[0], skip_special_tokens=True)
                        translations[target_lang] = translated
                except Exception as e:
                    self.logger.error(f"Translation error {source_lang}->{target_lang}: {str(e)}")
                    translations[target_lang] = text  # Fall back to original text
                    
        return translations

    def _can_crawl_url(self, url: str, depth: int) -> bool:
        """Check if a URL should be crawled based on configuration"""
        if depth > self.config.max_depth:
            return False
            
        domain = urlparse(url).netloc
        
        # Initialize domain tracking if needed
        if domain not in self.seen_urls:
            self.seen_urls[domain] = set()
            
        # Check domain page limit
        if len(self.seen_urls[domain]) >= self.config.max_pages_per_domain:
            return False
            
        # Check if already crawled
        if url in self.seen_urls[domain]:
            return False
            
        return True

    def _extract_and_clean_content(self, response: requests.Response) -> Optional[Tuple[str, str]]:
        """Extract and clean content from response, handling both HTML and PDF"""
        content_type = response.headers.get('content-type', '').lower()
        
        if 'text/html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = None
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                title = title_tag.string.strip()
            
            if not title:
                h1_tag = soup.find('h1')
                if h1_tag:
                    title = h1_tag.get_text(strip=True)
            
            if not title:
                title = urlparse(response.url).path.split('/')[-1] or "Untitled"
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'form']):
                element.decompose()
            
            content = soup.get_text(separator=' ', strip=True)
            content = self._clean_content(content)
            
            return (title, content)
            
        elif 'application/pdf' in content_type:
            return self._extract_pdf_content(response)
            
        return None

    def _extract_pdf_content(self, response: requests.Response) -> Optional[Tuple[str, str]]:
        """Extract content from PDF response"""
        try:
            text, metadata = self.pdf_processor.process_pdf(response.content)
            
            if not text:
                return None
            
            title = (metadata.get('title') or 
                    os.path.splitext(os.path.basename(response.url))[0] or 
                    "Untitled PDF")
            
            text = self._clean_content(text)
            
            return (title, text)
            
        except Exception as e:
            self.logger.error(f"Error processing PDF from {response.url}: {str(e)}")
            return None

    def _respect_crawl_delay(self) -> None:
        """Implement polite crawling with rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.config.request_delay:
            time.sleep(self.config.request_delay - elapsed)
        self.last_request_time = time.time()

    def crawl_url(self, url: str, depth: int = 0) -> Dict[str, Set[str]]:
        """Crawl a single URL and its linked pages"""
        result = {'processed': set(), 'failed': set()}
        domain = urlparse(url).netloc
        
        if not self._can_crawl_url(url, depth):
            return result
            
        try:
            self._respect_crawl_delay()
            
            response = self._make_request(url)
            if not response:
                result['failed'].add(url)
                return result
            
            content_tuple = self._extract_and_clean_content(response)
            
            if content_tuple:
                title, content = content_tuple
                if len(content) >= self.config.min_content_length:
                    if self.semantic_indexer.index_document({
                        'url': url,
                        'title': {'original': title},
                        'content': {'original': content},
                        'timestamp': int(time.time() * 1000),
                        'domain': domain,
                        'id': hashlib.sha256(f"{url}{content}".encode()).hexdigest()
                    }):
                        result['processed'].add(url)
                        self.seen_urls[domain].add(url)
                    else:
                        result['failed'].add(url)
            
            if depth < self.config.max_depth and 'text/html' in response.headers.get('content-type', '').lower():
                soup = BeautifulSoup(response.content, 'html.parser')
                links = self._extract_links(soup, url)
                for link in links:
                    if urlparse(link).netloc == domain:
                        sub_result = self.crawl_url(link, depth + 1)
                        result['processed'].update(sub_result['processed'])
                        result['failed'].update(sub_result['failed'])
                        
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            result['failed'].add(url)
            
        return result

    def crawl_site(self, url: str) -> Dict[str, Set[str]]:
        """Crawl an entire site starting from the given URL"""
        domain = urlparse(url).netloc
        self.seen_urls[domain] = set()
        
        self.logger.info(f"Starting crawl of {url}")
        result = self.crawl_url(url)
        
        self.logger.info(
            f"Completed crawl of {url}:\n"
            f"  Processed: {len(result['processed'])} pages\n"
            f"  Failed: {len(result['failed'])} pages"
        )
        
        return result

    def crawl_sites_parallel(self, urls: List[str], max_workers: Optional[int] = None) -> Dict[str, Dict[str, Set[str]]]:
        """Crawl multiple sites in parallel"""
        results = {}
        max_workers = max_workers or min(len(urls), os.cpu_count())
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.crawl_site, url): url for url in urls}
            
            for future in tqdm(future_to_url, desc="Crawling sites"):
                url = future_to_url[future]
                try:
                    results[url] = future.result()
                except Exception as e:
                    self.logger.error(f"Error crawling {url}: {str(e)}")
                    results[url] = {'processed': set(), 'failed': {url}}
                    
        return results
def main():
    logger = setup_logging()
    
    config = CrawlConfig(
        max_depth=3,
        max_pages_per_domain=100,
        supported_languages=['en', 'ar', 'ur']
    )
    
    crawler = RobustCrawler(config, logger=logger)
    print("Crawling sites...")
    
    try:
        with open('urls.json') as f:
            urls_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading URLs configuration: {str(e)}")
        return

    # Perform crawling
    crawl_results = {}
    total_processed = 0
    total_failed = 0
    
    for site in urls_config['sites']:
        url = site['url']
        try:
            result = crawler.crawl_site(url)
            crawl_results[url] = result
            total_processed += len(result['processed'])
            total_failed += len(result['failed'])
        except Exception as e:
            logger.error(f"Error processing site {url}: {str(e)}")
            total_failed += 1

    logger.info(f"\nCrawling completed:")
    logger.info(f"Total pages processed: {total_processed}")
    logger.info(f"Total pages failed: {total_failed}")

    # Perform PageRank analysis
    analyzer = PageRankAnalyzer(logger=logger)
    analyzer.add_pages(crawl_results)
    
    # Get and display top pages
    print("\nTop 10 Pages by PageRank:")
    for url, score in analyzer.get_top_pages(10):
        print(f"{url}: {score:.4f}")
    
    # Get and display domain scores
    print("\nDomain Rankings:")
    domain_scores = analyzer.get_domain_scores()
    for domain, score in sorted(domain_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{domain}: {score:.4f}")

if __name__ == "__main__":
    main()