import hashlib
import json
import re
from charset_normalizer import detect
import justext
import aiohttp
from bs4 import BeautifulSoup
import asyncio
import os
from tqdm import tqdm
from redis import Redis
from typing import Set, Dict, List, Optional, Tuple
from transformers import MarianMTModel, MarianTokenizer
from elasticsearch import AsyncElasticsearch
from urllib.parse import urljoin, urlparse
import time
import logging
from config_utils import CrawlConfig, PDFProcessor, setup_logging
from enhanced_index import EnhancedMultilingualIndexer
from random_user_agent.user_agent import UserAgent

codeToLang = {
    "en": "English",
    "ar": "Arabic",
    "ur": "Urdu"
}

ELASTICSEARCH_URL=os.getenv("ELASTICSEARCH_URL")
ELASTICSEARCH_API_KEY=os.getenv("ELASTICSEARCH_API_KEY")

class RobustCrawler:
    def __init__(
        self,
        config: CrawlConfig,
        redis_client: Optional[Redis] = None,
        logger: Optional[logging.Logger] = None,
        max_concurrent_requests: int = 40
    ):
        self.config = config
        self.redis = redis_client
        self.logger = logger or logging.getLogger(__name__)
        self.seen_urls = {}  # Domain -> URLs mapping
        self.last_request_time = 0
        self._setup_elasticsearch()
        self._setup_translation_models()
        self.pdf_processor = PDFProcessor(self.logger)
        self.ua = UserAgent()
        
        # Request limiting
        self.max_concurrent_requests = max_concurrent_requests
        self.request_semaphore = None  # Will be initialized in initialize()
        self.domain_semaphores = {}  # Domain-specific semaphores
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout))

        # Domain request tracking
        self.domain_last_request = {} 

        # Compile excluded patterns
        self.excluded_patterns = [re.compile(pattern) for pattern in self.config.excluded_patterns]

    async def initialize(self):
        """Initialize async components"""
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        self.semantic_indexer = await self._setup_semantic_search()

    async def cleanup(self):
        """Cleanup async resources"""
        if self.session:
            await self.session.close()
        await self.es.close()

    async def _make_request(self, url: str) -> Optional[Tuple[bytes, str]]:
        """Make HTTP request with error handling and session management"""
        current_url = url
        redirect_count = 0
        domain = urlparse(current_url).netloc
        domain_semaphore = self._get_domain_semaphore(domain)
        
        try:
            async with self.request_semaphore, domain_semaphore:
                while True:
                    if redirect_count >= 10:
                        self.logger.warning(f"Too many redirects, started from {url}")
                        return None
                    
                    await self._respect_domain_rate_limit(domain)
                    
                    if not current_url or not urlparse(current_url).scheme or not urlparse(current_url).netloc:
                        self.logger.warning(f"Invalid URL format: {current_url}")
                        return None

                    for attempt in range(self.config.max_retries):
                        try:
                            if attempt > 0:
                                await asyncio.sleep(2 ** attempt)
                                
                            headers = {
                                'User-Agent': self.ua.get_random_user_agent(),
                            }
                            
                            async with self.session.get(
                                current_url,
                                headers=headers,
                                allow_redirects=False,
                                proxy=os.getenv("PROXY"),
                                ssl=False,
                            ) as response:
                                if response.status == 200:
                                    try:
                                        content = await response.read()
                                        content_type = response.headers.get('content-type', '')
                                        
                                        if not content:
                                            self.logger.warning(f"Empty response from {current_url}")
                                            return None

                                        return content, content_type
                                    except Exception as e:
                                        self.logger.error(f"Error reading response from {current_url}: {str(e)}")
                                        return None
                                    
                                elif response.status in [301, 302, 303, 307, 308]:
                                    redirect_url = response.headers.get('Location')
                                    
                                    if redirect_url and not urlparse(redirect_url).netloc:
                                        base_url = urlparse(current_url)
                                        redirect_url = f"{base_url.scheme}://{base_url.netloc}{redirect_url}"
                                    
                                    if redirect_url:
                                        self.logger.info(f"Following redirect from {current_url} to {redirect_url}")
                                        current_url = redirect_url
                                        redirect_count += 1
                                        # Break the retry loop to handle the redirect
                                        break
                                    else:
                                        self.logger.warning(f"Redirect without Location header from {current_url}")
                                        return None
                                
                                else:
                                    self.logger.warning(
                                        f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): "
                                        f"Status {response.status} for {current_url}"
                                    )
                        
                        except aiohttp.ClientError as e:
                            self.logger.warning(
                                f"Connection error (attempt {attempt + 1}/{self.config.max_retries}): "
                                f"{str(e)} for {current_url}"
                            )
                            
                        except asyncio.TimeoutError:
                            self.logger.warning(
                                f"Timeout error (attempt {attempt + 1}/{self.config.max_retries}): {current_url}"
                            )
                            
                        except Exception as e:
                            self.logger.error(f"Unexpected error requesting {current_url}: {str(e)}")
                            return None
                    
                    # If we've exhausted all retries without success or a redirect
                    if response.status not in [301, 302, 303, 307, 308]:
                        return None
                        
        except Exception as e:
            self.logger.error(f"Critical error making request to {current_url}: {str(e)}")
            return None

    async def crawl_sites_parallel(self, urls: List[str], chunk_size: int = 3) -> Dict[str, Dict[str, Set[str]]]:
        """Crawl multiple sites in parallel with controlled concurrency"""
        results = {}
        
        # Process URLs in chunks to avoid overwhelming resources
        for i in range(0, len(urls), chunk_size):
            chunk = urls[i:i + chunk_size]
            tasks = [self.crawl_site(url) for url in chunk]
            chunk_results = await asyncio.gather(*tasks)
            results.update(dict(zip(chunk, chunk_results)))
            
            # Small delay between chunks
            await asyncio.sleep(1)
            
        return results

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

    def _get_domain_semaphore(self, domain: str) -> asyncio.Semaphore:
        """Get or create a domain-specific semaphore"""
        if domain not in self.domain_semaphores:
            self.domain_semaphores[domain] = asyncio.Semaphore(self.config.max_concurrent_requests_per_domain)
        return self.domain_semaphores[domain]
    
    async def _respect_domain_rate_limit(self, domain: str) -> None:
        """Respect domain-specific rate limiting"""
        last_request = self.domain_last_request.get(domain, 0)
        elapsed = time.time() - last_request
        if elapsed < self.config.request_delay:
            print(f"Respecting domain rate limit for {domain} - sleeping for {self.config.request_delay - elapsed} seconds")
            await asyncio.sleep(self.config.request_delay - elapsed)
        self.domain_last_request[domain] = time.time()

    def _setup_elasticsearch(self) -> None:
        """Initialize Elasticsearch client"""
        self.es = AsyncElasticsearch(
            ELASTICSEARCH_URL,
            api_key=ELASTICSEARCH_API_KEY,
        )

    async def _setup_semantic_search(self) -> EnhancedMultilingualIndexer:
        """Initialize semantic search indexer"""
        indexer = EnhancedMultilingualIndexer(
            es_client=self.es,
        )
        await indexer.create_index()
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
        
        # Remove control characters while preserving newlines
        content = ''.join(char for char in content if char == '\n' or char >= ' ')
        
        # Remove any remaining special characters while preserving Arabic/Urdu text
        content = re.sub(r'[^\w\s\.,!?\-\'\":،؟\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', content)
        
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

    async def _extract_and_clean_content(self, content: bytes, content_type: str, url: str) -> Optional[Tuple[str, str, str]]:
        """Extract and clean content from response, handling both HTML and PDF"""
        if 'text/html' in content_type.lower():
            # Detect encoding from content-type header
            encoding = None
            if 'charset=' in content_type.lower():
                encoding = content_type.lower().split('charset=')[-1].strip()
            
            # Try to detect encoding from content if not specified in headers
            if not encoding:
                result = detect(content)
                encoding = result['encoding'] if result else 'utf-8'
            
            try:
                # Try decoding with detected encoding
                decoded_content = content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                try:
                    # Fallback to utf-8
                    decoded_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        # Fallback to ISO-8859-1
                        decoded_content = content.decode('iso-8859-1')
                    except UnicodeDecodeError:
                        # Last resort: decode with errors ignored
                        decoded_content = content.decode('utf-8', errors='ignore')

            soup = BeautifulSoup(decoded_content, 'html.parser')
            
            # Try to get charset from meta tags if not found earlier
            if not encoding:
                meta_charset = soup.find('meta', charset=True)
                if meta_charset:
                    encoding = meta_charset.get('charset')
                else:
                    meta_content_type = soup.find('meta', {'http-equiv': lambda x: x and x.lower() == 'content-type'})
                    if meta_content_type and 'charset=' in meta_content_type.get('content', '').lower():
                        encoding = meta_content_type['content'].lower().split('charset=')[-1].strip()

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
                title = urlparse(url).path.split('/')[-1] or "Untitled"
            
            # Get text content
            text_content = soup.get_text()
            lang = self._detect_language(text_content)

            # Use justext for better content extraction
            paragraphs = justext.justext(
                content,
                justext.get_stoplist(codeToLang.get(lang, 'english')),
                encoding=encoding
            )
            
            content_text = "\n".join([p.text for p in paragraphs if not p.is_boilerplate])
            
            return (title, content_text, lang)
            
        elif 'application/pdf' in content_type.lower():
            return await self._extract_pdf_content(content, url)
            
        return None

    async def _extract_pdf_content(self, content: bytes, url: str) -> Optional[Tuple[str, str, str]]:
        """Extract content from PDF response"""
        try:
            text, metadata = await self.pdf_processor.process_pdf(content)
            
            if not text:
                return None
            
            title = (metadata.get('title') or 
                    os.path.splitext(os.path.basename(url))[0] or 
                    "Untitled PDF")
            
            text = self._clean_content(text)
            lang = self._detect_language(text)
            
            return (title, text, lang)
            
        except Exception as e:
            self.logger.error(f"Error processing PDF from {url}: {str(e)}")
            return None

    async def _respect_crawl_delay(self) -> None:
        """Implement polite crawling with rate limiting"""
        await asyncio.sleep(self.config.request_delay)

    async def crawl_url(self, url: str, depth: int = 0) -> Dict[str, Set[str]]:
        """Crawl a single URL and its linked pages"""
        result = {'processed': set(), 'failed': set()}
        
        try:
            # Validate URL before processing
            if not url or not urlparse(url).scheme or not urlparse(url).netloc:
                self.logger.warning(f"Skipping invalid URL: {url}")
                result['failed'].add(url)
                return result
                
            domain = urlparse(url).netloc
            
            if not self._can_crawl_url(url, depth):
                return result

            # Make request and handle response
            response = await self._make_request(url)
            if not response:
                result['failed'].add(url)
                return result
                
            content, content_type = response
            
            # Process content
            content_tuple = await self._extract_and_clean_content(content, content_type, url)
            if not content_tuple:
                self.logger.warning(f"Could not extract content from {url}")
                result['failed'].add(url)
                return result
                
            title, content_text, lang = content_tuple
            
            # Index content if meets minimum length
            if len(content_text) >= self.config.min_content_length:
                doc_id = hashlib.sha256(f"{url}{content_text}".encode()).hexdigest()
                
                if await self.semantic_indexer.index_document({
                    'url': url,
                    'title': {'original': title},
                    'content': {'original': content_text},
                    'timestamp': int(time.time() * 1000),
                    'domain': domain,
                    'language': lang,
                    'id': doc_id
                }):
                    result['processed'].add(url)
                    self.seen_urls[domain].add(url)
                else:
                    result['failed'].add(url)
            
            # Extract and process links if HTML content
            if 'text/html' in content_type.lower() and depth < self.config.max_depth:
                soup = BeautifulSoup(content, 'html.parser')
                links = self._extract_links(soup, url)
                
                # Process links in chunks to avoid overwhelming resources
                chunk_size = 5
                for i in range(0, len(links), chunk_size):
                    chunk = list(links)[i:i + chunk_size]
                    tasks = []
                    
                    for link in chunk:
                        if urlparse(link).netloc == domain:
                            tasks.append(self.crawl_url(link, depth + 1))
                    
                    if tasks:
                        sub_results = await asyncio.gather(*tasks, return_exceptions=True)
                        for sub_result in sub_results:
                            if isinstance(sub_result, dict):
                                result['processed'].update(sub_result['processed'])
                                result['failed'].update(sub_result['failed'])
                            else:
                                self.logger.error(f"Error in subcrawl: {str(sub_result)}")
                    
                    # Small delay between chunks
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Error crawling {url}: {str(e)}")
            result['failed'].add(url)
            
        return result

    async def crawl_site(self, url: str) -> Dict[str, Set[str]]:
        """Crawl an entire site starting from the given URL"""
        domain = urlparse(url).netloc
        self.seen_urls[domain] = set()
        
        self.logger.info(f"Starting crawl of {url}")
        result = await self.crawl_url(url)
        
        self.logger.info(
            f"Completed crawl of {url}:\n"
            f"  Processed: {len(result['processed'])} pages\n"
            f"  Failed: {len(result['failed'])} pages"
        )
        
        return result

async def main():
    logger = setup_logging()
    
    config = CrawlConfig(
        max_depth=3,
        supported_languages=['en', 'ar', 'ur']
    )
    
    crawler = RobustCrawler(config, logger=logger)
    await crawler.initialize()
    print("Crawling sites...")
    
    try:
        with open('urls.json') as f:
            urls_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading URLs configuration: {str(e)}")
        return

    # Perform crawling
    total_processed = 0
    total_failed = 0

    # use parallel crawling
    urls = [site['url'] for site in urls_config['sites']]
    crawl_results = await crawler.crawl_sites_parallel(urls)
    
    for url, result in crawl_results.items():
        total_processed += len(result['processed'])
        total_failed += len(result['failed'])

    logger.info(f"\nCrawling completed:")
    logger.info(f"Total pages processed: {total_processed}")
    logger.info(f"Total pages failed: {total_failed}")

    # Cleanup
    await crawler.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user")
    except Exception as e:
        print(f"Error during crawling: {str(e)}")
