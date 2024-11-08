import networkx as nx
from typing import Dict, Set, List, Tuple
from urllib.parse import urlparse
import logging
from collections import defaultdict

class PageRankAnalyzer:
    def __init__(self, alpha: float = 0.85, max_iter: int = 100, logger: logging.Logger = None):
        """
        Initialize PageRank analyzer
        
        Args:
            alpha: Damping parameter (default: 0.85)
            max_iter: Maximum number of iterations (default: 100)
            logger: Logger instance
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.logger = logger or logging.getLogger(__name__)
        self.graph = nx.DiGraph()
        
    def add_pages(self, crawl_results: Dict[str, Dict[str, Set[str]]]) -> None:
        """
        Add pages and their relationships to the graph from crawler results
        
        Args:
            crawl_results: Dictionary mapping site URLs to their crawl results
                         Format: {site_url: {'processed': set(), 'failed': set()}}
        """
        try:
            # First pass: Add all nodes
            for site_results in crawl_results.values():
                for url in site_results['processed']:
                    self.graph.add_node(url, domain=urlparse(url).netloc)
            
            # Second pass: Add edges based on link structure
            for url, node_data in self.graph.nodes(data=True):
                domain = node_data['domain']
                # Add edges to all other pages in same domain
                # This assumes pages in same domain are linked (simplified model)
                for other_url, other_data in self.graph.nodes(data=True):
                    if other_url != url and other_data['domain'] == domain:
                        self.graph.add_edge(url, other_url)
            
            self.logger.info(f"Created graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            self.logger.error(f"Error building graph: {str(e)}")

    def calculate_pagerank(self) -> Dict[str, float]:
        """
        Calculate PageRank scores for all pages
        
        Returns:
            Dictionary mapping URLs to their PageRank scores
        """
        try:
            if self.graph.number_of_nodes() == 0:
                return {}
                
            pagerank_scores = nx.pagerank(
                self.graph,
                alpha=self.alpha,
                max_iter=self.max_iter
            )
            
            self.logger.info("PageRank calculation completed successfully")
            return pagerank_scores
        except Exception as e:
            self.logger.error(f"Error calculating PageRank: {str(e)}")
            return {}
    
    def get_top_pages(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N pages by PageRank score
        
        Args:
            n: Number of top pages to return (default: 10)
            
        Returns:
            List of tuples (url, score) sorted by score descending
        """
        try:
            pagerank_scores = self.calculate_pagerank()
            sorted_pages = sorted(
                pagerank_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_pages[:n]
        except Exception as e:
            self.logger.error(f"Error getting top pages: {str(e)}")
            return []
    
    def get_domain_scores(self) -> Dict[str, float]:
        """
        Calculate aggregate PageRank scores by domain
        
        Returns:
            Dictionary mapping domains to their aggregate scores
        """
        try:
            pagerank_scores = self.calculate_pagerank()
            domain_scores = defaultdict(float)
            
            for url, score in pagerank_scores.items():
                domain = urlparse(url).netloc
                domain_scores[domain] += score
            
            return dict(domain_scores)
        except Exception as e:
            self.logger.error(f"Error calculating domain scores: {str(e)}")
            return {}