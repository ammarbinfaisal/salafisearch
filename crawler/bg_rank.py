import networkx as nx
from threading import Thread, Lock
from queue import Queue
from typing import Dict, Set, Optional, List, Tuple
from urllib.parse import urlparse
import logging
import time
from collections import defaultdict

class BackgroundPageRankCalculator:
    def __init__(
        self,
        update_interval: int = 300,  # 5 minutes
        alpha: float = 0.85,
        max_iter: int = 100,
        min_nodes_for_update: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize background PageRank calculator
        
        Args:
            update_interval: Seconds between PageRank updates
            alpha: Damping parameter for PageRank
            max_iter: Maximum iterations for PageRank calculation
            min_nodes_for_update: Minimum number of new nodes before forcing update
            logger: Logger instance
        """
        self.update_interval = update_interval
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_nodes_for_update = min_nodes_for_update
        self.logger = logger or logging.getLogger(__name__)
        
        # Graph and scores
        self.graph = nx.DiGraph()
        self.scores: Dict[str, float] = {}
        self.domain_scores: Dict[str, float] = {}
        
        # Thread safety
        self.graph_lock = Lock()
        self.scores_lock = Lock()
        self.queue = Queue()
        self.pending_updates = set()
        
        # Control flags
        self.running = False
        self.worker_thread: Optional[Thread] = None

    def start(self) -> None:
        """Start the background PageRank calculation thread"""
        if not self.running:
            self.running = True
            self.worker_thread = Thread(target=self._background_worker, daemon=True)
            self.worker_thread.start()
            self.logger.info("Background PageRank calculator started")

    def stop(self) -> None:
        """Stop the background PageRank calculation thread"""
        if self.running:
            self.running = False
            if self.worker_thread:
                self.worker_thread.join()
            self.logger.info("Background PageRank calculator stopped")

    def _background_worker(self) -> None:
        """Background worker that periodically updates PageRank scores"""
        last_update = 0
        
        while self.running:
            current_time = time.time()
            force_update = len(self.pending_updates) >= self.min_nodes_for_update
            
            if force_update or (current_time - last_update >= self.update_interval):
                self._update_pagerank()
                last_update = current_time
                self.pending_updates.clear()
            
            time.sleep(1)  # Prevent tight loop

    def _update_pagerank(self) -> None:
        """Calculate and update PageRank scores"""
        try:
            with self.graph_lock:
                if self.graph.number_of_nodes() == 0:
                    return
                
                # Calculate PageRank
                new_scores = nx.pagerank(
                    self.graph,
                    alpha=self.alpha,
                    max_iter=self.max_iter
                )
                
                # Calculate domain scores
                new_domain_scores = defaultdict(float)
                for url, score in new_scores.items():
                    domain = urlparse(url).netloc
                    new_domain_scores[domain] += score
                
                # Update scores atomically
                with self.scores_lock:
                    self.scores = new_scores
                    self.domain_scores = dict(new_domain_scores)
                
                self.logger.debug(
                    f"Updated PageRank scores for {len(new_scores)} pages "
                    f"and {len(new_domain_scores)} domains"
                )
                
        except Exception as e:
            self.logger.error(f"Error updating PageRank: {str(e)}")

    def add_pages(self, urls: Set[str], links: Dict[str, Set[str]]) -> None:
        """
        Add new pages and their relationships to the graph
        
        Args:
            urls: Set of URLs to add
            links: Dictionary mapping URLs to their outbound links
        """
        try:
            with self.graph_lock:
                # Add nodes
                for url in urls:
                    self.graph.add_node(url, domain=urlparse(url).netloc)
                
                # Add edges
                for source, destinations in links.items():
                    for dest in destinations:
                        if dest in urls:  # Only add edges to known pages
                            self.graph.add_edge(source, dest)
                
                self.pending_updates.update(urls)
                
            self.logger.debug(f"Added {len(urls)} pages to graph")
            
        except Exception as e:
            self.logger.error(f"Error adding pages: {str(e)}")

    def get_scores(self, n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get current PageRank scores
        
        Args:
            n: Optional number of top scores to return
            
        Returns:
            List of (url, score) tuples, sorted by score descending
        """
        with self.scores_lock:
            sorted_scores = sorted(
                self.scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_scores[:n] if n else sorted_scores

    def get_domain_scores(self, n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get current domain scores
        
        Args:
            n: Optional number of top domain scores to return
            
        Returns:
            List of (domain, score) tuples, sorted by score descending
        """
        with self.scores_lock:
            sorted_scores = sorted(
                self.domain_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_scores[:n] if n else sorted_scores