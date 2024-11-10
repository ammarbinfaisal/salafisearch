from dataclasses import dataclass
from typing import List, Optional
import logging
import time

@dataclass
class CrawlConfig:
    max_depth: int = 3
    max_pages_per_domain: int = 100000
    min_content_length: int = 100
    request_delay: float = 0
    timeout: int = 3000
    max_concurrent_requests_per_domain = 2
    max_retries: int = 2
    respect_robots: bool = True
    user_agents: List[str] = None
    excluded_patterns: List[str] = None
    supported_languages: List[str] = None
    elasticsearch_index: str = "enhanced"
    allowed_mime_types: List[str] = None
    
    def __post_init__(self):
        if self.user_agents is None:
            self.user_agents = [
                'Mozilla/5.0 (compatible; CustomBot/1.0; +http://example.com/bot)',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            ]
        if self.excluded_patterns is None:
            self.excluded_patterns = [
                r'\.(jpg|jpeg|png|gif|ico|css|js|doc|docx|zip|tar|gz|mp3|mp4|mkv)$',
                r'(calendar|login|signup|cart|checkout)',
                r'#.*$'
            ]
        if self.supported_languages is None:
            self.supported_languages = ['en', 'ar', "ur"]
        if self.allowed_mime_types is None:
            self.allowed_mime_types = ['text/html', 'application/pdf']

class PDFProcessor:
    """Helper class for processing PDF documents"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def extract_text_pdfminer(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PDFMiner"""
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.pdfparser import PDFSyntaxError
            from io import BytesIO
            return extract_text(BytesIO(pdf_content))
        except PDFSyntaxError as e:
            self.logger.error(f"PDFMiner extraction error: {str(e)}")
            return None

    def extract_text_pymupdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF (fallback method)"""
        try:
            import fitz
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
                tmp.write(pdf_content)
                tmp.flush()
                
                doc = fitz.open(tmp.name)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction error: {str(e)}")
            return None

    def extract_metadata(self, pdf_content: bytes) -> dict:
        """Extract PDF metadata using PyMuPDF"""
        try:
            import fitz
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
                tmp.write(pdf_content)
                tmp.flush()
                
                doc = fitz.open(tmp.name)
                metadata = doc.metadata
                doc.close()
                return metadata
        except Exception as e:
            self.logger.error(f"Metadata extraction error: {str(e)}")
            return {}

    def process_pdf(self, pdf_content: bytes) -> tuple[str, dict]:
        """
        Process PDF content and return extracted text and metadata
        Returns tuple of (text, metadata) or (None, {}) on failure
        """
        # Try PDFMiner first
        text = self.extract_text_pdfminer(pdf_content)
        
        # Fall back to PyMuPDF if PDFMiner fails
        if not text:
            text = self.extract_text_pymupdf(pdf_content)
        
        if not text:
            return None, {}
            
        metadata = self.extract_metadata(pdf_content)
        return text, metadata

def setup_logging():
    """Configure logging for the crawler"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crawler.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)