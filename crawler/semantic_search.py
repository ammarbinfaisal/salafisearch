from elasticsearch import Elasticsearch
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import Dict, List
import time
from urllib.parse import urlparse

class SemanticSearchIndexer:
    def __init__(
        self, 
        es_client: Elasticsearch,
        index_name: str = "multilingual_content",
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        embedding_dim: int = 768,
        chunk_size: int = 512,
        batch_size: int = 8  # Added for batch embedding processing
    ):
        self.es = es_client
        self.index_name = index_name
        self.model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        self.chunk_size = chunk_size
        self.batch_size = batch_size  # For batch processing of embeddings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG for detailed logs

    def create_index(self) -> None:
        """Create Elasticsearch index with semantic search mapping"""
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(
                index=self.index_name,
                body={
                    "settings": {
                        "analysis": {
                            "filter": {
                                "arabic_stop": {
                                    "type": "stop",
                                    "stopwords": "_arabic_"
                                },
                                "arabic_stemmer": {
                                    "type": "stemmer",
                                    "language": "arabic"
                                },
                                "arabic_normalization": {
                                    "type": "arabic_normalization"
                                }
                            },
                            "analyzer": {
                                "arabic_analyzer": {
                                    "tokenizer": "standard",
                                    "filter": [
                                        "lowercase",
                                        "decimal_digit",
                                        "arabic_stop",
                                        "arabic_normalization",
                                        "arabic_stemmer"
                                    ]
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "url": {"type": "keyword"},
                            "original_language": {"type": "keyword"},
                            "title": {
                                "type": "object",
                                "properties": {
                                    "original": {"type": "text"},
                                    "translations": {
                                        "type": "object",
                                        "properties": {
                                            "en": {
                                                "type": "text",
                                                "analyzer": "english"
                                            },
                                            "ar": {
                                                "type": "text",
                                                "analyzer": "arabic_analyzer"
                                            }
                                        }
                                    },
                                    "vector": {
                                        "type": "dense_vector",
                                        "dims": self.embedding_dim,
                                        "index": True,
                                        "similarity": "cosine"
                                    }
                                }
                            },
                            "content": {
                                "type": "object",
                                "properties": {
                                    "original": {"type": "text"},
                                    "translations": {
                                        "type": "object",
                                        "properties": {
                                            "en": {
                                                "type": "text",
                                                "analyzer": "english"
                                            },
                                            "ar": {
                                                "type": "text",
                                                "analyzer": "arabic_analyzer"
                                            }
                                        }
                                    },
                                    "chunks": {
                                        "type": "nested",
                                        "properties": {
                                            "text": {"type": "text"},
                                            "vector": {
                                                "type": "dense_vector",
                                                "dims": self.embedding_dim,
                                                "index": True,
                                                "similarity": "cosine"
                                            }
                                        }
                                    }
                                }
                            },
                            "timestamp": {
                                "type": "date",
                                "format": "epoch_millis"
                            },
                            "domain": {"type": "keyword"}
                        }
                    }
                }
            )
            self.logger.info(f"Created index {self.index_name} with semantic search mapping")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of approximately equal size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_length += len(word) + 1
            if current_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word) + 1
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        self.logger.debug(f"Text chunked into {len(chunks)} parts")
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches to improve memory usage"""
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            try:
                batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
                embeddings.extend(batch_embeddings.tolist())
            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch: {str(e)}")
                embeddings.extend([np.zeros(self.embedding_dim).tolist()] * len(batch_texts))

        return embeddings

    def prepare_document(self, doc: Dict) -> Dict:
        """Prepare document with semantic embeddings"""
        try:
            prepared_doc = {
                'url': doc['url'],
                'timestamp': doc['timestamp'],
                'domain': doc['domain'],
                'id': doc.get('id'),
                'title': {
                    'original': doc['title']['original'],
                    'translations': doc['title'].get('translations', {}),
                    'vector': None
                },
                'content': {
                    'original': doc['content']['original'],
                    'translations': doc['content'].get('translations', {}),
                    'chunks': []
                }
            }
            
            title_embedding = self.generate_embeddings([doc['title']['original']])[0]
            prepared_doc['title']['vector'] = title_embedding

            content_chunks = self.chunk_text(doc['content']['original'])
            chunk_embeddings = self.generate_embeddings(content_chunks)

            prepared_doc['content']['chunks'] = [
                {'text': chunk, 'vector': embedding}
                for chunk, embedding in zip(content_chunks, chunk_embeddings)
            ]
            
            return prepared_doc
        except Exception as e:
            self.logger.error(f"Error preparing document: {str(e)}")
            raise

    def index_document(self, doc: Dict) -> bool:
        """Index a document with semantic search capabilities"""
        try:
            if doc.get('id'):
                try:
                    self.es.delete(index=self.index_name, id=doc['id'], ignore=[404])
                except Exception as e:
                    self.logger.debug(f"Error deleting existing document: {str(e)}")

            prepared_doc = self.prepare_document(doc)
            self.es.index(
                index=self.index_name,
                id=prepared_doc['id'],
                document=prepared_doc,
                refresh=True
            )
            self.logger.info(f"Indexed document with ID {prepared_doc['id']}")
            return True
        except Exception as e:
            self.logger.error(f"Error indexing document: {str(e)}")
            return False

    def search_similar(self, query: str, size: int = 10, title_weight: float = 1.5, content_weight: float = 1.0) -> List[Dict]:
        """Search for similar documents using weighted semantic search"""
        try:
            query_vector = self.generate_embeddings([query])[0]

            response = self.es.search(
                index=self.index_name,
                body={
                    "size": size,
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"{title_weight} * cosineSimilarity(params.query_vector, 'title.vector') + {content_weight} * cosineSimilarity(params.query_vector, 'content.chunks.vector')",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                }
            )

            self.logger.info(f"Search for query '{query}' returned {len(response['hits']['hits'])} results")
            return [hit['_source'] for hit in response['hits']['hits']]
        except Exception as e:
            self.logger.error(f"Error performing semantic search: {str(e)}")
            return []
