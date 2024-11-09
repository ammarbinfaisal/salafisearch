from sentence_transformers import SentenceTransformer
import torch
from elasticsearch import AsyncElasticsearch
from typing import Dict, Optional, List

class EnhancedMultilingualIndexer:
    def __init__(self, es_client: AsyncElasticsearch, index_name: str = "enhanced"):
        self.es = es_client
        self.index_name = index_name
        
        # Keep the multilingual model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',
                                        device='cuda' if torch.cuda.is_available() else 'cpu')

    def get_embeddings(self, text: str) -> Optional[list]:
        """Generate embeddings for the given text"""
        if not text:
            return None
        
        try:
            embeddings = self.model.encode(text, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return None

    def chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into chunks of approximately equal size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_size += len(word) + 1  # +1 for space
            if current_size > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def prepare_document(self, doc: Dict) -> Optional[Dict]:
        """Prepare document with embeddings for both original and translations"""
        try:
            # Initialize the document structure
            prepared_doc = {
                "url": doc.get("url"),
                "original_language": doc.get("original_language"),
                "domain": doc.get("domain"),
                "id": doc.get("id"),
                "timestamp": doc.get("timestamp"),
                "title": {
                    "original": doc.get("title", {}).get("original"),
                    "translations": doc.get("title", {}).get("translations", {}),
                    "vector": self.get_embeddings(doc.get("title", {}).get("original", ""))
                },
                "content": {
                    "original": doc.get("content", {}).get("original"),
                    "translations": doc.get("content", {}).get("translations", {}),
                    "chunks": []
                }
            }

            # Process original content
            original_content = doc.get("content", {}).get("original", "")
            original_chunks = self.chunk_text(original_content)
            
            # Process translations
            translations = doc.get("content", {}).get("translations", {})
            
            # Create chunks for each language version
            for chunk_idx, original_chunk in enumerate(original_chunks):
                chunk_entry = {
                    "text": original_chunk,
                    "vector": self.get_embeddings(original_chunk),
                    "translations": {}
                }
                
                # Add translations for this chunk
                for lang, translated_content in translations.items():
                    translated_chunks = self.chunk_text(translated_content)
                    if chunk_idx < len(translated_chunks):
                        chunk_entry["translations"][lang] = {
                            "text": translated_chunks[chunk_idx],
                            "vector": self.get_embeddings(translated_chunks[chunk_idx])
                        }
                
                prepared_doc["content"]["chunks"].append(chunk_entry)
            
            return prepared_doc
        except Exception as e:
            print(f"Error preparing document: {str(e)}")
            return None

    async def create_index(self) -> bool:
        """Create the Elasticsearch index with appropriate mappings"""
        mappings = {
            "mappings": {
                "properties": {
                    "url": {"type": "keyword"},
                    "original_language": {"type": "keyword"},
                    "domain": {"type": "keyword"},
                    "id": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "timestamp": {"type": "date", "format": "epoch_millis"},
                    "title": {
                        "properties": {
                            "original": {"type": "text"},
                            "translations": {
                                "properties": {
                                    "en": {"type": "text", "analyzer": "english"},
                                    "ar": {"type": "text", "analyzer": "arabic_analyzer"}
                                }
                            },
                            "vector": {
                                "type": "dense_vector",
                                "dims": 384,
                                "index": True,
                                "similarity": "cosine",
                                "index_options": {
                                    "type": "int8_hnsw",
                                    "m": 16,
                                    "ef_construction": 100
                                }
                            }
                        }
                    },
                    "content": {
                        "properties": {
                            "original": {"type": "text"},
                            "translations": {
                                "properties": {
                                    "en": {"type": "text", "analyzer": "english"},
                                    "ar": {"type": "text", "analyzer": "arabic_analyzer"}
                                }
                            },
                            "chunks": {
                                "type": "nested",
                                "properties": {
                                    "text": {"type": "text"},
                                    "vector": {
                                        "type": "dense_vector",
                                        "dims": 384,
                                        "index": True,
                                        "similarity": "cosine",
                                        "index_options": {
                                            "type": "int8_hnsw",
                                            "m": 16,
                                            "ef_construction": 100
                                        }
                                    },
                                    "translations": {
                                        "properties": {
                                            "en": {
                                                "properties": {
                                                    "text": {"type": "text", "analyzer": "english"},
                                                    "vector": {
                                                        "type": "dense_vector",
                                                        "dims": 384,
                                                        "index": True,
                                                        "similarity": "cosine",
                                                        "index_options": {
                                                            "type": "int8_hnsw",
                                                            "m": 16,
                                                            "ef_construction": 100
                                                        }
                                                    }
                                                }
                                            },
                                            "ar": {
                                                "properties": {
                                                    "text": {"type": "text", "analyzer": "arabic_analyzer"},
                                                    "vector": {
                                                        "type": "dense_vector",
                                                        "dims": 384,
                                                        "index": True,
                                                        "similarity": "cosine",
                                                        "index_options": {
                                                            "type": "int8_hnsw",
                                                            "m": 16,
                                                            "ef_construction": 100
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                    "analysis": {
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
                        },
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
                        }
                    }
                }
            }
        }
        
        try:
            exists = await self.es.indices.exists(index=self.index_name)
            if not exists:
                await self.es.indices.create(
                    index=self.index_name,
                    body=mappings,
                )
                print(f"Created index {self.index_name}")
            return True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False

    async def search_similar(self, query_text: str, language: str = 'original', size: int = 10) -> list:
        """Search for similar documents using vector similarity on chunks"""
        try:
            query_vector = self.get_embeddings(query_text)
            if not query_vector:
                return []

            # Construct nested query for chunks
            search_query = {
                "size": size,
                "query": {
                    "nested": {
                        "path": "content.chunks",
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, " + 
                                             ("doc['content.chunks.vector']" if language == 'original' else 
                                              f"doc['content.chunks.translations.{language}.vector']") + ") + 1.0",
                                    "params": {
                                        "query_vector": query_vector
                                    }
                                }
                            }
                        },
                        "inner_hits": {
                            "size": 1,
                            "_source": ["content.chunks.text", f"content.chunks.translations.{language}.text"]
                        }
                    }
                }
            }

            results = await self.es.search(
                index=self.index_name,
                body=search_query
            )

            return [{
                'id': hit['_id'],
                'score': hit['_score'],
                'chunk': hit['inner_hits']['content.chunks']['hits']['hits'][0]['_source'],
                'url': hit['_source']['url']
            } for hit in results['hits']['hits']]

        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []

    async def delete_index(self) -> bool:
        """Delete the index if it exists"""
        try:
            exists = await self.es.indices.exists(index=self.index_name)
            if exists:
                await self.es.indices.delete(index=self.index_name)
                print(f"Deleted index {self.index_name}")
            return True
        except Exception as e:
            print(f"Error deleting index: {str(e)}")
            return False

    async def index_document(self, doc: Dict, doc_id: Optional[str] = None) -> bool:
        """
        Index a document into Elasticsearch.
        
        Args:
            doc (Dict): The document to index
            doc_id (Optional[str]): Optional custom document ID. If not provided, Elasticsearch will generate one.
            
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        try:
            # First ensure the index exists
            index_exists = await self.es.indices.exists(index=self.index_name)
            if not index_exists:
                created = await self.create_index()
                if not created:
                    return False
            
            # Prepare the document with embeddings
            prepared_doc = self.prepare_document(doc)
            if not prepared_doc:
                print("Failed to prepare document")
                return False
                
            # Index the document
            if doc_id:
                await self.es.index(
                    index=self.index_name,
                    id=doc_id,
                    document=prepared_doc,
                    refresh=True  # Make the document immediately searchable
                )
            else:
                await self.es.index(
                    index=self.index_name,
                    document=prepared_doc,
                    refresh=True
                )
                
            return True
            
        except Exception as e:
            print(f"Error indexing document: {str(e)}")
            return False