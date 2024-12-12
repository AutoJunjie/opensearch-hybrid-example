import pandas as pd
from opensearchpy import OpenSearch, RequestsHttpConnection
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables upfront
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global environment variables
OPENSEARCH_URL = os.getenv('OPENSEARCH_URL')
OPENSEARCH_USERNAME = os.getenv('OPENSEARCH_USERNAME')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD')

class BiomedicalSearch:
    def __init__(self):
        self.index_name = "medical_articles10"
        self.search_pipeline_id = "nlp-search-pipeline"
        
        # Initialize OpenSearch client
        self.opensearch = OpenSearch(
            hosts=[{'host': OPENSEARCH_URL, 'port': 443}],
            http_auth=(OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD),
            use_ssl=True,
            connection_class=RequestsHttpConnection
        )

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using sentence transformer"""
        try:
            #model = SentenceTransformer('GanjinZero/coder_all')
            model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
            embeddings_1 = model.encode([text], normalize_embeddings=True)
            return embeddings_1[0].tolist()
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def create_search_pipeline(self):
        """Create search pipeline for hybrid search normalization"""
        pipeline_config = {
            "description": "Search pipeline for hybrid search normalization",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {
                            "technique": "min_max"
                        },
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {
                                "weights": [0.3, 0.7]  # 30% keyword, 70% semantic
                            }
                        }
                    }
                }
            ]
        }
        
        try:
            self.opensearch.transport.perform_request(
                'PUT',
                f'/_search/pipeline/{self.search_pipeline_id}',
                body=pipeline_config
            )
            print(f"Search pipeline {self.search_pipeline_id} created successfully")
        except Exception as e:
            print(f"Error creating search pipeline: {e}")

    def create_index_mapping(self):
        """Create index with appropriate mappings"""
        mapping = {
            "settings": {
                "index.knn": True,
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "keywords": {"type": "text", "analyzer": "standard"},
                    "abstract": {"type": "text", "analyzer": "standard"},
                    "abstract_embedding": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "lucene"
                        }
                    }
                }
            }
        }
        
        try:
            if self.opensearch.indices.exists(index=self.index_name):
                self.opensearch.indices.delete(index=self.index_name)
                print(f"Existing index {self.index_name} deleted")
            
            self.opensearch.indices.create(
                index=self.index_name,
                body=mapping
            )
            print(f"Index {self.index_name} created successfully")
        except Exception as e:
            print(f"Error creating index: {e}")

    def ingest_data(self, csv_path: str):
        """Ingest data from CSV"""
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            embedding = self.get_embedding(row['abstract'])
            
            if embedding:
                keywords = row['keyword1']
                if pd.notna(row['keyword2']):
                    keywords += f" {row['keyword2']}"
                
                document = {
                    "id": row['id'],
                    "keywords": keywords,
                    "abstract": row['abstract'],
                    "abstract_embedding": embedding
                }
                
                try:
                    self.opensearch.index(
                        index=self.index_name,
                        body=document,
                        id=str(row['id'])
                    )
                    print(f"Indexed document {row['id']}")
                except Exception as e:
                    print(f"Error indexing document {row['id']}: {e}")

    def hybrid_search(self, query: str, k: int = 5):
        """Perform hybrid search using search pipeline"""
        query_embedding = self.get_embedding(query)
        
        if not query_embedding:
            return []
        
        search_query = {
            "size": k,
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "keywords": {
                                    "query": query
                                }
                            }
                        },
                        {
                            "knn": {
                                "abstract_embedding": {
                                    "vector": query_embedding,
                                    "k": k
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        try:
            response = self.opensearch.search(
                index=self.index_name,
                body=search_query,
                params={"search_pipeline": self.search_pipeline_id}
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_source']['id'],
                    'keywords': hit['_source']['keywords'],
                    'abstract': hit['_source']['abstract'],
                    'score': hit['_score']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error performing search: {e}")
            return []

    def vector_search(self, query: str, k: int = 5):
        """Perform vector search using only semantic embeddings"""
        query_embedding = self.get_embedding(query)
        
        if not query_embedding:
            return []
        
        search_query = {
            "size": k,
            "query": {
                "knn": {
                    "abstract_embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            }
        }
        
        try:
            response = self.opensearch.search(
                index=self.index_name,
                body=search_query,
                params={"search_pipeline": self.search_pipeline_id}
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_source']['id'],
                    'keywords': hit['_source']['keywords'],
                    'abstract': hit['_source']['abstract'],
                    'score': hit['_score']
                }
                results.append(result)
            
            return results
                
        except Exception as e:
            print(f"Error performing vector search: {e}")
            return []

    def keyword_search(self, query: str, k: int = 5):
        """Perform keyword search using only text matching"""
        search_query = {
            "size": k,
            "query": {
                "match": {
                    "keywords": {
                        "query": query
                    }
                }
            }
        }
        
        try:
            response = self.opensearch.search(
                index=self.index_name,
                body=search_query,
                params={"search_pipeline": self.search_pipeline_id}
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    'id': hit['_source']['id'],
                    'keywords': hit['_source']['keywords'],
                    'abstract': hit['_source']['abstract'],
                    'score': hit['_score']
                }
                results.append(result)
            
            return results
                
        except Exception as e:
            print(f"Error performing keyword search: {e}")
            return []

def main():
    # Initialize the search system
    search_system = BiomedicalSearch()
    
    # Set up pipelines and index
    search_system.create_search_pipeline()
    search_system.create_index_mapping()
    
    # Ingest data
    search_system.ingest_data('sample.csv')
    
    # Example searches
    test_queries = [
        "重组人黄体生成素",
        "胚胎移植",
    ]
    
    # Perform different types of searches
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"查询词: {query}")
        
        #print("\n1. 混合搜索结果:")
        #hybrid_results = search_system.hybrid_search(query)
        #for i, result in enumerate(hybrid_results, 1):
        #    print(f"\n{i}. 文档ID: {result['id']}")
        #    print(f"得分: {result['score']}")
        #    print(f"摘要: {result['abstract']}")
        
        print("\n2. 纯向量搜��结果:")
        vector_results = search_system.vector_search(query)
        for i, result in enumerate(vector_results, 1):
            print(f"\n{i}. 文档ID: {result['id']}")
            print(f"得分: {result['score']}")
            print(f"摘要: {result['abstract']}")
        #
        #print("\n3. 纯关键词搜索结果:")
        #keyword_results = search_system.keyword_search(query)
        ##print(keyword_results)
        #for i, result in enumerate(keyword_results, 1):
        #    print(f"\n{i}. 文档ID: {result['id']}")
        #    #print(f"关键词: {result['keywords']}")
        #    print(f"得分: {result['score']}")
        #    print(f"摘要: {result['abstract']}")
            
if __name__ == "__main__":
    main()