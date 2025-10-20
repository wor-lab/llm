"""
WB AI Corporation - Dataset Management System
Loads and processes HuggingFace datasets into ChromaDB
"""

from datasets import load_dataset
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

class DatasetManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Dataset configurations
        self.datasets_config = {
            "swe_bench": {
                "name": "princeton-nlp/SWE-bench_Verified",
                "split": "test",
                "collection": "swe_bench",
                "fields": ["problem_statement", "hints", "created_at"]
            },
            "humaneval": {
                "name": "openai/openai_humaneval",
                "split": "test", 
                "collection": "humaneval",
                "fields": ["prompt", "canonical_solution", "test"]
            },
            "mbpp": {
                "name": "google-research-datasets/mbpp",
                "split": "test",
                "collection": "mbpp",
                "fields": ["text", "code", "test_list"]
            },
            "bigcodebench": {
                "name": "bigcode/bigcodebench",
                "split": "v0.1.2",
                "collection": "bigcodebench",
                "fields": ["instruct_prompt", "canonical_solution"]
            },
            "stack_v2": {
                "name": "bigcode/the-stack-v2-train-smol-ids",
                "split": "train",
                "collection": "stack_v2",
                "fields": ["content"],
                "sample_size": 10000  # Sample for efficiency
            }
        }
        
    def load_all_datasets(self):
        """Load all configured datasets in parallel"""
        print("[DATA] Initializing WB AI Knowledge Base...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for dataset_key, config in self.datasets_config.items():
                future = executor.submit(self._load_dataset, dataset_key, config)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"[DATA] {result}")
                except Exception as e:
                    print(f"[DATA] Error: {e}")
                    
    def _load_dataset(self, key: str, config: Dict) -> str:
        """Load individual dataset into ChromaDB collection"""
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=config["collection"],
                embedding_function=self.embedding_fn
            )
            
            # Check if already loaded
            if collection.count() > 0:
                return f"{key}: Already loaded ({collection.count()} documents)"
            
            # Load from HuggingFace
            print(f"[DATA] Loading {config['name']}...")
            dataset = load_dataset(
                config["name"],
                split=config["split"],
                streaming=True if "sample_size" in config else False
            )
            
            documents = []
            metadatas = []
            ids = []
            
            # Process dataset
            sample_size = config.get("sample_size", float('inf'))
            for idx, item in enumerate(tqdm(dataset, desc=key, total=min(sample_size, 1000))):
                if idx >= sample_size:
                    break
                    
                # Build document from available fields
                doc_parts = []
                metadata = {"source": key, "index": idx}
                
                for field in config["fields"]:
                    if field in item and item[field]:
                        value = item[field]
                        if isinstance(value, list):
                            value = "\n".join(map(str, value))
                        doc_parts.append(f"{field}: {value}")
                        metadata[field[:50]] = str(value)[:500]  # Truncate for metadata
                
                if doc_parts:
                    document = "\n".join(doc_parts)
                    documents.append(document)
                    metadatas.append(metadata)
                    ids.append(hashlib.md5(document.encode()).hexdigest())
                
                # Batch insert
                if len(documents) >= 100:
                    collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents, metadatas, ids = [], [], []
            
            # Final batch
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            return f"{key}: Loaded {collection.count()} documents"
            
        except Exception as e:
            return f"{key}: Failed - {str(e)}"
            
    def search(self, query: str, collection_name: str = None, n_results: int = 5) -> List[Dict]:
        """Search across collections"""
        results = []
        
        collections = [collection_name] if collection_name else [
            c["collection"] for c in self.datasets_config.values()
        ]
        
        for coll_name in collections:
            try:
                collection = self.chroma_client.get_collection(coll_name)
                res = collection.query(
                    query_texts=[query],
                    n_results=n_results
                )
                
                for i in range(len(res["documents"][0])):
                    results.append({
                        "collection": coll_name,
                        "document": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "distance": res["distances"][0][i]
                    })
            except:
                continue
                
        return sorted(results, key=lambda x: x["distance"])[:n_results]
