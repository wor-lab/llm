"""
WB AI CORPORATION - Data Pipeline
Real Dataset Integration from HuggingFace
═══════════════════════════════════════════
"""

from datasets import load_dataset
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import json
from tqdm import tqdm

class DataPipeline:
    """Manages dataset loading and vector store operations"""
    
    def __init__(self, persist_dir: str = "./chromadb"):
        self.persist_dir = persist_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        self.collections = {}
        
        # Dataset configurations
        self.datasets_config = {
            'humaneval': {
                'path': 'openai/openai_humaneval',
                'split': 'test',
                'text_field': 'prompt',
                'metadata_fields': ['task_id', 'canonical_solution', 'test', 'entry_point']
            },
            'mbpp': {
                'path': 'google-research-datasets/mbpp',
                'split': 'train',
                'text_field': 'text',
                'metadata_fields': ['task_id', 'code', 'test_list']
            },
            'mbpp_sanitized': {
                'path': 'Muennighoff/mbpp',
                'split': 'sanitized',
                'text_field': 'text',
                'metadata_fields': ['task_id', 'code', 'test_list']
            },
            'the_stack': {
                'path': 'bigcode/the-stack-v2',
                'split': 'train',
                'text_field': 'content',
                'metadata_fields': ['repo_name', 'path', 'language'],
                'streaming': True,
                'max_samples': 5000  # Limit for Colab
            }
        }
    
    def load_datasets(self):
        """Load datasets from HuggingFace"""
        print("📥 Loading datasets from HuggingFace...")
        
        for name, config in self.datasets_config.items():
            try:
                print(f"\n  → Loading {name}...")
                
                if config.get('streaming'):
                    dataset = load_dataset(
                        config['path'],
                        split=config['split'],
                        streaming=True,
                        trust_remote_code=True
                    )
                    # Take limited samples for streaming datasets
                    dataset = list(dataset.take(config.get('max_samples', 1000)))
                else:
                    dataset = load_dataset(
                        config['path'],
                        split=config['split'],
                        trust_remote_code=True
                    )
                
                print(f"    ✓ Loaded {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples")
                
                # Create ChromaDB collection
                self._create_collection(name, dataset, config)
                
            except Exception as e:
                print(f"    ⚠ Warning: Could not load {name}: {e}")
                continue
        
        print("\n✅ Dataset loading complete")
    
    def _create_collection(self, name: str, dataset, config: dict):
        """Create and populate ChromaDB collection"""
        
        # Get or create collection
        try:
            collection = self.client.get_collection(name)
            print(f"    → Using existing collection: {name}")
        except:
            collection = self.client.create_collection(
                name=name,
                metadata={"description": f"WB AI - {name} dataset"}
            )
            print(f"    → Created new collection: {name}")
        
        self.collections[name] = collection
        
        # Check if already populated
        if collection.count() > 0:
            print(f"    → Collection already contains {collection.count()} items")
            return
        
        # Prepare data for insertion
        documents = []
        metadatas = []
        ids = []
        
        text_field = config['text_field']
        metadata_fields = config['metadata_fields']
        
        for idx, item in enumerate(tqdm(dataset[:1000] if hasattr(dataset, '__getitem__') else dataset[:1000], desc=f"    Processing {name}")):
            # Extract text
            text = str(item.get(text_field, ''))
            if not text or len(text) < 10:
                continue
            
            documents.append(text)
            
            # Extract metadata
            metadata = {
                'source': name,
                'index': idx
            }
            for field in metadata_fields:
                if field in item:
                    value = item[field]
                    # Convert to string if not JSON-serializable
                    if isinstance(value, (list, dict)):
                        metadata[field] = json.dumps(value)
                    else:
                        metadata[field] = str(value)
            
            metadatas.append(metadata)
            ids.append(f"{name}_{idx}")
        
        # Batch insert into ChromaDB
        if documents:
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                collection.add(
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size],
                    ids=ids[i:i+batch_size]
                )
            
            print(f"    ✓ Inserted {len(documents)} documents into {name}")
    
    def create_embeddings(self):
        """Embeddings are handled automatically by ChromaDB"""
        print("🔄 ChromaDB uses default embeddings (can be customized)")
        print(f"✅ Collections ready: {list(self.collections.keys())}")
    
    def search(self, query: str, collection_name: str = None, n_results: int = 5) -> List[Dict]:
        """Search across collections"""
        
        if collection_name:
            collections = [self.collections[collection_name]]
        else:
            collections = self.collections.values()
        
        all_results = []
        
        for collection in collections:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            for i in range(len(results['documents'][0])):
                all_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
        
        # Sort by distance
        all_results.sort(key=lambda x: x['distance'])
        
        return all_results[:n_results]
    
    def get_collection_stats(self):
        """Get statistics for all collections"""
        stats = {}
        for name, collection in self.collections.items():
            stats[name] = {
                'count': collection.count(),
                'name': name
            }
        return stats
