"""
WB AI CORPORATION — DATA DIVISION
Dataset Ingestion Pipeline: HuggingFace → ChromaDB

MISSION: Load real code datasets into vector database
AGENT: DataSynth
"""

import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages HuggingFace dataset ingestion to ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Dataset configurations
        self.dataset_configs = {
            'humaneval': {
                'path': 'openai/openai_humaneval',
                'split': 'test',
                'text_field': 'prompt',
                'metadata_fields': ['task_id', 'canonical_solution', 'entry_point']
            },
            'mbpp': {
                'path': 'google-research-datasets/mbpp',
                'split': 'train',
                'text_field': 'text',
                'metadata_fields': ['task_id', 'code', 'test_list']
            },
            'swe_bench': {
                'path': 'princeton-nlp/SWE-bench_Verified',
                'split': 'test',
                'text_field': 'problem_statement',
                'metadata_fields': ['instance_id', 'repo', 'base_commit']
            },
            'bigcodebench': {
                'path': 'bigcode/bigcodebench',
                'split': 'v0.1.2',
                'text_field': 'instruct_prompt',
                'metadata_fields': ['task_id', 'complete_prompt', 'code']
            }
        }
    
    def create_collection(self, name: str) -> chromadb.Collection:
        """Create or get ChromaDB collection"""
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            raise
    
    def load_humaneval(self):
        """Load OpenAI HumanEval dataset"""
        logger.info("📥 Loading HumanEval dataset...")
        
        dataset = load_dataset('openai/openai_humaneval', split='test')
        collection = self.create_collection('humaneval')
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(tqdm(dataset)):
            doc_text = f"{item['prompt']}\n\n# Solution:\n{item.get('canonical_solution', '')}"
            documents.append(doc_text)
            
            metadatas.append({
                'task_id': item['task_id'],
                'entry_point': item['entry_point'],
                'source': 'humaneval',
                'test': item.get('test', '')
            })
            
            ids.append(f"humaneval_{idx}")
        
        # Batch insert
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            embeddings = self.embedder.encode(batch_docs).tolist()
            
            collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids
            )
        
        logger.info(f"✅ Loaded {len(documents)} HumanEval samples")
    
    def load_mbpp(self):
        """Load MBPP dataset"""
        logger.info("📥 Loading MBPP dataset...")
        
        dataset = load_dataset('google-research-datasets/mbpp', 'sanitized', split='train')
        collection = self.create_collection('mbpp')
        
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(tqdm(dataset)):
            doc_text = f"Task: {item['text']}\n\nCode:\n{item['code']}"
            documents.append(doc_text)
            
            metadatas.append({
                'task_id': str(item['task_id']),
                'source': 'mbpp',
                'test_cases': json.dumps(item.get('test_list', []))
            })
            
            ids.append(f"mbpp_{idx}")
        
        # Batch insert
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            embeddings = self.embedder.encode(batch_docs).tolist()
            
            collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids
            )
        
        logger.info(f"✅ Loaded {len(documents)} MBPP samples")
    
    def load_swe_bench(self):
        """Load SWE-bench Verified dataset"""
        logger.info("📥 Loading SWE-bench Verified...")
        
        try:
            dataset = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
            collection = self.create_collection('swe_bench')
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, item in enumerate(tqdm(dataset)):
                doc_text = f"Issue: {item['problem_statement']}\n\nRepository: {item['repo']}"
                documents.append(doc_text)
                
                metadatas.append({
                    'instance_id': item['instance_id'],
                    'repo': item['repo'],
                    'base_commit': item.get('base_commit', ''),
                    'source': 'swe_bench'
                })
                
                ids.append(f"swe_{idx}")
            
            # Batch insert
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                embeddings = self.embedder.encode(batch_docs).tolist()
                
                collection.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
            
            logger.info(f"✅ Loaded {len(documents)} SWE-bench samples")
        except Exception as e:
            logger.warning(f"⚠️ SWE-bench load failed: {e}")
    
    def load_bigcodebench(self):
        """Load BigCodeBench dataset"""
        logger.info("📥 Loading BigCodeBench...")
        
        try:
            dataset = load_dataset('bigcode/bigcodebench', split='v0.1.2')
            collection = self.create_collection('bigcodebench')
            
            documents = []
            metadatas = []
            ids = []
            
            for idx, item in enumerate(tqdm(dataset)):
                doc_text = item.get('instruct_prompt', item.get('complete_prompt', ''))
                documents.append(doc_text)
                
                metadatas.append({
                    'task_id': item.get('task_id', f'task_{idx}'),
                    'source': 'bigcodebench'
                })
                
                ids.append(f"bcb_{idx}")
            
            # Batch insert
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                embeddings = self.embedder.encode(batch_docs).tolist()
                
                collection.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
            
            logger.info(f"✅ Loaded {len(documents)} BigCodeBench samples")
        except Exception as e:
            logger.warning(f"⚠️ BigCodeBench load failed: {e}")
    
    def load_all_datasets(self):
        """Load all configured datasets"""
        logger.info("🚀 Starting full dataset ingestion pipeline...")
        
        self.load_humaneval()
        self.load_mbpp()
        self.load_swe_bench()
        self.load_bigcodebench()
        
        logger.info("✅ All datasets loaded successfully")
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded datasets"""
        stats = {}
        collections = self.client.list_collections()
        
        for col in collections:
            stats[col.name] = col.count()
        
        return stats

if __name__ == "__main__":
    dm = DatasetManager()
    dm.load_all_datasets()
    print("\n📊 Dataset Statistics:")
    print(json.dumps(dm.get_stats(), indent=2))
