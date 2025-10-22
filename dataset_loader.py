"""
WB AI Enterprise - Dataset Management & ChromaDB Integration
Handles loading, preprocessing, and embedding of knowledge sources
"""

import logging
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import chromadb
from chromadb.config import Settings
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

logger = logging.getLogger("WB.DataLoader")


class DatasetManager:
    """Manages dataset ingestion and ChromaDB operations"""
    
    DATASETS = {
        'swe_bench': {
            'name': 'princeton-nlp/SWE-bench_Verified',
            'split': 'test',
            'text_field': 'problem_statement',
            'metadata_fields': ['repo', 'instance_id', 'version'],
            'sample_limit': 500  # Limit for faster loading
        },
        'the_stack': {
            'name': 'bigcode/the-stack-v2-dedup',
            'split': 'train',
            'text_field': 'content',
            'metadata_fields': ['lang', 'max_stars_repo_name'],
            'sample_limit': 1000,
            'streaming': True,
            'filters': {'lang': ['Python', 'JavaScript', 'Rust', 'Go']}
        },
        'rstar_coder': {
            'name': 'microsoft/rStar-Coder',
            'split': 'train',
            'text_field': 'solution',
            'metadata_fields': ['difficulty', 'language'],
            'sample_limit': 500
        }
    }
    
    def __init__(
        self,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "wb_ai_knowledge",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 512
    ):
        self.chroma_dir = Path(chroma_dir)
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def is_initialized(self) -> bool:
        """Check if collection already has data"""
        count = self.collection.count()
        logger.info(f"Current collection count: {count}")
        return count > 0
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
    
    async def load_all_datasets(self):
        """Load all configured datasets into ChromaDB"""
        total_loaded = 0
        
        for dataset_key, config in self.DATASETS.items():
            try:
                logger.info(f"📥 Loading dataset: {config['name']}")
                count = await self._load_dataset(dataset_key, config)
                total_loaded += count
                logger.info(f"✅ Loaded {count} documents from {dataset_key}")
            except Exception as e:
                logger.error(f"❌ Failed to load {dataset_key}: {str(e)}")
                continue
        
        logger.info(f"🎉 Total documents loaded: {total_loaded}")
        return total_loaded
    
    async def _load_dataset(self, dataset_key: str, config: Dict[str, Any]) -> int:
        """Load a single dataset"""
        
        # Load dataset from HuggingFace
        try:
            if config.get('streaming'):
                dataset = load_dataset(
                    config['name'],
                    split=config['split'],
                    streaming=True
                )
                # Take limited samples for streaming datasets
                dataset = dataset.take(config['sample_limit'])
                data = list(dataset)
            else:
                dataset = load_dataset(
                    config['name'],
                    split=config['split']
                )
                # Sample if limit specified
                if config['sample_limit'] and len(dataset) > config['sample_limit']:
                    indices = np.random.choice(
                        len(dataset),
                        config['sample_limit'],
                        replace=False
                    )
                    data = [dataset[int(i)] for i in indices]
                else:
                    data = list(dataset)
        except Exception as e:
            logger.error(f"Failed to load dataset {config['name']}: {str(e)}")
            return 0
        
        # Prepare documents
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(data):
            try:
                # Extract text
                text = item.get(config['text_field'], '')
                if not text or len(text.strip()) < 10:
                    continue
                
                # Truncate long documents
                text = text[:5000]
                
                # Extract metadata
                metadata = {
                    'source': dataset_key,
                    'dataset': config['name']
                }
                for field in config['metadata_fields']:
                    if field in item:
                        metadata[field] = str(item[field])
                
                documents.append(text)
                metadatas.append(metadata)
                ids.append(f"{dataset_key}_{idx}")
                
            except Exception as e:
                logger.warning(f"Skipping item {idx}: {str(e)}")
                continue
        
        # Batch insert into ChromaDB
        if documents:
            await self._batch_insert(documents, metadatas, ids)
        
        return len(documents)
    
    async def _batch_insert(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Insert documents in batches with embeddings"""
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(documents), self.batch_size), desc="Embedding & Inserting"):
            batch_docs = documents[i:i + self.batch_size]
            batch_meta = metadatas[i:i + self.batch_size]
            batch_ids = ids[i:i + self.batch_size]
            
            # Generate embeddings in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self.embedding_model.encode,
                batch_docs
            )
            
            # Insert into ChromaDB
            self.collection.add(
                documents=batch_docs,
                embeddings=embeddings.tolist(),
                metadatas=batch_meta,
                ids=batch_ids
            )
    
    def search(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """Search ChromaDB collection"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters
        )
        
        # Format results
        formatted = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
