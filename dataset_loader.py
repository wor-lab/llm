"""
WB AI Corporation - Dataset Loader Module
Handles loading, processing, and vectorization of HuggingFace datasets.
Architecture: Data Access Layer with streaming optimization.
"""

import os
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, Dataset, IterableDataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import chromadb
from chromadb.config import Settings


@dataclass
class DatasetConfig:
    """Dataset configuration with validation."""
    name: str
    split: str = "train"
    max_samples: int = 5000
    text_field: Optional[str] = None
    streaming: bool = True


class OptimizedDatasetLoader:
    """
    Enterprise-grade dataset loader with:
    - Streaming for memory efficiency
    - Parallel processing
    - Automatic text field detection
    - Error recovery
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "wb_ai_codebase",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        device: str = "cuda"
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        logger.info(f"Initializing embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vectorstore: Optional[Chroma] = None
        logger.success("DatasetLoader initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def load_dataset_stream(self, config: DatasetConfig) -> Iterator[Dict[str, Any]]:
        """Load dataset with streaming and retry logic."""
        logger.info(f"Loading dataset: {config.name} (streaming={config.streaming})")
        
        try:
            dataset = load_dataset(
                config.name,
                split=config.split,
                streaming=config.streaming,
                trust_remote_code=True
            )
            
            count = 0
            for item in dataset:
                if count >= config.max_samples:
                    break
                yield item
                count += 1
                
            logger.success(f"Loaded {count} samples from {config.name}")
            
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {e}")
            raise

    def _detect_text_field(self, sample: Dict[str, Any]) -> Optional[str]:
        """Auto-detect the main text field in dataset samples."""
        priority_fields = [
            "text", "content", "code", "solution", "problem_statement",
            "instruction", "response", "patch", "instance_id"
        ]
        
        for field in priority_fields:
            if field in sample and isinstance(sample[field], str):
                return field
        
        # Fallback: first string field
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 50:
                return key
        
        return None

    def process_dataset_to_documents(self, config: DatasetConfig) -> List[Document]:
        """Convert dataset to LangChain documents with metadata."""
        documents = []
        
        for idx, sample in enumerate(self.load_dataset_stream(config)):
            text_field = config.text_field or self._detect_text_field(sample)
            
            if not text_field:
                logger.warning(f"No text field found in sample {idx}")
                continue
            
            text = sample[text_field]
            if not text or len(text) < 10:
                continue
            
            metadata = {
                "source": config.name,
                "index": idx,
                "field": text_field,
                **{k: v for k, v in sample.items() if k != text_field and isinstance(v, (str, int, float, bool))}
            }
            
            documents.append(Document(page_content=text, metadata=metadata))
        
        logger.info(f"Processed {len(documents)} documents from {config.name}")
        return documents

    def build_vectorstore(self, datasets_config: List[DatasetConfig]) -> Chroma:
        """Build ChromaDB vectorstore from multiple datasets with parallel processing."""
        all_documents = []
        
        # Parallel dataset loading
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.process_dataset_to_documents, config): config
                for config in datasets_config
            }
            
            for future in as_completed(futures):
                config = futures[future]
                try:
                    docs = future.result()
                    all_documents.extend(docs)
                except Exception as e:
                    logger.error(f"Failed processing {config.name}: {e}")
        
        logger.info(f"Total documents collected: {len(all_documents)}")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(all_documents)
        logger.success(f"Created {len(split_docs)} chunks")
        
        # Build ChromaDB vectorstore with batching
        logger.info("Building ChromaDB vectorstore...")
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name
        )
        
        logger.success(f"Vectorstore built: {len(split_docs)} chunks indexed")
        return self.vectorstore

    def get_vectorstore(self) -> Chroma:
        """Get or load existing vectorstore."""
        if self.vectorstore:
            return self.vectorstore
        
        if (self.persist_dir / "chroma.sqlite3").exists():
            logger.info("Loading existing vectorstore...")
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.success("Vectorstore loaded from disk")
        else:
            logger.warning("No vectorstore found. Build one first.")
        
        return self.vectorstore


def initialize_datasets() -> OptimizedDatasetLoader:
    """Factory function to initialize dataset loader with env configs."""
    from dotenv import load_dotenv
    load_dotenv()
    
    loader = OptimizedDatasetLoader(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "wb_ai_codebase"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", 1024)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 128)),
        device=os.getenv("DEVICE", "cuda")
    )
    
    configs = [
        DatasetConfig(
            name=os.getenv("DATASET_1", "princeton-nlp/SWE-bench_Verified"),
            max_samples=int(os.getenv("DATASET_MAX_SAMPLES", 5000))
        ),
        DatasetConfig(
            name=os.getenv("DATASET_2", "bigcode/the-stack-v2-dedup"),
            max_samples=int(os.getenv("DATASET_MAX_SAMPLES", 5000))
        ),
        DatasetConfig(
            name=os.getenv("DATASET_3", "microsoft/rStar-Coder"),
            max_samples=int(os.getenv("DATASET_MAX_SAMPLES", 5000))
        )
    ]
    
    # Check if vectorstore exists, otherwise build
    if not (Path(loader.persist_dir) / "chroma.sqlite3").exists():
        logger.info("Building new vectorstore...")
        loader.build_vectorstore(configs)
    else:
        logger.info("Using existing vectorstore")
        loader.get_vectorstore()
    
    return loader
