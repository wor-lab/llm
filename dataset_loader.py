"""
WB AI Corporation - Dataset Loader Module (FIXED)
Handles loading, processing, and vectorization of HuggingFace datasets.
Architecture: Data Access Layer with robust error handling.
"""

import os
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass

from datasets import load_dataset, Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class DatasetConfig:
    """Dataset configuration with validation."""
    name: str
    split: str = "train"
    max_samples: int = 500  # Reduced for faster initial load
    text_field: Optional[str] = None
    streaming: bool = False  # Disabled streaming for reliability
    subset: Optional[str] = None


class OptimizedDatasetLoader:
    """
    Enterprise-grade dataset loader with robust error handling.
    """
    
    # Predefined configurations for known datasets
    DATASET_CONFIGS = {
        "princeton-nlp/SWE-bench_Verified": {
            "split": "test",
            "text_field": "problem_statement",
            "max_samples": 100,
            "streaming": False
        },
        "bigcode/the-stack-v2-dedup": {
            "subset": "Python",
            "split": "train",
            "text_field": "content",
            "max_samples": 200,
            "streaming": False
        },
        "microsoft/rStar-Coder": {
            "split": "train",
            "text_field": "solution",
            "max_samples": 200,
            "streaming": False
        }
    }
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "wb_ai_codebase",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 512,  # Reduced for faster processing
        chunk_overlap: int = 64,
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
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.vectorstore: Optional[Chroma] = None
        logger.success("DatasetLoader initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def load_dataset_safe(self, config: DatasetConfig) -> List[Dict[str, Any]]:
        """Load dataset with comprehensive error handling."""
        logger.info(f"Loading dataset: {config.name}")
        
        try:
            # Get predefined config if available
            if config.name in self.DATASET_CONFIGS:
                preset = self.DATASET_CONFIGS[config.name]
                config.text_field = config.text_field or preset.get("text_field")
                config.split = preset.get("split", config.split)
                config.subset = preset.get("subset")
                logger.info(f"Using preset config for {config.name}")
            
            # Load dataset
            load_kwargs = {
                "path": config.name,
                "split": config.split,
                "streaming": config.streaming,
                "trust_remote_code": True
            }
            
            if config.subset:
                load_kwargs["name"] = config.subset
            
            dataset = load_dataset(**load_kwargs)
            
            # Convert to list
            samples = []
            if config.streaming:
                for i, item in enumerate(dataset):
                    if i >= config.max_samples:
                        break
                    samples.append(item)
            else:
                # Non-streaming: take slice
                total = min(len(dataset), config.max_samples)
                samples = [dataset[i] for i in range(total)]
            
            logger.success(f"Loaded {len(samples)} samples from {config.name}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load {config.name}: {e}")
            logger.warning(f"Skipping dataset {config.name}")
            return []

    def _detect_text_field(self, sample: Dict[str, Any]) -> Optional[str]:
        """Enhanced text field detection."""
        if not sample:
            return None
        
        # Priority fields for code datasets
        priority_fields = [
            "content", "text", "code", "solution", "problem_statement",
            "instruction", "response", "patch", "input", "output",
            "prompt", "completion", "source_code", "body"
        ]
        
        # Check priority fields
        for field in priority_fields:
            if field in sample:
                value = sample[field]
                if isinstance(value, str) and len(value.strip()) > 20:
                    return field
        
        # Fallback: longest string field
        longest_field = None
        longest_length = 20
        
        for key, value in sample.items():
            if isinstance(value, str):
                length = len(value.strip())
                if length > longest_length:
                    longest_field = key
                    longest_length = length
        
        return longest_field

    def process_dataset_to_documents(self, config: DatasetConfig) -> List[Document]:
        """Convert dataset to LangChain documents with validation."""
        samples = self.load_dataset_safe(config)
        
        if not samples:
            logger.warning(f"No samples loaded from {config.name}")
            return []
        
        documents = []
        
        for idx, sample in enumerate(samples):
            try:
                # Detect text field
                text_field = config.text_field or self._detect_text_field(sample)
                
                if not text_field:
                    logger.debug(f"No text field found in sample {idx} of {config.name}")
                    continue
                
                # Extract text
                text = sample.get(text_field, "")
                
                # Validate text
                if not isinstance(text, str):
                    text = str(text)
                
                text = text.strip()
                
                if len(text) < 20:  # Minimum viable text length
                    continue
                
                # Create metadata
                metadata = {
                    "source": config.name,
                    "index": idx,
                    "field": text_field,
                }
                
                # Add additional metadata (limit to simple types)
                for k, v in sample.items():
                    if k != text_field and isinstance(v, (str, int, float, bool)):
                        # Truncate long strings in metadata
                        if isinstance(v, str) and len(v) > 200:
                            v = v[:200] + "..."
                        metadata[k] = v
                
                documents.append(Document(page_content=text, metadata=metadata))
                
            except Exception as e:
                logger.debug(f"Error processing sample {idx} from {config.name}: {e}")
                continue
        
        logger.info(f"Processed {len(documents)} documents from {config.name}")
        return documents

    def build_vectorstore(self, datasets_config: List[DatasetConfig]) -> Chroma:
        """Build ChromaDB vectorstore with validation."""
        all_documents = []
        
        # Sequential processing with error isolation
        for config in datasets_config:
            try:
                logger.info(f"Processing dataset: {config.name}")
                docs = self.process_dataset_to_documents(config)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed processing {config.name}: {e}")
                continue
        
        # Validation
        if not all_documents:
            logger.error("No documents collected from any dataset!")
            raise ValueError("Failed to load any documents. Check dataset configurations.")
        
        logger.info(f"Total documents collected: {len(all_documents)}")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        split_docs = self.text_splitter.split_documents(all_documents)
        
        if not split_docs:
            logger.error("No chunks created after splitting!")
            raise ValueError("Document splitting produced no chunks.")
        
        logger.success(f"Created {len(split_docs)} chunks")
        
        # Build ChromaDB vectorstore
        logger.info("Building ChromaDB vectorstore (this may take a few minutes)...")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=str(self.persist_dir),
                collection_name=self.collection_name
            )
            
            logger.success(f"Vectorstore built: {len(split_docs)} chunks indexed")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Failed to build vectorstore: {e}")
            raise

    def get_vectorstore(self) -> Chroma:
        """Get or load existing vectorstore."""
        if self.vectorstore:
            return self.vectorstore
        
        if (self.persist_dir / "chroma.sqlite3").exists():
            logger.info("Loading existing vectorstore...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                logger.success("Vectorstore loaded from disk")
            except Exception as e:
                logger.error(f"Failed to load existing vectorstore: {e}")
                logger.warning("Will rebuild vectorstore...")
                self.vectorstore = None
        
        return self.vectorstore


def initialize_datasets() -> OptimizedDatasetLoader:
    """Factory function to initialize dataset loader with env configs."""
    from dotenv import load_dotenv
    load_dotenv()
    
    loader = OptimizedDatasetLoader(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "wb_ai_codebase"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        chunk_size=int(os.getenv("CHUNK_SIZE", 512)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 64)),
        device=os.getenv("DEVICE", "cuda")
    )
    
    # Robust dataset configurations
    configs = [
        DatasetConfig(
            name="princeton-nlp/SWE-bench_Verified",
            split="test",
            max_samples=100,
            text_field="problem_statement"
        ),
        DatasetConfig(
            name="bigcode/the-stack-v2-dedup",
            subset="Python",
            split="train",
            max_samples=200,
            text_field="content"
        ),
        DatasetConfig(
            name="microsoft/rStar-Coder",
            split="train",
            max_samples=200,
            text_field="solution"
        )
    ]
    
    # Check if vectorstore exists
    vectorstore_path = Path(loader.persist_dir) / "chroma.sqlite3"
    
    if not vectorstore_path.exists():
        logger.info("Building new vectorstore...")
        try:
            loader.build_vectorstore(configs)
        except Exception as e:
            logger.error(f"Failed to build vectorstore: {e}")
            logger.warning("Falling back to minimal demo dataset...")
            
            # Fallback: Create minimal demo dataset
            demo_docs = [
                Document(
                    page_content="""
# Python FastAPI Example
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item.name, "price": item.price}
                    """,
                    metadata={"source": "demo", "type": "fastapi"}
                ),
                Document(
                    page_content="""
# Python Data Analysis with Pandas
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': np.random.randn(100),
    'B': np.random.randn(100)
})

result = df.describe()
correlation = df.corr()
                    """,
                    metadata={"source": "demo", "type": "pandas"}
                ),
                Document(
                    page_content="""
# Docker Compose for Python App
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dbname
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: pass
                    """,
                    metadata={"source": "demo", "type": "docker"}
                )
            ]
            
            logger.info("Creating demo vectorstore with sample code...")
            loader.vectorstore = Chroma.from_documents(
                documents=demo_docs,
                embedding=loader.embeddings,
                persist_directory=str(loader.persist_dir),
                collection_name=loader.collection_name
            )
            logger.success("Demo vectorstore created")
    else:
        logger.info("Using existing vectorstore")
        loader.get_vectorstore()
    
    return loader
