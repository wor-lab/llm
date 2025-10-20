"""
WB AI Corporation - Dataset Ingestion Module
Agent: DataSynth
Purpose: Load, process, and vectorize code datasets into ChromaDB
"""

import os
from typing import List, Dict, Any
from datasets import load_dataset
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class DatasetLoader:
    """Enterprise-grade dataset ingestion pipeline"""
    
    DATASET_CONFIGS = [
        {"name": "princeton-nlp/SWE-bench_Verified", "split": "test", "text_field": "problem_statement"},
        {"name": "openai/humaneval", "split": "test", "text_field": "prompt"},
        {"name": "mbpp", "split": "test", "text_field": "text"},
        {"name": "bigcode/bigcodebench", "split": "v0.1.0_hf", "text_field": "instruct_prompt"},
        {"name": "livecodebench/code_generation_lite", "split": "test", "text_field": "question_content"},
    ]
    
    def __init__(self):
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.max_samples = int(os.getenv("MAX_SAMPLES_PER_DATASET", 500))
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        logger.info(f"DatasetLoader initialized | Persist: {self.persist_dir}")
    
    def load_and_process(self) -> Chroma:
        """Load all datasets and create vector store"""
        all_documents = []
        
        for config in self.DATASET_CONFIGS:
            docs = self._load_single_dataset(config)
            all_documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {config['name']}")
        
        logger.info(f"Total documents: {len(all_documents)} | Creating vector store...")
        
        vectorstore = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="wb_code_intelligence"
        )
        
        logger.success(f"Vector store created | Documents: {len(all_documents)}")
        return vectorstore
    
    def _load_single_dataset(self, config: Dict[str, str]) -> List[Document]:
        """Load and convert single dataset to LangChain documents"""
        documents = []
        
        try:
            dataset = load_dataset(
                config["name"],
                split=config["split"],
                trust_remote_code=True
            )
            
            # Handle dataset size
            dataset = dataset.select(range(min(len(dataset), self.max_samples)))
            
            for idx, item in enumerate(dataset):
                # Extract text based on dataset structure
                text_content = self._extract_text(item, config)
                
                if text_content:
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": config["name"],
                            "index": idx,
                            "dataset_type": "code_benchmark"
                        }
                    )
                    documents.append(doc)
            
        except Exception as e:
            logger.warning(f"Failed to load {config['name']}: {e}")
        
        return documents
    
    def _extract_text(self, item: Dict[str, Any], config: Dict[str, str]) -> str:
        """Extract text content from dataset item"""
        text_field = config.get("text_field", "prompt")
        
        # Try primary field
        if text_field in item:
            return str(item[text_field])
        
        # Fallback strategies
        fallback_fields = ["prompt", "text", "question", "problem_statement", "instruction"]
        for field in fallback_fields:
            if field in item:
                return str(item[field])
        
        # Last resort: combine all string fields
        text_parts = [str(v) for v in item.values() if isinstance(v, str)]
        return " ".join(text_parts) if text_parts else ""
    
    def load_existing(self) -> Chroma:
        """Load existing vector store"""
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(f"Vector store not found at {self.persist_dir}")
        
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            collection_name="wb_code_intelligence"
        )
        
        logger.info("Loaded existing vector store")
        return vectorstore


if __name__ == "__main__":
    loader = DatasetLoader()
    vectorstore = loader.load_and_process()
    logger.success("Dataset loading complete")
