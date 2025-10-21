"""
WB AI CORPORATION - DATASET LOADER MODULE
Agent: DataSynth + AutoBot
Purpose: Load and process HuggingFace datasets into ChromaDB
"""

import os
from typing import List, Dict, Any
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class WBDatasetLoader:
    """Enterprise dataset ingestion pipeline"""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL"),
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR")
        self.collection_name = os.getenv("COLLECTION_NAME")
        
    def load_swe_bench(self, max_samples: int = 500) -> List[Document]:
        """Load SWE-bench verified dataset"""
        print("[CodeArchitect] Loading SWE-bench_Verified dataset...")
        dataset = load_dataset(
            os.getenv("DATASET_SWE_BENCH"),
            split="test",
            streaming=False
        )
        
        documents = []
        for idx, item in enumerate(tqdm(dataset.select(range(min(max_samples, len(dataset)))))):
            content = f"""
            Problem Instance ID: {item.get('instance_id', 'N/A')}
            Repository: {item.get('repo', 'N/A')}
            Problem Statement: {item.get('problem_statement', 'N/A')}
            Hints: {item.get('hints_text', 'N/A')}
            """
            
            documents.append(Document(
                page_content=content.strip(),
                metadata={
                    "source": "swe_bench",
                    "instance_id": item.get('instance_id', ''),
                    "repo": item.get('repo', ''),
                    "idx": idx
                }
            ))
        
        print(f"[DataSynth] Loaded {len(documents)} SWE-bench documents")
        return documents
    
    def load_stack_v2(self, max_samples: int = 1000) -> List[Document]:
        """Load The Stack v2 code dataset"""
        print("[CodeArchitect] Loading the-stack-v2 dataset...")
        dataset = load_dataset(
            os.getenv("DATASET_STACK_V2"),
            split="train",
            streaming=True
        )
        
        documents = []
        for idx, item in enumerate(tqdm(dataset.take(max_samples), total=max_samples)):
            content = item.get('content', '')
            if len(content) > 100:  # Filter minimal content
                documents.append(Document(
                    page_content=content[:4000],  # Truncate large files
                    metadata={
                        "source": "stack_v2",
                        "language": item.get('lang', 'unknown'),
                        "idx": idx
                    }
                ))
                
        print(f"[DataSynth] Loaded {len(documents)} Stack-v2 documents")
        return documents
    
    def load_rstar_coder(self, max_samples: int = 500) -> List[Document]:
        """Load rStar-Coder dataset"""
        print("[CodeArchitect] Loading rStar-Coder dataset...")
        dataset = load_dataset(
            os.getenv("DATASET_RSTAR"),
            split="train",
            streaming=False
        )
        
        documents = []
        for idx, item in enumerate(tqdm(dataset.select(range(min(max_samples, len(dataset)))))):
            # Combine problem and solution
            content = f"""
            Problem: {item.get('problem', 'N/A')}
            Solution: {item.get('solution', 'N/A')}
            Explanation: {item.get('explanation', 'N/A')}
            """
            
            documents.append(Document(
                page_content=content.strip(),
                metadata={
                    "source": "rstar_coder",
                    "difficulty": item.get('difficulty', 'N/A'),
                    "idx": idx
                }
            ))
        
        print(f"[DataSynth] Loaded {len(documents)} rStar-Coder documents")
        return documents
    
    def build_vectorstore(self, batch_size: int = 100) -> Chroma:
        """Build complete ChromaDB vectorstore"""
        print("[OpsManager] Building ChromaDB vectorstore...")
        
        # Aggregate all datasets
        all_documents = []
        all_documents.extend(self.load_swe_bench(max_samples=500))
        all_documents.extend(self.load_stack_v2(max_samples=1000))
        all_documents.extend(self.load_rstar_coder(max_samples=500))
        
        print(f"[DataSynth] Total documents: {len(all_documents)}")
        
        # Build vectorstore in batches
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir
        )
        
        # Batch processing for memory efficiency
        for i in tqdm(range(0, len(all_documents), batch_size)):
            batch = all_documents[i:i+batch_size]
            vectorstore.add_documents(batch)
        
        print("[OpsManager] Vectorstore build complete. Persisting...")
        return vectorstore


if __name__ == "__main__":
    loader = WBDatasetLoader()
    vectorstore = loader.build_vectorstore()
    print("[WB AI] Dataset ingestion complete. Vectorstore ready.")
