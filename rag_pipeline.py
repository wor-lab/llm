"""
WB AI CORPORATION - RAG Engine
ChromaDB Integration + Retrieval Pipeline
Production Vector Search System
"""

import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger(__name__)

class RAGEngine:
    """Enterprise RAG pipeline with ChromaDB"""
    
    def __init__(self, chroma_path: str, model_name: str):
        self.chroma_path = chroma_path
        self.model_name = model_name
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embeddings
        logger.info("Loading embedding model...")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="code_intelligence",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize LLM
        logger.info(f"Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )
        
        logger.info("✅ RAG Engine initialized")
    
    def index_documents(self, documents: List[Any]) -> None:
        """Index documents in ChromaDB"""
        logger.info(f"Indexing {len(documents)} documents...")
        
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [doc.id for doc in batch]
            texts = [doc.embedding_text for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            if (i + batch_size) % 500 == 0:
                logger.info(f"Indexed {i + batch_size} documents...")
        
        logger.info(f"✅ Indexed {self.collection.count()} total documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return documents
    
    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate response using retrieved context"""
        
        # Build prompt with context
        context_text = "\n\n".join([
            f"[Source: {doc['metadata'].get('source', 'unknown')}]\n{doc['content'][:500]}"
            for doc in context[:3]
        ])
        
        prompt = f"""You are an expert code assistant. Answer the following query using the provided context.

Context:
{context_text}

Query: {query}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = response.split("Answer:")[-1].strip()
        
        return answer
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """End-to-end RAG query"""
        logger.info(f"Processing query: {question[:100]}...")
        
        # Retrieve
        context = self.retrieve(question, top_k=top_k)
        
        # Generate
        answer = self.generate_response(question, context)
        
        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'source': doc['metadata'].get('source'),
                    'relevance': 1 - doc['distance'] if doc['distance'] else None
                }
                for doc in context
            ]
        }
