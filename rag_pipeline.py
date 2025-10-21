"""
WB AI CORPORATION - RAG PIPELINE MODULE
Agent: CodeArchitect
Purpose: Core RAG retrieval and generation pipeline
"""

import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv

load_dotenv()


class WBRAGPipeline:
    """Enterprise-grade RAG inference pipeline"""
    
    def __init__(self):
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.model_name = os.getenv("MODEL_NAME")
        
        print("[CodeArchitect] Initializing RAG pipeline...")
        self._load_embeddings()
        self._load_vectorstore()
        self._load_llm()
        self._build_chain()
        
    def _load_embeddings(self):
        """Load embedding model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL"),
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("[OpsManager] Embeddings loaded")
        
    def _load_vectorstore(self):
        """Load persisted ChromaDB"""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        print(f"[DataSynth] Vectorstore loaded: {self.vectorstore._collection.count()} documents")
        
    def _load_llm(self):
        """Load Qwen3-1.7B local model"""
        print(f"[CodeArchitect] Loading {self.model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            load_in_8bit=True  # Memory optimization
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=int(os.getenv("MAX_TOKENS", 2048)),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        print("[OpsManager] LLM loaded successfully")
        
    def _build_chain(self):
        """Build RetrievalQA chain"""
        template = """### WB AI CORPORATION - AGENTIC RAG SYSTEM ###

Context from knowledge base:
{context}

User Query: {question}

Instructions: You are an elite AI engineer at WB AI Corporation. Provide precise, actionable technical responses based on the retrieved context. Focus on code quality, scalability, and enterprise best practices.

Response:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("[CodeArchitect] RAG chain assembled")
        
    def query(self, question: str) -> Dict[str, Any]:
        """Execute RAG query"""
        print(f"[Analyst] Processing query: {question[:100]}...")
        
        result = self.qa_chain.invoke({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:200],
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }
    
    def search_vectorstore(self, query: str, k: int = 5) -> List[Dict]:
        """Direct vectorstore search"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]


if __name__ == "__main__":
    rag = WBRAGPipeline()
    response = rag.query("How do I implement a FastAPI endpoint with authentication?")
    print("\n[WB AI Response]:")
    print(response["answer"])
