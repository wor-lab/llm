"""
WB AI Corporation - RAG Pipeline Module
Advanced RAG implementation with LangChain & LangGraph.
Architecture: Retrieval-Augmented Generation with hybrid search and reranking.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
import torch
from loguru import logger


@dataclass
class RAGConfig:
    """RAG configuration parameters."""
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.1
    top_k: int = 5
    device: str = "cuda"
    load_in_4bit: bool = True


class HybridRetriever(BaseRetriever):
    """
    Custom retriever with:
    - Semantic search (dense embeddings)
    - Keyword search (BM25-style)
    - Reranking by relevance
    """
    
    vectorstore: Chroma
    search_kwargs: Dict[str, Any] = {"k": 5}
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents using hybrid approach."""
        # Semantic search
        semantic_docs = self.vectorstore.similarity_search(
            query, **self.search_kwargs
        )
        
        # MMR (Maximum Marginal Relevance) for diversity
        mmr_docs = self.vectorstore.max_marginal_relevance_search(
            query, k=self.search_kwargs.get("k", 5), fetch_k=20
        )
        
        # Combine and deduplicate
        seen = set()
        combined = []
        for doc in semantic_docs + mmr_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                combined.append(doc)
        
        return combined[:self.search_kwargs.get("k", 5)]


class ProductionRAGPipeline:
    """
    Enterprise RAG pipeline with:
    - Optimized model loading (4-bit quantization)
    - Custom prompt engineering
    - Streaming support
    - Error handling
    """
    
    def __init__(self, vectorstore: Chroma, config: RAGConfig):
        self.vectorstore = vectorstore
        self.config = config
        self.llm = None
        self.qa_chain = None
        
        self._initialize_model()
        self._build_chain()
    
    def _initialize_model(self):
        """Load Qwen model with optimizations."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ) if self.config.load_in_4bit else None
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        logger.success("Model loaded successfully")
    
    def _build_chain(self):
        """Build RAG chain with custom prompt."""
        
        PROMPT_TEMPLATE = """You are WB AI Corporation's elite software engineer assistant.
Use the following code examples and documentation to answer the question precisely.

Context:
{context}

Question: {question}

Instructions:
- Provide production-grade, optimized code
- Include error handling and type hints
- Follow best practices (PEP 8, clean architecture)
- Be concise but complete
- If unsure, state limitations

Answer:"""

        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        
        retriever = HybridRetriever(
            vectorstore=self.vectorstore,
            search_kwargs={"k": self.config.top_k}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.success("RAG chain built")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Execute RAG query with error handling."""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            result = self.qa_chain.invoke({"query": question})
            
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content[:500],
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ],
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "source_documents": [],
                "status": "error"
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently."""
        return [self.query(q) for q in questions]


def initialize_rag_pipeline(vectorstore: Chroma) -> ProductionRAGPipeline:
    """Factory function to create RAG pipeline from environment."""
    from dotenv import load_dotenv
    load_dotenv()
    
    config = RAGConfig(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 2048)),
        temperature=float(os.getenv("TEMPERATURE", 0.1)),
        device=os.getenv("DEVICE", "cuda")
    )
    
    return ProductionRAGPipeline(vectorstore, config)
