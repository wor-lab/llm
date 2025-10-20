"""
WB AI Corporation - RAG Pipeline Module
Agent: CodeArchitect
Purpose: Implement retrieval-augmented generation for code intelligence
"""

import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """Production RAG system for code intelligence"""
    
    SYSTEM_PROMPT = """You are WB AI Code Intelligence Assistant - an expert code analysis and generation system.

Context from knowledge base:
{context}

User Query: {question}

Instructions:
- Provide precise, executable code solutions
- Reference relevant examples from context
- Follow best practices and enterprise standards
- Include brief explanations for complex logic

Response:"""
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.model_server = os.getenv("MODEL_SERVER")
        self.api_key = os.getenv("HF_API_KEY")
        self.temperature = float(os.getenv("TEMPERATURE", 0.7))
        self.top_k = int(os.getenv("TOP_K_RESULTS", 5))
        self.max_length = int(os.getenv("MAX_CONTEXT_LENGTH", 2048))
        
        # Initialize LLM
        self.llm = HuggingFaceEndpoint(
            endpoint_url=self.model_server,
            huggingfacehub_api_token=self.api_key,
            task="text-generation",
            model_kwargs={
                "temperature": self.temperature,
                "max_new_tokens": 512,
                "return_full_text": False,
            }
        )
        
        # Setup retrieval chain
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        self.prompt = PromptTemplate(
            template=self.SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
        
        logger.info("RAG Pipeline initialized | Model: Qwen3-1.7B")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Execute RAG query"""
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            response = {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:200],
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ],
                "status": "success"
            }
            
            logger.success("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "status": "error"
            }
    
    def retrieve_context(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents without generation"""
        k = k or self.top_k
        docs = self.retriever.get_relevant_documents(query)[:k]
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": idx
            }
            for idx, doc in enumerate(docs)
        ]


if __name__ == "__main__":
    from dataset_loader import DatasetLoader
    
    loader = DatasetLoader()
    vectorstore = loader.load_existing()
    
    pipeline = RAGPipeline(vectorstore)
    result = pipeline.query("Write a Python function to reverse a linked list")
    
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
