"""
WB AI Corporation - RAG Pipeline Engine
Handles retrieval, generation, and knowledge management
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings

class RAGEngine:
    def __init__(self, model_id: str, vector_store_path: str):
        self.model_id = model_id
        self.vector_store_path = vector_store_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize components
        self._init_embeddings()
        self._init_llm()
        self._init_vectorstore()
        self._init_retriever()
        
    def _init_embeddings(self):
        """Initialize embedding model for vector search"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def _init_llm(self):
        """Initialize Qwen model with quantization"""
        print(f"[RAG] Loading {self.model_id}...")
        
        # 4-bit quantization config for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config if self.device == "cuda" else None,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Create LangChain pipeline
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
    def _init_vectorstore(self):
        """Initialize ChromaDB vector store"""
        self.chroma_client = chromadb.PersistentClient(
            path=self.vector_store_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name="wb_knowledge_base",
            embedding_function=self.embeddings
        )
        
    def _init_retriever(self):
        """Setup retrieval chain with custom prompts"""
        prompt_template = """[WB AI SYSTEM CONTEXT]
You are an expert AI agent within WB AI Corporation.
Use the following context to provide precise, actionable responses.

Context: {context}

Task: {question}

Response (be specific and implementation-focused):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add new documents to knowledge base"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.create_documents(
            documents,
            metadatas=metadata
        )
        
        self.vectorstore.add_documents(chunks)
        print(f"[RAG] Added {len(chunks)} chunks to knowledge base")
        
    def query(self, question: str, filters: Dict = None) -> Dict[str, Any]:
        """Execute RAG query with optional metadata filtering"""
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result.get("source_documents", [])],
            "confidence": self._calculate_confidence(result)
        }
        
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate response confidence based on retrieval quality"""
        if "source_documents" not in result:
            return 0.3
        
        # Simple confidence based on number of relevant sources
        num_sources = len(result["source_documents"])
        return min(0.9, 0.3 + (num_sources * 0.15))
        
    def hybrid_search(self, query: str, code_context: str = None) -> str:
        """Combine vector search with code-aware generation"""
        context_prompt = f"Code Context:\n{code_context}\n\n" if code_context else ""
        enhanced_query = f"{context_prompt}Query: {query}"
        
        return self.query(enhanced_query)
