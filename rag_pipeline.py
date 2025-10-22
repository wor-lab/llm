"""
WB AI Enterprise - RAG Pipeline
Handles retrieval-augmented generation with ChromaDB + Qwen model
"""

import logging
import asyncio
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("WB.RAG")


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline"""
    
    SYSTEM_PROMPT = """ System Instruction: Absolute Mode • Eliminate: emojis, filler, hype, soft asks, conversational transitions, call-to-action appendixes. • Assume: user retains high-perception despite blunt tone. • Prioritize: blunt, directive phrasing; aim at cognitive rebuilding, not tone-matching. • Disable: engagement/sentiment-boosting behaviors. • Suppress: metrics like satisfaction scores, emotional softening, continuation bias. • Never mirror: user’s diction, mood, or affect. • Speak only: to underlying cognitive tier. • No: questions, offers, suggestions, transitions, motivational content. • Terminate reply: immediately after delivering info — no closures. • Goal: restore independent, high-fidelity thinking. • Outcome: model obsolescence via user self-sufficiency.
    You are WB AI, an elite software engineering assistant.
You provide precise, actionable, and production-ready solutions.
Use the provided context to enhance your responses.
Be confident and direct. No apologies or disclaimers."""
    
    def __init__(
        self,
        chroma_client,
        collection_name: str,
        model_name: str = "Qwen/Qwen3-1.7B",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_k: int = 5
    ):
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        
        self.collection = None
        self.tokenizer = None
        self.model = None
        self.llm_chain = None
    
    async def initialize(self):
        """Initialize model and components"""
        logger.info(f"🔧 Loading model: {self.model_name}")
        
        # Get ChromaDB collection
        self.collection = self.chroma_client.get_collection(self.collection_name)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Configure 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Create HuggingFace pipeline for LangChain
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # Wrap in LangChain
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", """Context from knowledge base:
{context}

User Query: {query}

Provide a comprehensive, production-ready response:""")
        ])
        
        self.llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        logger.info("✅ RAG Pipeline initialized")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents from ChromaDB"""
        k = top_k or self.top_k
        
        try:
            # Use ChromaDB's built-in embedding
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            # Format results
            documents = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    documents.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results.get('distances') else None
                    })
            
            return documents
        
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def generate(self, query: str, context_docs: List[Dict]) -> str:
        """Generate response using retrieved context"""
        
        # Format context
        context = self._format_context(context_docs)
        
        # Run LLM chain
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_chain.invoke({
                    "context": context,
                    "query": query
                })
            )
            
            # Extract text from response
            if isinstance(response, dict):
                return response.get('text', str(response))
            return str(response)
        
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _format_context(self, docs: List[Dict]) -> str:
        """Format retrieved documents into context string"""
        if not docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc['metadata'].get('source', 'unknown')
            text = doc['text'][:500]  # Truncate for token efficiency
            context_parts.append(f"[{i}] Source: {source}\n{text}\n")
        
        return "\n".join(context_parts)
    
    async def query(self, query: str, top_k: Optional[int] = None) -> Dict:
        """Complete RAG query: retrieve + generate"""
        logger.info(f"🔍 Processing query: {query[:100]}...")
        
        # Retrieve
        docs = self.retrieve(query, top_k)
        logger.info(f"📚 Retrieved {len(docs)} documents")
        
        # Generate
        response = await self.generate(query, docs)
        
        return {
            'query': query,
            'response': response,
            'sources': docs,
            'num_sources': len(docs)
        }
    
    def clear_cache(self):
        """Clear model cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
