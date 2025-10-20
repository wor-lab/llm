"""
WB AI CORPORATION — DATA INTELLIGENCE PIPELINE
Agentic RAG Implementation with Multi-Step Reasoning

MISSION: Advanced retrieval-augmented generation
AGENT: DataSynth + CodeArchitect
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Structure for retrieval results"""
    document: str
    metadata: dict
    score: float
    source: str

class AgenticRAGPipeline:
    """
    Multi-stage RAG with:
    - Query decomposition
    - Multi-source retrieval
    - Reranking
    - Answer synthesis
    """
    
    def __init__(
        self,
        chroma_client: chromadb.PersistentClient,
        embedder: SentenceTransformer,
        llm
    ):
        self.client = chroma_client
        self.embedder = embedder
        self.llm = llm
        
        self.collections = ['humaneval', 'mbpp', 'swe_bench', 'bigcodebench']
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Break complex queries into sub-queries
        """
        logger.info("🔍 Decomposing query...")
        
        # Simple decomposition (can be enhanced with LLM)
        sub_queries = [query]
        
        # Extract key programming concepts
        keywords = []
        code_patterns = ['function', 'class', 'algorithm', 'implement', 'solve']
        
        for pattern in code_patterns:
            if pattern in query.lower():
                keywords.append(pattern)
        
        if keywords:
            sub_queries.append(f"Examples of {' and '.join(keywords)}")
        
        return sub_queries
    
    def retrieve_multi_source(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve from multiple collections and merge results
        """
        logger.info(f"📚 Retrieving from {len(self.collections)} sources...")
        
        all_results = []
        
        for query in queries:
            query_embedding = self.embedder.encode([query]).tolist()
            
            for col_name in self.collections:
                try:
                    collection = self.client.get_collection(col_name)
                    
                    results = collection.query(
                        query_embeddings=query_embedding,
                        n_results=top_k
                    )
                    
                    if results['documents']:
                        for doc, meta, dist in zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['distances'][0]
                        ):
                            all_results.append(
                                RetrievalResult(
                                    document=doc,
                                    metadata=meta,
                                    score=1 - dist,  # Convert distance to similarity
                                    source=col_name
                                )
                            )
                except Exception as e:
                    logger.warning(f"⚠️ Error querying {col_name}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:top_k * 2]
    
    def rerank_results(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder or relevance scoring
        """
        logger.info("🎯 Reranking results...")
        
        # Simple reranking based on keyword overlap
        query_words = set(query.lower().split())
        
        for result in results:
            doc_words = set(result.document.lower().split())
            overlap = len(query_words & doc_words)
            result.score = result.score * (1 + overlap * 0.1)
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def synthesize_answer(
        self,
        query: str,
        retrieved_docs: List[RetrievalResult]
    ) -> Dict:
        """
        Generate final answer with citations
        """
        logger.info("✍️ Synthesizing answer...")
        
        # Build context from top documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Source {i} - {doc.source}]:\n{doc.document}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are WB AI Corporation's code intelligence system.
Use the provided sources to answer the question accurately.

Sources:
{context}

Question: {query}

Provide a comprehensive answer with code examples where appropriate:
"""
        
        answer = self.llm.invoke(prompt)
        
        return {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'text': doc.document[:200] + '...',
                    'source': doc.source,
                    'score': round(doc.score, 3)
                }
                for doc in retrieved_docs
            ],
            'confidence': sum(doc.score for doc in retrieved_docs) / len(retrieved_docs)
        }
    
    def process(self, query: str, top_k: int = 5) -> Dict:
        """
        Full agentic RAG pipeline
        """
        logger.info(f"🚀 Processing query: {query[:50]}...")
        
        # Step 1: Decompose query
        sub_queries = self.decompose_query(query)
        
        # Step 2: Multi-source retrieval
        retrieved_docs = self.retrieve_multi_source(sub_queries, top_k=top_k)
        
        # Step 3: Rerank
        reranked_docs = self.rerank_results(query, retrieved_docs, top_k=top_k)
        
        # Step 4: Synthesize answer
        result = self.synthesize_answer(query, reranked_docs)
        
        logger.info("✅ Pipeline complete")
        
        return result

class RAGOrchestrator:
    """High-level RAG interface"""
    
    def __init__(self):
        logger.info("🔧 Initializing RAG Pipeline...")
        
        # Load dependencies
        from agent_system import ModelLoader
        
        self.llm, _ = ModelLoader.load_model()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize pipeline
        self.pipeline = AgenticRAGPipeline(
            chroma_client=self.chroma_client,
            embedder=self.embedder,
            llm=self.llm
        )
    
    def query(self, question: str) -> Dict:
        """Process RAG query"""
        return self.pipeline.process(question)

if __name__ == "__main__":
    rag = RAGOrchestrator()
    
    test_queries = [
        "How do I implement binary search in Python?",
        "Explain the time complexity of quicksort",
        "Write a function to reverse a linked list"
    ]
    
    for query in test_queries:
        result = rag.query(query)
        
        print(f"\n{'='*70}")
        print(f"Q: {result['query']}")
        print(f"\nA: {result['answer'][:500]}...")
        print(f"\nConfidence: {result['confidence']:.2%}")
        print(f"Sources: {len(result['sources'])}")
        print(f"{'='*70}\n")
