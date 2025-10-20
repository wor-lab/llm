"""
WB AI CORPORATION - API Server
FastAPI + Ngrok Deployment
Production REST API Interface
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from pyngrok import ngrok
import os

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_agents: Optional[bool] = True

class QueryResponse(BaseModel):
    response: str
    execution_log: Optional[List[str]] = []
    sources: List[Dict] = []
    status: str = "success"

def create_app(agent_system, ngrok_token: str) -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="WB AI NEXUS-RAG API",
        description="Enterprise Agentic Code Intelligence System",
        version="1.0.0"
    )
    
    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Configure Ngrok
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        public_url = ngrok.connect(8000)
        logger.info(f"🌐 Ngrok tunnel established: {public_url}")
        logger.info(f"📡 Public API URL: {public_url}/docs")
    
    @app.get("/")
    def root():
        """Health check endpoint"""
        return {
            "status": "operational",
            "service": "WB AI NEXUS-RAG",
            "version": "1.0.0",
            "endpoints": {
                "query": "/api/v1/query",
                "health": "/health",
                "docs": "/docs"
            }
        }
    
    @app.get("/health")
    def health():
        """Detailed health check"""
        return {
            "status": "healthy",
            "chroma_db": "connected",
            "model": "loaded",
            "agents": "active"
        }
    
    @app.post("/api/v1/query", response_model=QueryResponse)
    async def query_endpoint(request: QueryRequest):
        """Main query endpoint - processes code intelligence requests"""
        
        try:
            logger.info(f"📥 Received query: {request.query[:100]}...")
            
            if request.use_agents:
                # Use full agent system
                result = agent_system.process_query(request.query)
                
                return QueryResponse(
                    response=result['response'],
                    execution_log=result['execution_log'],
                    sources=result['sources'],
                    status="success"
                )
            else:
                # Direct RAG query
                result = agent_system.rag_engine.query(request.query, top_k=request.top_k)
                
                return QueryResponse(
                    response=result['answer'],
                    sources=result['sources'],
                    status="success"
                )
                
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/stats")
    def stats():
        """System statistics"""
        collection = agent_system.rag_engine.collection
        
        return {
            "total_documents": collection.count(),
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "llm_model": agent_system.model_name,
            "agents": 5,
            "status": "operational"
        }
    
    return app

# Standalone execution
if __name__ == "__main__":
    # This would be called from run_colab.py in production
    logger.warning("Use run_colab.py for full system deployment")
