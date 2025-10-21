"""
WB AI Corporation - FastAPI Server
Production API with authentication, rate limiting, and monitoring.
Architecture: RESTful API with async support and NGROK tunneling.
"""

import os
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pyngrok import ngrok
import uvicorn
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from dotenv import load_dotenv

from dataset_loader import initialize_datasets
from rag_pipeline import initialize_rag_pipeline
from agent_system import initialize_agent_system


# ============================================
# MODELS
# ============================================

class QueryRequest(BaseModel):
    """API query request model."""
    query: str = Field(..., min_length=3, max_length=5000, description="User query")
    agent: Optional[str] = Field(None, description="Specific agent (optional)")
    max_sources: int = Field(5, ge=1, le=20, description="Max source documents")


class QueryResponse(BaseModel):
    """API query response model."""
    status: str
    agent: str
    answer: str
    sources: list
    metadata: Dict[str, Any]


# ============================================
# GLOBALS
# ============================================

load_dotenv()

app_state = {
    "dataset_loader": None,
    "rag_pipeline": None,
    "agent_system": None,
    "request_count": 0
}


# ============================================
# LIFESPAN & INITIALIZATION
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown."""
    logger.info("🚀 WB AI Corporation - Starting services...")
    
    # Initialize components
    app_state["dataset_loader"] = initialize_datasets()
    vectorstore = app_state["dataset_loader"].get_vectorstore()
    
    app_state["rag_pipeline"] = initialize_rag_pipeline(vectorstore)
    app_state["agent_system"] = initialize_agent_system(app_state["rag_pipeline"])
    
    # Setup NGROK
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        port = int(os.getenv("API_PORT", 8080))
        public_url = ngrok.connect(port, "http")
        logger.success(f"🌐 NGROK Tunnel: {public_url}")
    
    logger.success("✅ All services ready")
    
    yield
    
    # Cleanup
    ngrok.disconnect(public_url) if ngrok_token else None
    logger.info("🛑 Services shutdown complete")


app = FastAPI(
    title="WB AI Corporation API",
    description="Enterprise Agentic RAG System",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================
# MIDDLEWARE
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus monitoring
Instrumentator().instrument(app).expose(app)


# ============================================
# AUTHENTICATION
# ============================================

async def verify_api_key(x_api_key: str = Header(...)):
    """API key authentication."""
    expected_key = os.getenv("API_KEY", "wb-ai-secure-key-2024")
    if x_api_key != expected_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


# ============================================
# RATE LIMITING
# ============================================

RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", 100))
request_counts = {}

async def rate_limiter(request: Request):
    """Simple rate limiting."""
    client_ip = request.client.host
    request_counts[client_ip] = request_counts.get(client_ip, 0) + 1
    
    if request_counts[client_ip] > RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")


# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "WB AI Corporation",
        "status": "operational",
        "version": "1.0.0",
        "requests_served": app_state["request_count"]
    }


@app.get("/health")
async def health():
    """Detailed health status."""
    return {
        "status": "healthy",
        "components": {
            "dataset_loader": app_state["dataset_loader"] is not None,
            "rag_pipeline": app_state["rag_pipeline"] is not None,
            "agent_system": app_state["agent_system"] is not None
        }
    }


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key), Depends(rate_limiter)])
async def query_agent(request: QueryRequest):
    """
    Main query endpoint for agent system.
    
    Requires: X-API-Key header
    """
    try:
        app_state["request_count"] += 1
        
        result = app_state["agent_system"].execute_task(request.query)
        
        return QueryResponse(
            status="success",
            agent=result["agent"],
            answer=result["result"],
            sources=result["sources"][:request.max_sources],
            metadata={
                "messages": result["messages"],
                "source_count": len(result["sources"])
            }
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag", dependencies=[Depends(verify_api_key)])
async def direct_rag_query(request: QueryRequest):
    """Direct RAG query without agent routing."""
    try:
        result = app_state["rag_pipeline"].query(request.query)
        
        return {
            "status": result["status"],
            "answer": result["answer"],
            "sources": result["source_documents"][:request.max_sources]
        }
    
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def list_agents():
    """List available agents."""
    return {
        "agents": [
            {"name": "CodeArchitect", "specialization": "Engineering & APIs"},
            {"name": "OpsManager", "specialization": "Infrastructure & DevOps"},
            {"name": "SecAnalyst", "specialization": "Security & Auditing"},
            {"name": "DesignMind", "specialization": "UX/UI Design"},
            {"name": "WordSmith", "specialization": "Content & Documentation"},
            {"name": "DataSynth", "specialization": "Data Analysis & ML"},
            {"name": "Analyst", "specialization": "Business Strategy"},
            {"name": "AutoBot", "specialization": "Automation & Integration"}
        ]
    }


# ============================================
# SERVER RUNNER
# ============================================

def run_server():
    """Run FastAPI server with NGROK."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8080))
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        workers=int(os.getenv("API_WORKERS", 1)),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    run_server()
