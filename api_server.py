"""
WB AI Corporation - API Server Module
Agent: AutoBot
Purpose: FastAPI server with NGROK tunnel for production deployment
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from pyngrok import ngrok
from loguru import logger
from dotenv import load_dotenv

from dataset_loader import DatasetLoader
from rag_pipeline import RAGPipeline
from agent_system import WBAgentSystem

load_dotenv()


# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Code-related query")
    use_agent: bool = Field(default=False, description="Use multi-agent system")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    vectorstore_docs: int


# ============================================
# APPLICATION LIFESPAN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("WB AI Corporation - Initializing services...")
    
    # Setup NGROK
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        port = int(os.getenv("API_PORT", 8000))
        public_url = ngrok.connect(port)
        logger.success(f"NGROK tunnel established: {public_url}")
        app.state.public_url = str(public_url)
    
    # Load vector store
    loader = DatasetLoader()
    
    if not os.path.exists(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")):
        logger.info("Vector store not found. Creating...")
        vectorstore = loader.load_and_process()
    else:
        vectorstore = loader.load_existing()
    
    # Initialize RAG and Agent System
    app.state.rag = RAGPipeline(vectorstore)
    app.state.agent_system = WBAgentSystem(app.state.rag)
    app.state.vectorstore = vectorstore
    
    logger.success("All systems operational")
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    ngrok.disconnect(app.state.public_url)
    ngrok.kill()


# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title=os.getenv("API_TITLE", "WB AI Code Intelligence API"),
    version=os.getenv("API_VERSION", "1.0.0"),
    description="Enterprise-grade Agentic RAG for code intelligence",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# ENDPOINTS
# ============================================

@app.get("/", response_model=HealthResponse)
async def health_check():
    """System health check"""
    return HealthResponse(
        status="operational",
        version=os.getenv("API_VERSION", "1.0.0"),
        model="Qwen3-1.7B",
        vectorstore_docs=app.state.vectorstore._collection.count()
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process code intelligence query"""
    try:
        if request.use_agent:
            # Use multi-agent system
            result = app.state.agent_system.execute(request.query)
            
            return QueryResponse(
                answer=result["response"],
                sources=[{"context": result["context"]}],
                metadata={
                    "mode": "agent_system",
                    "iterations": result["iterations"],
                    "analysis": result["analysis"]
                },
                status=result["status"]
            )
        else:
            # Use standard RAG
            result = app.state.rag.query(request.query)
            
            return QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                metadata={"mode": "rag_pipeline"},
                status=result["status"]
            )
            
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve")
async def retrieve_context(query: str, k: int = 5):
    """Retrieve relevant documents without generation"""
    try:
        docs = app.state.rag.retrieve_context(query, k)
        return {
            "query": query,
            "documents": docs,
            "count": len(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """System statistics"""
    return {
        "total_documents": app.state.vectorstore._collection.count(),
        "model": os.getenv("MODEL_NAME"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
        "public_url": getattr(app.state, "public_url", "Not available")
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
