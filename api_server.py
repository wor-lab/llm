"""
WB AI CORPORATION — AUTOMATION HUB
FastAPI Server + NGROK Public Endpoint

MISSION: Expose agent system via REST API
AGENT: AutoBot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from pyngrok import ngrok
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """Request model for agent queries"""
    query: str
    mode: Optional[str] = "auto"  # auto, rag, agent
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    """Response model"""
    query: str
    answer: str
    mode: str
    confidence: float
    sources: Optional[List[Dict]] = []
    timestamp: str
    processing_time: float

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    model: str
    collections: List[str]
    uptime: float

# ══════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ══════════════════════════════════════════════════════

app = FastAPI(
    title="WB AI Corporation API",
    description="Multi-Agent Code Intelligence System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class AppState:
    agent_system = None
    rag_pipeline = None
    start_time = None

state = AppState()

# ══════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ══════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    logger.info("🚀 Starting WB AI Corporation API...")
    
    from agent_system import AgentOrchestrator
    from rag_pipeline import RAGOrchestrator
    import time
    
    state.start_time = time.time()
    
    logger.info("Loading agent system...")
    state.agent_system = AgentOrchestrator()
    
    logger.info("Loading RAG pipeline...")
    state.rag_pipeline = RAGOrchestrator()
    
    logger.info("✅ All systems operational")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🔴 Shutting down WB AI Corporation API...")

# ══════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════

@app.get("/", response_model=Dict)
async def root():
    """API root endpoint"""
    return {
        "name": "WB AI Corporation",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "agent": "/api/agent",
            "rag": "/api/rag",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import time
    import chromadb
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collections = [col.name for col in chroma_client.list_collections()]
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model="Qwen2.5-1.5B-Instruct",
        collections=collections,
        uptime=time.time() - state.start_time if state.start_time else 0
    )

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint - automatically routes to best system
    """
    import time
    start = time.time()
    
    try:
        if request.mode == "agent":
            # Use agent system
            result = state.agent_system.process_query(request.query)
            response = QueryResponse(
                query=result['query'],
                answer=result['answer'],
                mode="agent",
                confidence=result['confidence'],
                sources=[],
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start
            )
        
        elif request.mode == "rag":
            # Use RAG pipeline
            result = state.rag_pipeline.query(request.query)
            response = QueryResponse(
                query=result['query'],
                answer=result['answer'],
                mode="rag",
                confidence=result['confidence'],
                sources=result['sources'],
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start
            )
        
        else:  # auto mode
            # Use RAG by default (faster and more reliable)
            result = state.rag_pipeline.query(request.query)
            response = QueryResponse(
                query=result['query'],
                answer=result['answer'],
                mode="auto(rag)",
                confidence=result['confidence'],
                sources=result['sources'],
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start
            )
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agent")
async def agent_query(request: QueryRequest):
    """Direct agent system query"""
    try:
        result = state.agent_system.process_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rag")
async def rag_query(request: QueryRequest):
    """Direct RAG pipeline query"""
    try:
        result = state.rag_pipeline.query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    import chromadb
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collections = chroma_client.list_collections()
    
    stats = {}
    for col in collections:
        stats[col.name] = col.count()
    
    return {
        "collections": stats,
        "total_documents": sum(stats.values()),
        "model": "Qwen2.5-1.5B-Instruct",
        "embedding_model": "all-MiniLM-L6-v2"
    }

# ══════════════════════════════════════════════════════
# NGROK LAUNCHER
# ══════════════════════════════════════════════════════

def launch_server(port: int = 8000):
    """
    Launch FastAPI server with NGROK tunnel
    """
    logger.info(f"🌐 Starting server on port {port}...")
    
    # Set NGROK auth token
    ngrok_token = os.getenv('NGROK_AUTH_TOKEN')
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        logger.info("✅ NGROK authenticated")
    
    # Start NGROK tunnel
    public_url = ngrok.connect(port)
    logger.info(f"🌍 Public URL: {public_url}")
    
    print(f"""
╔════════════════════════════════════════════════════╗
║   🏢 WB AI CORPORATION — API OPERATIONAL           ║
╠════════════════════════════════════════════════════╣
║   Public Endpoint: {public_url}           
║   Documentation:   {public_url}/docs       
║   Health Check:    {public_url}/health     
╚════════════════════════════════════════════════════╝
    """)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

if __name__ == "__main__":
    launch_server()
