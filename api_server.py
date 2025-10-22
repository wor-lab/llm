"""
WB AI Enterprise - FastAPI Server with Ngrok
Exposes agents and RAG pipeline via REST API
"""

import logging
import asyncio
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from pyngrok import ngrok
import os

logger = logging.getLogger("WB.API")


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    agent_type: Optional[str] = Field(None, description="Specific agent to use")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    query: str
    response: str
    agent_used: Optional[str]
    sources: List[Dict]
    metadata: Dict


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    vector_db_count: int
    agents_available: int


class WBAPIServer:
    """FastAPI server for WB AI Enterprise"""
    
    def __init__(
        self,
        agent_system,
        rag_pipeline,
        host: str = "0.0.0.0",
        port: int = 8000,
        ngrok_token: Optional[str] = None
    ):
        self.agent_system = agent_system
        self.rag_pipeline = rag_pipeline
        self.host = host
        self.port = port
        self.ngrok_token = ngrok_token
        self.public_url = None
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        
        app = FastAPI(
            title="WB AI Enterprise API",
            description="Elite AI Agent System with RAG",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Exception handler
        @app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "detail": str(exc)}
            )
        
        # Routes
        @app.get("/", response_model=Dict)
        async def root():
            """API root"""
            return {
                "service": "WB AI Enterprise",
                "status": "operational",
                "docs": "/docs",
                "health": "/health"
            }
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                version="1.0.0",
                model=self.rag_pipeline.model_name,
                vector_db_count=self.rag_pipeline.collection.count(),
                agents_available=len(self.agent_system.AGENT_CONFIGS)
            )
        
        @app.post("/query", response_model=QueryResponse)
        async def process_query(request: QueryRequest):
            """Process user query through agent system"""
            try:
                logger.info(f"📨 Received query: {request.query[:100]}...")
                
                # Execute through agent system
                response = await self.agent_system.execute_task(request.query)
                
                # Get context docs
                docs = self.rag_pipeline.retrieve(request.query, request.top_k)
                
                return QueryResponse(
                    query=request.query,
                    response=response,
                    agent_used=None,  # Would be populated from agent system
                    sources=docs,
                    metadata={
                        "top_k": request.top_k,
                        "num_sources": len(docs)
                    }
                )
            
            except Exception as e:
                logger.error(f"Query processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/rag", response_model=Dict)
        async def rag_query(request: QueryRequest):
            """Direct RAG query (bypass agent routing)"""
            try:
                result = await self.rag_pipeline.query(request.query, request.top_k)
                return result
            
            except Exception as e:
                logger.error(f"RAG query failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/agents", response_model=Dict)
        async def list_agents():
            """List available agents and their capabilities"""
            return self.agent_system.get_agent_capabilities()
        
        @app.post("/search", response_model=Dict)
        async def search_knowledge_base(request: QueryRequest):
            """Search vector database directly"""
            try:
                docs = self.rag_pipeline.retrieve(request.query, request.top_k)
                return {
                    "query": request.query,
                    "results": docs,
                    "count": len(docs)
                }
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        return app
    
    async def start(self):
        """Start FastAPI server with Ngrok tunnel"""
        
        # Setup Ngrok if token provided
        if self.ngrok_token:
            logger.info("🔗 Setting up Ngrok tunnel...")
            ngrok.set_auth_token(self.ngrok_token)
            
            # Create tunnel
            tunnel = ngrok.connect(self.port, bind_tls=True)
            self.public_url = tunnel.public_url
            
            logger.info(f"✅ Public URL: {self.public_url}")
            logger.info(f"📖 API Docs: {self.public_url}/docs")
        else:
            logger.info("⚠️  No Ngrok token - running locally only")
        
        # Print local URL
        logger.info(f"🌐 Local URL: http://{self.host}:{self.port}")
        
        # Run server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
