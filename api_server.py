"""
WB AI CORPORATION - FASTAPI SERVER
Agent: AutoBot + SecAnalyst
Purpose: Production API with NGROK tunneling
"""

import os
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from agent_system import WBAgentSystem
from rag_pipeline import WBRAGPipeline
from dotenv import load_dotenv
from pyngrok import ngrok
import uvicorn

load_dotenv()

# Initialize core systems
app = FastAPI(
    title="WB AI Corporation - Agentic RAG API",
    description="Enterprise-grade RAG system with LangGraph agents",
    version="1.0.0"
)

agent_system = None
rag_pipeline = None

# Security
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    """API key authentication"""
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


class QueryRequest(BaseModel):
    query: str
    use_agent: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list
    metadata: dict


@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    global agent_system, rag_pipeline
    
    print("[OpsManager] Starting WB AI API server...")
    
    # Check if vectorstore exists
    if not os.path.exists(os.getenv("CHROMA_PERSIST_DIR")):
        print("[WARNING] Vectorstore not found. Run dataset_loader.py first.")
    
    rag_pipeline = WBRAGPipeline()
    agent_system = WBAgentSystem()
    
    print("[WB AI] Systems online. API ready.")


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "service": "WB AI Corporation - Agentic RAG",
        "version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Main query endpoint"""
    try:
        if request.use_agent:
            # Use LangGraph agent system
            result = agent_system.execute(request.query)
            return QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                metadata={
                    "workflow": "agent_system",
                    "iterations": result["iterations"],
                    "trace": result["workflow_trace"]
                }
            )
        else:
            # Direct RAG pipeline
            result = rag_pipeline.query(request.query)
            return QueryResponse(
                answer=result["answer"],
                sources=result["source_documents"],
                metadata={"workflow": "direct_rag"}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/search")
async def search_endpoint(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Vector search endpoint"""
    try:
        results = rag_pipeline.search_vectorstore(request.query, k=10)
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def setup_ngrok():
    """Configure NGROK tunnel"""
    token = os.getenv("NGROK_AUTH_TOKEN")
    if token and token != "your_ngrok_token_here":
        ngrok.set_auth_token(token)
        port = int(os.getenv("API_PORT", 8080))
        public_url = ngrok.connect(port)
        print(f"[AutoBot] NGROK tunnel established: {public_url}")
        return public_url
    else:
        print("[WARNING] NGROK_AUTH_TOKEN not configured. Running local only.")
        return None


if __name__ == "__main__":
    # Setup NGROK
    ngrok_url = setup_ngrok()
    
    # Run server
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8080)),
        log_level="info"
    )
