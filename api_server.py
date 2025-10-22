# Algorithms & FULLCODE for api_server.py
# Algorithm: API Server Workflow
# 1. Define FastAPI endpoints for project execution and RAG queries.
# 2. Integrate with agent_system and rag_pipeline.
# 3. Handle requests asynchronously.
# 4. Return structured responses (goal, process, output, next steps).
# Optimization: Async endpoints, rate-limiting optional via middleware.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_system import create_agent_graph, run_project
from rag_pipeline import get_rag_chain, run_rag_query
from dataset_loader import load_datasets_to_chroma

app = FastAPI(title="WB AI Corporation API")

class ProjectRequest(BaseModel):
    request: str

class RAGQuery(BaseModel):
    query: str

# Load shared resources on startup (optimized singleton)
collection = load_datasets_to_chroma(subset_size=100)
rag_chain = get_rag_chain(collection)
agent_graph = create_agent_graph(rag_chain)

@app.post("/execute_project")
async def execute_project(req: ProjectRequest):
    try:
        result = run_project(agent_graph, req.request)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag_query")
async def rag_query(req: RAGQuery):
    try:
        result = run_rag_query(rag_chain, req.query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
