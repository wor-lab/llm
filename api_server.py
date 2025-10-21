import os
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from rag_pipeline import RAGResources, env
from dataset_loader import ingest_all
from agent_system import WBCore

from pyngrok import ngrok

app = FastAPI(title="WB AI Core", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESOURCES: Optional[RAGResources] = None
CORE: Optional[WBCore] = None
NGROK_URL: Optional[str] = None

@app.on_event("startup")
def on_startup():
    global RESOURCES, CORE, NGROK_URL
    RESOURCES = RAGResources()
    CORE = WBCore(RESOURCES)
    token = env("NGROK_AUTH_TOKEN", "")
    if token:
        ngrok.set_auth_token(token)
        port = int(env("API_PORT", "8000"))
        # open http tunnel
        NGROK_URL = ngrok.connect(port, "http").public_url
        print(f"[ngrok] public_url: {NGROK_URL}")

@app.get("/health")
def health():
    return {"status": "ok", "ngrok": NGROK_URL}

@app.post("/ingest")
def ingest(payload: Dict[str, Any] = Body(...)):
    """
    POST body:
    {
      "swe_max": 500,
      "stack_max": 200,
      "rstar_max": 200,
      "stack_langs": ["python","javascript"]
    }
    """
    assert RESOURCES is not None
    swe_max = int(payload.get("swe_max", 500))
    stack_max = int(payload.get("stack_max", 200))
    rstar_max = int(payload.get("rstar_max", 200))
    stack_langs = payload.get("stack_langs")
    result = ingest_all(RESOURCES, swe_max, stack_max, rstar_max, stack_langs)
    return {"ok": True, "result": result}

@app.get("/chroma/stats")
def chroma_stats():
    assert RESOURCES is not None
    return {"stats": RESOURCES.vs.stats()}

@app.post("/query")
def query(payload: Dict[str, Any] = Body(...)):
    """
    POST body:
    { "query": "...", "collections": ["swe_bench","the_stack"], "k": 5 }
    """
    assert RESOURCES is not None
    q = payload["query"]
    cols = payload.get("collections")
    k = int(payload.get("k", 5))
    res = RESOURCES.rag_answer(q, collections=cols, k=k)
    return res

@app.post("/agent/run")
def agent_run(payload: Dict[str, Any] = Body(...)):
    """
    POST body:
    { "task": "...", "agent": "CodeArchitect", "params": { ... } }
    """
    assert CORE is not None
    task = payload["task"]
    agent = payload.get("agent")
    params = payload.get("params", {})
    res = CORE.run(task=task, agent=agent, params=params)
    return res


def main():
    host = env("API_HOST", "0.0.0.0")
    port = int(env("API_PORT", "8000"))
    uvicorn.run("api_server:app", host=host, port=port, reload=bool(int(env("UVICORN_RELOAD", "0"))))


if __name__ == "__main__":
    main()
