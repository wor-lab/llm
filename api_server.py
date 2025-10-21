import os
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn

from rag_pipeline import create_agentic_rag, run_agentic_rag
from agent_system import build_agents


class QueryIn(BaseModel):
    query: str
    k: Optional[int] = 8


class AgentTaskIn(BaseModel):
    task_type: str
    payload: Dict[str, Any] = {}


def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _require_key(x_api_key: Optional[str]):
    api_key = _get_env("API_KEY")
    if not x_api_key or x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


def build_app() -> FastAPI:
    chroma_dir = _get_env("CHROMA_DIR", "./storage/chroma")
    app = FastAPI()
    graph, llm, retriever = create_agentic_rag(chroma_dir)
    store = retriever.vectorstore if hasattr(retriever, "vectorstore") else None  # type: ignore
    agents = build_agents(llm, store)

    app.state.graph = graph
    app.state.agents = agents

    @app.post("/query")
    def query(inp: QueryIn, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
        _require_key(x_api_key)
        res = run_agentic_rag(app.state.graph, inp.query)
        return res

    @app.post("/agent/execute")
    def agent_execute(inp: AgentTaskIn, x_api_key: Optional[str] = Header(default=None, convert_underscores=False)):
        _require_key(x_api_key)
        res = app.state.agents["AutoBot"].execute(inp.task_type, inp.payload)
        return res

    return app


def _start_ngrok(port: int):
    token = _get_env("NGROK_AUTH_TOKEN")
    ngrok.set_auth_token(token)
    tunnel = ngrok.connect(addr=port, proto="http")
    public_url = tunnel.public_url
    os.environ["PUBLIC_API_BASE"] = public_url
    print(f"PUBLIC_API_BASE={public_url}")


def main():
    port = int(os.getenv("PORT", "8000"))
    threading.Thread(target=_start_ngrok, args=(port,), daemon=True).start()
    uvicorn.run(build_app(), host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
