import os
import json
from typing import Dict, Any, List, Optional, Literal

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from rag_pipeline import RAGResources, env

import subprocess
import shlex
import requests
from pathlib import Path

# Workspace setup
WORKSPACE = Path(os.getenv("WORKSPACE_DIR", "./workspace"))
WORKSPACE.mkdir(parents=True, exist_ok=True)

# Restricted command execution
DEFAULT_ALLOWED_CMDS = [
    "pytest",
    "bandit",
    "pip-audit",
    "python",  # for linters or module checks, no arbitrary code by default
]

def run_command(command: str, cwd: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Execute whitelisted shell commands safely.
    """
    allowed = set([c.strip() for c in os.getenv("ALLOWED_CMDS", ",".join(DEFAULT_ALLOWED_CMDS)).split(",")])
    parts = shlex.split(command)
    if not parts:
        return {"ok": False, "error": "Empty command"}
    if parts[0] not in allowed:
        return {"ok": False, "error": f"Command '{parts[0]}' not allowed"}
    try:
        proc = subprocess.run(
            parts,
            cwd=cwd or str(WORKSPACE),
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return {"ok": proc.returncode == 0, "code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "TimeoutExpired"}

# File tools

def read_file(path: str) -> str:
    p = WORKSPACE / path
    return p.read_text(encoding="utf-8")

def write_file(path: str, content: str, mode: str = "w") -> str:
    p = WORKSPACE / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open(mode, encoding="utf-8") as f:
        f.write(content)
    return str(p)

def list_dir(path: str = ".") -> List[str]:
    p = WORKSPACE / path
    return [str(x) for x in p.glob("**/*") if x.is_file()]

# HTTP tools

def http_get(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    r = requests.get(url, headers=headers, timeout=30)
    return {"status": r.status_code, "headers": dict(r.headers), "text": r.text[:20000]}

def http_post(url: str, json_body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    r = requests.post(url, json=json_body, headers=headers, timeout=30)
    return {"status": r.status_code, "headers": dict(r.headers), "text": r.text[:20000]}

# RAG tool

def rag_search(resources: RAGResources, query: str, collections: Optional[List[str]] = None, k: int = 5) -> Dict[str, Any]:
    return resources.rag_answer(query, collections=collections, k=k)

# Agent prompts per role

ROLE_SYSTEM_PROMPTS: Dict[str, str] = {
    "CodeArchitect": "You are CodeArchitect. Design and write high-quality code, tests, and repo ops. Use tools precisely.",
    "OpsManager": "You are OpsManager. Manage services, health checks, ngrok, and CI/CD considerations.",
    "SecAnalyst": "You are SecAnalyst. Perform security audits with bandit and pip-audit. Report concise, actionable findings.",
    "DesignMind": "You are DesignMind. Improve UX copy and component specs with Tailwind/Figma logic where relevant.",
    "WordSmith": "You are WordSmith. Create professional docs: API specs, release notes, and brief summaries.",
    "DataSynth": "You are DataSynth. Retrieve, summarize and produce dataset-backed insights. Keep it factual.",
    "Analyst": "You are Analyst. Build strategic summaries grounded in retrieved corpus. Be precise.",
    "AutoBot": "You are AutoBot. Trigger automation webhooks and glue workflows.",
}

# Build tools for agents

def build_tools(resources: RAGResources) -> List[Tool]:
    return [
        Tool(
            name="read_file",
            description="Read a file from workspace. Input: relative path string.",
            func=read_file,
        ),
        Tool(
            name="write_file",
            description="Write content to a file. Input: JSON {path, content, mode='w'}",
            func=lambda s: _write_file_from_json(s),
        ),
        Tool(
            name="list_dir",
            description="List files in workspace path. Input: relative path or '.'",
            func=list_dir,
        ),
        Tool(
            name="run_command",
            description="Run a whitelisted shell command. Input: JSON {command, cwd?, timeout?}",
            func=lambda s: _run_cmd_from_json(s),
        ),
        Tool(
            name="rag_search",
            description="RAG: Search across collections. Input: JSON {query, collections?, k?}",
            func=lambda s: _rag_search_from_json(resources, s),
        ),
        Tool(
            name="http_get",
            description="HTTP GET. Input: JSON {url, headers?}",
            func=lambda s: _http_get_from_json(s),
        ),
        Tool(
            name="http_post",
            description="HTTP POST. Input: JSON {url, json, headers?}",
            func=lambda s: _http_post_from_json(s),
        ),
    ]

def _write_file_from_json(s: str) -> str:
    o = json.loads(s)
    return write_file(o["path"], o["content"], o.get("mode", "w"))

def _run_cmd_from_json(s: str) -> str:
    o = json.loads(s)
    res = run_command(o["command"], cwd=o.get("cwd"), timeout=int(o.get("timeout", 60)))
    return json.dumps(res, ensure_ascii=False)

def _rag_search_from_json(resources: RAGResources, s: str) -> str:
    o = json.loads(s)
    res = rag_search(resources, o["query"], o.get("collections"), int(o.get("k", 5)))
    return json.dumps(res, ensure_ascii=False)

def _http_get_from_json(s: str) -> str:
    o = json.loads(s)
    return json.dumps(http_get(o["url"], headers=o.get("headers")), ensure_ascii=False)

def _http_post_from_json(s: str) -> str:
    o = json.loads(s)
    return json.dumps(http_post(o["url"], json_body=o.get("json", {}), headers=o.get("headers")), ensure_ascii=False)

# Agent Factory

def build_agent(resources: RAGResources, role: str):
    tools = build_tools(resources)
    llm = resources.llm
    system_prompt = ROLE_SYSTEM_PROMPTS.get(role, "You are an effective enterprise agent.")
    # Use Zero-shot ReAct agent with LLM (HF pipeline)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={
            "system_message": system_prompt
        },
    )
    return agent

# LangGraph Orchestrator

class WBState(BaseModel):
    task: str = Field(..., description="High-level task request")
    agent: Optional[str] = Field(default=None, description="Agent to run")
    input: Optional[dict] = Field(default=None, description="Additional params")
    result: Optional[dict] = Field(default=None, description="Final result")


class WBCore:
    """
    WB AI Core with LangGraph router -> agent -> finalize.
    """

    def __init__(self, resources: RAGResources):
        self.resources = resources
        self.agents = {name: build_agent(resources, name) for name in ROLE_SYSTEM_PROMPTS.keys()}
        self.graph = self._build_graph()

    def _router(self, state: WBState) -> str:
        # Simple route: if agent provided, use it; else choose based on keywords
        if state.agent in self.agents:
            return state.agent
        t = (state.task or "").lower()
        if any(k in t for k in ["deploy", "serve", "ngrok", "infra", "uptime", "health"]):
            return "OpsManager"
        if any(k in t for k in ["security", "audit", "vuln", "cve", "bandit", "dependency"]):
            return "SecAnalyst"
        if any(k in t for k in ["design", "ux", "tailwind", "ui"]):
            return "DesignMind"
        if any(k in t for k in ["doc", "docs", "documentation", "release", "api spec", "readme"]):
            return "WordSmith"
        if any(k in t for k in ["data", "metrics", "dashboard", "analysis"]):
            return "DataSynth"
        if any(k in t for k in ["market", "strategy", "competitor", "forecast"]):
            return "Analyst"
        if any(k in t for k in ["automation", "webhook", "n8n", "trigger"]):
            return "AutoBot"
        if any(k in t for k in ["code", "build", "test", "refactor", "fix"]):
            return "CodeArchitect"
        return "CodeArchitect"

    def _agent_exec(self, state: WBState) -> WBState:
        role = self._router(state)
        agent = self.agents[role]
        # Compose a compact instruction for the agent
        instruction = state.task
        if state.input:
            instruction = f"{state.task}\n\nExtra:\n{json.dumps(state.input, ensure_ascii=False)}"
        output = agent.invoke(instruction)
        state.result = {"agent": role, "output": output}
        return state

    def _build_graph(self):
        g = StateGraph(WBState)
        g.add_node("run_agent", self._agent_exec)
        g.set_entry_point("run_agent")
        g.add_edge("run_agent", END)
        return g.compile()

    def run(self, task: str, agent: Optional[str] = None, params: Optional[dict] = None) -> dict:
        init = WBState(task=task, agent=agent, input=params or {})
        out = self.graph.invoke(init)
        return out.result or {}
