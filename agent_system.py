# Algorithms & FULLCODE for agent_system.py
# Algorithm: Multi-Agent Orchestration with LangGraph
# 1. Define 8 agents as LangChain tools/classes.
# 2. Create LangGraph: Nodes for each agent, edges for workflow (sequential/conditional).
# 3. WB AI Core: Orchestrates graph execution based on project request.
# 4. Execute: Assemble agents -> Run graph -> Document output.
# Optimization: Conditional routing in graph for efficiency; parallel edges where possible.

from langgraph.graph import StateGraph, END
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import HuggingFacePipeline
from rag_pipeline import run_rag_query
from transformers import pipeline  # Reuse from rag_pipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Local LLM (shared)
llm = HuggingFacePipeline.from_model_id(
    model_id=os.getenv("MODEL_NAME", "Qwen/Qwen2-1.5B"),
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 256}
)

# Define 8 agents as tools (modular classes)
class CodeArchitect:
    def __call__(self, input): return f"Generated code: {input}"  # Placeholder; integrate real coding logic

class OpsManager:
    def __call__(self, input): return f"Deployed infra: {input}"

class SecAnalyst:
    def __call__(self, input): return f"Security audit: {input}"

class DesignMind:
    def __call__(self, input): return f"UI design: {input}"

class WordSmith:
    def __call__(self, input): return f"Content: {input}"

class DataSynth:
    def __call__(self, input): return f"Data analysis: {input}"

class Analyst:
    def __call__(self, input): return f"Strategy: {input}"

class AutoBot:
    def __call__(self, input): return f"Automated workflow: {input}"

# Tools list for agents
tools = [
    Tool(name="CodeArchitect", func=CodeArchitect(), description="Code complex systems"),
    Tool(name="OpsManager", func=OpsManager(), description="Manage infra"),
    Tool(name="SecAnalyst", func=SecAnalyst(), description="Security analysis"),
    Tool(name="DesignMind", func=DesignMind(), description="UX/UI design"),
    Tool(name="WordSmith", func=WordSmith(), description="Content creation"),
    Tool(name="DataSynth", func=DataSynth(), description="Data tasks"),
    Tool(name="Analyst", func=Analyst(), description="Strategy"),
    Tool(name="AutoBot", func=AutoBot(), description="Automation")
]

# Create LangGraph for orchestration
def create_agent_graph(rag_chain):
    graph = StateGraph()

    # Add nodes (agents)
    for tool in tools:
        agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        graph.add_node(tool.name, lambda state: agent.run(state["input"]))

    # Add RAG node for data retrieval
    graph.add_node("RAG", lambda state: run_rag_query(rag_chain, state["input"]))

    # Edges: Simple sequential for demo; optimize with conditions
    graph.set_entry_point("RAG")
    prev = "RAG"
    for tool in tools:
        graph.add_edge(prev, tool.name)
        prev = tool.name
    graph.add_edge(prev, END)

    return graph.compile()

# WB AI Core: Run project
def run_project(agent_graph, request):
    # Workflow steps
    scope = f"Scope: {request}"
    process = "Assembling agents..."
    output = agent_graph.invoke({"input": request})
    next_steps = "Iterate based on output."

    return {
        "goal": scope,
        "process": process,
        "output": output,
        "next_steps": next_steps
    }
