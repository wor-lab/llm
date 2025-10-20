"""
WB AI CORPORATION - Agentic RAG System
Multi-Agent Workflow with LangGraph
═══════════════════════════════════════
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from typing import TypedDict, List, Annotated
import operator
from data_pipeline import DataPipeline

# ═══════════════════════════════════════════════
# STATE DEFINITION
# ═══════════════════════════════════════════════

class AgentState(TypedDict):
    """Shared state across all agents"""
    query: str
    retrieved_contexts: List[dict]
    generated_code: str
    evaluation_score: float
    feedback: str
    refinement_count: int
    agent_path: Annotated[List[str], operator.add]
    final_output: str

# ═══════════════════════════════════════════════
# AGENTIC RAG SYSTEM
# ═══════════════════════════════════════════════

class AgenticRAG:
    """Multi-agent RAG system for code generation"""
    
    def __init__(self, api_base: str, api_key: str):
        self.api_base = api_base
        self.api_key = api_key
        
        # Initialize LLM client
        self.llm = ChatOpenAI(
            base_url=f"{api_base}/v1",
            api_key=api_key,
            model="qwen3-1.7b",
            temperature=0.7
        )
        
        # Initialize data pipeline
        self.data_pipeline = DataPipeline()
        
        # Build agent graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow"""
        
        from agents import (
            retriever_agent,
            generator_agent,
            evaluator_agent,
            refiner_agent
        )
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("retrieve", lambda state: retriever_agent(state, self.data_pipeline))
        workflow.add_node("generate", lambda state: generator_agent(state, self.llm))
        workflow.add_node("evaluate", lambda state: evaluator_agent(state, self.llm))
        workflow.add_node("refine", lambda state: refiner_agent(state, self.llm))
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")
        
        # Conditional edge: refine or end
        workflow.add_conditional_edges(
            "evaluate",
            lambda state: "refine" if state["evaluation_score"] < 0.7 and state["refinement_count"] < 2 else "end",
            {
                "refine": "refine",
                "end": END
            }
        )
        
        workflow.add_edge("refine", "generate")
        
        return workflow
    
    def execute(self, query: str) -> dict:
        """Execute agentic RAG workflow"""
        
        initial_state = {
            "query": query,
            "retrieved_contexts": [],
            "generated_code": "",
            "evaluation_score": 0.0,
            "feedback": "",
            "refinement_count": 0,
            "agent_path": [],
            "final_output": ""
        }
        
        # Run workflow
        final_state = self.app.invoke(initial_state)
        
        return {
            "code": final_state["generated_code"],
            "score": final_state["evaluation_score"],
            "agent_path": final_state["agent_path"],
            "contexts": final_state["retrieved_contexts"],
            "refinements": final_state["refinement_count"]
        }
