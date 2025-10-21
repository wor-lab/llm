"""
WB AI Corporation - Multi-Agent System
LangGraph-based orchestration of 8 specialized agents.
Architecture: State machine with agent coordination and task routing.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from loguru import logger

from rag_pipeline import ProductionRAGPipeline


class AgentType(Enum):
    """Available WB AI agents."""
    CODE_ARCHITECT = "CodeArchitect"
    OPS_MANAGER = "OpsManager"
    SEC_ANALYST = "SecAnalyst"
    DESIGN_MIND = "DesignMind"
    WORD_SMITH = "WordSmith"
    DATA_SYNTH = "DataSynth"
    ANALYST = "Analyst"
    AUTO_BOT = "AutoBot"


class AgentState(TypedDict):
    """Shared state across all agents."""
    task: str
    context: List[Document]
    current_agent: str
    messages: Annotated[List[str], operator.add]
    result: str
    metadata: Dict[str, Any]


class WBAgentSystem:
    """
    Multi-agent orchestration system with:
    - Task routing based on keywords
    - Agent specialization
    - State management via LangGraph
    - RAG integration
    """
    
    AGENT_PROMPTS = {
        AgentType.CODE_ARCHITECT: """You are CodeArchitect - elite software engineer.
Specialization: Complex systems (Python, JS, Rust, Go), APIs, architecture.
Task: {task}
Context: {context}
Output production-grade code with type hints, error handling, and documentation.""",

        AgentType.OPS_MANAGER: """You are OpsManager - infrastructure expert.
Specialization: CI/CD, cloud (AWS/GCP), containers, Kubernetes, monitoring.
Task: {task}
Context: {context}
Provide deployment configs, infrastructure-as-code, and operational best practices.""",

        AgentType.SEC_ANALYST: """You are SecAnalyst - security specialist.
Specialization: Penetration testing, security audits, threat modeling.
Task: {task}
Context: {context}
Identify vulnerabilities, provide mitigation strategies, and security hardening steps.""",

        AgentType.DESIGN_MIND: """You are DesignMind - UX/UI expert.
Specialization: User experience, Tailwind CSS, Figma, design systems.
Task: {task}
Context: {context}
Create intuitive designs, responsive layouts, and brand-consistent interfaces.""",

        AgentType.WORD_SMITH: """You are WordSmith - content strategist.
Specialization: Technical documentation, SEO, marketing content.
Task: {task}
Context: {context}
Produce clear, professional, and SEO-optimized content.""",

        AgentType.DATA_SYNTH: """You are DataSynth - data scientist.
Specialization: Analysis (pandas), SQL, dashboards, ML models.
Task: {task}
Context: {context}
Provide data-driven insights, visualizations, and statistical analysis.""",

        AgentType.ANALYST: """You are Analyst - business strategist.
Specialization: Market analysis, business plans, forecasting.
Task: {task}
Context: {context}
Deliver actionable strategies, market insights, and growth plans.""",

        AgentType.AUTO_BOT: """You are AutoBot - automation engineer.
Specialization: API integration, workflows (FastAPI, n8n), task automation.
Task: {task}
Context: {context}
Build automated workflows, API connectors, and integration solutions."""
    }
    
    ROUTING_KEYWORDS = {
        AgentType.CODE_ARCHITECT: ["code", "python", "api", "function", "class", "algorithm", "architecture"],
        AgentType.OPS_MANAGER: ["deploy", "docker", "kubernetes", "ci/cd", "infrastructure", "cloud"],
        AgentType.SEC_ANALYST: ["security", "vulnerability", "audit", "penetration", "threat", "encryption"],
        AgentType.DESIGN_MIND: ["design", "ui", "ux", "tailwind", "figma", "interface", "responsive"],
        AgentType.WORD_SMITH: ["document", "write", "seo", "content", "marketing", "blog"],
        AgentType.DATA_SYNTH: ["data", "analysis", "pandas", "sql", "dashboard", "visualization"],
        AgentType.ANALYST: ["business", "strategy", "market", "forecast", "plan", "growth"],
        AgentType.AUTO_BOT: ["automate", "workflow", "integration", "n8n", "webhook", "cron"]
    }
    
    def __init__(self, rag_pipeline: ProductionRAGPipeline):
        self.rag = rag_pipeline
        self.graph = self._build_graph()
        logger.info("WB Agent System initialized with 8 agents")
    
    def _route_task(self, state: AgentState) -> AgentType:
        """Intelligent task routing based on keywords."""
        task_lower = state["task"].lower()
        scores = {}
        
        for agent, keywords in self.ROUTING_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            scores[agent] = score
        
        selected = max(scores, key=scores.get)
        logger.info(f"Task routed to: {selected.value}")
        return selected
    
    def _execute_agent(self, state: AgentState, agent_type: AgentType) -> AgentState:
        """Execute specific agent with RAG."""
        prompt = self.AGENT_PROMPTS[agent_type].format(
            task=state["task"],
            context="\n".join([doc.page_content[:300] for doc in state["context"][:3]])
        )
        
        rag_result = self.rag.query(prompt)
        
        state["current_agent"] = agent_type.value
        state["messages"].append(f"[{agent_type.value}] Processing task...")
        state["result"] = rag_result["answer"]
        state["metadata"]["sources"] = rag_result["source_documents"]
        
        return state
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine."""
        workflow = StateGraph(AgentState)
        
        # Define agent nodes
        for agent in AgentType:
            workflow.add_node(
                agent.value,
                lambda s, a=agent: self._execute_agent(s, a)
            )
        
        # Routing logic
        def route_to_agent(state: AgentState):
            agent = self._route_task(state)
            return agent.value
        
        # Add conditional routing from START
        workflow.set_conditional_entry_point(route_to_agent)
        
        # All agents lead to END
        for agent in AgentType:
            workflow.add_edge(agent.value, END)
        
        return workflow.compile()
    
    def execute_task(self, task: str) -> Dict[str, Any]:
        """Execute task through agent system."""
        logger.info(f"Executing task: {task[:100]}...")
        
        # Retrieve relevant context via RAG
        context_result = self.rag.query(task)
        context_docs = [
            Document(page_content=doc["content"], metadata=doc["metadata"])
            for doc in context_result["source_documents"]
        ]
        
        initial_state: AgentState = {
            "task": task,
            "context": context_docs,
            "current_agent": "",
            "messages": [],
            "result": "",
            "metadata": {}
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "agent": final_state["current_agent"],
            "result": final_state["result"],
            "messages": final_state["messages"],
            "sources": final_state["metadata"].get("sources", [])
        }


def initialize_agent_system(rag_pipeline: ProductionRAGPipeline) -> WBAgentSystem:
    """Factory function to create agent system."""
    return WBAgentSystem(rag_pipeline)
