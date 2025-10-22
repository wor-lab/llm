"""
WB AI Enterprise - 8-Agent System with LangGraph
Orchestrates specialized agents for complex task execution
"""

import logging
from typing import Dict, List, Optional, TypedDict, Annotated
from enum import Enum
import asyncio

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("WB.Agents")


class AgentType(str, Enum):
    """Available agent types"""
    CODE_ARCHITECT = "CodeArchitect"
    OPS_MANAGER = "OpsManager"
    SEC_ANALYST = "SecAnalyst"
    DESIGN_MIND = "DesignMind"
    WORD_SMITH = "WordSmith"
    DATA_SYNTH = "DataSynth"
    ANALYST = "Analyst"
    AUTO_BOT = "AutoBot"


class AgentState(TypedDict):
    """State passed between agents"""
    query: str
    agent_type: Optional[str]
    context: List[Dict]
    intermediate_results: Dict[str, str]
    final_response: str
    metadata: Dict


class WBAgentSystem:
    """Orchestrates 8 specialized agents using LangGraph"""
    
    AGENT_CONFIGS = {
        AgentType.CODE_ARCHITECT: {
            "role": "Elite Software Engineer",
            "expertise": "Python, JS, Rust, Go, system architecture, algorithms, API design",
            "keywords": ["code", "implement", "architect", "api", "algorithm", "debug", "refactor"]
        },
        AgentType.OPS_MANAGER: {
            "role": "DevOps & Infrastructure Expert",
            "expertise": "CI/CD, Docker, Kubernetes, AWS, monitoring, scaling",
            "keywords": ["deploy", "infra", "docker", "kubernetes", "ci/cd", "scale"]
        },
        AgentType.SEC_ANALYST: {
            "role": "Security Specialist",
            "expertise": "Penetration testing, threat modeling, security audits, compliance",
            "keywords": ["security", "audit", "vulnerability", "pentest", "threat"]
        },
        AgentType.DESIGN_MIND: {
            "role": "UX/UI Designer",
            "expertise": "Tailwind CSS, Figma, user experience, branding, design systems",
            "keywords": ["design", "ui", "ux", "tailwind", "figma", "brand"]
        },
        AgentType.WORD_SMITH: {
            "role": "Technical Writer & Content Strategist",
            "expertise": "Documentation, SEO, marketing copy, technical writing",
            "keywords": ["write", "document", "content", "seo", "marketing"]
        },
        AgentType.DATA_SYNTH: {
            "role": "Data Engineer & Analyst",
            "expertise": "Pandas, SQL, data pipelines, visualization, ML models",
            "keywords": ["data", "analyze", "sql", "pandas", "dashboard", "model"]
        },
        AgentType.ANALYST: {
            "role": "Business Strategist",
            "expertise": "Market analysis, business plans, forecasting, competitive analysis",
            "keywords": ["strategy", "business", "market", "forecast", "plan"]
        },
        AgentType.AUTO_BOT: {
            "role": "Automation Specialist",
            "expertise": "API integration, FastAPI, n8n, workflow automation, webhooks",
            "keywords": ["automate", "api", "integrate", "workflow", "webhook"]
        }
    }
    
    def __init__(self, rag_pipeline, model_name: str):
        self.rag_pipeline = rag_pipeline
        self.model_name = model_name
        self.graph = None
    
    async def initialize(self):
        """Initialize LangGraph workflow"""
        logger.info("🤖 Building agent workflow graph...")
        self.graph = self._build_graph()
        logger.info("✅ Agent system ready")
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph state machine"""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("synthesizer", self._synthesizer_node)
        
        # Add edges
        workflow.set_entry_point("router")
        workflow.add_edge("router", "executor")
        workflow.add_edge("executor", "synthesizer")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    async def _router_node(self, state: AgentState) -> AgentState:
        """Route query to appropriate agent"""
        query = state['query'].lower()
        
        # Score each agent based on keyword matching
        scores = {}
        for agent_type, config in self.AGENT_CONFIGS.items():
            score = sum(1 for keyword in config['keywords'] if keyword in query)
            scores[agent_type] = score
        
        # Select agent with highest score
        selected_agent = max(scores, key=scores.get)
        
        # If no strong match, default to CodeArchitect
        if scores[selected_agent] == 0:
            selected_agent = AgentType.CODE_ARCHITECT
        
        logger.info(f"🎯 Routed to: {selected_agent}")
        
        state['agent_type'] = selected_agent
        state['metadata'] = {
            'scores': {k.value: v for k, v in scores.items()},
            'selected': selected_agent.value
        }
        
        return state
    
    async def _executor_node(self, state: AgentState) -> AgentState:
        """Execute agent-specific logic with RAG"""
        agent_type = state['agent_type']
        config = self.AGENT_CONFIGS[agent_type]
        
        # Retrieve relevant context
        context_docs = self.rag_pipeline.retrieve(state['query'], top_k=5)
        state['context'] = context_docs
        
        # Build agent-specific prompt
        agent_prompt = self._build_agent_prompt(
            query=state['query'],
            role=config['role'],
            expertise=config['expertise'],
            context_docs=context_docs
        )
        
        # Generate response using RAG
        response = await self.rag_pipeline.generate(state['query'], context_docs)
        
        state['intermediate_results'][agent_type] = response
        
        return state
    
    async def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final response"""
        # For now, use the primary agent's response
        agent_type = state['agent_type']
        final_response = state['intermediate_results'].get(agent_type, "No response generated")
        
        # Add metadata
        state['final_response'] = final_response
        
        return state
    
    def _build_agent_prompt(self, query: str, role: str, expertise: str, context_docs: List[Dict]) -> str:
        """Build specialized prompt for agent"""
        context = self.rag_pipeline._format_context(context_docs)
        
        return f"""You are {role} at WB AI Enterprise.

Your expertise: {expertise}

Context from knowledge base:
{context}

User Query: {query}

Provide a production-ready, actionable response. Be confident and precise."""
    
    async def execute_task(self, query: str) -> str:
        """Execute task through agent system"""
        
        # Initialize state
        initial_state: AgentState = {
            'query': query,
            'agent_type': None,
            'context': [],
            'intermediate_results': {},
            'final_response': '',
            'metadata': {}
        }
        
        # Run graph
        try:
            final_state = await self.graph.ainvoke(initial_state)
            response = final_state['final_response']
            
            # Log execution
            logger.info(f"✅ Task completed by {final_state['agent_type']}")
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Task execution failed: {str(e)}")
            return f"Error executing task: {str(e)}"
    
    def get_agent_capabilities(self) -> Dict:
        """Return agent capabilities for API documentation"""
        return {
            agent.value: {
                'role': config['role'],
                'expertise': config['expertise'],
                'keywords': config['keywords']
            }
            for agent, config in self.AGENT_CONFIGS.items()
        }
