"""
WB AI CORPORATION - LANGGRAPH AGENT SYSTEM
Agent: CodeArchitect + AutoBot
Purpose: Multi-agent orchestration with LangGraph
"""

import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from rag_pipeline import WBRAGPipeline
from dotenv import load_dotenv
import operator

load_dotenv()


class AgentState(TypedDict):
    """Agent state schema"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    rag_result: dict
    iterations: int
    final_answer: str


class WBAgentSystem:
    """LangGraph-based agentic orchestration"""
    
    def __init__(self):
        self.rag_pipeline = WBRAGPipeline()
        self.max_iterations = int(os.getenv("MAX_ITERATIONS", 5))
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Construct agent workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("validate_output", self.validate_output)
        
        # Define edges
        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "validate_output")
        
        # Conditional edge: retry or finish
        workflow.add_conditional_edges(
            "validate_output",
            self.should_continue,
            {
                "continue": "retrieve_context",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def analyze_query(self, state: AgentState) -> AgentState:
        """Query analysis node"""
        print("[Analyst] Analyzing query intent...")
        state["iterations"] = state.get("iterations", 0) + 1
        
        # Add analysis message
        state["messages"].append(
            AIMessage(content=f"[Query Analysis] Intent: Technical coding assistance")
        )
        return state
    
    def retrieve_context(self, state: AgentState) -> AgentState:
        """RAG retrieval node"""
        print("[DataSynth] Retrieving relevant context...")
        
        docs = self.rag_pipeline.search_vectorstore(state["query"], k=5)
        context = "\n\n".join([doc["content"][:500] for doc in docs])
        
        state["messages"].append(
            AIMessage(content=f"[Context Retrieved] {len(docs)} documents found")
        )
        return state
    
    def generate_response(self, state: AgentState) -> AgentState:
        """LLM generation node"""
        print("[CodeArchitect] Generating response...")
        
        result = self.rag_pipeline.query(state["query"])
        state["rag_result"] = result
        state["final_answer"] = result["answer"]
        
        state["messages"].append(
            AIMessage(content=f"[Generation] Response generated")
        )
        return state
    
    def validate_output(self, state: AgentState) -> AgentState:
        """Output validation node"""
        print("[SecAnalyst] Validating output quality...")
        
        # Simple validation: check response length
        if len(state["final_answer"]) < 50:
            state["messages"].append(
                AIMessage(content="[Validation] Response too short - retry")
            )
        else:
            state["messages"].append(
                AIMessage(content="[Validation] Output validated - complete")
            )
        
        return state
    
    def should_continue(self, state: AgentState) -> str:
        """Decision function for graph routing"""
        if state["iterations"] >= self.max_iterations:
            return "end"
        
        if len(state["final_answer"]) < 50:
            return "continue"
        
        return "end"
    
    def execute(self, query: str) -> dict:
        """Execute agent workflow"""
        print(f"\n[WB AI CORE] Initializing agent workflow...")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "rag_result": {},
            "iterations": 0,
            "final_answer": ""
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "answer": final_state["final_answer"],
            "iterations": final_state["iterations"],
            "sources": final_state["rag_result"].get("source_documents", []),
            "workflow_trace": [msg.content for msg in final_state["messages"]]
        }


if __name__ == "__main__":
    agent = WBAgentSystem()
    result = agent.execute("Write a Python function to implement JWT authentication")
    
    print("\n" + "="*80)
    print("[WB AI AGENT RESPONSE]")
    print("="*80)
    print(result["answer"])
    print("\n[Workflow Trace]:", result["workflow_trace"])
