# agent_system.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from rag_pipeline import get_rag_chain

# --- Agent State Definition ---
class AgentState(TypedDict):
    """Defines the state of the agent's thought process."""
    question: str
    generation: str
    context: list[str]

# --- Agent Nodes ---
def retrieve_context(state: AgentState):
    """
    Node to retrieve relevant context using the RAG pipeline.
    This is a simplified node; the full RAG chain is invoked in the 'generate' node.
    """
    print("--- Node: Retrieve Context ---")
    question = state["question"]
    # In a more complex graph, this node would just call the retriever part.
    # For this linear graph, we'll keep it simple and just pass the question along.
    return {"question": question}

def generate_answer(state: AgentState):
    """Node to generate an answer using the full RAG chain."""
    print("--- Node: Generate Answer ---")
    question = state["question"]
    rag_chain = get_rag_chain()
    
    # The rag_chain already encapsulates retrieval, prompt, and LLM call.
    generation = rag_chain.invoke(question)
    
    return {"generation": generation}

# --- Graph Assembly ---
def get_agent_graph():
    """Constructs and returns the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("generate", generate_answer)

    # Define the workflow edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph into a runnable
    agent_graph = workflow.compile()
    print("Agent graph compiled successfully.")
    return agent_graph
