"""
WB AI CORPORATION - Agent Definitions
Specialized Agents for RAG Workflow
═══════════════════════════════════════
"""

from typing import Dict, List
from langchain.schema import HumanMessage, SystemMessage
import re

# ═══════════════════════════════════════════════
# RETRIEVER AGENT
# ═══════════════════════════════════════════════

def retriever_agent(state: Dict, data_pipeline) -> Dict:
    """Retrieves relevant code examples from vector store"""
    
    query = state["query"]
    
    # Search across all collections
    results = data_pipeline.search(query, n_results=5)
    
    state["retrieved_contexts"] = results
    state["agent_path"].append("Retriever")
    
    return state

# ═══════════════════════════════════════════════
# GENERATOR AGENT
# ═══════════════════════════════════════════════

def generator_agent(state: Dict, llm) -> Dict:
    """Generates code based on query and retrieved contexts"""
    
    query = state["query"]
    contexts = state["retrieved_contexts"]
    feedback = state.get("feedback", "")
    
    # Build context string
    context_str = "\n\n".join([
        f"Example {i+1}:\n{ctx['text'][:500]}"
        for i, ctx in enumerate(contexts[:3])
    ])
    
    # Build prompt
    if feedback:
        system_prompt = f"""You are a code generation expert. Generate improved Python code based on the feedback.

Previous feedback: {feedback}

Use these examples as reference:
{context_str}"""
    else:
        system_prompt = f"""You are a code generation expert. Generate clean, efficient Python code.

Use these examples as reference:
{context_str}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Task: {query}\n\nGenerate only the Python code with docstring:")
    ]
    
    response = llm.invoke(messages)
    generated_code = response.content.strip()
    
    # Extract code block if wrapped in markdown
    code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
    if code_match:
        generated_code = code_match.group(1)
    
    state["generated_code"] = generated_code
    state["agent_path"].append("Generator")
    
    return state

# ═══════════════════════════════════════════════
# EVALUATOR AGENT
# ═══════════════════════════════════════════════

def evaluator_agent(state: Dict, llm) -> Dict:
    """Evaluates generated code quality"""
    
    code = state["generated_code"]
    query = state["query"]
    
    system_prompt = """You are a code quality evaluator. Assess the code on:
1. Correctness (does it solve the task?)
2. Code quality (is it clean and efficient?)
3. Completeness (are edge cases handled?)

Respond with:
SCORE: <0.0-1.0>
FEEDBACK: <specific improvements needed>"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Task: {query}\n\nCode:\n{code}")
    ]
    
    response = llm.invoke(messages)
    evaluation = response.content.strip()
    
    # Parse score and feedback
    score_match = re.search(r'SCORE:\s*([\d.]+)', evaluation)
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', evaluation, re.DOTALL)
    
    score = float(score_match.group(1)) if score_match else 0.5
    feedback = feedback_match.group(1).strip() if feedback_match else "No specific feedback"
    
    state["evaluation_score"] = score
    state["feedback"] = feedback
    state["agent_path"].append("Evaluator")
    
    return state

# ═══════════════════════════════════════════════
# REFINER AGENT
# ═══════════════════════════════════════════════

def refiner_agent(state: Dict, llm) -> Dict:
    """Refines code based on evaluation feedback"""
    
    state["refinement_count"] += 1
    state["agent_path"].append("Refiner")
    
    # Feedback is already in state, generator will use it
    return state
