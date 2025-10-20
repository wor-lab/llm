# api_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent_system import get_agent_graph

# --- API Application ---
app = FastAPI(
    title="WB AI Corporation - Agentic RAG API",
    description="API for interacting with the Phoenix Agentic System.",
    version="1.0.0"
)

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# --- Agent Initialization ---
# The agent graph is compiled once on startup for efficiency.
agent_executor = get_agent_graph()

# --- API Endpoints ---
@app.post("/invoke", response_model=QueryResponse)
def invoke_agent(request: QueryRequest):
    """
    Receives a question, invokes the agent graph, and returns the generated answer.
    """
    print(f"Received query: {request.question}")
    inputs = {"question": request.question}
    
    # The agent_executor.stream() method can be used for streaming responses.
    # For a simple request/response, we use invoke().
    result_state = agent_executor.invoke(inputs)
    
    answer = result_state.get("generation", "Error: No generation found in final state.")
    
    return QueryResponse(answer=answer)

@app.get("/")
def root():
    return {"status": "WB AI Corporation Phoenix Agent is operational."}

# To run this server: uvicorn api_server:app --host 0.0.0.0 --port 8000
