# Algorithms & FULLCODE for run_colab.py
# Algorithm: Main Execution Workflow
# 1. Load environment variables.
# 2. Initialize dataset loading into ChromaDB.
# 3. Set up agent system with LangGraph orchestration.
# 4. Start FastAPI server and expose via Ngrok.
# 5. Run a demo project to demonstrate orchestration.
# Optimization: Asynchronous setup for non-blocking execution.

import os
from dotenv import load_dotenv
from pyngrok import ngrok
import uvicorn
from dataset_loader import load_datasets_to_chroma
from agent_system import create_agent_graph, run_project
from api_server import app  # FastAPI app
from rag_pipeline import get_rag_chain  # For RAG integration

# Load environment
load_dotenv()

# Main entry point for Colab
def main():
    print("WB AI Corporation: Initializing system...")

    # Step 1: Load datasets into ChromaDB (real HF data, subset for efficiency)
    collection = load_datasets_to_chroma(subset_size=100)  # Modular call

    # Step 2: Get RAG chain (optimized for quick retrieval)
    rag_chain = get_rag_chain(collection)

    # Step 3: Create agent graph (LangGraph for orchestration)
    agent_graph = create_agent_graph(rag_chain)

    # Step 4: Start API server in background
    def start_api():
        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        server.run()

    import threading
    api_thread = threading.Thread(target=start_api)
    api_thread.start()

    # Expose via Ngrok
    ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))
    public_url = ngrok.connect(8000)
    print(f"API exposed at: {public_url}")

    # Step 5: Demo project execution
    demo_request = "Audit a Python codebase for security vulnerabilities."
    result = run_project(agent_graph, demo_request)
    print("Demo Project Result:", result)

if __name__ == "__main__":
    main()
