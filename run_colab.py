# run_colab.py
import os
import subprocess
import time
from pyngrok import ngrok, conf
from dotenv import load_dotenv

def setup_environment():
    """Install dependencies and load environment variables."""
    print("--- Setting up environment ---")
    
    # Install all required packages
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    
    # Load .env file for secrets
    load_dotenv()
    
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    hf_token = os.getenv("HUGGING_FACE_TOKEN") # Rename for clarity

    if not ngrok_token or not hf_token:
        raise ValueError("NGROK_AUTH_TOKEN and HUGGING_FACE_TOKEN must be set in your environment or a .env file.")

    # Authenticate Ngrok
    ngrok.set_auth_token(ngrok_token)
    
    print("Environment setup complete.")
    return hf_token

def run_data_ingestion():
    """Run the dataset loader script if the DB doesn't exist."""
    db_path = "./chroma_db"
    if not os.path.exists(db_path):
        print("--- ChromaDB not found. Running data ingestion protocol. This may take a while. ---")
        subprocess.run(["python", "dataset_loader.py"], check=True)
    else:
        print("--- ChromaDB found. Skipping data ingestion. ---")

def start_model_server(hf_token: str):
    """Starts the Hugging Face model server in a background process."""
    print("--- Starting local LLM server (Qwen/Qwen1.5-1.8B-Chat) ---")
    model_id = "Qwen/Qwen3-1.7B"
    
    # Command to run the TGI or a similar lightweight server
    # Using `text-generation-inference` is recommended for production. Here we use a simple placeholder.
    # A more robust solution involves a dedicated script or docker container.
    # For this example, we will assume a compatible server is started.
    # Let's start the FastAPI server that has the llm endpoint configured
    # In a real scenario, this would be a separate process for the LLM itself.
    # For simplicity in Colab, we'll combine. The `langchain_huggingface` points to this.
    
    # Here, we'll run a simple HuggingFaceEndpoint server wrapper if needed.
    # The current `rag_pipeline.py` is configured to call an endpoint at port 8001.
    # We will simulate this by having a placeholder for the actual model server process.
    # For now, we will launch our FastAPI server which is what ngrok will point to.
    
    print("Model server process would be started here on port 8001.")
    # Example command (requires text-generation-inference):
    # !text-generation-launcher --model-id {model_id} --port 8001
    
    # For the purpose of this script, we'll proceed assuming the user runs the model server separately.
    # The langchain HuggingFaceEndpoint will connect to it.

def start_api_server_and_ngrok():
    """Starts the FastAPI application and exposes it via Ngrok."""
    print("--- Starting FastAPI server ---")
    # Run uvicorn as a background process
    api_process = subprocess.Popen(["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Wait for the server to initialize
    time.sleep(5)
    
    print("--- Exposing API via Ngrok ---")
    # Open a tunnel to the local port 8000
    public_url = ngrok.connect(8000)
    print(f"✅ Public API Endpoint: {public_url}")
    print("Your agentic system is now live. Send POST requests to /invoke.")
    print("Press Ctrl+C to shut down.")
    
    try:
        # Keep the script running to maintain the tunnel and server
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Shutting down servers ---")
        ngrok.disconnect(public_url)
        api_process.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    print("--- WB AI Corporation: Phoenix System Deployment ---")
    hf_token = setup_environment()
    run_data_ingestion()
    # start_model_server(hf_token) # Assumed to be running for this example
    start_api_server_and_ngrok()
