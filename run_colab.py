"""
WB AI Corporation - Colab Runtime Orchestrator
Handles environment setup, model initialization, and system bootstrap
"""

import os
import sys
import subprocess
from pathlib import Path
import torch
from pyngrok import ngrok
import nest_asyncio

class ColabOrchestrator:
    def __init__(self):
        self.base_dir = Path("/")
        self.model_id = "Qwen/Qwen3-1.7B"  # Optimized variant
        self.ngrok_token = None
        
    def setup_environment(self):
        """Configure Colab runtime with required dependencies"""
        packages = [
            "langchain==0.3.7",
            "langgraph==0.2.38", 
            "chromadb==0.5.20",
            "transformers==4.46.3",
            "sentence-transformers==3.3.1",
            "fastapi==0.115.5",
            "uvicorn==0.32.1",
            "pyngrok==7.2.1",
            "datasets==3.2.0",
            "torch==2.5.1",
            "accelerate==1.2.1",
            "bitsandbytes==0.44.1"
        ]
        
        print("[WB-CORE] Installing enterprise dependencies...")
        for pkg in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
        
        # Setup directories
        self.base_dir.mkdir(exist_ok=True, parents=True)
        (self.base_dir / "vectordb").mkdir(exist_ok=True)
        (self.base_dir / "cache").mkdir(exist_ok=True)
        
        # Enable async in notebooks
        nest_asyncio.apply()
        
    def configure_ngrok(self, auth_token: str):
        """Setup NGROK tunnel for API exposure"""
        self.ngrok_token = auth_token
        ngrok.set_auth_token(auth_token)
        return ngrok
        
    def initialize_gpu(self):
        """Optimize GPU memory for model loading"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.85)
            print(f"[WB-CORE] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[WB-CORE] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("[WB-CORE] Running on CPU - performance will be limited")
            
    def launch_system(self):
        """Bootstrap all WB AI Corporation components"""
        from dataset_loader import DatasetManager
        from rag_pipeline import RAGEngine
        from agent_system import WBAgentNetwork
        from api_server import WBAPIServer
        
        print("[WB-CORE] Initializing WB AI Corporation Systems...")
        
        # Phase 1: Data Infrastructure
        data_mgr = DatasetManager(self.base_dir / "vectordb")
        data_mgr.load_all_datasets()
        
        # Phase 2: RAG Engine
        rag = RAGEngine(
            model_id=self.model_id,
            vector_store_path=str(self.base_dir / "vectordb")
        )
        
        # Phase 3: Agent Network
        agents = WBAgentNetwork(rag_engine=rag)
        
        # Phase 4: API Server
        server = WBAPIServer(
            agent_network=agents,
            port=8000
        )
        
        # Phase 5: Expose via NGROK
        if self.ngrok_token:
            public_url = ngrok.connect(8000)
            print(f"[WB-CORE] Public API: {public_url}")
        
        return {
            "data_manager": data_mgr,
            "rag_engine": rag,
            "agents": agents,
            "api_server": server
        }

# EXECUTION
if __name__ == "__main__":
    orchestrator = ColabOrchestrator()
    orchestrator.setup_environment()
    orchestrator.initialize_gpu()
    
    # Get NGROK token from env or prompt
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN", "1vikehg18jsR9XrEzKEybCifEr9_AWWFzoCD58Xa151mXfLd")
    if ngrok_token:
        orchestrator.configure_ngrok(ngrok_token)
    
    system = orchestrator.launch_system()
    print("[WB-CORE] WB AI Corporation operational.")
