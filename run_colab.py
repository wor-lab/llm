"""
WB AI CORPORATION - GOOGLE COLAB ORCHESTRATOR
Agent: OpsManager
Purpose: One-command deployment for Colab environment
"""

import os
import subprocess
import sys
from pathlib import Path


class WBColabRunner:
    """Automated Colab deployment pipeline"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        
    def setup_environment(self):
        """Install dependencies"""
        print("[OpsManager] Installing dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
        ], check=True)
        print("[OpsManager] Dependencies installed")
        
    def verify_env_file(self):
        """Check .env configuration"""
        env_path = self.project_root / ".env"
        if not env_path.exists():
            print("[ERROR] .env file not found. Create it with required variables.")
            sys.exit(1)
            
        # Load and validate
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = [
            "MODEL_NAME", "EMBEDDING_MODEL", "CHROMA_PERSIST_DIR",
            "COLLECTION_NAME", "API_KEY"
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            print(f"[ERROR] Missing environment variables: {missing}")
            sys.exit(1)
            
        print("[SecAnalyst] Environment configuration validated")
        
    def build_vectorstore(self):
        """Execute dataset loading"""
        print("\n[DataSynth] Building vectorstore (this may take 10-20 minutes)...")
        from dataset_loader import WBDatasetLoader
        
        loader = WBDatasetLoader()
        vectorstore = loader.build_vectorstore()
        print("[OpsManager] Vectorstore build complete")
        
    def test_rag_pipeline(self):
        """Test RAG functionality"""
        print("\n[CodeArchitect] Testing RAG pipeline...")
        from rag_pipeline import WBRAGPipeline
        
        rag = WBRAGPipeline()
        test_query = "How to implement authentication in Python?"
        result = rag.query(test_query)
        
        print(f"\n[Test Query]: {test_query}")
        print(f"[Test Response]: {result['answer'][:200]}...")
        print("[OpsManager] RAG pipeline operational")
        
    def test_agent_system(self):
        """Test LangGraph agents"""
        print("\n[AutoBot] Testing agent system...")
        from agent_system import WBAgentSystem
        
        agent = WBAgentSystem()
        result = agent.execute("Write a function to validate JWT tokens")
        
        print(f"\n[Agent Response]: {result['answer'][:200]}...")
        print(f"[Iterations]: {result['iterations']}")
        print("[OpsManager] Agent system operational")
        
    def start_api_server(self):
        """Launch FastAPI server with NGROK"""
        print("\n[AutoBot] Starting API server...")
        print("[INFO] API will run in foreground. Use Ctrl+C to stop.")
        print("[INFO] Access endpoints at /docs for Swagger UI")
        
        subprocess.run([
            sys.executable, "api_server.py"
        ])
        
    def run_full_pipeline(self):
        """Execute complete deployment"""
        print("="*80)
        print("WB AI CORPORATION - COLAB DEPLOYMENT PIPELINE")
        print("="*80)
        
        # Step 1: Environment setup
        self.setup_environment()
        self.verify_env_file()
        
        # Step 2: Build vectorstore
        vectorstore_exists = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")).exists()
        if not vectorstore_exists:
            print("\n[Decision] Vectorstore not found. Building from datasets...")
            self.build_vectorstore()
        else:
            print("\n[Decision] Vectorstore found. Skipping build.")
        
        # Step 3: Test systems
        self.test_rag_pipeline()
        self.test_agent_system()
        
        # Step 4: Launch API
        print("\n" + "="*80)
        print("DEPLOYMENT COMPLETE - LAUNCHING API SERVER")
        print("="*80)
        self.start_api_server()


# Colab-specific helpers
def colab_quick_start():
    """Single-command startup for Colab notebooks"""
    runner = WBColabRunner()
    runner.run_full_pipeline()


def colab_rebuild_vectorstore():
    """Force rebuild vectorstore"""
    runner = WBColabRunner()
    runner.setup_environment()
    runner.verify_env_file()
    runner.build_vectorstore()


def colab_api_only():
    """Start API without rebuilding"""
    runner = WBColabRunner()
    runner.setup_environment()
    runner.verify_env_file()
    runner.start_api_server()


if __name__ == "__main__":
    # Default: Full pipeline
    colab_quick_start()
