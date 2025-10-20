"""
WB AI CORPORATION - Agentic RAG System
Main Orchestrator for Google Colab Deployment
═══════════════════════════════════════════════
"""

import os
import sys
import subprocess
from pathlib import Path
import threading
import time

class WBOrchestrator:
    """Central command for WB AI Corporation's Agentic RAG System"""
    
    def __init__(self):
        self.ngrok_token = os.getenv('1vikehg18jsR9XrEzKEybCifEr9_AWWFzoCD58Xa151mXfLd')
        self.project_root = Path('/')
        self.model_name = "Qwen/Qwen3-1.7B"  # Using available Qwen model
        self.api_base = None
        self.api_key = "wb-ai-internal-key"
        
    def setup_environment(self):
        """Install dependencies and configure environment"""
        print("🏢 WB AI Corporation - System Initialization")
        print("=" * 60)
        
        # Create project structure
        self.project_root.mkdir(exist_ok=True)
        os.chdir(self.project_root)
        
        # Install dependencies
        dependencies = [
            "langchain langchain-community langchain-openai",
            "langgraph",
            "chromadb",
            "sentence-transformers",
            "datasets",
            "transformers torch accelerate bitsandbytes",
            "pyngrok",
            "fastapi uvicorn",
            "tiktoken",
            "openai"
        ]
        
        print("\n📦 Installing production dependencies...")
        for dep in dependencies:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", dep])
        
        print("✅ Environment configured\n")
    
    def configure_ngrok(self):
        """Setup NGROK tunnel for model server"""
        if not self.ngrok_token:
            raise ValueError("❌ NGROK_AUTH_TOKEN not found in environment")
        
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = self.ngrok_token
        print("✅ NGROK authenticated\n")
    
    def start_model_server(self):
        """Launch Qwen3-1.7B server with NGROK tunnel"""
        print("🚀 Launching Model Server (Qwen3-1.7B)...")
        
        # Start server in background thread
        server_thread = threading.Thread(target=self._run_server, daemon=True)
        server_thread.start()
        
        # Wait for server startup
        time.sleep(10)
        
        # Create NGROK tunnel
        from pyngrok import ngrok
        tunnel = ngrok.connect(8000, bind_tls=True)
        self.api_base = tunnel.public_url
        
        print(f"✅ Model Server Online")
        print(f"📡 API Endpoint: {self.api_base}")
        print(f"🔑 API Key: {self.api_key}\n")
        
        # Save configuration
        with open('config.txt', 'w') as f:
            f.write(f"API_BASE={self.api_base}\n")
            f.write(f"API_KEY={self.api_key}\n")
    
    def _run_server(self):
        """Internal server runner"""
        from model_server import app
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    
    def initialize_data_pipeline(self):
        """Load and embed code datasets into ChromaDB"""
        print("📊 Initializing Data Pipeline...")
        from data_pipeline import DataPipeline
        
        pipeline = DataPipeline()
        pipeline.load_datasets()
        pipeline.create_embeddings()
        
        print("✅ Data pipeline ready\n")
    
    def launch_agentic_rag(self):
        """Start the multi-agent RAG system"""
        print("🧠 Launching Agentic RAG System...")
        from agentic_rag import AgenticRAG
        
        rag = AgenticRAG(
            api_base=self.api_base,
            api_key=self.api_key
        )
        
        print("✅ Agentic RAG System Online\n")
        return rag
    
    def run_demo(self, rag):
        """Execute demonstration queries"""
        print("=" * 60)
        print("🎯 DEMO: Code Generation with Agentic RAG")
        print("=" * 60)
        
        test_queries = [
            "Write a Python function to implement binary search",
            "Create a function to reverse a linked list in-place",
            "Implement a function to check if a string is a valid palindrome"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[QUERY {i}] {query}")
            print("-" * 60)
            
            result = rag.execute(query)
            
            print(f"📝 Generated Code:\n{result['code']}\n")
            print(f"⚡ Agent Path: {' → '.join(result['agent_path'])}")
            print(f"📊 Evaluation Score: {result.get('score', 'N/A')}")
            print("=" * 60)
    
    def execute(self):
        """Main execution pipeline"""
        try:
            self.setup_environment()
            self.configure_ngrok()
            self.start_model_server()
            self.initialize_data_pipeline()
            
            rag = self.launch_agentic_rag()
            self.run_demo(rag)
            
            print("\n✅ WB AI Corporation - System Fully Operational")
            print(f"📡 API Endpoint: {self.api_base}")
            print("🔄 System ready for production queries\n")
            
            return rag
            
        except Exception as e:
            print(f"❌ SYSTEM FAILURE: {e}")
            raise


# ═══════════════════════════════════════════════
# EXECUTION ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    # Set NGROK token (replace with your token)
    # os.environ['NGROK_AUTH_TOKEN'] = 'your_token_here'
    
    orchestrator = WBOrchestrator()
    rag_system = orchestrator.execute()
    
    # Keep system running for interactive queries
    print("\n💡 System ready. Use rag_system.execute(your_query) for custom queries")
