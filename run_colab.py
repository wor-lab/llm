"""
WB AI CORPORATION - NEXUS-RAG
Main Orchestrator - Google Colab Environment
Enterprise Execution Controller
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='[WB AI] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NexusOrchestrator:
    """Central execution controller for NEXUS-RAG system"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.config = {
            'model_name': 'Qwen/Qwen3-1.7B',
            'ngrok_token': os.getenv('NGROK_AUTH_TOKEN'),
            'hf_token': os.getenv('HF_TOKEN'),
            'chroma_path': './chroma_db',
            'datasets': [
                'princeton-nlp/SWE-bench_Verified',
                'openai/humaneval',
                'google-research-datasets/mbpp',
                'bigcode/bigcodebench',
                'bigcode/the-stack-v2-dedup',
            ]
        }
        
    def install_dependencies(self):
        """Install production dependencies"""
        logger.info("🔧 Installing dependencies...")
        
        packages = [
            'langchain',
            'langgraph',
            'chromadb',
            'transformers',
            'torch',
            'fastapi',
            'uvicorn',
            'pyngrok',
            'datasets',
            'sentence-transformers',
            'accelerate',
            'bitsandbytes',
            'pydantic',
            'python-dotenv',
        ]
        
        for package in packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
        
        logger.info("✅ Dependencies installed")
    
    def setup_environment(self):
        """Configure environment variables and paths"""
        logger.info("⚙️ Setting up environment...")
        
        # Validate credentials
        if not self.config['ngrok_token']:
            raise ValueError("NGROK_AUTH_TOKEN not found in environment")
        
        if not self.config['hf_token']:
            logger.warning("HF_TOKEN not found - some datasets may be restricted")
        
        # Create directories
        os.makedirs(self.config['chroma_path'], exist_ok=True)
        os.makedirs('./logs', exist_ok=True)
        
        logger.info("✅ Environment configured")
    
    def initialize_system(self):
        """Initialize all subsystems"""
        logger.info("🚀 Initializing NEXUS-RAG subsystems...")
        
        # Import after dependencies installed
        from dataset_loader import DatasetManager
        from rag_pipeline import RAGEngine
        from agent_system import AgentOrchestrator
        
        # Initialize dataset loader
        logger.info("📊 Initializing Dataset Manager...")
        self.dataset_manager = DatasetManager(
            datasets=self.config['datasets'],
            hf_token=self.config['hf_token']
        )
        
        # Initialize RAG pipeline
        logger.info("🔍 Initializing RAG Engine...")
        self.rag_engine = RAGEngine(
            chroma_path=self.config['chroma_path'],
            model_name=self.config['model_name']
        )
        
        # Initialize agent system
        logger.info("🤖 Initializing Agent System...")
        self.agent_system = AgentOrchestrator(
            rag_engine=self.rag_engine,
            model_name=self.config['model_name']
        )
        
        logger.info("✅ All subsystems initialized")
    
    def load_and_index_datasets(self):
        """Load datasets and build vector index"""
        logger.info("📥 Loading and indexing datasets...")
        
        # Load datasets
        documents = self.dataset_manager.load_all_datasets()
        logger.info(f"📚 Loaded {len(documents)} documents")
        
        # Index in ChromaDB
        self.rag_engine.index_documents(documents)
        logger.info("✅ Datasets indexed in ChromaDB")
    
    def start_api_server(self):
        """Launch FastAPI server with Ngrok tunnel"""
        logger.info("🌐 Starting API server...")
        
        from api_server import create_app
        
        app = create_app(
            agent_system=self.agent_system,
            ngrok_token=self.config['ngrok_token']
        )
        
        logger.info("✅ API server ready")
        
        return app
    
    def run(self):
        """Execute full system deployment"""
        logger.info("=" * 60)
        logger.info("WB AI CORPORATION - NEXUS-RAG SYSTEM")
        logger.info("=" * 60)
        
        try:
            # Phase 1: Setup
            self.install_dependencies()
            self.setup_environment()
            
            # Phase 2: Initialize
            self.initialize_system()
            
            # Phase 3: Data Loading
            self.load_and_index_datasets()
            
            # Phase 4: API Deployment
            app = self.start_api_server()
            
            logger.info("=" * 60)
            logger.info("✅ NEXUS-RAG SYSTEM FULLY OPERATIONAL")
            logger.info("=" * 60)
            
            return app
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}", exc_info=True)
            raise

# Execute if run directly
if __name__ == "__main__":
    orchestrator = NexusOrchestrator()
    app = orchestrator.run()
    
    # Start server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
