"""
WB AI Corporation - Colab Orchestrator
Agent: OpsManager
Purpose: Single-command deployment for Google Colab environment
"""

import os
import sys
import subprocess
from pathlib import Path
from loguru import logger


class ColabOrchestrator:
    """Production deployment orchestrator for Colab"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        logger.info("WB AI Corporation - Deployment Initiated")
    
    def setup_environment(self):
        """Install dependencies"""
        logger.info("Step 1/5: Installing dependencies...")
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
        ], check=True)
        
        logger.success("Dependencies installed")
    
    def validate_config(self):
        """Validate environment configuration"""
        logger.info("Step 2/5: Validating configuration...")
        
        from dotenv import load_dotenv
        load_dotenv()
        
        required_vars = ["HF_API_KEY", "NGROK_AUTH_TOKEN", "MODEL_SERVER"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            logger.info("Please update .env file with required credentials")
            return False
        
        logger.success("Configuration validated")
        return True
    
    def initialize_vectorstore(self, force_reload: bool = False):
        """Load or create vector store"""
        logger.info("Step 3/5: Initializing vector store...")
        
        from dataset_loader import DatasetLoader
        
        loader = DatasetLoader()
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        
        if os.path.exists(persist_dir) and not force_reload:
            logger.info("Loading existing vector store...")
            vectorstore = loader.load_existing()
        else:
            logger.info("Creating vector store from datasets (this may take 5-10 minutes)...")
            vectorstore = loader.load_and_process()
        
        logger.success(f"Vector store ready | Documents: {vectorstore._collection.count()}")
        return vectorstore
    
    def test_rag_pipeline(self):
        """Test RAG functionality"""
        logger.info("Step 4/5: Testing RAG pipeline...")
        
        from dataset_loader import DatasetLoader
        from rag_pipeline import RAGPipeline
        
        loader = DatasetLoader()
        vectorstore = loader.load_existing()
        rag = RAGPipeline(vectorstore)
        
        test_query = "Write a function to check if a number is prime"
        result = rag.query(test_query)
        
        if result["status"] == "success":
            logger.success("RAG pipeline operational")
            logger.info(f"Test response: {result['answer'][:200]}...")
            return True
        else:
            logger.error("RAG pipeline test failed")
            return False
    
    def start_api_server(self):
        """Launch FastAPI server with NGROK"""
        logger.info("Step 5/5: Starting API server...")
        
        import uvicorn
        from api_server import app
        
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", 8000))
        
        logger.success("API server launching...")
        logger.info("=" * 60)
        logger.info("WB AI CORPORATION - SYSTEM OPERATIONAL")
        logger.info("=" * 60)
        
        uvicorn.run(app, host=host, port=port, log_level="info")
    
    def deploy(self, force_reload: bool = False, skip_tests: bool = False):
        """Execute full deployment pipeline"""
        try:
            # Step 1: Dependencies
            self.setup_environment()
            
            # Step 2: Config validation
            if not self.validate_config():
                logger.error("Deployment aborted - configuration invalid")
                return False
            
            # Step 3: Vector store
            self.initialize_vectorstore(force_reload)
            
            # Step 4: Testing
            if not skip_tests:
                if not self.test_rag_pipeline():
                    logger.warning("Tests failed but continuing deployment...")
            
            # Step 5: API server
            self.start_api_server()
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
        
        return True


def main():
    """Main execution entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WB AI Corporation - Deployment System")
    parser.add_argument("--force-reload", action="store_true", help="Force reload datasets")
    parser.add_argument("--skip-tests", action="store_true", help="Skip validation tests")
    
    args = parser.parse_args()
    
    orchestrator = ColabOrchestrator()
    orchestrator.deploy(
        force_reload=args.force_reload,
        skip_tests=args.skip_tests
    )


if __name__ == "__main__":
    main()
