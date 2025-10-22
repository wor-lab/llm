#!/usr/bin/env python3
"""
WB AI Enterprise - Main Orchestration Entry Point
Handles initialization, setup, and execution coordination
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("WB.Core")

# Load environment variables
load_dotenv()

# Import WB AI modules
from dataset_loader import DatasetManager
from rag_pipeline import RAGPipeline
from agent_system import WBAgentSystem
from api_server import WBAPIServer


class WBCore:
    """Main orchestrator for WB AI Enterprise System"""
    
    def __init__(self):
        self.config = self._load_config()
        self.dataset_manager: Optional[DatasetManager] = None
        self.rag_pipeline: Optional[RAGPipeline] = None
        self.agent_system: Optional[WBAgentSystem] = None
        self.api_server: Optional[WBAPIServer] = None
        
    def _load_config(self) -> dict:
        """Load and validate configuration"""
        config = {
            'ngrok_token': os.getenv('NGROK_AUTH_TOKEN'),
            'model_name': os.getenv('MODEL_NAME', 'Qwen/Qwen2.5-1.5B-Instruct'),
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            'chroma_dir': os.getenv('CHROMA_PERSIST_DIR', './chroma_db'),
            'collection_name': os.getenv('CHROMA_COLLECTION_NAME', 'wb_ai_knowledge'),
            'api_host': os.getenv('API_HOST', '0.0.0.0'),
            'api_port': int(os.getenv('API_PORT', 8000)),
            'batch_size': int(os.getenv('BATCH_SIZE', 512)),
            'max_tokens': int(os.getenv('MAX_TOKENS', 2048)),
            'temperature': float(os.getenv('TEMPERATURE', 0.7)),
            'top_k': int(os.getenv('TOP_K_RETRIEVAL', 5)),
        }
        
        # Validate critical config
        if not config['ngrok_token']:
            logger.warning("NGROK_AUTH_TOKEN not set - API will be local only")
            
        return config
    
    async def initialize_system(self, force_reload: bool = False):
        """Initialize all WB AI components"""
        try:
            logger.info("🚀 WB AI ENTERPRISE INITIALIZATION STARTED")
            
            # Step 1: Initialize Dataset Manager
            logger.info("📊 Initializing Dataset Manager...")
            self.dataset_manager = DatasetManager(
                chroma_dir=self.config['chroma_dir'],
                collection_name=self.config['collection_name'],
                embedding_model=self.config['embedding_model'],
                batch_size=self.config['batch_size']
            )
            
            # Step 2: Load datasets into ChromaDB
            if force_reload or not self.dataset_manager.is_initialized():
                logger.info("📥 Loading datasets (this may take time on first run)...")
                await self.dataset_manager.load_all_datasets()
            else:
                logger.info("✅ Using existing ChromaDB collection")
            
            # Step 3: Initialize RAG Pipeline
            logger.info("🔍 Initializing RAG Pipeline...")
            self.rag_pipeline = RAGPipeline(
                chroma_client=self.dataset_manager.chroma_client,
                collection_name=self.config['collection_name'],
                model_name=self.config['model_name'],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                top_k=self.config['top_k']
            )
            await self.rag_pipeline.initialize()
            
            # Step 4: Initialize Agent System
            logger.info("🤖 Initializing 8-Agent System...")
            self.agent_system = WBAgentSystem(
                rag_pipeline=self.rag_pipeline,
                model_name=self.config['model_name']
            )
            await self.agent_system.initialize()
            
            # Step 5: Initialize API Server
            logger.info("🌐 Initializing API Server...")
            self.api_server = WBAPIServer(
                agent_system=self.agent_system,
                rag_pipeline=self.rag_pipeline,
                host=self.config['api_host'],
                port=self.config['api_port'],
                ngrok_token=self.config['ngrok_token']
            )
            
            logger.info("✅ WB AI ENTERPRISE INITIALIZATION COMPLETE")
            self._print_system_status()
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}", exc_info=True)
            raise
    
    def _print_system_status(self):
        """Print system status and configuration"""
        status = f"""
╔══════════════════════════════════════════════════════════════╗
║           WB AI ENTERPRISE - SYSTEM READY                   ║
╠══════════════════════════════════════════════════════════════╣
║ Model:         {self.config['model_name']:<40} ║
║ Vector Store:  ChromaDB ({self.dataset_manager.get_collection_count()} documents){' ':<19} ║
║ Agents:        8 Specialized Agents Active{' ':<23} ║
║ API Status:    Ready on port {self.config['api_port']}{' ':<27} ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(status)
    
    async def run_interactive_mode(self):
        """Run interactive CLI mode for testing"""
        logger.info("🎮 Starting Interactive Mode (type 'exit' to quit)")
        
        while True:
            try:
                user_input = input("\n🔵 WB AI > ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    logger.info("👋 Shutting down...")
                    break
                
                if not user_input:
                    continue
                
                # Route to agent system
                response = await self.agent_system.execute_task(user_input)
                print(f"\n🤖 Response:\n{response}\n")
                
            except KeyboardInterrupt:
                logger.info("\n👋 Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
    
    async def start_api_server(self):
        """Start the FastAPI server with Ngrok"""
        logger.info("🚀 Starting API Server...")
        await self.api_server.start()


async def main():
    """Main execution function"""
    core = WBCore()
    
    # Parse command line arguments
    force_reload = '--reload' in sys.argv
    interactive = '--interactive' in sys.argv
    
    # Initialize system
    await core.initialize_system(force_reload=force_reload)
    
    if interactive:
        # Run interactive mode
        await core.run_interactive_mode()
    else:
        # Start API server (default)
        await core.start_api_server()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 WB AI Enterprise shutdown complete")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)
