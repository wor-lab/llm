"""
WB AI Corporation - Main Orchestration Script
Complete system initialization and execution for Google Colab.
Architecture: Entry point with comprehensive setup and testing.
"""

import os
import sys
from pathlib import Path
import subprocess
from typing import Optional

from loguru import logger
from dotenv import load_dotenv


# ============================================
# CONFIGURATION
# ============================================

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>")


def setup_environment():
    """Configure environment and install dependencies."""
    logger.info("=" * 60)
    logger.info("WB AI CORPORATION - AUTONOMOUS AI ENTERPRISE")
    logger.info("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Create directories
    dirs = ["./chroma_db", "./cache", "./logs"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.success(f"Directory ready: {d}")
    
    # GPU check
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        logger.info(f"GPU Available: {gpu_available}")
        if gpu_available:
            logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("PyTorch not installed yet")


def install_dependencies():
    """Install all requirements."""
    logger.info("Installing dependencies...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
            check=True
        )
        logger.success("Dependencies installed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)


def initialize_system():
    """Initialize all system components."""
    logger.info("Initializing WB AI System...")
    
    from dataset_loader import initialize_datasets
    from rag_pipeline import initialize_rag_pipeline
    from agent_system import initialize_agent_system
    
    # Step 1: Dataset & Vectorstore
    logger.info("Step 1/3: Loading datasets and building vectorstore...")
    dataset_loader = initialize_datasets()
    vectorstore = dataset_loader.get_vectorstore()
    logger.success("Vectorstore ready")
    
    # Step 2: RAG Pipeline
    logger.info("Step 2/3: Initializing RAG pipeline...")
    rag_pipeline = initialize_rag_pipeline(vectorstore)
    logger.success("RAG pipeline ready")
    
    # Step 3: Agent System
    logger.info("Step 3/3: Initializing multi-agent system...")
    agent_system = initialize_agent_system(rag_pipeline)
    logger.success("Agent system ready")
    
    return dataset_loader, rag_pipeline, agent_system


def run_tests(agent_system):
    """Execute test queries across all agents."""
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING SYSTEM TESTS")
    logger.info("=" * 60 + "\n")
    
    test_queries = [
        "Create a Python FastAPI server with authentication and rate limiting",
        "Design a CI/CD pipeline for deploying a Django app to AWS",
        "Perform a security audit on this SQL query: SELECT * FROM users WHERE id = ${user_input}",
        "Design a responsive dashboard layout using Tailwind CSS",
        "Write technical documentation for a REST API",
        "Analyze this dataset and create a pandas visualization",
        "Create a business plan for an AI SaaS product",
        "Build an automated workflow to sync data between two APIs"
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n[TEST {i}/8] {query[:60]}...")
        result = agent_system.execute_task(query)
        results.append(result)
        logger.success(f"✓ Agent: {result['agent']}")
        logger.info(f"Answer preview: {result['result'][:200]}...")
    
    logger.info("\n" + "=" * 60)
    logger.success(f"ALL TESTS COMPLETED: {len(results)}/8 successful")
    logger.info("=" * 60)
    
    return results


def start_api_server():
    """Launch FastAPI server with NGROK."""
    logger.info("\n" + "=" * 60)
    logger.info("STARTING API SERVER")
    logger.info("=" * 60 + "\n")
    
    from api_server import run_server
    
    logger.warning("Starting server in 3 seconds...")
    import time
    time.sleep(3)
    
    run_server()


def interactive_mode(agent_system):
    """Interactive CLI for querying the system."""
    logger.info("\n" + "=" * 60)
    logger.info("INTERACTIVE MODE")
    logger.info("Type 'exit' to quit, 'agents' to list agents")
    logger.info("=" * 60 + "\n")
    
    while True:
        try:
            query = input("\n🤖 WB AI > ").strip()
            
            if query.lower() == 'exit':
                logger.info("Shutting down...")
                break
            
            if query.lower() == 'agents':
                logger.info("\nAvailable Agents:")
                for i, agent in enumerate([
                    "CodeArchitect", "OpsManager", "SecAnalyst", "DesignMind",
                    "WordSmith", "DataSynth", "Analyst", "AutoBot"
                ], 1):
                    logger.info(f"  {i}. {agent}")
                continue
            
            if not query:
                continue
            
            result = agent_system.execute_task(query)
            
            logger.success(f"\n[{result['agent']}]")
            logger.info(f"\n{result['result']}\n")
            logger.info(f"Sources: {len(result['sources'])}")
        
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


# ============================================
# MAIN EXECUTION
# ============================================

def main(mode: str = "test"):
    """
    Main execution function.
    
    Modes:
    - test: Run system tests
    - server: Start API server
    - interactive: CLI mode
    - full: Setup + tests + server
    """
    
    # Setup
    setup_environment()
    install_dependencies()
    
    # Initialize
    dataset_loader, rag_pipeline, agent_system = initialize_system()
    
    # Execute based on mode
    if mode == "test":
        run_tests(agent_system)
    
    elif mode == "server":
        start_api_server()
    
    elif mode == "interactive":
        interactive_mode(agent_system)
    
    elif mode == "full":
        run_tests(agent_system)
        logger.info("\nTests complete. Starting server...")
        start_api_server()
    
    else:
        logger.error(f"Unknown mode: {mode}")
        logger.info("Valid modes: test, server, interactive, full")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WB AI Corporation - System Runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "server", "interactive", "full"],
        help="Execution mode"
    )
    
    args = parser.parse_args()
    
    try:
        main(mode=args.mode)
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)
