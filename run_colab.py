import os
from dotenv import load_dotenv

from dataset_loader import load_and_index_datasets
from rag_pipeline import create_agentic_rag
from agent_system import build_agents
from api_server import main as start_server


def _get_env(name: str, default=None):
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing env: {name}")
    return v


def prepare():
    load_dotenv()
    chroma_dir = _get_env("CHROMA_DIR", "./chroma")
    emb_model = _get_env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    max_samples = int(os.getenv("MAX_SAMPLES_PER_DATASET", "300"))

    # Ingest datasets into Chroma (idempotent: repeated adds will deduplicate by ids provided)
    load_and_index_datasets(
        chroma_dir=chroma_dir,
        embedding_model=emb_model,
        max_samples_per_dataset=max_samples,
    )

    # Preload model and graph once to warm caches
    graph, llm, retriever = create_agentic_rag(chroma_dir)
    build_agents(llm, getattr(retriever, "vectorstore", None))  # warm


if __name__ == "__main__":
    prepare()
    start_server()
