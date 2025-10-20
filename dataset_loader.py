# Filename: dataset_loader.py
# Department: Data Division (DataSynth), Engineering Division (CodeArchitect)
# Purpose: Ingests and processes code datasets into a persistent ChromaDB vector store.

import chromadb
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# --- CONFIGURATION ---
DATASETS = [
    "princeton-nlp/SWE-bench_Verified",
    "zai-org/humaneval-x",
    "Muennighoff/mbpp",
    "bigcode/bigcodebench",
    "microsoft/r-star-coder", # Corrected name from rStar-Coder
    "bigcode/the-stack-v2",
    "livecodebench/code_generation_lite"
]

# Use a specific subset/split for large datasets to manage resources
# For 'the-stack-v2', specify a language subset, e.g., 'data/python'
DATASET_CONFIGS = {
    "bigcode/the-stack-v2": {"data_dir": "data/python", "split": "train[:1%]"} # Example: 1% of python data
}

CHROMA_PATH = "chroma_db_code"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "code_collection"

def initialize_components():
    """Initializes embeddings model and ChromaDB client."""
    print("Initializing components...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return embeddings, collection

def process_and_embed(collection, embeddings, documents):
    """Processes documents and embeds them into the collection in batches."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Batch processing for efficiency
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_texts = [chunk.page_content for chunk in batch_chunks]
        batch_metadatas = [chunk.metadata for chunk in batch_chunks]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")
        
        embedded_vectors = embeddings.embed_documents(batch_texts)
        ids = [f"id_{i+j}" for j in range(len(batch_chunks))]
        
        collection.add(
            embeddings=embedded_vectors,
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=ids
        )

def main():
    """Main function to load datasets and populate ChromaDB."""
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f"ChromaDB already exists at '{CHROMA_PATH}'. Skipping data loading.")
        return

    embeddings, collection = initialize_components()
    
    all_docs = []
    for repo_id in DATASETS:
        try:
            print(f"\nLoading dataset: {repo_id}")
            config = DATASET_CONFIGS.get(repo_id, {})
            # Use 'all' split if available, otherwise 'train'
            split = config.get("split", "train")
            dataset = load_dataset(repo_id, split=split, **{k:v for k,v in config.items() if k != 'split'})
            
            # Identify the most likely text/code column
            content_column = next((col for col in ['content', 'text', 'code', 'prompt', 'canonical_solution'] if col in dataset.column_names), None)
            
            if not content_column:
                print(f"Warning: Could not find a suitable content column for {repo_id}. Skipping.")
                continue

            for item in dataset:
                if item[content_column]:
                    all_docs.append({"page_content": item[content_column], "metadata": {"source": repo_id}})

        except Exception as e:
            print(f"Failed to load or process {repo_id}. Error: {e}")

    # Convert dicts to LangChain Document objects for splitter
    from langchain_core.documents import Document
    langchain_docs = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in all_docs]

    if not langchain_docs:
        print("No documents were loaded. Exiting.")
        return

    process_and_embed(collection, embeddings, langchain_docs)
    print("\n✅ Data ingestion complete. ChromaDB is ready.")

if __name__ == "__main__":
    main()
