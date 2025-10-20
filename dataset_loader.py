# dataset_loader.py
import os
import chromadb
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# --- Configuration ---
DATASETS_TO_LOAD = [
    "princeton-nlp/SWE-bench_Verified",
    # "bigcode/the-stack-v2",  # Note: This dataset is massive. Load a subset for practical use.
    # "livecodebench/code_generation_lite"
    # Add other smaller datasets for initial testing
    "zai-org/humaneval-x",
    "Muennighoff/mbpp"
]

# Use a standard, high-performance embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "code_assistant_knowledge"

def initialize_embeddings(model_name: str):
    """Initializes the sentence transformer embedding model."""
    print(f"Initializing embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

def initialize_chroma_client(path: str):
    """Initializes the persistent ChromaDB client."""
    print(f"Initializing ChromaDB client at: {path}")
    return chromadb.PersistentClient(path=path)

def process_and_store(client, collection, documents, metadatas, batch_size=100):
    """Processes documents in batches and stores them in ChromaDB."""
    for i in tqdm(range(0, len(documents), batch_size), desc="Storing documents"):
        batch_docs = documents[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        ids = [f"doc_{collection.count() + j}" for j in range(len(batch_docs))]
        collection.add(
            ids=ids,
            documents=batch_docs,
            metadatas=batch_metas
        )

def load_and_process_datasets():
    """
    Main function to load datasets from Hugging Face, process their content,
    and store them in a persistent ChromaDB collection.
    """
    # 1. Initialize services
    embeddings = initialize_embeddings(EMBEDDING_MODEL_NAME)
    chroma_client = initialize_chroma_client(CHROMA_DB_PATH)

    # 2. Create or get the collection
    # The embedding function is passed at collection creation
    embedding_function_wrapper = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function_wrapper
    )
    
    print(f"Collection '{COLLECTION_NAME}' loaded. Current count: {collection.count()} documents.")

    # 3. Load, process, and store each dataset
    for dataset_name in DATASETS_TO_LOAD:
        print(f"\nProcessing dataset: {dataset_name}")
        try:
            # Adjust 'split' and 'name' as per dataset specifics on Hugging Face Hub
            dataset = load_dataset(dataset_name, split="train", streaming=False) # streaming=False for easier processing
            
            documents = []
            metadatas = []
            
            # This part requires adaptation per dataset's structure
            # Example for a generic dataset with 'text' and 'source' columns
            # You MUST inspect each dataset's structure and adapt this logic.
            # For now, we'll try a few common column names.
            text_column = "content" if "content" in dataset.column_names else "text" if "text" in dataset.column_names else "prompt"

            for item in tqdm(dataset, desc=f"Reading {dataset_name}"):
                if item[text_column]:
                    documents.append(str(item[text_column]))
                    metadatas.append({"source": dataset_name})

            if documents:
                process_and_store(chroma_client, collection, documents, metadatas)
                print(f"Successfully processed and stored {len(documents)} documents from {dataset_name}.")
                print(f"Collection '{COLLECTION_NAME}' new count: {collection.count()} documents.")

        except Exception as e:
            print(f"Failed to process dataset {dataset_name}. Error: {e}")

if __name__ == "__main__":
    print("--- WB AI Corporation: Data Ingestion Protocol ---")
    load_and_process_datasets()
    print("--- Data Ingestion Complete ---")
