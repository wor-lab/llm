# Algorithms & FULLCODE for dataset_loader.py
# Algorithm: Dataset Loading to ChromaDB
# 1. Load real HF datasets (subsets for optimization).
# 2. Preprocess: Extract text fields, chunk if large.
# 3. Embed using sentence-transformers.
# 4. Upsert to ChromaDB collection in batches.
# Optimization: Parallel loading with datasets, batch upserts to avoid memory issues.

import os
from datasets import load_dataset
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def load_datasets_to_chroma(subset_size=None):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load real HF datasets (no mocks)
    datasets = {
        "swe_bench": load_dataset("princeton-nlp/SWE-bench_Verified", split="train"),
        "the_stack": load_dataset("bigcode/the-stack-v2", split="train"),  # Huge; subset heavily
        "rstarcoder": load_dataset("microsoft/rStar-Coder", split="train")  # Assuming typo; use StarCoder if not
    }
    
    documents = []
    for name, ds in datasets.items():
        subset = ds.select(range(subset_size)) if subset_size else ds
        for item in subset:
            text = item.get("text", "") or item.get("content", "")  # Adapt to dataset schema
            if text:
                splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
                chunks = splitter.split_text(text)
                documents.extend(chunks)
    
    # Create or load Chroma collection
    vectorstore = Chroma(
        collection_name="wb_ai_datasets",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Batch upsert (optimized for large data)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vectorstore.add_texts(batch)
    
    vectorstore.persist()
    return vectorstore.get_collection("wb_ai_datasets")
