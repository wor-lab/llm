# Algorithms & FULLCODE for rag_pipeline.py
# Algorithm: Agentic RAG Pipeline
# 1. Initialize embeddings (sentence-transformers for efficiency).
# 2. Create retriever from ChromaDB collection.
# 3. Set up LangChain RetrievalQA chain with local Qwen model.
# 4. Execute query: Retrieve -> Augment -> Generate.
# Optimization: Batch embeddings, top-k retrieval for speed.

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()

# Initialize local model (Qwen2-1.5B for lightweight inference)
def get_local_llm():
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=0 if torch.cuda.is_available() else -1)
    return HuggingFacePipeline(pipeline=pipe)

# Create RAG chain
def get_rag_chain(collection):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        collection_name="wb_ai_datasets",
        embedding_function=embeddings,
        persist_directory="./chroma_db"  # Persist for optimization
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Optimized top-k

    llm = get_local_llm()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Efficient for small contexts
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain

# Example usage (modular, can be called externally)
def run_rag_query(rag_chain, query):
    result = rag_chain({"query": query})
    return result["result"]
