# rag_pipeline.py
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

# --- Configuration ---
# Match the config from dataset_loader.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "code_assistant_knowledge"

# API configuration for the LLM
# IMPORTANT: This assumes a local server is running, which will be started by api_server.py
# We will use the HuggingFaceEndpoint class to interface with it.
LLM_API_URL = "http://127.0.0.1:8001" 
HF_TOKEN = "your_huggingface_token_here" # Required for some models, even if self-hosted

def get_rag_chain():
    """Constructs and returns the complete RAG chain."""
    
    # 1. Initialize the embedding model for the retriever
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 2. Connect to the existing ChromaDB collection
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5}) # Retrieve top 5 documents

    # 3. Define the prompt template
    template = """
    You are an expert programmer and code assistant. Use the following retrieved context to answer the user's question.
    If you don't know the answer from the context, state that you do not have enough information.
    Provide concise, accurate, and code-centric answers.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Initialize the LLM via its API endpoint
    llm = HuggingFaceEndpoint(
        endpoint_url=LLM_API_URL,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        model_kwargs={"max_new_tokens": 512}
    )

    # 5. Build the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
