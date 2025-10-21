import os
import json
from typing import Optional, List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Optional OpenAI-compatible path if MODEL_SERVER_API_BASE is provided
OPENAI_COMPAT_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI
    OPENAI_COMPAT_AVAILABLE = True
except Exception:
    OPENAI_COMPAT_AVAILABLE = False


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    return v


def as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).lower() in {"1", "true", "yes", "y"}


class ModelFactory:
    """
    Provides an LLM for LangChain usage.
    - Default: local HuggingFace pipeline (Qwen3-1.7B).
    - Optional: OpenAI-compatible base URL via MODEL_SERVER_API_BASE + MODEL_SERVER_API_KEY.
    """

    @staticmethod
    def build_llm() -> Any:
        api_base = env("MODEL_SERVER_API_BASE", "").strip()
        api_key = env("MODEL_SERVER_API_KEY", "").strip()
        model_name = env("MODEL_ID", "Qwen/Qwen3-1.7B-Instruct")
        temperature = float(env("TEMPERATURE", "0.2"))
        max_new_tokens = int(env("MAX_NEW_TOKENS", "512"))

        # Remote (OpenAI-compatible) path if configured
        if api_base and api_key and OPENAI_COMPAT_AVAILABLE:
            # Make it compatible with langchain_openai
            os.environ["OPENAI_API_KEY"] = api_key
            # langchain_openai uses "base_url"
            llm = ChatOpenAI(
                base_url=api_base,
                model=env("OPENAI_COMPAT_MODEL", model_name),
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return llm

        # Local HF pipeline path
        device_map = env("DEVICE_MAP", "auto")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        gen_pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        llm = HuggingFacePipeline(pipeline=gen_pipe)
        return llm


class EmbeddingFactory:
    """
    HuggingFace Embeddings (CPU-friendly by default).
    Default: BAAI/bge-small-en-v1.5 with normalize embeddings.
    """

    @staticmethod
    def build_embeddings():
        embed_model = env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        normalize = as_bool(env("EMBEDDING_NORMALIZE", "1"), True)
        return HuggingFaceEmbeddings(
            model_name=embed_model,
            encode_kwargs={"normalize_embeddings": normalize},
        )


class VectorStoreManager:
    """
    Manages Chroma vector stores per collection.
    """

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.persist_dir = env("CHROMA_DIR", "./data/chroma")
        os.makedirs(self.persist_dir, exist_ok=True)
        self._stores: Dict[str, Chroma] = {}

    def get_store(self, collection: str) -> Chroma:
        if collection not in self._stores:
            self._stores[collection] = Chroma(
                collection_name=collection,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._stores[collection]

    def add_texts(
        self,
        collection: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> Tuple[int, int]:
        store = self.get_store(collection)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        store.persist()
        count = store._collection.count()  # type: ignore
        return len(texts), count

    def search(
        self, collection: str, query: str, k: int = 5
    ) -> List[Document]:
        store = self.get_store(collection)
        return store.similarity_search(query, k=k)

    def retriever(self, collection: str, k: int = 5):
        return self.get_store(collection).as_retriever(search_kwargs={"k": k})

    def stats(self) -> Dict[str, Any]:
        stats = {}
        for name, store in self._stores.items():
            try:
                stats[name] = {"count": store._collection.count()}  # type: ignore
            except Exception:
                stats[name] = {"count": None}
        return stats


def build_rag_chain(llm: Any):
    """
    Simple RAG chain given a prompt and retrieved context.
    """
    template = (
        "You are an enterprise assistant. Use the provided context to answer precisely.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer concisely and cite dataset and source if applicable."
    )
    prompt = PromptTemplate.from_template(template)
    return LLMChain(llm=llm, prompt=prompt)


class RAGResources:
    """
    Shared RAG resources for agents/API.
    """

    def __init__(self):
        self.llm = ModelFactory.build_llm()
        self.embeddings = EmbeddingFactory.build_embeddings()
        self.vs = VectorStoreManager(self.embeddings)

    def default_retriever(self):
        default_collection = env("DEFAULT_COLLECTION", "swe_bench")
        k = int(env("TOP_K", "5"))
        return self.vs.retriever(default_collection, k=k)

    def rag_answer(self, question: str, collections: Optional[List[str]] = None, k: int = 5) -> Dict[str, Any]:
        collections = collections or [env("DEFAULT_COLLECTION", "swe_bench")]
        all_docs: List[Document] = []
        for c in collections:
            try:
                all_docs.extend(self.vs.search(c, question, k=k))
            except Exception:
                continue

        context = "\n---\n".join([f"[{d.metadata.get('dataset','unknown')}] {d.page_content[:2000]}" for d in all_docs])
        chain = build_rag_chain(self.llm)
        answer = chain.run({"context": context, "question": question})
        return {"answer": answer, "context_docs": [d.metadata for d in all_docs]}
