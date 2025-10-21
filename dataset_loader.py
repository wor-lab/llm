import os
import hashlib
from typing import Dict, Any, Iterable, List, Tuple, Optional

from datasets import load_dataset, DatasetDict, IterableDataset
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, CodeSplitter
from rag_pipeline import RAGResources, env

# Helpers

def _hash_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _chunker_for_kind(kind: str) -> RecursiveCharacterTextSplitter:
    # Code-aware defaults
    if kind == "code":
        return CodeSplitter.from_language(
            language=Language.PYTHON,
            # Python as default; we also fallback to generic chunker when lang unknown
            chunk_size=800,
            chunk_overlap=80,
        )
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

def _add_docs(resources: RAGResources, collection: str, docs: List[Document]) -> Tuple[int, int]:
    if not docs:
        return (0, 0)
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    ids = [_hash_id(f"{metas[i].get('source','')}-{texts[i][:64]}") for i in range(len(docs))]
    return resources.vs.add_texts(collection=collection, texts=texts, metadatas=metas, ids=ids)

# SWE-bench Verified

def ingest_swe_bench(resources: RAGResources, max_docs: int = 500) -> Tuple[int, int]:
    """
    Ingests SWE-bench Verified dataset into 'swe_bench' collection.
    """
    collection = "swe_bench"
    docs: List[Document] = []
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="train", streaming=True)
    except Exception:
        # Fallback to default split if needed
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="train")

    chunker = _chunker_for_kind("text")
    count = 0
    for item in ds:
        if count >= max_docs:
            break
        # Common fields in SWE-bench Verified include: repo, problem_statement, patch, test, etc.
        # We will combine key text fields if present.
        parts = []
        for key in ["title", "problem_statement", "summary", "issue_url", "patch", "test", "repo", "instance_id"]:
            if key in item and item[key]:
                parts.append(f"{key.upper()}: {item[key]}")
        content = "\n\n".join(parts).strip()
        if not content:
            continue
        for chunk in chunker.split_text(content):
            meta = {
                "dataset": "SWE-bench_Verified",
                "source": item.get("issue_url", item.get("repo", "swe-bench")),
                "repo": item.get("repo"),
                "instance_id": item.get("instance_id"),
                "kind": "issue_patch",
            }
            docs.append(Document(page_content=chunk, metadata=meta))
        count += 1

    return _add_docs(resources, collection, docs)

# The Stack v2 (filtered sampling)

def ingest_the_stack(resources: RAGResources, max_docs: int = 200, languages: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    Ingest filtered subset of bigcode/the-stack-v2 into 'the_stack' collection.
    Use streaming and optional language filter.
    """
    collection = "the_stack"
    docs: List[Document] = []
    langs = set([l.strip().lower() for l in (languages or env("THE_STACK_LANGS", "python,javascript").split(",")) if l.strip()])

    try:
        ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
    except Exception:
        ds = load_dataset("bigcode/the-stack-v2", split="train")

    # Heuristic chunker for code
    code_chunker = _chunker_for_kind("code")
    count = 0
    for item in ds:
        if count >= max_docs:
            break
        lang = (item.get("lang") or item.get("language") or "").lower()
        if langs and lang and lang not in langs:
            continue
        content = item.get("content") or item.get("text") or ""
        if not content:
            continue
        # Split code into chunks
        for chunk in code_chunker.split_text(content):
            meta = {
                "dataset": "the-stack-v2",
                "source": item.get("path", item.get("repo_name", "the-stack")),
                "lang": lang,
                "kind": "code",
            }
            docs.append(Document(page_content=chunk, metadata=meta))
        count += 1

    return _add_docs(resources, collection, docs)

# rStar-Coder (if available)

def ingest_rstar(resources: RAGResources, max_docs: int = 200) -> Tuple[int, int]:
    """
    Attempt to ingest microsoft/rStar-Coder if available on HF as a dataset.
    If not available as a dataset, this will no-op safely.
    """
    collection = "rstar"
    docs: List[Document] = []
    try:
        ds = load_dataset("microsoft/rStar-Coder", split="train", streaming=True)
    except Exception:
        try:
            ds = load_dataset("microsoft/rStar-Coder", split="train")
        except Exception:
            return (0, resources.vs.get_store(collection).__dict__.get("_collection", None).count() if hasattr(resources.vs.get_store(collection), "_collection") else 0)  # type: ignore

    chunker = _chunker_for_kind("code")
    count = 0
    for item in ds:
        if count >= max_docs:
            break
        content = item.get("content") or item.get("text") or ""
        if not content:
            continue
        for chunk in chunker.split_text(content):
            meta = {
                "dataset": "rStar-Coder",
                "source": item.get("path", "rstar"),
                "kind": "code",
            }
            docs.append(Document(page_content=chunk, metadata=meta))
        count += 1

    return _add_docs(resources, collection, docs)

# Public API

def ingest_all(
    resources: RAGResources,
    swe_max: int = 500,
    stack_max: int = 200,
    rstar_max: int = 200,
    stack_langs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    swe_added, swe_total = ingest_swe_bench(resources, swe_max)
    stack_added, stack_total = ingest_the_stack(resources, stack_max, stack_langs)
    rstar_added, rstar_total = ingest_rstar(resources, rstar_max)
    return {
        "swe_bench": {"added": swe_added, "total": swe_total},
        "the_stack": {"added": stack_added, "total": stack_total},
        "rstar": {"added": rstar_added, "total": rstar_total},
    }
