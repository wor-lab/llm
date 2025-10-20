import os
import json
from typing import Dict, Iterable, List, Optional, Tuple
from datasets import load_dataset, IterableDataset, DatasetDict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm


DEFAULT_DATASETS = [
    "princeton-nlp/SWE-bench_Verified",
    "zai-org/humaneval-x",
    "Muennighoff/mbpp",
    "bigcode/bigcodebench",
    "microsoft/rStar-Coder",
    "bigcode/the-stack-v2",
    "livecodebench/code_generation_lite",
]


def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _try_load_streaming(name: str, split_candidates: List[str]) -> Tuple[Optional[IterableDataset], Optional[str]]:
    for split in split_candidates:
        try:
            ds = load_dataset(name, split=split, streaming=True)
            return ds, split
        except Exception:
            continue
    return None, None


def _try_load_disk(name: str, split_candidates: List[str]) -> Tuple[Optional[IterableDataset], Optional[str]]:
    for split in split_candidates:
        try:
            ds = load_dataset(name, split=split)
            return ds, split
        except Exception:
            continue
    return None, None


def _iter_dataset(name: str, max_samples: int) -> Iterable[Dict]:
    split_candidates = ["train", "test", "validation", "dev", "all"]

    # Prefer streaming to avoid huge memory/disk pulls
    ds, used_split = _try_load_streaming(name, split_candidates)
    if ds is None:
        ds, used_split = _try_load_disk(name, split_candidates)
    if ds is None:
        raise RuntimeError(f"Failed to load dataset: {name}")

    count = 0
    for ex in ds:
        yield ex
        count += 1
        if count >= max_samples:
            break


KEY_PRIORITY = [
    "prompt", "problem", "question", "description", "text", "docstring", "instruction",
    "title", "body", "context", "repo", "language", "task_id",
    "code", "canonical_solution", "solution", "solutions", "ground_truth", "reference_solution",
    "diff", "patch", "unit_tests", "tests", "test", "entry_point",
]


def _extract_text_blob(dataset_name: str, sample: Dict) -> Tuple[str, Dict]:
    selected = []
    for k in KEY_PRIORITY:
        if k in sample and sample[k] is not None:
            try:
                v = sample[k]
                if isinstance(v, (list, tuple)):
                    v = "\n".join([str(x) for x in v])
                elif isinstance(v, dict):
                    v = json.dumps(v, ensure_ascii=False)
                else:
                    v = str(v)
                if v.strip():
                    selected.append(f"{k.upper()}:\n{v}")
            except Exception:
                continue

    # Fallback: include any additional short scalar strings for recall
    if not selected:
        for k, v in sample.items():
            try:
                if isinstance(v, str) and v.strip():
                    selected.append(f"{k.upper()}:\n{v}")
            except Exception:
                continue

    content = "\n\n".join(selected).strip()
    metadata = {
        "dataset": dataset_name,
        "keys": [k for k in sample.keys()],
        "has_code": any(kk in sample for kk in ["code", "solution", "canonical_solution"]),
    }
    return content, metadata


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120,
        separators=[
            "\nclass ", "\ndef ", "\n#", "\n\n", "\n", " ", "",
        ],
        length_function=len,
        is_separator_regex=False,
    )


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        multi_process=False,
    )


def load_and_index_datasets(
    chroma_dir: str,
    embedding_model: str,
    datasets: Optional[List[str]] = None,
    max_samples_per_dataset: int = 500,
    collection_name: str = "code-corpus",
) -> Chroma:
    datasets = datasets or DEFAULT_DATASETS
    _ensure_dir(chroma_dir)

    embeddings = get_embeddings(embedding_model)
    vectordb = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=chroma_dir)

    splitter = build_text_splitter()

    for dname in datasets:
        texts: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []
        i = 0

        for sample in tqdm(_iter_dataset(dname, max_samples_per_dataset), desc=f"Ingest {dname}"):
            try:
                blob, meta = _extract_text_blob(dname, sample)
                if not blob:
                    continue

                chunks = splitter.split_text(blob)
                for idx, ch in enumerate(chunks):
                    texts.append(ch)
                    md = dict(meta)
                    md["chunk_idx"] = idx
                    md["sample_idx"] = i
                    metadatas.append(md)
                    ids.append(f"{dname}::s{i}::c{idx}")

                i += 1

                if len(texts) >= 512:  # batch write
                    vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                    texts, metadatas, ids = [], [], []

            except Exception:
                # Skip malformed samples silently
                continue

        if texts:
            vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        vectordb.persist()

    return vectordb
