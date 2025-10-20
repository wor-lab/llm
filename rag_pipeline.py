import os
import json
from typing import Dict, List, TypedDict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END


def _get_env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def build_llm() -> HuggingFacePipeline:
    model_id = _get_env("MODEL_ID")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    gen_pipe = pipeline(
        task="text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=1024,
        temperature=0.2,
        top_p=0.95,
        repetition_penalty=1.05,
        do_sample=True,
        pad_token_id=tok.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)


def build_retriever(chroma_dir: str, collection: str = "code-corpus", k: int = 8):
    db = Chroma(collection_name=collection, persist_directory=chroma_dir)
    return db.as_retriever(search_kwargs={"k": k})


class RAGState(TypedDict):
    question: str
    subqueries: List[str]
    contexts: List[str]
    draft: str
    final: str
    citations: List[Dict]


PLAN_PROMPT = PromptTemplate.from_template(
    "You are a senior engineer. Decompose the user query into 1-4 focused sub-queries for retrieval.\n"
    "Return a JSON list of strings only.\n\n"
    "Query: {question}\n"
)

GEN_PROMPT = PromptTemplate.from_template(
    "System:\n"
    "You are an enterprise code assistant. Use only the provided context to answer.\n"
    "If the answer is uncertain, state the uncertainty and what evidence is missing.\n\n"
    "Context:\n{context}\n\n"
    "User:\n{question}\n\n"
    "Answer:"
)

CRITIQUE_PROMPT = PromptTemplate.from_template(
    "Act as a security and quality auditor. Identify unsupported claims, missing citations, and risks.\n"
    "Return a JSON with keys: issues (list[str]), risk (low|medium|high).\n\n"
    "Draft:\n{draft}\n"
)


def _llm_json(llm: HuggingFacePipeline, prompt: str) -> List[str]:
    out = llm.invoke(prompt)
    text = out if isinstance(out, str) else getattr(out, "content", str(out))
    try:
        js = json.loads(text.strip().split("```json")[-1].split("```")[0]) if "```json" in text else json.loads(text)
    except Exception:
        # naive list extraction fallback
        lines = [l.strip("-* \n") for l in text.splitlines() if l.strip()]
        js = [l for l in lines if len(l) > 2][:4]
    if isinstance(js, list):
        return [str(s) for s in js][:4]
    return [str(js)]


def _join_contexts(ctxs: List[str]) -> str:
    if not ctxs:
        return "No relevant context."
    return "\n\n----\n\n".join(ctxs)


def create_agentic_rag(chroma_dir: str):
    retriever = build_retriever(chroma_dir)
    llm = build_llm()

    def plan_node(state: RAGState) -> RAGState:
        prompt = PLAN_PROMPT.format(question=state["question"])
        subs = _llm_json(llm, prompt)
        return {**state, "subqueries": subs or [state["question"]]}

    def retrieve_node(state: RAGState) -> RAGState:
        contexts: List[str] = []
        citations: List[Dict] = []
        for q in state["subqueries"]:
            docs = retriever.get_relevant_documents(q)
            for d in docs:
                contexts.append(d.page_content)
                citations.append(d.metadata)
        return {**state, "contexts": contexts, "citations": citations}

    def generate_node(state: RAGState) -> RAGState:
        context = _join_contexts(state.get("contexts", []))
        prompt = GEN_PROMPT.format(context=context, question=state["question"])
        draft = llm.invoke(prompt)
        draft = draft if isinstance(draft, str) else getattr(draft, "content", str(draft))
        return {**state, "draft": draft}

    def critique_refine_node(state: RAGState) -> RAGState:
        critique = llm.invoke(CRITIQUE_PROMPT.format(draft=state["draft"]))
        critique = critique if isinstance(critique, str) else getattr(critique, "content", str(critique))
        refine_prompt = (
            "Revise the answer based on the critique while staying grounded in the context.\n"
            f"Critique:\n{critique}\n\n"
            f"Original Draft:\n{state['draft']}\n\n"
            "Final Answer:"
        )
        final = llm.invoke(refine_prompt)
        final = final if isinstance(final, str) else getattr(final, "content", str(final))
        return {**state, "final": final}

    graph = StateGraph(RAGState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique_refine", critique_refine_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique_refine")
    graph.add_edge("critique_refine", END)

    app = graph.compile()
    return app, llm, retriever


def run_agentic_rag(app, question: str) -> Dict:
    state: RAGState = {"question": question, "subqueries": [], "contexts": [], "draft": "", "final": "", "citations": []}
    result = app.invoke(state)
    return {
        "answer": result.get("final", result.get("draft", "")),
        "citations": result.get("citations", []),
        "subqueries": result.get("subqueries", []),
    }
