from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma


@dataclass
class CodeArchitect:
    llm: HuggingFacePipeline

    def synthesize(self, spec: str) -> str:
        prompt = (
            "You are CodeArchitect. Produce robust, minimal, production-grade code for the following spec.\n"
            "Return only code blocks and necessary comments.\n\n"
            f"Spec:\n{spec}\n"
        )
        return self.llm.invoke(prompt)


@dataclass
class OpsManager:
    store: Optional[Chroma]
    def status(self) -> Dict[str, Any]:
        info = {"collections": [], "count": 0}
        if self.store is not None:
            try:
                info["collections"] = [self.store._collection.name]  # type: ignore
                info["count"] = self.store._collection.count()  # type: ignore
            except Exception:
                pass
        return info


@dataclass
class SecAnalyst:
    llm: HuggingFacePipeline
    def audit_text(self, text: str) -> Dict[str, Any]:
        prompt = (
            "Security Audit: identify secrets, hardcoded credentials, insecure patterns, and supply a list of risks.\n"
            "Return JSON with keys: secrets(list), insecure_patterns(list), risk(low|medium|high).\n\n"
            f"Text:\n{text}\n"
        )
        return {"report": self.llm.invoke(prompt)}


@dataclass
class DesignMind:
    def rag_answer_template(self) -> str:
        return (
            "Format answers with concise sections: Overview, Key Points, References.\n"
            "Avoid verbosity. Keep to-the-point engineering language.\n"
        )


@dataclass
class WordSmith:
    llm: HuggingFacePipeline
    def reframe(self, text: str) -> str:
        prompt = (
            "Rewrite the following into clear, executive-brief engineering prose.\n\n"
            f"{text}\n"
        )
        return self.llm.invoke(prompt)


@dataclass
class DataSynth:
    store: Optional[Chroma]
    def diagnostics(self, queries: List[str]) -> Dict[str, Any]:
        stats = {"queries": len(queries), "retrieval": []}
        if self.store is None:
            return stats
        retriever = self.store.as_retriever(search_kwargs={"k": 4})
        for q in queries:
            docs = retriever.get_relevant_documents(q)
            stats["retrieval"].append({"q": q, "hits": len(docs)})
        return stats


@dataclass
class Analyst:
    def corpus_strategy(self) -> str:
        return (
            "Prioritize HumanEval-X and MBPP for function-level tasks; incorporate SWE-bench for repository-level context.\n"
            "Deduplicate overlapping samples and cap Stack v2 ingestion via streaming head to manage footprint.\n"
        )


@dataclass
class AutoBot:
    components: Dict[str, Any]
    def execute(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if task_type == "codegen":
            return {"result": self.components["CodeArchitect"].synthesize(payload.get("spec", ""))}
        if task_type == "audit":
            return {"result": self.components["SecAnalyst"].audit_text(payload.get("text", ""))}
        if task_type == "rewrite":
            return {"result": self.components["WordSmith"].reframe(payload.get("text", ""))}
        if task_type == "ops_status":
            return {"result": self.components["OpsManager"].status()}
        if task_type == "diag":
            return {"result": self.components["DataSynth"].diagnostics(payload.get("queries", []))}
        if task_type == "strategy":
            return {"result": self.components["Analyst"].corpus_strategy()}
        return {"error": f"unknown task_type: {task_type}"}


def build_agents(llm: HuggingFacePipeline, store: Optional[Chroma]) -> Dict[str, Any]:
    agents = {
        "CodeArchitect": CodeArchitect(llm=llm),
        "OpsManager": OpsManager(store=store),
        "SecAnalyst": SecAnalyst(llm=llm),
        "DesignMind": DesignMind(),
        "WordSmith": WordSmith(llm=llm),
        "DataSynth": DataSynth(store=store),
        "Analyst": Analyst(),
        "AutoBot": None,  # fill after instantiation
    }
    agents["AutoBot"] = AutoBot(components=agents)
    return agents
