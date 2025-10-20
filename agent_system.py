"""
WB AI CORPORATION — ENGINEERING DIVISION
Multi-Agent Orchestration using LangGraph

MISSION: Coordinate specialized agents for code intelligence
AGENT: CodeArchitect
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import operator
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# AGENT STATE DEFINITION
# ══════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Shared state across all agents"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    task_type: str
    context: dict
    retrieved_docs: list
    final_answer: str
    confidence: float

class TaskType(Enum):
    """Task classification"""
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    DEBUG = "debug"
    OPTIMIZATION = "optimization"
    GENERAL = "general"

# ══════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ══════════════════════════════════════════════════════

class ModelLoader:
    """Load Qwen2.5-1.5B model with optimization"""
    
    @staticmethod
    def load_model():
        logger.info("🔄 Loading Qwen2.5-1.5B-Instruct...")
        
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        logger.info("✅ Model loaded successfully")
        return llm, tokenizer

# ══════════════════════════════════════════════════════
# SPECIALIZED AGENTS
# ══════════════════════════════════════════════════════

class RouterAgent:
    """Routes tasks to appropriate specialist agents"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "RouterAgent"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("🧭 [RouterAgent] Analyzing task...")
        
        last_message = state['messages'][-1].content
        
        # Simple keyword-based routing (can be enhanced with LLM classification)
        if any(kw in last_message.lower() for kw in ['write', 'create', 'generate', 'implement']):
            task_type = TaskType.CODE_GENERATION.value
            next_agent = "CodeGenerator"
        elif any(kw in last_message.lower() for kw in ['explain', 'what does', 'how does', 'understand']):
            task_type = TaskType.CODE_EXPLANATION.value
            next_agent = "ExplainerAgent"
        elif any(kw in last_message.lower() for kw in ['bug', 'error', 'fix', 'debug', 'wrong']):
            task_type = TaskType.DEBUG.value
            next_agent = "DebugAgent"
        elif any(kw in last_message.lower() for kw in ['optimize', 'improve', 'faster', 'better']):
            task_type = TaskType.OPTIMIZATION.value
            next_agent = "OptimizerAgent"
        else:
            task_type = TaskType.GENERAL.value
            next_agent = "GeneralAgent"
        
        state['task_type'] = task_type
        state['current_agent'] = next_agent
        
        logger.info(f"→ Routing to {next_agent} ({task_type})")
        return state

class RetrieverAgent:
    """Retrieves relevant context from ChromaDB"""
    
    def __init__(self, chroma_client, embedder):
        self.client = chroma_client
        self.embedder = embedder
        self.name = "RetrieverAgent"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("🔍 [RetrieverAgent] Searching knowledge base...")
        
        query = state['messages'][-1].content
        task_type = state.get('task_type', 'general')
        
        # Determine which collection to search
        if 'humaneval' in task_type or 'code_generation' in task_type:
            collections = ['humaneval', 'mbpp']
        elif 'swe' in task_type or 'debug' in task_type:
            collections = ['swe_bench', 'bigcodebench']
        else:
            collections = ['humaneval', 'mbpp', 'bigcodebench']
        
        all_results = []
        
        for col_name in collections:
            try:
                collection = self.client.get_collection(col_name)
                query_embedding = self.embedder.encode([query]).tolist()
                
                results = collection.query(
                    query_embeddings=query_embedding,
                    n_results=3
                )
                
                if results['documents']:
                    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                        all_results.append({
                            'document': doc,
                            'metadata': metadata,
                            'source': col_name
                        })
            except Exception as e:
                logger.warning(f"⚠️ Error querying {col_name}: {e}")
        
        state['retrieved_docs'] = all_results
        logger.info(f"✅ Retrieved {len(all_results)} relevant documents")
        
        return state

class CodeGeneratorAgent:
    """Generates code solutions"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "CodeGeneratorAgent"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("💻 [CodeGenerator] Creating solution...")
        
        query = state['messages'][-1].content
        context_docs = state.get('retrieved_docs', [])
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"Example {i+1}:\n{doc['document']}"
            for i, doc in enumerate(context_docs[:3])
        ])
        
        prompt = f"""You are an expert programmer. Generate clean, efficient code.

Context Examples:
{context}

User Request: {query}

Provide a complete, working solution with comments:
"""
        
        response = self.llm.invoke(prompt)
        
        state['final_answer'] = response
        state['confidence'] = 0.85
        state['messages'].append(AIMessage(content=response))
        
        return state

class ExplainerAgent:
    """Explains code and concepts"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "ExplainerAgent"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("📖 [Explainer] Analyzing and explaining...")
        
        query = state['messages'][-1].content
        context_docs = state.get('retrieved_docs', [])
        
        context = "\n".join([doc['document'] for doc in context_docs[:2]])
        
        prompt = f"""You are a technical educator. Explain code clearly and thoroughly.

Reference Material:
{context}

Question: {query}

Provide a detailed explanation:
"""
        
        response = self.llm.invoke(prompt)
        
        state['final_answer'] = response
        state['confidence'] = 0.80
        state['messages'].append(AIMessage(content=response))
        
        return state

class GeneralAgent:
    """Handles general queries"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "GeneralAgent"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("🤖 [GeneralAgent] Processing query...")
        
        query = state['messages'][-1].content
        context_docs = state.get('retrieved_docs', [])
        
        context = "\n".join([doc['document'] for doc in context_docs[:3]])
        
        prompt = f"""You are a helpful AI assistant with expertise in programming.

Available Context:
{context}

User Query: {query}

Response:
"""
        
        response = self.llm.invoke(prompt)
        
        state['final_answer'] = response
        state['confidence'] = 0.75
        state['messages'].append(AIMessage(content=response))
        
        return state

# ══════════════════════════════════════════════════════
# LANGGRAPH ORCHESTRATION
# ══════════════════════════════════════════════════════

class AgentOrchestrator:
    """WB AI Corporation Multi-Agent System"""
    
    def __init__(self):
        logger.info("🏗️ Building agent graph...")
        
        # Load model
        self.llm, self.tokenizer = ModelLoader.load_model()
        
        # Load ChromaDB and embedder
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize agents
        self.router = RouterAgent(self.llm)
        self.retriever = RetrieverAgent(self.chroma_client, self.embedder)
        self.code_gen = CodeGeneratorAgent(self.llm)
        self.explainer = ExplainerAgent(self.llm)
        self.general = GeneralAgent(self.llm)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self.router)
        workflow.add_node("retriever", self.retriever)
        workflow.add_node("code_generator", self.code_gen)
        workflow.add_node("explainer", self.explainer)
        workflow.add_node("general", self.general)
        
        # Define edges
        workflow.set_entry_point("router")
        
        workflow.add_edge("router", "retriever")
        
        # Conditional routing from retriever
        def route_after_retrieval(state):
            return state['current_agent'].lower().replace("agent", "")
        
        workflow.add_conditional_edges(
            "retriever",
            route_after_retrieval,
            {
                "codegenerator": "code_generator",
                "explainer": "explainer",
                "general": "general"
            }
        )
        
        # All specialist agents end
        workflow.add_edge("code_generator", END)
        workflow.add_edge("explainer", END)
        workflow.add_edge("general", END)
        
        return workflow.compile()
    
    def process_query(self, query: str) -> dict:
        """Process user query through agent system"""
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_agent": "router",
            "task_type": "",
            "context": {},
            "retrieved_docs": [],
            "final_answer": "",
            "confidence": 0.0
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "answer": result['final_answer'],
            "confidence": result['confidence'],
            "task_type": result['task_type'],
            "sources": len(result['retrieved_docs'])
        }

if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    
    test_query = "Write a Python function to check if a number is prime"
    result = orchestrator.process_query(test_query)
    
    print(f"\n{'='*60}")
    print(f"Query: {result['query']}")
    print(f"Task Type: {result['task_type']}")
    print(f"Sources Used: {result['sources']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"{'='*60}\n")
