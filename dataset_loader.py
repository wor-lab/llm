"""
WB AI CORPORATION - Data Division
Dataset Loading & Processing Pipeline
Real HuggingFace Dataset Integration
"""

import logging
from typing import List, Dict, Any
from datasets import load_dataset
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class CodeDocument:
    """Structured code document for RAG indexing"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding_text: str

class DatasetManager:
    """Enterprise dataset loading and preprocessing"""
    
    def __init__(self, datasets: List[str], hf_token: str = None):
        self.dataset_configs = datasets
        self.hf_token = hf_token
        self.documents = []
        
    def _generate_id(self, content: str, source: str) -> str:
        """Generate unique document ID"""
        hash_input = f"{source}:{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _load_swe_bench(self) -> List[CodeDocument]:
        """Load SWE-bench Verified dataset"""
        logger.info("Loading SWE-bench_Verified...")
        
        try:
            dataset = load_dataset(
                "princeton-nlp/SWE-bench_Verified",
                split="test",
                token=self.hf_token
            )
            
            docs = []
            for idx, item in enumerate(dataset):
                if idx >= 500:  # Limit for performance
                    break
                    
                content = f"""
Problem: {item.get('problem_statement', '')}

Patch:
{item.get('patch', '')}

Test:
{item.get('test_patch', '')}
"""
                
                doc = CodeDocument(
                    id=self._generate_id(content, 'swe_bench'),
                    content=content,
                    metadata={
                        'source': 'swe_bench',
                        'repo': item.get('repo', ''),
                        'instance_id': item.get('instance_id', ''),
                    },
                    embedding_text=content
                )
                docs.append(doc)
            
            logger.info(f"✅ Loaded {len(docs)} SWE-bench documents")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load SWE-bench: {e}")
            return []
    
    def _load_humaneval(self) -> List[CodeDocument]:
        """Load HumanEval dataset"""
        logger.info("Loading HumanEval...")
        
        try:
            dataset = load_dataset("openai/humaneval", split="test")
            
            docs = []
            for item in dataset:
                content = f"""
Task: {item['prompt']}

Canonical Solution:
{item['canonical_solution']}

Test Cases:
{item['test']}
"""
                
                doc = CodeDocument(
                    id=self._generate_id(content, 'humaneval'),
                    content=content,
                    metadata={
                        'source': 'humaneval',
                        'task_id': item['task_id'],
                        'entry_point': item['entry_point'],
                    },
                    embedding_text=f"{item['prompt']} {item['canonical_solution']}"
                )
                docs.append(doc)
            
            logger.info(f"✅ Loaded {len(docs)} HumanEval documents")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            return []
    
    def _load_mbpp(self) -> List[CodeDocument]:
        """Load MBPP dataset"""
        logger.info("Loading MBPP...")
        
        try:
            dataset = load_dataset(
                "google-research-datasets/mbpp",
                "sanitized",
                split="test"
            )
            
            docs = []
            for idx, item in enumerate(dataset):
                if idx >= 300:  # Limit
                    break
                    
                content = f"""
Problem: {item['text']}

Code:
{item['code']}

Test Cases:
{chr(10).join(item['test_list'])}
"""
                
                doc = CodeDocument(
                    id=self._generate_id(content, 'mbpp'),
                    content=content,
                    metadata={
                        'source': 'mbpp',
                        'task_id': item['task_id'],
                    },
                    embedding_text=f"{item['text']} {item['code']}"
                )
                docs.append(doc)
            
            logger.info(f"✅ Loaded {len(docs)} MBPP documents")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load MBPP: {e}")
            return []
    
    def _load_bigcodebench(self) -> List[CodeDocument]:
        """Load BigCodeBench dataset"""
        logger.info("Loading BigCodeBench...")
        
        try:
            dataset = load_dataset(
                "bigcode/bigcodebench",
                split="v0.1.2",
                token=self.hf_token
            )
            
            docs = []
            for idx, item in enumerate(dataset):
                if idx >= 200:  # Limit
                    break
                    
                content = f"""
Task: {item.get('instruct_prompt', '')}

Complete Prompt:
{item.get('complete_prompt', '')}

Code Context:
{item.get('code_context', '')}
"""
                
                doc = CodeDocument(
                    id=self._generate_id(content, 'bigcodebench'),
                    content=content,
                    metadata={
                        'source': 'bigcodebench',
                        'task_id': item.get('task_id', ''),
                    },
                    embedding_text=content
                )
                docs.append(doc)
            
            logger.info(f"✅ Loaded {len(docs)} BigCodeBench documents")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load BigCodeBench: {e}")
            return []
    
    def _load_the_stack(self) -> List[CodeDocument]:
        """Load The Stack v2 dataset (sampled)"""
        logger.info("Loading The Stack v2 (sampled)...")
        
        try:
            dataset = load_dataset(
                "bigcode/the-stack-v2-dedup",
                data_dir="data/python",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            
            docs = []
            for idx, item in enumerate(dataset):
                if idx >= 1000:  # Sample limit
                    break
                
                content = item.get('content', '')
                if len(content) < 100 or len(content) > 5000:  # Filter
                    continue
                    
                doc = CodeDocument(
                    id=self._generate_id(content, 'the_stack'),
                    content=content,
                    metadata={
                        'source': 'the_stack_v2',
                        'language': 'python',
                        'repo': item.get('max_stars_repo_name', ''),
                        'stars': item.get('max_stars_count', 0),
                    },
                    embedding_text=content[:1000]  # Limit for embedding
                )
                docs.append(doc)
            
            logger.info(f"✅ Loaded {len(docs)} The Stack documents")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load The Stack: {e}")
            return []
    
    def load_all_datasets(self) -> List[CodeDocument]:
        """Load all configured datasets"""
        logger.info("🔄 Starting dataset loading pipeline...")
        
        all_docs = []
        
        # Load each dataset
        all_docs.extend(self._load_swe_bench())
        all_docs.extend(self._load_humaneval())
        all_docs.extend(self._load_mbpp())
        all_docs.extend(self._load_bigcodebench())
        all_docs.extend(self._load_the_stack())
        
        logger.info(f"📚 Total documents loaded: {len(all_docs)}")
        
        self.documents = all_docs
        return all_docs
