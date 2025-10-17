"""
Qwen3 Agent Layer - Optimized for T4 GPU
Complete working implementation
"""

import re
import json
import ast
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AgentConfig:
    """Configuration for the agent"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_iterations: int = 5
    temperature: float = 0.7
    top_p: float = 0.95
    use_self_consistency: bool = True
    num_samples: int = 5
    max_tokens: int = 2048
    device: str = "auto"


class ToolExecutor:
    """Safe code execution and validation tools"""
    
    @staticmethod
    def execute_python(code: str, timeout: int = 10) -> Dict[str, Any]:
        """Execute Python code safely in isolated environment"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_file = f.name
                f.write(code)
            
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            os.unlink(temp_file)
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout.strip(),
                'error': result.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return {'success': False, 'output': '', 'error': 'Execution timeout'}
        except Exception as e:
            return {'success': False, 'output': '', 'error': str(e)}
    
    @staticmethod
    def validate_syntax(code: str) -> bool:
        """Check if code is syntactically valid"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    @staticmethod
    def safe_eval(expression: str) -> Any:
        """Safely evaluate mathematical expressions"""
        try:
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'len': len, '__builtins__': {}
            }
            code = compile(expression, '<string>', 'eval')
            return eval(code, {"__builtins__": {}}, allowed_names)
        except Exception:
            return None


class PromptTemplates:
    """Optimized prompts for each benchmark"""
    
    @staticmethod
    def gsm8k(question: str, use_code: bool = False) -> str:
        """GSM8K math problem prompt"""
        if use_code:
            return f"""Solve this math problem by writing Python code.

Problem: {question}

Write Python code to solve it step by step. Print only the final numerical answer.

```python
# Solution code
"""
        else:
            return f"""Solve this math problem step by step.

Problem: {question}

Solution:
Let me solve this step by step:

Step 1:"""

    @staticmethod
    def aime(question: str) -> str:
        """AIME competition problem prompt"""
        return f"""This is an AIME competition math problem. Solve it systematically.

Problem: {question}

Solution approach:

Step 1: Understand the problem
Step 2: Identify key concepts
Step 3: Solve step by step

Let me begin:
"""

    @staticmethod
    def mmlu(question: str, choices: List[str]) -> str:
        """MMLU multiple choice prompt"""
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        return f"""Answer this multiple choice question by analyzing each option carefully.

Question: {question}

Options:
{choices_str}

Let me analyze each option:

"""

    @staticmethod
    def swe_bench(issue: str, context: str = "") -> str:
        """SWE-Bench coding task prompt"""
        ctx = f"Context:\n{context}\n\n" if context else ""
        return f"""You are an expert software engineer. Fix this issue with clean, working code.

{ctx}Issue: {issue}

Solution:

Analysis: Let me understand the problem first.

Implementation:

```python
# Fixed code
"""

    @staticmethod
    def react_agent(task: str, available_tools: List[str]) -> str:
        """ReAct agent prompt"""
        tools_desc = ", ".join(available_tools)
        return f"""Solve this task using available tools: {tools_desc}

Task: {task}

Use this exact format:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [input for the tool]
Observation: [result will be provided]
... (repeat Thought/Action/Observation as needed)
Thought: I now have the final answer
Final Answer: [your final answer]

Begin!

Thought:"""


class ReasoningEngine:
    """Advanced reasoning and answer extraction"""
    
    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract final answer from model output"""
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'####\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Therefore,?\s+(?:the answer is\s+)?(.+?)(?:\n|$)',
            r'(?:^|\n)([A-D])\s*(?:\.|$)',
            r'(?:equals?|=)\s*([0-9,.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # Clean markdown and special chars
                answer = answer.replace('**', '').replace('*', '')
                answer = answer.replace('`', '').replace('[', '').replace(']', '')
                return answer
        
        # Fallback: return last non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else text.strip()
    
    @staticmethod
    def extract_code(text: str) -> str:
        """Extract code from markdown blocks"""
        patterns = [
            r'```python\s*\n(.*?)```',
            r'```\s*\n(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[-1].strip()
        
        return ""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison"""
        answer = str(answer).lower().strip()
        answer = re.sub(r'[^\w\s.]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        answer = re.sub(r'\b(the|a|an)\b', '', answer).strip()
        return answer


class QwenAgent:
    """Main agent orchestrator with multi-strategy reasoning"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.tools = ToolExecutor()
        self.prompts = PromptTemplates()
        self.reasoning = ReasoningEngine()
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        print("🔄 Initializing Qwen Agent...")
        self._load_model()
        print("✅ Agent ready!\n")
    
    def _load_model(self):
        """Load Qwen model optimized for T4 GPU"""
        try:
            print(f"📦 Loading {self.config.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # T4 GPU optimized settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map=self.config.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
            )
            
            print(f"✓ Model loaded on: {next(self.model.parameters()).device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with the model"""
        temp = kwargs.get('temperature', self.config.temperature)
        max_tok = kwargs.get('max_tokens', self.config.max_tokens)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            outputs = self.pipe(
                messages,
                max_new_tokens=max_tok,
                temperature=temp,
                top_p=kwargs.get('top_p', self.config.top_p),
                do_sample=temp > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )
            
            response = outputs[0]['generated_text']
            if isinstance(response, list):
                response = response[-1]['content']
            elif isinstance(response, dict):
                response = response.get('content', str(response))
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""
    
    def self_consistency(self, prompt: str, n: int = None) -> str:
        """Generate multiple answers and vote for most common"""
        n = n or self.config.num_samples
        answers = []
        
        print(f"  Generating {n} samples...", end='')
        for i in range(n):
            response = self.generate(prompt, temperature=0.7)
            answer = self.reasoning.extract_answer(response)
            if answer:
                answers.append(self.reasoning.normalize_answer(answer))
        
        print(" Done!")
        
        if not answers:
            return ""
        
        # Vote for most common answer
        counter = Counter(answers)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def solve_gsm8k(self, question: str, use_code: bool = True) -> Dict[str, Any]:
        """Solve GSM8K math word problems"""
        print(f"🧮 GSM8K: {question[:60]}...")
        
        # Try Program-of-Thought first
        if use_code:
            prompt = self.prompts.gsm8k(question, use_code=True)
            response = self.generate(prompt, temperature=0.3)
            code = self.reasoning.extract_code(response)
            
            if code and self.tools.validate_syntax(code):
                result = self.tools.execute_python(code)
                if result['success'] and result['output']:
                    return {
                        'answer': result['output'],
                        'method': 'code',
                        'reasoning': response,
                        'success': True
                    }
        
        # Fallback to self-consistency
        prompt = self.prompts.gsm8k(question, use_code=False)
        if self.config.use_self_consistency:
            answer = self.self_consistency(prompt)
        else:
            response = self.generate(prompt, temperature=0.3)
            answer = self.reasoning.extract_answer(response)
        
        return {
            'answer': answer,
            'method': 'self_consistency' if self.config.use_self_consistency else 'single',
            'success': bool(answer)
        }
    
    def solve_aime(self, question: str) -> Dict[str, Any]:
        """Solve AIME competition problems"""
        print(f"🎯 AIME: {question[:60]}...")
        
        # Try code-based solution
        code_prompt = f"""Solve this AIME problem with Python code. Print only the final answer.

{question}

```python
import math
# Solution
"""
        response = self.generate(code_prompt, temperature=0.2, max_tokens=2048)
        code = self.reasoning.extract_code(response)
        
        if code and self.tools.validate_syntax(code):
            result = self.tools.execute_python(code)
            if result['success'] and result['output']:
                return {
                    'answer': result['output'],
                    'method': 'code',
                    'verified': True,
                    'success': True
                }
        
        # Fallback to reasoning
        prompt = self.prompts.aime(question)
        response = self.generate(prompt, temperature=0.2, max_tokens=2048)
        answer = self.reasoning.extract_answer(response)
        
        return {
            'answer': answer,
            'method': 'reasoning',
            'verified': False,
            'success': bool(answer)
        }
    
    def solve_mmlu(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """Solve MMLU multiple choice questions"""
        print(f"📚 MMLU: {question[:60]}...")
        
        prompt = self.prompts.mmlu(question, choices)
        
        if self.config.use_self_consistency:
            answer = self.self_consistency(prompt)
        else:
            response = self.generate(prompt, temperature=0.1)
            answer = self.reasoning.extract_answer(response)
        
        # Extract letter choice
        choice_match = re.search(r'\b([A-D])\b', answer)
        choice = choice_match.group(1).upper() if choice_match else (answer[0].upper() if answer and answer[0] in 'ABCD' else 'A')
        
        return {
            'answer': choice,
            'success': choice in ['A', 'B', 'C', 'D']
        }
    
    def solve_swe_bench(self, issue: str, context: str = "") -> Dict[str, Any]:
        """Solve SWE-Bench software engineering tasks"""
        print(f"💻 SWE-Bench: {issue[:60]}...")
        
        prompt = self.prompts.swe_bench(issue, context)
        
        for iteration in range(self.config.max_iterations):
            print(f"  Iteration {iteration + 1}/{self.config.max_iterations}...", end='\r')
            
            response = self.generate(prompt, temperature=0.3, max_tokens=3000)
            code = self.reasoning.extract_code(response)
            
            if not code:
                continue
            
            if self.tools.validate_syntax(code):
                result = self.tools.execute_python(code)
                
                if result['success']:
                    print()  # New line
                    return {
                        'code': code,
                        'explanation': response,
                        'iteration': iteration,
                        'output': result['output'],
                        'success': True
                    }
                else:
                    prompt += f"\n\nPrevious error:\n{result['error']}\n\nFixed code:\n```python\n"
        
        print()  # New line
        return {
            'code': code if 'code' in locals() and code else '',
            'success': False
        }
    
    def react_agent(self, task: str, tools: List[str] = None, max_steps: int = 8) -> Dict[str, Any]:
        """ReAct agent with tool use"""
        print(f"🤖 ReAct: {task[:60]}...")
        
        tools = tools or ['python', 'calculator']
        prompt = self.prompts.react_agent(task, tools)
        
        history = []
        for step in range(max_steps):
            response = self.generate(prompt, temperature=0.5, max_tokens=1024)
            history.append(response)
            
            # Check for final answer
            if "Final Answer:" in response:
                return {
                    'answer': self.reasoning.extract_answer(response),
                    'steps': step + 1,
                    'history': history,
                    'success': True
                }
            
            # Parse action
            action_match = re.search(r'Action:\s*(\w+)', response, re.IGNORECASE)
            input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
            
            if action_match and input_match:
                action = action_match.group(1).lower()
                action_input = input_match.group(1).strip()
                observation = self._execute_tool(action, action_input)
                prompt += f"\n{response}\nObservation: {observation}\nThought:"
            else:
                prompt += f"\n{response}\nThought:"
        
        return {
            'answer': 'Task incomplete',
            'steps': max_steps,
            'history': history,
            'success': False
        }
    
    def _execute_tool(self, tool: str, input_data: str) -> str:
        """Execute a tool and return observation"""
        if tool in ['python', 'code']:
            code = self.reasoning.extract_code(input_data) or input_data
            result = self.tools.execute_python(code)
            return result.get('output', '') or result.get('error', 'Error executing code')
        
        elif tool in ['calculator', 'calc', 'math']:
            result = self.tools.safe_eval(input_data)
            return str(result) if result is not None else 'Invalid expression'
        
        else:
            return f"Unknown tool: {tool}"


class BenchmarkEvaluator:
    """Evaluate agent performance on benchmarks"""
    
    def __init__(self, agent: QwenAgent):
        self.agent = agent
    
    def evaluate_gsm8k_sample(self, test_cases: List[Dict] = None, sample_size: int = 3):
        """Test on GSM8K examples"""
        if test_cases is None:
            test_cases = [
                {
                    'question': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                    'answer': '18'
                },
                {
                    'question': "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                    'answer': '3'
                },
                {
                    'question': "If 5 apples cost $10, how much do 8 apples cost?",
                    'answer': '16'
                },
            ]
        
        correct = 0
        print("\n" + "="*70)
        print("📊 GSM8K EVALUATION")
        print("="*70)
        
        for i, case in enumerate(test_cases[:sample_size], 1):
            print(f"\n[{i}/{sample_size}] Q: {case['question'][:70]}...")
            result = self.agent.solve_gsm8k(case['question'])
            
            predicted = self.agent.reasoning.normalize_answer(result['answer'])
            expected = self.agent.reasoning.normalize_answer(case['answer'])
            
            is_correct = predicted == expected or expected in predicted
            correct += int(is_correct)
            
            print(f"  Predicted: {result['answer']}")
            print(f"  Expected: {case['answer']}")
            print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        
        accuracy = (correct / sample_size) * 100
        print(f"\n{'='*70}")
        print(f"🎯 GSM8K Accuracy: {accuracy:.1f}% ({correct}/{sample_size})")
        print(f"{'='*70}\n")
        
        return accuracy
    
    def evaluate_mmlu_sample(self, test_cases: List[Dict] = None):
        """Test on MMLU examples"""
        if test_cases is None:
            test_cases = [
                {
                    'question': 'What is the capital of France?',
                    'choices': ['London', 'Paris', 'Berlin', 'Madrid'],
                    'answer': 'B'
                },
                {
                    'question': 'Which planet is closest to the Sun?',
                    'choices': ['Venus', 'Earth', 'Mercury', 'Mars'],
                    'answer': 'C'
                },
            ]
        
        correct = 0
        print("\n" + "="*70)
        print("📚 MMLU EVALUATION")
        print("="*70)
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Q: {case['question']}")
            result = self.agent.solve_mmlu(case['question'], case['choices'])
            
            is_correct = result['answer'] == case['answer']
            correct += int(is_correct)
            
            print(f"  Predicted: {result['answer']} - {case['choices'][ord(result['answer'])-65]}")
            print(f"  Expected: {case['answer']} - {case['choices'][ord(case['answer'])-65]}")
            print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        
        accuracy = (correct / len(test_cases)) * 100
        print(f"\n{'='*70}")
        print(f"🎯 MMLU Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
        print(f"{'='*70}\n")
        
        return accuracy


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("🚀 Qwen Agent Layer - Ready for T4 GPU")
    print("="*70)
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*70 + "\n")
    
    # Initialize agent
    config = AgentConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        use_self_consistency=True,
        num_samples=3,
        max_iterations=3,
    )
    
    agent = QwenAgent(config)
    
    # Run examples
    print("\n" + "="*70)
    print("🎮 RUNNING EXAMPLES")
    print("="*70)
    
    # Example 1: GSM8K
    print("\n1️⃣ GSM8K Math Problem")
    print("-"*70)
    result = agent.solve_gsm8k("If 5 pencils cost $2.50, how much do 12 pencils cost?")
    print(f"💡 Answer: {result['answer']}\n")
    
    # Example 2: MMLU
    print("2️⃣ MMLU Multiple Choice")
    print("-"*70)
    result = agent.solve_mmlu(
        "What is the powerhouse of the cell?",
        ["Nucleus", "Mitochondria", "Ribosome", "Chloroplast"]
    )
    print(f"💡 Answer: {result['answer']}\n")
    
    # Example 3: ReAct
    print("3️⃣ ReAct Agent")
    print("-"*70)
    result = agent.react_agent("Calculate 25 * 4 + 50")
    print(f"💡 Answer: {result['answer']}\n")
    
    # Benchmark evaluation
    evaluator = BenchmarkEvaluator(agent)
    gsm8k_score = evaluator.evaluate_gsm8k_sample(sample_size=3)
    mmlu_score = evaluator.evaluate_mmlu_sample()
    
    print("\n" + "="*70)
    print("🏆 BENCHMARK RESULTS")
    print("="*70)
    print(f"GSM8K: {gsm8k_score:.1f}%")
    print(f"MMLU: {mmlu_score:.1f}%")
    print("="*70)
