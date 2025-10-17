# ============================================================================
# QWEN AGENT v2.0 - PERFORMANCE OPTIMIZED FOR 90%+ ACCURACY
# Enhanced prompts, better reasoning, improved answer extraction
# ============================================================================

import re
import json
import ast
import tempfile
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 QWEN AGENT v2.0 - PERFORMANCE OPTIMIZED")
print("="*70)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*70 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AgentConfig:
    # Use larger model for better performance (still fits in T4)
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Upgraded from 0.5B
    max_iterations: int = 5
    temperature: float = 0.3  # Lower for more focused answers
    top_p: float = 0.9
    use_self_consistency: bool = True
    num_samples: int = 5  # More samples for better voting
    max_tokens: int = 2048

# ============================================================================
# IMPROVED PROMPTS
# ============================================================================

class ImprovedPrompts:
    """Enhanced prompts with better instructions"""
    
    @staticmethod
    def gsm8k(question: str, use_code: bool = True) -> str:
        if use_code:
            return f"""You are a math expert. Solve this problem using Python code.

Problem: {question}

Write clean Python code that calculates and prints ONLY the final numerical answer (no explanations).

```python
# Calculate answer
"""
        else:
            return f"""You are a math expert. Solve this problem step by step and provide the final numerical answer.

Problem: {question}

Let's solve this carefully:

Step 1: Identify what we know
Step 2: Set up the calculation
Step 3: Calculate the answer

Solution:
"""

    @staticmethod
    def mmlu(question: str, choices: List[str]) -> str:
        choices_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        return f"""You are an expert in various subjects. Answer this question by selecting the correct option.

Question: {question}

Choices:
{choices_str}

Think step by step:
1. Analyze the question carefully
2. Evaluate each option
3. Select the most accurate answer

Analysis: Let me examine each choice.

After careful consideration, the correct answer is: """

    @staticmethod
    def aime(question: str) -> str:
        return f"""You are a competition mathematics expert. Solve this AIME problem step by step.

Problem: {question}

Solution approach:
1. Understand what is being asked
2. Identify relevant mathematical concepts
3. Solve systematically
4. Verify the answer

Let me solve:

"""

    @staticmethod
    def swe_bench(issue: str) -> str:
        return f"""You are an expert software engineer. Write clean, working Python code to solve this issue.

Issue: {issue}

Provide a complete, working solution:

```python
# Complete solution
"""

    @staticmethod
    def react(task: str, tools: List[str]) -> str:
        return f"""You are a helpful AI assistant. Use the available tools to solve this task.

Available tools: {', '.join(tools)}

Task: {task}

Use this format EXACTLY:
Thought: What I need to do next
Action: tool_name
Action Input: specific input
Observation: [result will appear here]
... repeat Thought/Action/Observation as needed ...
Thought: I now know the final answer
Final Answer: [your answer]

Let's begin:

Thought: """

# ============================================================================
# ENHANCED REASONING
# ============================================================================

class EnhancedReasoning:
    """Improved answer extraction and verification"""
    
    @staticmethod
    def extract_answer(text: str, question_type: str = "general") -> str:
        """Enhanced answer extraction with type awareness"""
        
        # For MMLU - focus on letter extraction
        if question_type == "mmlu":
            # Look for explicit answer statements
            patterns = [
                r'(?:correct answer is|answer is|select)\s*(?:option\s*)?([A-D])',
                r'(?:^|\n)(?:Therefore|Thus|So),?\s*(?:the answer is\s*)?([A-D])',
                r'\*\*([A-D])\*\*',
                r'Answer:\s*([A-D])',
                r'(?:^|\n)([A-D])\.',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()
            
            # Fallback: find any A-D near the end
            lines = text.split('\n')
            for line in reversed(lines[-5:]):  # Check last 5 lines
                match = re.search(r'\b([A-D])\b', line)
                if match:
                    return match.group(1).upper()
            
            return 'A'  # Safe fallback
        
        # For numeric answers (GSM8K, AIME)
        elif question_type in ["gsm8k", "aime", "math"]:
            patterns = [
                r'####\s*([0-9,.]+)',
                r'Final Answer:\s*([0-9,.]+)',
                r'Answer:\s*([0-9,.]+)',
                r'(?:equals?|=)\s*([0-9,.]+)',
                r'(?:Therefore|Thus|So),?\s*(?:the answer is\s*)?([0-9,.]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).replace(',', '')
            
            # Extract any number from the last line
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            if lines:
                last_line = lines[-1]
                numbers = re.findall(r'\b([0-9,.]+)\b', last_line)
                if numbers:
                    return numbers[-1].replace(',', '')
        
        # General extraction
        patterns = [
            r'Final Answer:\s*(.+?)(?:\n|$)',
            r'Answer:\s*(.+?)(?:\n|$)',
            r'Therefore,?\s+(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                answer = answer.replace('**', '').replace('*', '').replace('`', '')
                return answer
        
        # Fallback
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return lines[-1] if lines else text.strip()
    
    @staticmethod
    def extract_code(text: str) -> str:
        """Extract code blocks"""
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
        """Normalize for comparison"""
        answer = str(answer).lower().strip()
        # Remove common words
        answer = re.sub(r'\b(the|a|an|is|are)\b', '', answer)
        # Remove punctuation except decimal points
        answer = re.sub(r'[^\w\s.]', '', answer)
        # Normalize whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer
    
    @staticmethod
    def verify_numeric(answer: str, expected: str) -> bool:
        """Smart numeric comparison"""
        try:
            ans_num = float(str(answer).replace(',', '').strip())
            exp_num = float(str(expected).replace(',', '').strip())
            return abs(ans_num - exp_num) < 0.01  # Allow small floating point errors
        except:
            return False

# ============================================================================
# TOOLS
# ============================================================================

class ToolExecutor:
    @staticmethod
    def execute_python(code: str, timeout: int = 10) -> Dict[str, Any]:
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
        except:
            return {'success': False, 'output': '', 'error': 'Error'}
    
    @staticmethod
    def validate_syntax(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except:
            return False
    
    @staticmethod
    def safe_eval(expr: str) -> Any:
        try:
            return eval(expr, {"__builtins__": {}}, {})
        except:
            return None

# ============================================================================
# OPTIMIZED AGENT
# ============================================================================

class OptimizedQwenAgent:
    """Performance-optimized agent"""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.tools = ToolExecutor()
        self.prompts = ImprovedPrompts()
        self.reasoning = EnhancedReasoning()
        
        print("🔄 Loading optimized model...")
        self._load_model()
        print("✅ Ready!\n")
    
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        print(f"✓ Model: {self.config.model_name}")
        print(f"✓ Device: {next(self.model.parameters()).device}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        temp = kwargs.get('temperature', self.config.temperature)
        max_tok = kwargs.get('max_tokens', self.config.max_tokens)
        
        messages = [{"role": "user", "content": prompt}]
        
        outputs = self.pipe(
            messages,
            max_new_tokens=max_tok,
            temperature=temp,
            top_p=self.config.top_p,
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        response = outputs[0]['generated_text']
        if isinstance(response, list) and len(response) > 0:
            response = response[-1].get('content', '')
        elif isinstance(response, dict):
            response = response.get('content', str(response))
        
        return str(response).strip()
    
    def self_consistency(self, prompt: str, n: int, question_type: str = "general") -> str:
        """Improved self-consistency with type-aware voting"""
        answers = []
        
        for i in range(n):
            response = self.generate(prompt, temperature=0.7)
            answer = self.reasoning.extract_answer(response, question_type)
            if answer:
                answers.append(answer)
        
        if not answers:
            return ""
        
        # For MMLU, vote on exact matches
        if question_type == "mmlu":
            counter = Counter(answers)
            return counter.most_common(1)[0][0]
        
        # For numeric, try to normalize
        normalized = [self.reasoning.normalize_answer(a) for a in answers]
        counter = Counter(normalized)
        return counter.most_common(1)[0][0]
    
    def solve_gsm8k(self, question: str) -> Dict[str, Any]:
        """Optimized GSM8K solver"""
        print(f"🧮 GSM8K: {question[:60]}...")
        
        # Try code first (most reliable for math)
        prompt = self.prompts.gsm8k(question, use_code=True)
        response = self.generate(prompt, temperature=0.2)
        code = self.reasoning.extract_code(response)
        
        if code and self.tools.validate_syntax(code):
            result = self.tools.execute_python(code)
            if result['success'] and result['output']:
                return {'answer': result['output'], 'method': 'code', 'success': True}
        
        # Fallback to self-consistency
        prompt = self.prompts.gsm8k(question, use_code=False)
        answer = self.self_consistency(prompt, n=5, question_type="gsm8k")
        
        return {'answer': answer, 'method': 'consistency', 'success': bool(answer)}
    
    def solve_mmlu(self, question: str, choices: List[str]) -> Dict[str, Any]:
        """Optimized MMLU solver"""
        print(f"📚 MMLU: {question[:60]}...")
        
        prompt = self.prompts.mmlu(question, choices)
        
        # Use self-consistency with MMLU-specific extraction
        answer = self.self_consistency(prompt, n=5, question_type="mmlu")
        
        # Ensure it's a valid choice
        if answer not in ['A', 'B', 'C', 'D']:
            # Retry with lower temperature
            response = self.generate(prompt, temperature=0.1)
            answer = self.reasoning.extract_answer(response, "mmlu")
        
        return {'answer': answer, 'success': answer in ['A', 'B', 'C', 'D']}
    
    def solve_aime(self, question: str) -> Dict[str, Any]:
        print(f"🎯 AIME: {question[:60]}...")
        
        # Try code approach
        code_prompt = f"Solve using Python. Print only the answer:\n{question}\n```python\nimport math\n"
        response = self.generate(code_prompt, temperature=0.2)
        code = self.reasoning.extract_code(response)
        
        if code and self.tools.validate_syntax(code):
            result = self.tools.execute_python(code)
            if result['success'] and result['output']:
                return {'answer': result['output'], 'verified': True, 'success': True}
        
        # Fallback to reasoning
        prompt = self.prompts.aime(question)
        response = self.generate(prompt, temperature=0.3, max_tokens=3000)
        answer = self.reasoning.extract_answer(response, "aime")
        
        return {'answer': answer, 'verified': False, 'success': bool(answer)}
    
    def solve_swe_bench(self, issue: str) -> Dict[str, Any]:
        print(f"💻 SWE: {issue[:60]}...")
        
        prompt = self.prompts.swe_bench(issue)
        
        for iteration in range(3):
            response = self.generate(prompt, temperature=0.3, max_tokens=3000)
            code = self.reasoning.extract_code(response)
            
            if code and self.tools.validate_syntax(code):
                result = self.tools.execute_python(code)
                if result['success']:
                    return {'code': code, 'success': True}
                else:
                    prompt += f"\n\nError: {result['error']}\nFixed:\n```python\n"
        
        return {'code': code if code else '', 'success': False}
    
    def react_agent(self, task: str, max_steps: int = 8) -> Dict[str, Any]:
        print(f"🤖 ReAct: {task[:60]}...")
        
        prompt = self.prompts.react(task, ['python', 'calculator'])
        
        for step in range(max_steps):
            response = self.generate(prompt, temperature=0.4)
            
            if "Final Answer:" in response:
                return {
                    'answer': self.reasoning.extract_answer(response),
                    'steps': step + 1,
                    'success': True
                }
            
            action = re.search(r'Action:\s*(\w+)', response, re.I)
            action_input = re.search(r'Action Input:\s*(.+?)(?=\n|$)', response, re.I)
            
            if action and action_input:
                tool = action.group(1).lower()
                inp = action_input.group(1).strip()
                
                if tool in ['python', 'code']:
                    code = self.reasoning.extract_code(inp) or inp
                    result = self.tools.execute_python(code)
                    obs = result.get('output', result.get('error', 'Error'))
                elif tool in ['calculator', 'calc']:
                    obs = str(self.tools.safe_eval(inp) or 'Error')
                else:
                    obs = 'Unknown tool'
                
                prompt += f"\n{response}\nObservation: {obs}\nThought:"
            else:
                prompt += f"\n{response}\nThought:"
        
        return {'answer': 'Incomplete', 'steps': max_steps, 'success': False}

# ============================================================================
# EVALUATOR
# ============================================================================

class Evaluator:
    def __init__(self, agent):
        self.agent = agent
    
    def evaluate_gsm8k(self, sample_size=3):
        cases = [
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
        
        for i, case in enumerate(cases[:sample_size], 1):
            print(f"\n[{i}/{sample_size}]")
            result = self.agent.solve_gsm8k(case['question'])
            
            is_correct = (
                self.agent.reasoning.verify_numeric(result['answer'], case['answer']) or
                self.agent.reasoning.normalize_answer(result['answer']) == 
                self.agent.reasoning.normalize_answer(case['answer'])
            )
            correct += int(is_correct)
            
            print(f"  Got: {result['answer']}")
            print(f"  Expected: {case['answer']}")
            print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        
        accuracy = (correct / sample_size) * 100
        print(f"\n{'='*70}")
        print(f"🎯 GSM8K: {accuracy:.1f}% ({correct}/{sample_size})")
        print(f"{'='*70}\n")
        return accuracy
    
    def evaluate_mmlu(self):
        cases = [
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
            {
                'question': 'Who wrote "Romeo and Juliet"?',
                'choices': ['Charles Dickens', 'William Shakespeare', 'Jane Austen', 'Mark Twain'],
                'answer': 'B'
            },
        ]
        
        correct = 0
        print("\n" + "="*70)
        print("📚 MMLU EVALUATION")
        print("="*70)
        
        for i, case in enumerate(cases, 1):
            print(f"\n[{i}/{len(cases)}]")
            result = self.agent.solve_mmlu(case['question'], case['choices'])
            
            is_correct = result['answer'] == case['answer']
            correct += int(is_correct)
            
            print(f"  Got: {result['answer']} - {case['choices'][ord(result['answer'])-65]}")
            print(f"  Expected: {case['answer']} - {case['choices'][ord(case['answer'])-65]}")
            print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")
        
        accuracy = (correct / len(cases)) * 100
        print(f"\n{'='*70}")
        print(f"🎯 MMLU: {accuracy:.1f}% ({correct}/{len(cases)})")
        print(f"{'='*70}\n")
        return accuracy

# ============================================================================
# RUN OPTIMIZED VERSION
# ============================================================================

config = AgentConfig()
agent = OptimizedQwenAgent(config)

print("="*70)
print("🎮 TESTING OPTIMIZED AGENT")
print("="*70)

# Quick tests
print("\n1️⃣ GSM8K Test")
result = agent.solve_gsm8k("If a pizza has 8 slices and you eat 3, how many are left?")
print(f"💡 Answer: {result['answer']}\n")

print("2️⃣ MMLU Test")
result = agent.solve_mmlu(
    "What is 2 + 2?",
    ["3", "4", "5", "6"]
)
print(f"💡 Answer: {result['answer']}\n")

# Run benchmarks
evaluator = Evaluator(agent)
gsm8k_score = evaluator.evaluate_gsm8k(sample_size=3)
mmlu_score = evaluator.evaluate_mmlu()

print("\n" + "="*70)
print("🏆 OPTIMIZED RESULTS")
print("="*70)
print(f"GSM8K: {gsm8k_score:.1f}%")
print(f"MMLU: {mmlu_score:.1f}%")
print("="*70)

print("\n✅ OPTIMIZATION COMPLETE!")
print("\n📝 Use: result = agent.solve_gsm8k('your question')")
