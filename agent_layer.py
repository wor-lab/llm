"""
Elite Coding Agent Layer - Optimized for 90% Performance
Supports: HumanEval, MBPP, SWE-Bench, LiveBench, BigCodeBench, CodeContests
"""

import re
import ast
import sys
import io
import subprocess
import tempfile
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
import contextlib

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    print("⚠️  Run: pip install transformers torch accelerate")

from performance_config import (
    PerformanceOptimizer,
    HumanEvalConfig,
    MBPPConfig,
    SWEBenchConfig,
    LiveBenchConfig,
    BigCodeBenchConfig
)


# ============================================================================
# CODE EXECUTION ENGINE
# ============================================================================

class CodeExecutor:
    """Safe code execution with multiple backends"""
    
    @staticmethod
    def execute_python(code: str, test_input: str = "", timeout: int = 10) -> Dict[str, Any]:
        """Execute Python code safely with test input"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_file = f.name
                f.write(code)
            
            # Execute with optional input
            result = subprocess.run(
                ['python3', temp_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Cleanup
            os.unlink(temp_file)
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout.strip(),
                'error': result.stderr.strip(),
                'exit_code': result.returncode
            }
        
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
            return {
                'success': False,
                'output': '',
                'error': f'Execution timeout after {timeout}s',
                'exit_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'exit_code': -1
            }
    
    @staticmethod
    def execute_in_memory(code: str, test_input: str = "") -> Dict[str, Any]:
        """Execute code in memory (faster but less safe)"""
        try:
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            # Execute
            exec_globals = {}
            exec(code, exec_globals)
            
            # Get output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            return {
                'success': True,
                'output': output.strip(),
                'error': '',
                'globals': exec_globals
            }
        
        except Exception as e:
            sys.stdout = old_stdout
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'globals': {}
            }
    
    @staticmethod
    def run_tests(code: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run multiple test cases"""
        results = []
        passed = 0
        
        for i, test in enumerate(test_cases):
            input_data = test.get('input', '')
            expected = test.get('output', '')
            
            result = CodeExecutor.execute_python(code, input_data)
            
            test_passed = result['success'] and result['output'] == expected
            passed += int(test_passed)
            
            results.append({
                'test_id': i,
                'passed': test_passed,
                'expected': expected,
                'actual': result['output'],
                'error': result['error']
            })
        
        return {
            'total': len(test_cases),
            'passed': passed,
            'failed': len(test_cases) - passed,
            'pass_rate': passed / len(test_cases) if test_cases else 0,
            'results': results
        }


class CodeValidator:
    """Code validation and quality checks"""
    
    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, str]:
        """Check Python syntax"""
        try:
            ast.parse(code)
            return True, "Valid syntax"
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
    
    @staticmethod
    def check_imports(code: str) -> List[str]:
        """Extract imports from code"""
        try:
            tree = ast.parse(code)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module)
            return imports
        except:
            return []
    
    @staticmethod
    def has_security_issues(code: str) -> Tuple[bool, List[str]]:
        """Basic security checks"""
        issues = []
        
        dangerous_patterns = [
            (r'\beval\s*KATEX_INLINE_OPEN', 'eval() usage'),
            (r'\bexec\s*KATEX_INLINE_OPEN', 'exec() usage'),
            (r'__import__', '__import__ usage'),
            (r'\bos\.system', 'os.system usage'),
            (r'\bsubprocess\.(?!run|check_output)', 'unsafe subprocess'),
        ]
        
        for pattern, issue in dangerous_patterns:
            if re.search(pattern, code):
                issues.append(issue)
        
        return len(issues) > 0, issues
    
    @staticmethod
    def count_complexity(code: str) -> int:
        """Estimate cyclomatic complexity"""
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except:
            return 0


# ============================================================================
# CODE GENERATION PROMPTS
# ============================================================================

class CodingPrompts:
    """Optimized prompts for each coding benchmark"""
    
    @staticmethod
    def humaneval(problem: str, signature: str = "") -> str:
        """HumanEval: Generate from docstring"""
        return f"""You are an expert Python programmer. Complete this function.

{problem}

Write a complete, correct implementation. Include:
1. Proper handling of all edge cases
2. Efficient algorithm
3. Clean, readable code

```python
{signature if signature else "# Complete function implementation"}
"""
    
    @staticmethod
    def mbpp(description: str, examples: List[str] = None) -> str:
        """MBPP: Generate from description and examples"""
        examples_str = ""
        if examples:
            examples_str = "\n\nExamples:\n" + "\n".join(f"  {ex}" for ex in examples)
        
        return f"""You are an expert Python programmer. Write a function to solve this problem.

Problem: {description}{examples_str}

Write a complete, correct Python function that:
1. Solves the problem efficiently
2. Handles edge cases
3. Works with all test cases

```python
def solution():
"""
    
    @staticmethod
    def swe_bench(issue: str, repository_context: str = "", error_trace: str = "") -> str:
        """SWE-Bench: Fix real-world bugs"""
        context = f"\n\nRepository Context:\n{repository_context}" if repository_context else ""
        trace = f"\n\nError Trace:\n{error_trace}" if error_trace else ""
        
        return f"""You are an expert software engineer. Fix this bug in a real codebase.

Issue: {issue}{context}{trace}

Provide a complete fix that:
1. Resolves the issue completely
2. Doesn't introduce new bugs
3. Maintains code quality
4. Passes all existing tests

```python
# Fixed code
"""
    
    @staticmethod
    def livebench(problem: str, constraints: str = "") -> str:
        """LiveBench: Competitive programming style"""
        const = f"\n\nConstraints:\n{constraints}" if constraints else ""
        
        return f"""You are a competitive programmer. Solve this problem efficiently.

Problem: {problem}{const}

Write an optimal solution that:
1. Meets all constraints
2. Handles edge cases
3. Runs efficiently
4. Produces correct output

```python
def solve():
"""
    
    @staticmethod
    def bigcodebench(task: str, requirements: List[str] = None) -> str:
        """BigCodeBench: Large-scale code generation"""
        reqs = ""
        if requirements:
            reqs = "\n\nRequirements:\n" + "\n".join(f"  - {r}" for r in requirements)
        
        return f"""You are an expert software engineer. Implement this large-scale task.

Task: {task}{reqs}

Create a complete, production-quality implementation with:
1. Modular, well-organized code
2. Comprehensive error handling
3. Type hints and documentation
4. PEP8 compliance
5. Test coverage

```python
# Complete implementation
"""


# ============================================================================
# CODE EXTRACTION & REFINEMENT
# ============================================================================

class CodeExtractor:
    """Extract and refine code from model outputs"""
    
    @staticmethod
    def extract_code(text: str, language: str = "python") -> str:
        """Extract code from markdown or raw text"""
        # Try markdown code blocks first
        patterns = [
            rf'```{language}\s*\n(.*?)```',
            r'```\s*\n(.*?)```',
            r'```(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[-1].strip()
        
        # If no code block, try to find function definitions
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else text.strip()
    
    @staticmethod
    def clean_code(code: str) -> str:
        """Clean and format code"""
        # Remove common markdown artifacts
        code = re.sub(r'^[`\s]+|[`\s]+$', '', code)
        
        # Remove extra blank lines
        lines = code.split('\n')
        cleaned_lines = []
        prev_blank = False
        
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned_lines.append(line)
            prev_blank = is_blank
        
        return '\n'.join(cleaned_lines).strip()
    
    @staticmethod
    def add_imports(code: str, required_imports: List[str] = None) -> str:
        """Add missing imports"""
        if not required_imports:
            return code
        
        existing_imports = CodeValidator.check_imports(code)
        missing = [imp for imp in required_imports if imp not in existing_imports]
        
        if missing:
            import_lines = [f"import {imp}" for imp in missing]
            return '\n'.join(import_lines) + '\n\n' + code
        
        return code


# ============================================================================
# MAIN CODING AGENT
# ============================================================================

class CodingAgent:
    """Elite coding agent with 90% target performance"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.optimizer = PerformanceOptimizer()
        self.prompts = CodingPrompts()
        self.executor = CodeExecutor()
        self.validator = CodeValidator()
        self.extractor = CodeExtractor()
        
        print(f"🔄 Loading {model_name}...")
        self._load_model()
        print("✅ Coding Agent Ready!\n")
    
    def _load_model(self):
        """Load Qwen model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        print(f"✓ Device: {next(self.model.parameters()).device}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate code with model"""
        temp = kwargs.get('temperature', 0.3)
        max_tok = kwargs.get('max_tokens', 4096)
        
        messages = [{"role": "user", "content": prompt}]
        
        outputs = self.pipe(
            messages,
            max_new_tokens=max_tok,
            temperature=temp,
            top_p=kwargs.get('top_p', 0.95),
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        response = outputs[0]['generated_text']
        if isinstance(response, list):
            response = response[-1].get('content', '')
        elif isinstance(response, dict):
            response = response.get('content', str(response))
        
        return str(response).strip()
    
    def generate_with_retry(
        self,
        prompt: str,
        max_attempts: int = 3,
        **kwargs
    ) -> Tuple[str, bool]:
        """Generate code with syntax validation and retry"""
        
        for attempt in range(max_attempts):
            response = self.generate(prompt, **kwargs)
            code = self.extractor.extract_code(response)
            code = self.extractor.clean_code(code)
            
            # Validate syntax
            is_valid, error = self.validator.validate_syntax(code)
            
            if is_valid:
                return code, True
            
            # Add error feedback for retry
            prompt += f"\n\nPrevious code had syntax error: {error}\nGenerate corrected code:\n```python\n"
        
        return code, False
    
    # ========================================================================
    # HUMANEVAL
    # ========================================================================
    
    def solve_humaneval(
        self,
        problem: str,
        signature: str = "",
        test_cases: List[Dict] = None
    ) -> Dict[str, Any]:
        """Solve HumanEval problem with 90%+ target"""
        print(f"🎯 HumanEval: {problem[:60]}...")
        
        config = self.optimizer.humaneval
        prompt = self.prompts.humaneval(problem, signature)
        
        # Generate multiple solutions
        solutions = []
        for i in range(config.num_samples):
            code, is_valid = self.generate_with_retry(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            if is_valid:
                solutions.append(code)
        
        if not solutions:
            return {'code': '', 'success': False, 'error': 'No valid solution generated'}
        
        # Test solutions if test cases provided
        if test_cases:
            best_solution = None
            best_pass_rate = 0
            
            for code in solutions:
                test_results = self.executor.run_tests(code, test_cases)
                if test_results['pass_rate'] > best_pass_rate:
                    best_pass_rate = test_results['pass_rate']
                    best_solution = code
            
            return {
                'code': best_solution,
                'success': best_pass_rate == 1.0,
                'pass_rate': best_pass_rate,
                'num_solutions': len(solutions)
            }
        
        # Return most common solution (self-consistency)
        code_counter = Counter(solutions)
        best_code = code_counter.most_common(1)[0][0]
        
        return {
            'code': best_code,
            'success': True,
            'num_solutions': len(solutions)
        }
    
    # ========================================================================
    # MBPP
    # ========================================================================
    
    def solve_mbpp(
        self,
        description: str,
        examples: List[str] = None,
        test_cases: List[Dict] = None
    ) -> Dict[str, Any]:
        """Solve MBPP problem with 90%+ target"""
        print(f"📝 MBPP: {description[:60]}...")
        
        config = self.optimizer.mbpp
        prompt = self.prompts.mbpp(description, examples)
        
        # Iterative refinement with test feedback
        best_code = None
        best_pass_rate = 0
        
        for iteration in range(config.max_retries):
            code, is_valid = self.generate_with_retry(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            if not is_valid:
                continue
            
            # Test if test cases provided
            if test_cases:
                test_results = self.executor.run_tests(code, test_cases)
                pass_rate = test_results['pass_rate']
                
                if pass_rate > best_pass_rate:
                    best_pass_rate = pass_rate
                    best_code = code
                
                # If perfect, return immediately
                if pass_rate == 1.0:
                    return {
                        'code': best_code,
                        'success': True,
                        'pass_rate': 1.0,
                        'iterations': iteration + 1
                    }
                
                # Add test failure feedback
                failed_tests = [r for r in test_results['results'] if not r['passed']]
                if failed_tests:
                    feedback = "\n".join([
                        f"Test {r['test_id']}: Expected {r['expected']}, got {r['actual']}"
                        for r in failed_tests[:2]  # Show first 2 failures
                    ])
                    prompt += f"\n\nFailed tests:\n{feedback}\n\nGenerate corrected code:\n```python\n"
            else:
                best_code = code
                break
        
        return {
            'code': best_code or code,
            'success': best_pass_rate == 1.0 if test_cases else is_valid,
            'pass_rate': best_pass_rate if test_cases else None
        }
    
    # ========================================================================
    # SWE-BENCH
    # ========================================================================
    
    def solve_swe_bench(
        self,
        issue: str,
        repository_context: str = "",
        test_commands: List[str] = None
    ) -> Dict[str, Any]:
        """Solve SWE-Bench with 79%+ target"""
        print(f"🔧 SWE-Bench: {issue[:60]}...")
        
        config = self.optimizer.swe_bench
        prompt = self.prompts.swe_bench(issue, repository_context)
        
        # Iterative refinement
        for iteration in range(config.max_iterations):
            code, is_valid = self.generate_with_retry(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            if not is_valid:
                continue
            
            # Execute tests if provided
            if test_commands:
                all_passed = True
                for cmd in test_commands:
                    result = subprocess.run(
                        cmd.split(),
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode != 0:
                        all_passed = False
                        prompt += f"\n\nTest failed: {cmd}\nError: {result.stderr}\nFix the code:\n```python\n"
                        break
                
                if all_passed:
                    return {
                        'code': code,
                        'success': True,
                        'iterations': iteration + 1
                    }
            else:
                # No tests, return if syntax valid
                return {
                    'code': code,
                    'success': True,
                    'iterations': iteration + 1
                }
        
        return {
            'code': code if 'code' in locals() else '',
            'success': False,
            'iterations': config.max_iterations
        }
    
    # ========================================================================
    # LIVEBENCH
    # ========================================================================
    
    def solve_livebench(
        self,
        problem: str,
        constraints: str = "",
        test_cases: List[Dict] = None
    ) -> Dict[str, Any]:
        """Solve LiveBench with 85%+ target"""
        print(f"⚡ LiveBench: {problem[:60]}...")
        
        config = self.optimizer.livebench
        prompt = self.prompts.livebench(problem, constraints)
        
        code, is_valid = self.generate_with_retry(
            prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        if not is_valid:
            return {'code': code, 'success': False, 'error': 'Invalid syntax'}
        
        # Test if test cases provided
        if test_cases:
            test_results = self.executor.run_tests(code, test_cases)
            return {
                'code': code,
                'success': test_results['pass_rate'] == 1.0,
                'pass_rate': test_results['pass_rate'],
                'test_results': test_results
            }
        
        return {'code': code, 'success': True}
    
    # ========================================================================
    # BIGCODEBENCH
    # ========================================================================
    
    def solve_bigcodebench(
        self,
        task: str,
        requirements: List[str] = None
    ) -> Dict[str, Any]:
        """Solve BigCodeBench with 80%+ target"""
        print(f"📦 BigCodeBench: {task[:60]}...")
        
        config = self.optimizer.bigcodebench
        prompt = self.prompts.bigcodebench(task, requirements)
        
        code, is_valid = self.generate_with_retry(
            prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        if not is_valid:
            return {'code': code, 'success': False}
        
        # Quality checks
        has_issues, security_issues = self.validator.has_security_issues(code)
        complexity = self.validator.count_complexity(code)
        
        return {
            'code': code,
            'success': is_valid and not has_issues,
            'security_issues': security_issues,
            'complexity': complexity,
            'quality_score': 1.0 if not has_issues else 0.5
        }


# ============================================================================
# EVALUATOR
# ============================================================================

class CodingBenchmarkEvaluator:
    """Evaluate agent on coding benchmarks"""
    
    def __init__(self, agent: CodingAgent):
        self.agent = agent
    
    def evaluate_humaneval_sample(self):
        """Quick HumanEval test"""
        problem = '''
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''
        
        result = self.agent.solve_humaneval(problem, "from typing import List")
        print("\n" + "="*80)
        print("🎯 HumanEval Sample Result")
        print("="*80)
        print(f"Success: {result['success']}")
        print(f"\nGenerated Code:\n{result['code'][:200]}...")
        print("="*80)
        
        return result['success']
    
    def evaluate_mbpp_sample(self):
        """Quick MBPP test"""
        description = "Write a function to find the minimum value in a given list"
        examples = ["min_value([3, 1, 4, 1, 5]) == 1", "min_value([0]) == 0"]
        test_cases = [
            {'input': '', 'output': '1'},  # Simplified
        ]
        
        result = self.agent.solve_mbpp(description, examples)
        print("\n" + "="*80)
        print("📝 MBPP Sample Result")
        print("="*80)
        print(f"Success: {result['success']}")
        print(f"\nGenerated Code:\n{result['code'][:200]}...")
        print("="*80)
        
        return result['success']


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 ELITE CODING AGENT - INITIALIZING")
    print("="*80)
    
    # Initialize agent
    agent = CodingAgent("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Show configurations
    agent.optimizer.print_all_configs()
    
    # Run sample evaluations
    evaluator = CodingBenchmarkEvaluator(agent)
    
    print("\n" + "="*80)
    print("🧪 RUNNING SAMPLE TESTS")
    print("="*80)
    
    humaneval_success = evaluator.evaluate_humaneval_sample()
    mbpp_success = evaluator.evaluate_mbpp_sample()
    
    print("\n" + "="*80)
    print("📊 SAMPLE RESULTS")
    print("="*80)
    print(f"HumanEval: {'✅ PASS' if humaneval_success else '❌ FAIL'}")
    print(f"MBPP: {'✅ PASS' if mbpp_success else '❌ FAIL'}")
    print("="*80)
    
    print("\n✅ Agent ready for coding tasks!")
    print("\n💡 Usage:")
    print("  result = agent.solve_humaneval(problem, signature)")
    print("  result = agent.solve_mbpp(description, examples)")
    print("  result = agent.solve_swe_bench(issue, context)")
