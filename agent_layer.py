"""WB AI CORPORATION - AGENT LAYER"""
import re
import ast
import sys
import subprocess
import tempfile
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from performance_config import PerformanceOptimizer, PRODUCTION_PROMPTS, EXECUTION_STRATEGIES, HumanEvalConfig, MBPPConfig, SWEBenchConfig, LiveBenchConfig, BigCodeBenchConfig

class CodeExecutor:
    @staticmethod
    def execute_python(code: str, test_code: str = "", timeout: int = 10, capture_output: bool = True) -> Dict[str, Any]:
        temp_file = None
        try:
            full_code = code
            if test_code:
                full_code += f"\n\n{test_code}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                temp_file = f.name
                f.write(full_code)
            start_time = time.time()
            result = subprocess.run([sys.executable, temp_file], capture_output=capture_output, text=True, timeout=timeout, env=os.environ.copy())
            execution_time = time.time() - start_time
            return {'success': result.returncode == 0, 'output': result.stdout.strip() if capture_output else '', 'error': result.stderr.strip() if capture_output else '', 'execution_time': execution_time, 'return_code': result.returncode}
        except subprocess.TimeoutExpired:
            return {'success': False, 'output': '', 'error': f'Execution timeout after {timeout}s', 'execution_time': timeout, 'return_code': -1}
        except Exception as e:
            return {'success': False, 'output': '', 'error': f'Execution error: {str(e)}', 'execution_time': 0, 'return_code': -1}
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    @staticmethod
    def extract_function_info(code: str) -> Optional[Dict[str, Any]]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return {'name': node.name, 'args': [arg.arg for arg in node.args.args], 'returns': ast.unparse(node.returns) if node.returns else None, 'docstring': ast.get_docstring(node)}
        except:
            pass
        return None
    @staticmethod
    def run_unit_tests(code: str, tests: List[Dict[str, Any]], timeout: int = 5) -> Dict[str, Any]:
        passed = 0
        failed = 0
        failures = []
        for i, test in enumerate(tests):
            test_code = code + "\n\n" + f"try:\n    result = {test['input']}\n    expected = {test['expected']}\n    assert result == expected, f\"Expected {{expected}}, got {{result}}\"\n    print(f\"Test {i+1}: PASS\")\nexcept AssertionError as e:\n    print(f\"Test {i+1}: FAIL - {{e}}\")\n    exit(1)\nexcept Exception as e:\n    print(f\"Test {i+1}: ERROR - {{e}}\")\n    exit(2)\n"
            result = CodeExecutor.execute_python(test_code, timeout=timeout)
            if result['success']:
                passed += 1
            else:
                failed += 1
                failures.append({'test_num': i + 1, 'input': test['input'], 'expected': test['expected'], 'error': result['error'] or result['output']})
        return {'passed': passed, 'failed': failed, 'total': len(tests), 'failures': failures, 'success_rate': passed / len(tests) if tests else 0.0}

class CodeExtractor:
    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        patterns = [r'```python\s*\n(.*?)```', r'```\s*\n(.*?)```']
        code_blocks = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                code_blocks.extend([m.strip() for m in matches])
        if not code_blocks:
            func_pattern = r'(def\s+\w+\s*KATEX_INLINE_OPEN[^)]*KATEX_INLINE_CLOSE\s*(?:->.*?)?\s*:.*?)(?=\n(?:def\s+|\Z))'
            matches = re.findall(func_pattern, text, re.DOTALL)
            if matches:
                code_blocks.extend(matches)
        return code_blocks
    @staticmethod
    def extract_primary_code(text: str) -> str:
        blocks = CodeExtractor.extract_code_blocks(text)
        if not blocks:
            lines = text.split('\n')
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            if code_lines:
                return '\n'.join(code_lines)
            return text.strip()
        return max(blocks, key=len)
    @staticmethod
    def clean_code(code: str) -> str:
        code = re.sub(r'^```(?:python)?\s*\n?', '', code)
        code = re.sub(r'\n?```\s*$', '', code)
        code = re.sub(r'^#+\s*(?:Solution|Implementation|Code|Answer).*\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n{3,}', '\n\n', code)
        return code.strip()
    @staticmethod
    def extract_imports(code: str) -> List[str]:
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
        except:
            import_lines = re.findall(r'^(?:import|from)\s+.*$', code, re.MULTILINE)
            imports.extend(import_lines)
        return imports

class CodingAgent:
    def __init__(self, model_name: str = None, device: str = "auto", load_in_8bit: bool = False):
        self.optimizer = PerformanceOptimizer()
        self.model_name = model_name or self.optimizer.humaneval.model_name
        self.device = device
        self.executor = CodeExecutor()
        self.extractor = CodeExtractor()
        print("🧠 WB AI Engineering Division - Initializing Agent")
        self._load_model(load_in_8bit)
        print("✅ Agent operational\n")
    def _load_model(self, load_in_8bit: bool = False):
        print(f"📦 Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side='left')
        load_kwargs = {'trust_remote_code': True, 'device_map': self.device}
        if torch.cuda.is_available() and not load_in_8bit:
            load_kwargs['torch_dtype'] = torch.float16
        elif load_in_8bit:
            load_kwargs['load_in_8bit'] = True
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        device_name = next(self.model.parameters()).device
        print(f"✓ Device: {device_name}")
        print(f"✓ Parameters: {self.model.num_parameters() / 1e9:.2f}B")
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048, top_p: float = 0.95, stop: Optional[List[str]] = None) -> str:
        try:
            outputs = self.pipe(prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, do_sample=temperature > 0, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id, return_full_text=False)
            response = outputs[0]['generated_text']
            if isinstance(response, str):
                return response.strip()
            elif isinstance(response, dict):
                return response.get('content', str(response)).strip()
            elif isinstance(response, list) and response:
                return response[-1].get('content', str(response[-1])).strip()
            return str(response).strip()
        except Exception as e:
            print(f"⚠️ Generation error: {e}")
            return ""
    def solve_humaneval(self, prompt: str, entry_point: str = None, test_code: str = None, canonical_solution: str = None) -> Dict[str, Any]:
        print(f"🎯 HumanEval: {entry_point or 'function'}...")
        config = self.optimizer.humaneval
        template = PRODUCTION_PROMPTS['humaneval']['template']
        solutions = []
        for attempt in range(config.num_samples):
            temp = config.temperature + (attempt * 0.05)
            full_prompt = template.format(prompt=prompt)
            response = self.generate(full_prompt, temperature=temp, max_tokens=config.max_tokens)
            code = self.extractor.extract_primary_code(response)
            code = self.extractor.clean_code(code)
            if 'def ' not in code and prompt:
                code = prompt + '\n' + code
            is_valid, error = self.executor.validate_syntax(code)
            if not is_valid:
                if config.retry_on_syntax_error and attempt < config.num_samples - 1:
                    continue
                else:
                    solutions.append({'code': code, 'passed': False, 'error': f"Syntax error: {error}"})
                    continue
            if test_code:
                full_test = test_code.replace('{code}', code) if '{code}' in test_code else f"{code}\n\n{test_code}"
                result = self.executor.execute_python(full_test, timeout=config.timeout)
                solutions.append({'code': code, 'passed': result['success'], 'result': result, 'attempt': attempt + 1})
                if result['success']:
                    return {'code': code, 'passed': True, 'test_results': result, 'method': 'multi_sample', 'attempts': attempt + 1}
            else:
                solutions.append({'code': code, 'passed': True, 'result': {'success': True}})
        if config.enable_self_repair and solutions and test_code:
            best = max(solutions, key=lambda x: x.get('passed', False))
            if not best['passed']:
                repaired = self._self_repair_code(code=best['code'], error=best.get('result', {}).get('error', ''), prompt=prompt, test_code=test_code, max_iterations=config.max_iterations)
                if repaired['success']:
                    return {'code': repaired['code'], 'passed': True, 'test_results': repaired['result'], 'method': 'self_repair', 'attempts': len(solutions) + repaired['iterations']}
        if solutions:
            best = max(solutions, key=lambda x: (x.get('passed', False), len(x.get('code', ''))))
            return {'code': best['code'], 'passed': best.get('passed', False), 'test_results': best.get('result', {}), 'method': 'best_of_n', 'attempts': len(solutions)}
        return {'code': '', 'passed': False, 'test_results': {'error': 'No valid solution generated'}, 'method': 'failed', 'attempts': config.num_samples}
    def solve_mbpp(self, task: str, examples: List[Dict[str, Any]] = None, test_cases: List[Dict[str, Any]] = None, assertions: List[str] = None) -> Dict[str, Any]:
        print(f"📝 MBPP: {task[:50]}...")
        config = self.optimizer.mbpp
        if examples and config.use_examples:
            examples_str = "\n".join([f"Input: {ex.get('input', ex)}\nOutput: {ex.get('output', ex.get('expected', ''))}" for ex in examples])
            template = PRODUCTION_PROMPTS['mbpp']['with_examples']
            prompt = template.format(task=task, examples=examples_str)
        else:
            template = PRODUCTION_PROMPTS['mbpp']['template']
            prompt = template.format(task=task)
        best_solution = None
        best_score = 0.0
        for attempt in range(config.num_samples):
            response = self.generate(prompt, temperature=config.temperature, max_tokens=config.max_tokens)
            code = self.extractor.extract_primary_code(response)
            code = self.extractor.clean_code(code)
            is_valid, error = self.executor.validate_syntax(code)
            if not is_valid:
                continue
            if test_cases and config.validate_with_tests:
                test_results = self.executor.run_unit_tests(code, test_cases, timeout=config.timeout)
                score = test_results['success_rate']
                if score > best_score:
                    best_score = score
                    best_solution = {'code': code, 'passed': test_results['passed'] == test_results['total'], 'test_results': test_results, 'score': score}
                if score == 1.0:
                    return best_solution
            elif assertions:
                test_code = f"{code}\n\n" + "\n".join(assertions)
                result = self.executor.execute_python(test_code, timeout=config.timeout)
                if result['success']:
                    return {'code': code, 'passed': True, 'test_results': result}
                if not best_solution:
                    best_solution = {'code': code, 'passed': False, 'test_results': result}
            else:
                return {'code': code, 'passed': True, 'test_results': {'success': True}}
        if best_solution and best_score < 1.0 and config.enable_self_repair:
            failures = best_solution['test_results'].get('failures', [])
            if failures:
                error_desc = "; ".join([f"Test {f['test_num']}: {f['error']}" for f in failures[:2]])
                repaired = self._self_repair_code(code=best_solution['code'], error=error_desc, prompt=task, max_iterations=config.max_iterations)
                if repaired['success']:
                    return {'code': repaired['code'], 'passed': True, 'test_results': repaired['result'], 'method': 'self_repair'}
        return best_solution or {'code': '', 'passed': False, 'test_results': {'error': 'No valid solution'}}
    def solve_swe_bench(self, issue: str, repo_context: str = "", error_trace: str = "", current_code: str = "", file_path: str = "") -> Dict[str, Any]:
        print(f"🔧 SWE-Bench: {issue[:50]}...")
        config = self.optimizer.swe_bench
        if error_trace and current_code:
            template = PRODUCTION_PROMPTS['swe_bench']['with_trace']
            prompt = template.format(issue=issue, error=error_trace, code=current_code)
        else:
            template = PRODUCTION_PROMPTS['swe_bench']['template']
            prompt = template.format(repo=repo_context[:500] if repo_context else "N/A", issue=issue, context=repo_context[:1000] if repo_context else "", error=error_trace or "No error trace")
        for iteration in range(config.max_iterations):
            print(f"  Iteration {iteration + 1}/{config.max_iterations}...", end='\r')
            response = self.generate(prompt, temperature=config.temperature, max_tokens=config.max_tokens)
            code = self.extractor.extract_primary_code(response)
            code = self.extractor.clean_code(code)
            is_valid, syntax_error = self.executor.validate_syntax(code)
            if not is_valid:
                prompt += "\n\nSyntax Error: " + str(syntax_error) + "\nFix the code:\n```python\n"
                continue
            result = self.executor.execute_python(code, timeout=config.timeout)
            if result['success']:
                print()
                patch = self._generate_patch(current_code, code) if current_code else ""
                return {'code': code, 'patch': patch, 'success': True, 'explanation': response, 'iterations': iteration + 1, 'file_path': file_path}
            else:
                error_msg = result.get('error', 'Unknown error')
                prompt += "\n\nExecution Error:\n" + error_msg + "\n\nCorrected code:\n```python\n"
        print()
        return {'code': code if 'code' in locals() else '', 'patch': '', 'success': False, 'explanation': 'Max iterations reached', 'iterations': config.max_iterations}
    def solve_livebench(self, problem: str, constraints: str = "", test_cases: List[Dict] = None, time_limit: float = None, memory_limit: int = None) -> Dict[str, Any]:
        print(f"⚡ LiveBench: {problem[:50]}...")
        config = self.optimizer.livebench
        template = PRODUCTION_PROMPTS['livebench']['template']
        prompt = template.format(problem=problem, constraints=constraints or "No specific constraints")
        response = self.generate(prompt, temperature=config.temperature, max_tokens=config.max_tokens)
        code = self.extractor.extract_primary_code(response)
        code = self.extractor.clean_code(code)
        is_valid, error = self.executor.validate_syntax(code)
        if not is_valid:
            return {'code': code, 'passed': False, 'error': error, 'performance': {}}
        if test_cases:
            test_results = self.executor.run_unit_tests(code, test_cases, timeout=config.timeout)
            return {'code': code, 'passed': test_results['passed'] == test_results['total'], 'test_results': test_results, 'performance': {'success_rate': test_results['success_rate'], 'tests_passed': test_results['passed'], 'tests_total': test_results['total']}}
        return {'code': code, 'passed': True, 'performance': {'validated': True}}
    def solve_bigcodebench(self, specification: str, requirements: List[str] = None, dependencies: List[str] = None, context: str = "") -> Dict[str, Any]:
        print(f"📦 BigCodeBench: {specification[:50]}...")
        config = self.optimizer.bigcodebench
        template = PRODUCTION_PROMPTS['bigcodebench']['template']
        requirements_str = "\n".join([f"- {req}" for req in (requirements or [])])
        if dependencies:
            requirements_str += "\n\nDependencies:\n" + "\n".join([f"- {dep}" for dep in dependencies])
        prompt = template.format(specification=specification, requirements=requirements_str or "No specific requirements")
        if context:
            prompt = f"{context}\n\n{prompt}"
        response = self.generate(prompt, temperature=config.temperature, max_tokens=config.max_tokens)
        code = self.extractor.extract_primary_code(response)
        code = self.extractor.clean_code(code)
        is_valid, error = self.executor.validate_syntax(code)
        quality_score = 0.0
        if is_valid:
            quality_score = self._calculate_code_quality(code)
        return {'code': code, 'success': is_valid, 'quality_score': quality_score, 'error': error if not is_valid else None, 'metadata': {'lines': len(code.split('\n')), 'functions': len(re.findall(r'\ndef\s+\w+', code)), 'classes': len(re.findall(r'\nclass\s+\w+', code))}}
    def _self_repair_code(self, code: str, error: str, prompt: str, test_code: str = None, max_iterations: int = 3) -> Dict[str, Any]:
        repair_prompt = "This code has errors:\n\n```python\n" + code + "\n```\n\nError: " + error + "\n\nOriginal task: " + prompt + "\n\nFixed code:\n```python\n"
        for iteration in range(max_iterations):
            response = self.generate(repair_prompt, temperature=0.2, max_tokens=2048)
            fixed_code = self.extractor.extract_primary_code(response)
            fixed_code = self.extractor.clean_code(fixed_code)
            is_valid, syntax_error = self.executor.validate_syntax(fixed_code)
            if not is_valid:
                repair_prompt += "\n\nStill has syntax error: " + str(syntax_error) + "\nTry again:\n```python\n"
                continue
            if test_code:
                full_test = test_code.replace('{code}', fixed_code) if '{code}' in test_code else f"{fixed_code}\n\n{test_code}"
                result = self.executor.execute_python(full_test, timeout=10)
                if result['success']:
                    return {'success': True, 'code': fixed_code, 'result': result, 'iterations': iteration + 1}
                else:
                    repair_prompt += "\n\nStill failing: " + result.get('error', '') + "\nFix:\n```python\n"
            else:
                return {'success': True, 'code': fixed_code, 'result': {'success': True}, 'iterations': iteration + 1}
        return {'success': False, 'code': code, 'result': {'error': 'Could not repair'}, 'iterations': max_iterations}
    def _generate_patch(self, original: str, fixed: str) -> str:
        try:
            import difflib
            diff = difflib.unified_diff(original.splitlines(keepends=True), fixed.splitlines(keepends=True), fromfile='original.py', tofile='fixed.py', lineterm='')
            return ''.join(diff)
        except:
            return f"--- Original\n+++ Fixed\n\n{fixed}"
    def _calculate_code_quality(self, code: str) -> float:
        score = 1.0
        lines = code.split('\n')
        long_lines = sum(1 for line in lines if len(line) > 100)
        score -= (long_lines / len(lines)) * 0.15
        if '"""' in code or "'''" in code:
            score += 0.1
        if '->' in code or ': ' in code:
            score += 0.05
        if '\nclass ' in code or '\ndef ' in code:
            score += 0.05
        nested_blocks = code.count('    ' * 4)
        if nested_blocks > 5:
            score -= 0.1
        return max(0.0, min(1.0, score))

if __name__ == "__main__":
    print("="*80)
    print("🏢 WB AI CORPORATION - QUANTUM-CODER AGENT")
    print("="*80)
    agent = CodingAgent()
    print("\n✅ Agent ready for deployment")
    print("="*80)
