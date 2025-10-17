"""
Advanced Coding Techniques for 90%+ Performance
Includes: Test-Driven Development, Self-Debugging, Multi-Solution Voting, Code Evolution
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import ast

from agent_layer import CodingAgent, CodeExecutor, CodeValidator, CodeExtractor


class TestDrivenDevelopment:
    """TDD approach for code generation"""
    
    def __init__(self, agent: CodingAgent):
        self.agent = agent
        self.executor = CodeExecutor()
    
    def generate_from_tests(
        self,
        function_name: str,
        test_cases: List[Dict[str, Any]],
        description: str = ""
    ) -> Dict[str, Any]:
        """Generate code that passes given tests"""
        
        # Create test suite description
        test_descriptions = []
        for i, test in enumerate(test_cases):
            inp = test.get('input', '')
            out = test.get('output', '')
            test_descriptions.append(f"Test {i+1}: {function_name}({inp}) should return {out}")
        
        tests_str = "\n".join(test_descriptions)
        
        prompt = f"""Write a Python function that passes these tests:

Function: {function_name}
{f"Description: {description}" if description else ""}

Tests:
{tests_str}

Generate a complete, correct implementation:

```python
def {function_name}():
"""
        
        # Iterative generation until all tests pass
        max_iterations = 5
        for iteration in range(max_iterations):
            response = self.agent.generate(prompt, temperature=0.2)
            code = self.agent.extractor.extract_code(response)
            
            # Run tests
            test_results = self.executor.run_tests(code, test_cases)
            
            if test_results['pass_rate'] == 1.0:
                return {
                    'code': code,
                    'success': True,
                    'iterations': iteration + 1,
                    'test_results': test_results
                }
            
            # Add feedback
            failed = [r for r in test_results['results'] if not r['passed']]
            feedback = f"\n\nFailed {len(failed)} tests. Examples:\n"
            for r in failed[:3]:
                feedback += f"  Expected {r['expected']}, got {r['actual']}\n"
            
            prompt += feedback + "\nGenerate corrected code:\n```python\n"
        
        return {
            'code': code,
            'success': False,
            'iterations': max_iterations,
            'test_results': test_results
        }


class SelfDebuggingAgent:
    """Agent that debugs its own code"""
    
    def __init__(self, agent: CodingAgent):
        self.agent = agent
        self.executor = CodeExecutor()
        self.validator = CodeValidator()
    
    def debug_code(
        self,
        code: str,
        error_message: str = "",
        test_case: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Debug code using error message and test case"""
        
        # Analyze the error
        debug_prompt = f"""Debug this Python code that has an error:

Code:
```python
{code}
