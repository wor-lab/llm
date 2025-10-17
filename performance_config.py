"""
Performance Configuration for Coding Benchmarks
Optimized settings for 90%+ accuracy on coding tasks
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class HumanEvalConfig:
    """Optimized for HumanEval (164 Python problems)
    Target: 90%+ pass@1
    """
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    temperature: float = 0.2  # Low for deterministic code
    top_p: float = 0.95
    max_tokens: int = 2048
    num_samples: int = 10  # For pass@k
    use_self_repair: bool = True
    max_repair_iterations: int = 3
    include_docstring: bool = True
    use_type_hints: bool = True
    validate_syntax: bool = True
    run_tests: bool = True
    timeout: int = 10  # seconds per test


@dataclass
class MBPPConfig:
    """Optimized for MBPP (Mostly Basic Programming Problems)
    Target: 85%+ pass@1
    """
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 1024
    num_samples: int = 5
    use_test_driven: bool = True  # Generate tests first
    use_self_repair: bool = True
    max_repair_iterations: int = 2
    validate_with_assertions: bool = True
    timeout: int = 5


@dataclass
class SWEBenchConfig:
    """Optimized for SWE-Bench (Real GitHub issues)
    Target: 40-50% (state-of-the-art is ~50%)
    """
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 4096
    num_samples: int = 1
    use_repo_context: bool = True
    max_context_files: int = 5
    use_iterative_editing: bool = True
    max_edit_iterations: int = 5
    run_tests: bool = True
    use_git_diff: bool = True
    lint_code: bool = True
    timeout: int = 30


@dataclass
class LiveBenchConfig:
    """Optimized for LiveBench (Live evaluation)
    Target: 80%+ on code tasks
    """
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    temperature: float = 0.25
    top_p: float = 0.95
    max_tokens: int = 3072
    num_samples: int = 3
    use_chain_of_thought: bool = True
    verify_output: bool = True
    use_self_consistency: bool = True
    timeout: int = 15


@dataclass
class BigCodeBenchConfig:
    """Optimized for BigCodeBench (Complex code generation)
    Target: 70%+ pass@1
    """
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 8192  # Larger for complex code
    num_samples: int = 5
    use_multi_file: bool = True
    use_planning: bool = True  # Plan before coding
    use_self_repair: bool = True
    max_repair_iterations: int = 4
    validate_imports: bool = True
    run_integration_tests: bool = True
    timeout: int = 30


@dataclass
class GeneralCodingConfig:
    """General-purpose coding configuration"""
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 4096
    num_samples: int = 1
    use_comments: bool = True
    use_error_handling: bool = True
    validate_syntax: bool = True
    timeout: int = 15


class BenchmarkConfigs:
    """Central configuration manager"""
    
    def __init__(self):
        self.humaneval = HumanEvalConfig()
        self.mbpp = MBPPConfig()
        self.swe_bench = SWEBenchConfig()
        self.live_bench = LiveBenchConfig()
        self.bigcode_bench = BigCodeBenchConfig()
        self.general = GeneralCodingConfig()
    
    def get_config(self, benchmark: str) -> Any:
        """Get configuration for specific benchmark"""
        configs = {
            'humaneval': self.humaneval,
            'mbpp': self.mbpp,
            'swe_bench': self.swe_bench,
            'swebench': self.swe_bench,
            'live_bench': self.live_bench,
            'livebench': self.live_bench,
            'bigcode_bench': self.bigcode_bench,
            'bigcodebench': self.bigcode_bench,
            'general': self.general,
        }
        return configs.get(benchmark.lower(), self.general)
    
    def print_all(self):
        """Print all configurations"""
        print("="*70)
        print("📊 CODING BENCHMARK CONFIGURATIONS")
        print("="*70)
        
        configs = [
            ("HumanEval", self.humaneval),
            ("MBPP", self.mbpp),
            ("SWE-Bench", self.swe_bench),
            ("LiveBench", self.live_bench),
            ("BigCodeBench", self.bigcode_bench),
        ]
        
        for name, config in configs:
            print(f"\n🎯 {name}")
            print("-"*70)
            for key, value in config.__dict__.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
        
        print("\n" + "="*70)


# Prompt templates for each benchmark
class PromptStrategies:
    """Optimal prompting strategies per benchmark"""
    
    HUMANEVAL_SYSTEM = """You are an expert Python programmer. Generate clean, efficient, and correct code that passes all test cases. Follow the function signature exactly."""
    
    MBPP_SYSTEM = """You are a Python programming expert. Write simple, clear code that solves the problem correctly. Include basic error handling."""
    
    SWE_BENCH_SYSTEM = """You are an expert software engineer working on a GitHub repository. Analyze the issue carefully, understand the codebase context, and provide a precise fix."""
    
    LIVE_BENCH_SYSTEM = """You are an expert programmer. Think step-by-step, write clean code, and verify your solution works correctly."""
    
    BIGCODE_BENCH_SYSTEM = """You are a senior software engineer. Plan your solution, write modular code, and handle edge cases properly."""


# Performance optimization tips
OPTIMIZATION_GUIDE = """
🚀 PERFORMANCE OPTIMIZATION GUIDE

1. HumanEval (Target: 90%+):
   ✓ Use temperature 0.2 for deterministic outputs
   ✓ Generate 10+ samples and use pass@k metrics
   ✓ Enable self-repair with test feedback
   ✓ Validate syntax before execution
   ✓ Use type hints for clarity

2. MBPP (Target: 85%+):
   ✓ Lower temperature (0.3) for basic problems
   ✓ Test-driven approach: generate tests first
   ✓ Self-repair on failed assertions
   ✓ Keep solutions simple and readable

3. SWE-Bench (Target: 40-50%):
   ✓ Use repository context (5 most relevant files)
   ✓ Iterative editing with git diff
   ✓ Run existing test suite
   ✓ Lint code before submission
   ✓ Focus on minimal changes

4. LiveBench (Target: 80%+):
   ✓ Use chain-of-thought reasoning
   ✓ Self-consistency across multiple samples
   ✓ Verify outputs against examples
   ✓ Temperature 0.25 for balance

5. BigCodeBench (Target: 70%+):
   ✓ Plan solution architecture first
   ✓ Support multi-file generation
   ✓ Validate imports and dependencies
   ✓ Run integration tests
   ✓ Max 8K tokens for complex problems

6. General Best Practices:
   ✓ Always validate syntax before execution
   ✓ Use timeouts to prevent infinite loops
   ✓ Implement self-repair mechanisms
   ✓ Generate multiple samples for uncertainty
   ✓ Use Qwen2.5-Coder models for best results
"""


if __name__ == "__main__":
    configs = BenchmarkConfigs()
    configs.print_all()
    print(OPTIMIZATION_GUIDE)
