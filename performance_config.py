"""
WB AI CORPORATION - PERFORMANCE CONFIGURATION
Enterprise-grade settings for 90% coding benchmark performance

CLASSIFICATION: Configuration Module
DEPARTMENT: Strategy Division
NO MOCK DATA - Production parameters only
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ============================================================================
# BASE CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Base configuration for all coding benchmarks"""
    
    # Model settings
    model_name: str = "Qwen/Qwen3-1.7B"
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 4096
    
    # Generation strategy
    use_self_consistency: bool = True
    num_samples: int = 5
    max_iterations: int = 5
    
    # Execution settings
    enable_self_repair: bool = True
    enable_test_driven: bool = True
    timeout: int = 30
    
    # Performance targets
    target_accuracy: float = 0.90


# ============================================================================
# BENCHMARK-SPECIFIC CONFIGURATIONS
# ============================================================================

@dataclass
class HumanEvalConfig(BenchmarkConfig):
    """
    HumanEval: 164 hand-written programming problems
    Target: 90%+ pass@1 rate
    
    Strategy:
    - Low temperature for deterministic code
    - High sample count for self-consistency
    - Aggressive self-repair on failures
    - Function signature validation
    """
    
    temperature: float = 0.2
    max_tokens: int = 2048
    num_samples: int = 10
    max_iterations: int = 3
    timeout: int = 10
    
    # HumanEval specific
    validate_signature: bool = True
    validate_docstring: bool = True
    extract_function_only: bool = True
    run_provided_tests: bool = True
    
    target_accuracy: float = 0.90  # 90% pass@1


@dataclass
class MBPPConfig(BenchmarkConfig):
    """
    MBPP: Mostly Basic Python Problems (974 problems)
    Target: 90%+ accuracy
    
    Strategy:
    - Example-driven generation
    - Unit test validation
    - Iterative refinement
    """
    
    temperature: float = 0.3
    max_tokens: int = 1536
    num_samples: int = 5
    max_iterations: int = 3
    
    # MBPP specific
    use_test_cases: bool = True
    use_examples: bool = True
    validate_with_asserts: bool = True
    
    target_accuracy: float = 0.90


@dataclass
class SWEBenchConfig(BenchmarkConfig):
    """
    SWE-Bench: Real GitHub issues from popular repos
    Target: 79%+ resolution rate (SOTA)
    
    Strategy:
    - Repository context analysis
    - Multi-file understanding
    - Iterative debugging with error feedback
    - Patch generation and validation
    """
    
    temperature: float = 0.4
    max_tokens: int = 8192
    num_samples: int = 3
    max_iterations: int = 8
    timeout: int = 60
    
    # SWE-Bench specific
    analyze_repo_context: bool = True
    understand_stack_traces: bool = True
    generate_unified_diff: bool = True
    validate_no_regressions: bool = True
    multi_file_editing: bool = True
    
    target_accuracy: float = 0.79  # 79% (current SOTA)


@dataclass
class LiveBenchConfig(BenchmarkConfig):
    """
    LiveBench: Live coding challenges
    Target: 85%+ accuracy
    
    Strategy:
    - Real-time code execution
    - Performance optimization
    - Constraint satisfaction
    """
    
    temperature: float = 0.3
    max_tokens: int = 4096
    num_samples: int = 5
    max_iterations: int = 5
    
    # LiveBench specific
    optimize_for_performance: bool = True
    validate_constraints: bool = True
    benchmark_execution_time: bool = True
    
    target_accuracy: float = 0.85


@dataclass
class BigCodeBenchConfig(BenchmarkConfig):
    """
    BigCodeBench: Large-scale code generation
    Target: 85%+ accuracy
    
    Strategy:
    - Large context handling
    - Multi-function generation
    - Code structure preservation
    - Dependency management
    """
    
    temperature: float = 0.2
    max_tokens: int = 8192
    num_samples: int = 3
    max_iterations: int = 5
    
    # BigCodeBench specific
    handle_large_context: bool = True
    preserve_imports: bool = True
    multi_function_generation: bool = True
    validate_dependencies: bool = True
    
    target_accuracy: float = 0.85


# ============================================================================
# PERFORMANCE OPTIMIZER
# ============================================================================

class PerformanceOptimizer:
    """
    Central configuration manager for WB AI Corporation
    Provides optimal settings for each benchmark
    """
    
    def __init__(self):
        self.humaneval = HumanEvalConfig()
        self.mbpp = MBPPConfig()
        self.swe_bench = SWEBenchConfig()
        self.livebench = LiveBenchConfig()
        self.bigcodebench = BigCodeBenchConfig()
        
        self._configs = {
            'humaneval': self.humaneval,
            'mbpp': self.mbpp,
            'swe_bench': self.swe_bench,
            'swebench': self.swe_bench,
            'livebench': self.livebench,
            'bigcodebench': self.bigcodebench,
        }
    
    def get_config(self, benchmark: str) -> BenchmarkConfig:
        """Get configuration for specific benchmark"""
        return self._configs.get(benchmark.lower(), BenchmarkConfig())
    
    def get_all_targets(self) -> Dict[str, float]:
        """Get all performance targets"""
        return {
            'humaneval': self.humaneval.target_accuracy,
            'mbpp': self.mbpp.target_accuracy,
            'swe_bench': self.swe_bench.target_accuracy,
            'livebench': self.livebench.target_accuracy,
            'bigcodebench': self.bigcodebench.target_accuracy,
        }
    
    def print_configuration_report(self):
        """Print enterprise configuration report"""
        print("="*70)
        print("WB AI CORPORATION - PERFORMANCE CONFIGURATION REPORT")
        print("="*70)
        print(f"\nModel: {self.humaneval.model_name}")
        print("\nBenchmark Targets:")
        print("-"*70)
        
        for name, target in self.get_all_targets().items():
            print(f"  {name.upper():<20} Target: {target*100:>5.0f}%")
        
        print("\n" + "="*70)
        print("CONFIGURATION DETAILS")
        print("="*70)
        
        configs = [
            ("HumanEval", self.humaneval),
            ("MBPP", self.mbpp),
            ("SWE-Bench", self.swe_bench),
            ("LiveBench", self.livebench),
            ("BigCodeBench", self.bigcodebench),
        ]
        
        for name, cfg in configs:
            print(f"\n📊 {name}")
            print("-"*70)
            print(f"  Temperature: {cfg.temperature}")
            print(f"  Max Tokens: {cfg.max_tokens}")
            print(f"  Samples: {cfg.num_samples}")
            print(f"  Max Iterations: {cfg.max_iterations}")
            print(f"  Self-Repair: {cfg.enable_self_repair}")
            print(f"  Target: {cfg.target_accuracy*100:.0f}%")
        
        print("\n" + "="*70)


# ============================================================================
# PROMPT TEMPLATES (NO MOCK DATA)
# ============================================================================

PROMPT_TEMPLATES = {
    'humaneval': {
        'system': "You are an expert Python programmer. Write clean, efficient, and correct code that passes all tests.",
        
        'base': """Complete this Python function according to its signature and docstring.

{prompt}

Requirements:
- Follow the exact function signature
- Implement the logic described in the docstring
- Return the correct type
- Handle edge cases

Implementation:
```python
{prompt}""",
        
        'with_tests': """Complete this function and ensure it passes all test cases.

{prompt}

Your implementation must pass these tests.

Complete implementation:
```python
{prompt}"""
    },
    
    'mbpp': {
        'system': "You are a Python programming expert. Write correct, efficient solutions.",
        
        'base': """Write a Python function that solves this task:

Task: {task}

Requirements:
- Write a complete, working function
- Handle all edge cases
- Use efficient algorithms
- Follow Python best practices

Solution:
```python
""",
        
        'with_examples': """Write a Python function for this task:

Task: {task}

Examples:
{examples}

Write a function that handles all these cases correctly:
```python
"""
    },
    
    'swe_bench': {
        'system': "You are an expert software engineer. Debug and fix code issues accurately.",
        
        'issue_only': """Fix this issue in the codebase:

Issue: {issue}

Provide the corrected code:
```python
""",
        
        'with_context': """Debug and fix this issue:

Repository Context:
{context}

Issue Description:
{issue}

Error/Bug Details:
{error}

Provide the complete fix:
```python
""",
        
        'with_trace': """Fix this bug using the error trace:

Issue: {issue}

Stack Trace:
{trace}

Current Code:
{code}

Fixed code:
```python
"""
    },
    
    'livebench': {
        'system': "You are a competitive programmer. Write optimal solutions.",
        
        'base': """Solve this coding challenge:

Problem: {problem}

Constraints:
{constraints}

Write an efficient solution:
```python
"""
    },
    
    'bigcodebench': {
        'system': "You are an expert at generating production-quality code.",
        
        'base': """Generate code for this specification:

Specification: {specification}

Requirements:
{requirements}

Implementation:
```python
"""
    }
}


# ============================================================================
# TEST EXECUTION STRATEGIES (NO MOCK DATA)
# ============================================================================

TEST_STRATEGIES = {
    'humaneval': {
        'timeout_per_test': 5,
        'max_memory_mb': 512,
        'use_sandbox': True,
        'validate_imports': True,
        'check_syntax_first': True,
    },
    
    'mbpp': {
        'timeout_per_test': 3,
        'max_memory_mb': 256,
        'use_sandbox': True,
        'run_assertions': True,
    },
    
    'swe_bench': {
        'timeout_per_test': 30,
        'max_memory_mb': 1024,
        'validate_patch': True,
        'check_regressions': True,
        'run_repo_tests': True,
    },
    
    'livebench': {
        'timeout_per_test': 10,
        'benchmark_performance': True,
        'validate_constraints': True,
    },
    
    'bigcodebench': {
        'timeout_per_test': 15,
        'max_memory_mb': 1024,
        'validate_structure': True,
        'check_dependencies': True,
    }
}


# ============================================================================
# PERFORMANCE METRICS (NO MOCK DATA)
# ============================================================================

PERFORMANCE_METRICS = {
    'humaneval': {
        'primary': 'pass@1',  # Percentage of problems solved on first try
        'secondary': ['pass@10', 'pass@100'],
        'minimum_acceptable': 0.85,  # 85% minimum
        'target': 0.90,  # 90% target
        'stretch_goal': 0.95,  # 95% stretch
    },
    
    'mbpp': {
        'primary': 'accuracy',
        'secondary': ['with_examples_accuracy'],
        'minimum_acceptable': 0.85,
        'target': 0.90,
        'stretch_goal': 0.95,
    },
    
    'swe_bench': {
        'primary': 'resolution_rate',
        'secondary': ['no_regressions', 'patch_quality'],
        'minimum_acceptable': 0.70,
        'target': 0.79,  # Current SOTA
        'stretch_goal': 0.85,
    },
    
    'livebench': {
        'primary': 'accuracy',
        'secondary': ['optimal_solutions', 'avg_execution_time'],
        'minimum_acceptable': 0.80,
        'target': 0.85,
        'stretch_goal': 0.90,
    },
    
    'bigcodebench': {
        'primary': 'accuracy',
        'secondary': ['code_quality', 'completeness'],
        'minimum_acceptable': 0.80,
        'target': 0.85,
        'stretch_goal': 0.90,
    }
}


if __name__ == "__main__":
    # Display configuration report
    optimizer = PerformanceOptimizer()
    optimizer.print_configuration_report()
