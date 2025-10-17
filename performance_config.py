"""
Performance Configurations for Coding Benchmarks
Optimized for 90%+ accuracy on SWE-Bench, HumanEval, MBPP, LiveBench, BigCodeBench
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class BenchmarkConfig:
    """Base configuration for all benchmarks"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 4096
    use_self_consistency: bool = True
    num_samples: int = 5
    timeout: int = 30


@dataclass
class HumanEvalConfig(BenchmarkConfig):
    """
    HumanEval: Python code generation from docstrings
    Target: 90%+ pass@1
    """
    temperature: float = 0.2  # Low temp for precise code
    max_tokens: int = 2048
    num_samples: int = 10  # High sampling for pass@k
    use_test_driven: bool = True
    validate_syntax: bool = True
    run_test_cases: bool = True
    
    # Code generation strategies
    use_docstring_analysis: bool = True
    use_type_hints: bool = True
    use_examples: bool = True
    
    # Optimization
    cache_imports: bool = True
    optimize_loops: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'num_samples': self.num_samples,
            'top_p': self.top_p,
        }


@dataclass
class MBPPConfig(BenchmarkConfig):
    """
    MBPP: Mostly Basic Python Problems
    Target: 90%+ pass@1
    """
    temperature: float = 0.25
    max_tokens: int = 1536
    num_samples: int = 8
    use_test_driven: bool = True
    validate_syntax: bool = True
    
    # MBPP-specific
    use_problem_decomposition: bool = True
    use_edge_case_handling: bool = True
    verify_with_examples: bool = True
    
    # Error handling
    max_retries: int = 3
    auto_debug: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'num_samples': self.num_samples,
            'top_p': self.top_p,
        }


@dataclass
class SWEBenchConfig(BenchmarkConfig):
    """
    SWE-Bench: Real-world software engineering tasks
    Target: 79%+ (very challenging)
    """
    temperature: float = 0.3
    max_tokens: int = 6144  # Large for complex fixes
    num_samples: int = 5
    max_iterations: int = 5
    
    # SWE-Bench specific
    use_repository_context: bool = True
    use_static_analysis: bool = True
    run_tests: bool = True
    
    # Advanced strategies
    use_git_diff: bool = True
    use_linting: bool = True
    use_type_checking: bool = True
    
    # Iterative improvement
    use_feedback_loop: bool = True
    verify_no_regression: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'num_samples': self.num_samples,
            'max_iterations': self.max_iterations,
        }


@dataclass
class LiveBenchConfig(BenchmarkConfig):
    """
    LiveBench: Live coding challenges
    Target: 85%+ accuracy
    """
    temperature: float = 0.2
    max_tokens: int = 3072
    num_samples: int = 5
    time_limit: int = 60  # seconds per problem
    
    # LiveBench specific
    use_competitive_programming_style: bool = True
    optimize_for_speed: bool = True
    use_standard_algorithms: bool = True
    
    # Strategy
    test_with_examples: bool = True
    verify_edge_cases: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'num_samples': self.num_samples,
        }


@dataclass
class BigCodeBenchConfig(BenchmarkConfig):
    """
    BigCodeBench: Large-scale code generation
    Target: 80%+ accuracy
    """
    temperature: float = 0.25
    max_tokens: int = 8192  # Very large for complex code
    num_samples: int = 3
    
    # BigCodeBench specific
    use_modular_design: bool = True
    use_code_reuse: bool = True
    add_documentation: bool = True
    
    # Quality
    enforce_pep8: bool = True
    add_type_hints: bool = True
    add_error_handling: bool = True
    
    # Optimization
    parallel_generation: bool = False  # For speed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'num_samples': self.num_samples,
        }


@dataclass
class CodeContestsConfig(BenchmarkConfig):
    """
    CodeContests: Competitive programming
    Target: 70%+ (very hard)
    """
    temperature: float = 0.2
    max_tokens: int = 4096
    num_samples: int = 10
    
    # Competitive programming specific
    use_algorithm_library: bool = True
    optimize_complexity: bool = True
    use_mathematical_analysis: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'num_samples': self.num_samples,
        }


class PerformanceOptimizer:
    """Central optimizer for all coding benchmarks"""
    
    def __init__(self):
        self.humaneval = HumanEvalConfig()
        self.mbpp = MBPPConfig()
        self.swe_bench = SWEBenchConfig()
        self.livebench = LiveBenchConfig()
        self.bigcodebench = BigCodeBenchConfig()
        self.codecontests = CodeContestsConfig()
    
    def get_config(self, benchmark: str) -> BenchmarkConfig:
        """Get configuration for specific benchmark"""
        configs = {
            'humaneval': self.humaneval,
            'mbpp': self.mbpp,
            'swe_bench': self.swe_bench,
            'swenbench': self.swe_bench,
            'livebench': self.livebench,
            'bigcodebench': self.bigcodebench,
            'codecontests': self.codecontests,
        }
        return configs.get(benchmark.lower(), BenchmarkConfig())
    
    def print_all_configs(self):
        """Display all benchmark configurations"""
        print("="*80)
        print("🎯 CODING BENCHMARK CONFIGURATIONS")
        print("="*80)
        
        configs = [
            ("HumanEval", self.humaneval, "90%"),
            ("MBPP", self.mbpp, "90%"),
            ("SWE-Bench", self.swe_bench, "79%"),
            ("LiveBench", self.livebench, "85%"),
            ("BigCodeBench", self.bigcodebench, "80%"),
            ("CodeContests", self.codecontests, "70%"),
        ]
        
        for name, config, target in configs:
            print(f"\n📊 {name} (Target: {target})")
            print("-"*80)
            print(f"  Temperature: {config.temperature}")
            print(f"  Max Tokens: {config.max_tokens}")
            print(f"  Samples: {config.num_samples}")
            if hasattr(config, 'max_iterations'):
                print(f"  Max Iterations: {config.max_iterations}")
        
        print("\n" + "="*80)


# Optimization strategies per benchmark
OPTIMIZATION_STRATEGIES = {
    'humaneval': {
        'priority': 'correctness',
        'key_techniques': [
            'Test-Driven Development',
            'Syntax Validation',
            'Multiple Solutions with Voting',
            'Type Hint Analysis'
        ],
        'pass_at_k': [1, 10, 100],
        'expected_performance': '90%+ pass@1'
    },
    
    'mbpp': {
        'priority': 'correctness',
        'key_techniques': [
            'Problem Decomposition',
            'Edge Case Handling',
            'Example-Based Verification',
            'Auto-Debugging'
        ],
        'expected_performance': '90%+ pass@1'
    },
    
    'swe_bench': {
        'priority': 'real_world_fixes',
        'key_techniques': [
            'Repository Context Analysis',
            'Iterative Refinement',
            'Test Execution',
            'Regression Prevention',
            'Git Diff Analysis'
        ],
        'expected_performance': '79%+ resolution rate'
    },
    
    'livebench': {
        'priority': 'speed_and_correctness',
        'key_techniques': [
            'Competitive Programming Patterns',
            'Algorithm Optimization',
            'Fast Testing',
            'Edge Case Verification'
        ],
        'expected_performance': '85%+ accuracy'
    },
    
    'bigcodebench': {
        'priority': 'large_scale_generation',
        'key_techniques': [
            'Modular Design',
            'Code Reuse',
            'Documentation',
            'PEP8 Compliance',
            'Error Handling'
        ],
        'expected_performance': '80%+ accuracy'
    },
}


# Code quality checklist
CODE_QUALITY_CHECKS = {
    'syntax': True,
    'pep8': True,
    'type_hints': True,
    'docstrings': True,
    'error_handling': True,
    'test_coverage': True,
    'security': True,
    'performance': True,
}


# Prompt optimization settings
PROMPT_SETTINGS = {
    'use_few_shot': True,
    'num_examples': 3,
    'use_chain_of_thought': True,
    'use_step_by_step': True,
    'use_reflection': True,
    'use_verification': True,
}


if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    optimizer.print_all_configs()
    
    print("\n🔧 OPTIMIZATION STRATEGIES:")
    print("="*80)
    for benchmark, strategy in OPTIMIZATION_STRATEGIES.items():
        print(f"\n{benchmark.upper()}:")
        print(f"  Priority: {strategy['priority']}")
        print(f"  Techniques: {', '.join(strategy['key_techniques'])}")
        print(f"  Target: {strategy['expected_performance']}")
    print("\n" + "="*80)
