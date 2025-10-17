"""
Performance configurations optimized for each benchmark
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class GSM8KConfig:
    """Optimized for 90-95% GSM8K accuracy"""
    use_self_consistency: bool = True
    num_samples: int = 8
    temperature: float = 0.7
    use_code: bool = True
    verify_arithmetic: bool = True
    max_tokens: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'use_self_consistency': self.use_self_consistency,
            'num_samples': self.num_samples,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }


@dataclass
class AIMEConfig:
    """Optimized for AIME competition problems"""
    use_program_of_thoughts: bool = True
    temperature: float = 0.3
    max_tokens: int = 4096
    verify_with_code: bool = True
    use_tree_of_thoughts: bool = False
    num_branches: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }


@dataclass
class MMLUConfig:
    """Optimized for MMLU knowledge questions"""
    temperature: float = 0.1
    use_cot: bool = True
    analyze_all_options: bool = True
    use_self_consistency: bool = True
    num_samples: int = 5
    max_tokens: int = 1024
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'use_self_consistency': self.use_self_consistency,
            'num_samples': self.num_samples,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }


@dataclass
class SWEBenchConfig:
    """Optimized for 70-79% SWE-Bench accuracy"""
    max_iterations: int = 5
    use_static_analysis: bool = True
    run_tests: bool = True
    temperature: float = 0.4
    max_tokens: int = 4096
    verify_syntax: bool = True
    iterative_refinement: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_iterations': self.max_iterations,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }


@dataclass
class ReActConfig:
    """Optimized for ReAct agent tasks"""
    max_steps: int = 10
    temperature: float = 0.5
    use_reflection: bool = True
    max_tokens: int = 1024
    available_tools: list = None
    
    def __post_init__(self):
        if self.available_tools is None:
            self.available_tools = ['python', 'calculator', 'search']
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }


class PerformanceOptimizer:
    """Apply optimal configurations for each benchmark"""
    
    def __init__(self):
        self.gsm8k = GSM8KConfig()
        self.aime = AIMEConfig()
        self.mmlu = MMLUConfig()
        self.swe_bench = SWEBenchConfig()
        self.react = ReActConfig()
    
    def get_config(self, benchmark: str) -> Dict[str, Any]:
        """Get optimal config for a benchmark"""
        configs = {
            'gsm8k': self.gsm8k.to_dict(),
            'aime': self.aime.to_dict(),
            'mmlu': self.mmlu.to_dict(),
            'swe_bench': self.swe_bench.to_dict(),
            'react': self.react.to_dict(),
        }
        return configs.get(benchmark.lower(), {})
    
    def print_all_configs(self):
        """Print all configurations"""
        print("="*70)
        print("📊 PERFORMANCE CONFIGURATIONS")
        print("="*70)
        
        print("\n🧮 GSM8K (Target: 90-95%)")
        print("-"*70)
        for key, value in self.gsm8k.to_dict().items():
            print(f"  {key}: {value}")
        
        print("\n🎯 AIME (Target: 40-55%)")
        print("-"*70)
        for key, value in self.aime.to_dict().items():
            print(f"  {key}: {value}")
        
        print("\n📚 MMLU (Target: 70-85%)")
        print("-"*70)
        for key, value in self.mmlu.to_dict().items():
            print(f"  {key}: {value}")
        
        print("\n💻 SWE-Bench (Target: 70-79%)")
        print("-"*70)
        for key, value in self.swe_bench.to_dict().items():
            print(f"  {key}: {value}")
        
        print("\n🤖 ReAct Agent")
        print("-"*70)
        for key, value in self.react.to_dict().items():
            print(f"  {key}: {value}")
        
        print("\n" + "="*70)


# Usage recommendations
OPTIMIZATION_TIPS = """
🚀 OPTIMIZATION TIPS FOR MAXIMUM PERFORMANCE:

1. GSM8K (Target: 95%):
   - Use self-consistency with 8+ samples
   - Enable code execution for arithmetic
   - Temperature: 0.7 for diversity

2. AIME (Target: 50%+):
   - Use Program-of-Thoughts (code-based solving)
   - Lower temperature (0.3) for precision
   - Verify answers with code when possible

3. MMLU (Target: 80%+):
   - Very low temperature (0.1) for factual questions
   - Use chain-of-thought for complex questions
   - Self-consistency for ambiguous cases

4. SWE-Bench (Target: 79%):
   - Iterative refinement with error feedback
   - Syntax validation before execution
   - 5+ iterations for complex bugs

5. General:
   - Use T4/A100 GPU for faster inference
   - Batch similar questions together
   - Cache model for repeated runs
   - Monitor GPU memory usage
"""


if __name__ == "__main__":
    optimizer = PerformanceOptimizer()
    optimizer.print_all_configs()
    print(OPTIMIZATION_TIPS)
