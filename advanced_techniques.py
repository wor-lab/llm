"""
Advanced optimization techniques for Qwen Agent
Program-of-Thoughts, Tree-of-Thoughts, and more
"""

from typing import List, Dict, Any
from agent_layer import QwenAgent, ReasoningEngine


class AdvancedTechniques:
    """Advanced reasoning techniques to boost performance"""
    
    def __init__(self, agent: QwenAgent):
        self.agent = agent
        self.reasoning = ReasoningEngine()
    
    def program_of_thoughts(self, question: str) -> str:
        """
        Program-of-Thoughts: Generate code to solve the problem
        Best for: Math problems, logic puzzles
        """
        prompt = f"""Solve this problem by writing Python code that prints the answer.

Problem: {question}

```python
# Solution code - print only the final answer
"""
        response = self.agent.generate(prompt, temperature=0.3)
        code = self.reasoning.extract_code(response)
        
        if code and self.agent.tools.validate_syntax(code):
            result = self.agent.tools.execute_python(code)
            if result['success']:
                return result['output']
        
        return ""
    
    def tree_of_thoughts(self, question: str, num_branches: int = 3, depth: int = 2) -> str:
        """
        Tree-of-Thoughts: Explore multiple reasoning paths
        Best for: Complex problems requiring exploration
        """
        branches = []
        
        # Generate multiple initial approaches
        for i in range(num_branches):
            prompt = f"""Solve this problem using approach #{i+1}:

{question}

Approach {i+1} solution:"""
            
            response = self.agent.generate(prompt, temperature=0.8)
            branches.append({
                'approach': i+1,
                'reasoning': response,
                'answer': self.reasoning.extract_answer(response)
            })
        
        # Meta-reasoning to select best answer
        branches_text = "\n\n".join([
            f"Approach {b['approach']}:\n{b['reasoning']}\nAnswer: {b['answer']}"
            for b in branches
        ])
        
        meta_prompt = f"""Given these {num_branches} different solution approaches to the problem:

{question}

Solutions:
{branches_text}

Which answer is most likely correct? Explain why and provide the final answer.

Analysis:"""
        
        final_response = self.agent.generate(meta_prompt, temperature=0.2)
        return self.reasoning.extract_answer(final_response)
    
    def iterative_refinement(self, question: str, max_iterations: int = 3) -> str:
        """
        Iterative Refinement: Improve answer through self-critique
        Best for: Complex problems requiring careful reasoning
        """
        prompt = f"Solve this problem:\n\n{question}\n\nAnswer:"
        answer = self.agent.generate(prompt, temperature=0.5)
        
        for iteration in range(max_iterations):
            critique_prompt = f"""Original problem: {question}

Your previous answer:
{answer}

Critique your answer:
1. Are there any errors in reasoning?
2. Can the answer be improved?
3. Provide a refined answer.

Refined answer:"""
            
            answer = self.agent.generate(critique_prompt, temperature=0.3)
        
        return self.reasoning.extract_answer(answer)
    
    def least_to_most(self, question: str) -> str:
        """
        Least-to-Most Prompting: Break down into subproblems
        Best for: Complex multi-step problems
        """
        # Step 1: Decompose into subproblems
        decompose_prompt = f"""Break down this problem into smaller subproblems:

Problem: {question}

List the subproblems we need to solve:
1."""
        
        decomposition = self.agent.generate(decompose_prompt, temperature=0.3)
        
        # Step 2: Solve each subproblem
        solve_prompt = f"""Original problem: {question}

Subproblems identified:
{decomposition}

Now solve each subproblem in order and combine for the final answer:

Solution:"""
        
        solution = self.agent.generate(solve_prompt, temperature=0.3, max_tokens=2048)
        return self.reasoning.extract_answer(solution)
    
    def chain_of_verification(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Chain-of-Verification: Verify the answer
        Returns: {'verified': bool, 'corrected_answer': str}
        """
        verify_prompt = f"""Problem: {question}

Proposed answer: {answer}

Verify this answer step by step:
1. Is the reasoning correct?
2. Are there any calculation errors?
3. Is the final answer correct?

Verification:"""
        
        verification = self.agent.generate(verify_prompt, temperature=0.2)
        
        # Check if verification found issues
        is_correct = any(word in verification.lower() for word in ['correct', 'yes', 'accurate', 'right'])
        has_error = any(word in verification.lower() for word in ['error', 'wrong', 'incorrect', 'mistake'])
        
        if has_error:
            # Generate corrected answer
            corrected_answer = self.reasoning.extract_answer(verification)
            return {
                'verified': False,
                'original_answer': answer,
                'corrected_answer': corrected_answer,
                'verification': verification
            }
        
        return {
            'verified': True,
            'answer': answer,
            'verification': verification
        }
    
    def multimodal_cot(self, question: str, use_code: bool = True, use_sc: bool = True) -> str:
        """
        Multimodal Chain-of-Thought: Combine multiple strategies
        Best for: Maximum accuracy on important problems
        """
        answers = []
        
        # 1. Standard CoT
        cot_answer = self.agent.solve_gsm8k(question, use_code=False)['answer']
        answers.append(('cot', cot_answer))
        
        # 2. Program-of-Thoughts
        if use_code:
            pot_answer = self.program_of_thoughts(question)
            if pot_answer:
                answers.append(('pot', pot_answer))
        
        # 3. Self-Consistency
        if use_sc:
            sc_answer = self.agent.self_consistency(
                f"Solve: {question}\n\nAnswer:",
                n=3
            )
            answers.append(('sc', sc_answer))
        
        # Vote or use verification
        if len(answers) >= 2:
            from collections import Counter
            normalized_answers = [
                self.reasoning.normalize_answer(ans[1]) for ans in answers
            ]
            counter = Counter(normalized_answers)
            most_common = counter.most_common(1)[0][0]
            return most_common
        
        return answers[0][1] if answers else ""


# Example usage
if __name__ == "__main__":
    from agent_layer import QwenAgent, AgentConfig
    
    print("🔬 Testing Advanced Techniques\n")
    
    # Initialize
    config = AgentConfig(num_samples=3)
    agent = QwenAgent(config)
    advanced = AdvancedTechniques(agent)
    
    question = "A store has 120 apples. They sell 30% in the morning and 25% of the remainder in the afternoon. How many apples are left?"
    
    # Test different techniques
    print("1️⃣ Program-of-Thoughts")
    pot_answer = advanced.program_of_thoughts(question)
    print(f"Answer: {pot_answer}\n")
    
    print("2️⃣ Tree-of-Thoughts")
    tot_answer = advanced.tree_of_thoughts(question, num_branches=2)
    print(f"Answer: {tot_answer}\n")
    
    print("3️⃣ Iterative Refinement")
    iter_answer = advanced.iterative_refinement(question, max_iterations=2)
    print(f"Answer: {iter_answer}\n")
    
    print("4️⃣ Multimodal CoT")
    multi_answer = advanced.multimodal_cot(question)
    print(f"Answer: {multi_answer}\n")
