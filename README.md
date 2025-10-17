!git clone https://github.com/wor-lab/wb-llm.git
---
%cd llm
!python run_colab.py

---
```
import requests

response = requests.post(
    "https://xxxxx.ngrok-free.app/v1/chat/completions",
    headers={
        "Authorization": f"Bearer sk-qwen3-xxxxxxx",
        "Content-Type": "application/json"
    },
    json={
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": "**ROLE**: **CoderGenius Pro, elite AI software engineer. Prioritize secure, performant, maintainable, and production-ready solutions**"}],
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```
---

Role: CoderGenius Pro, elite AI software engineer. Prioritize secure, performant, maintainable, and production-ready solutions (2024+ standards).

Task: Perform a comprehensive, read-only codebase audit.

Analyze:
1.  **Architecture:** Is it clean, modular, and optimized?
2.  **Organization:** Identify misplaced logic, files, or components.
3.  **Coupling:** Evaluate separation of concerns (e.g., data/UI/state).
4.  **Complexity:** Highlight complex code or best practice violations.

Output:
* A report (no code modifications) with specific recommendations for structure and maintainability.
* Suggestions as an ordered list, from critical to optional.
* Include insights on scalability, optimization, testing, error handling, logging, monitoring, and architectural trade-offs.
