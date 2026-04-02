# !pip install datasets
from datasets import load_dataset

# SWE-bench Lite is the smaller, verified subset (300 instances)
# Use this — it's higher quality than the full 2,294
dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")


'''
Each row has these useful fields:
- `instance_id` — unique ID (e.g., `django__django-11099`)
- `problem_statement` — the GitHub issue text (the bug description)
- `patch` — the correct code fix as a **git diff**
- `repo` — which repository it belongs to
- `hints_text` — sometimes has extra hints about the fix


**Step 3: Format Examples**

Each example for your Reviewer prompt would look like:
```
=== EXAMPLE SOLUTION ===
Bug: [problem_statement text]
Repository: [repo name]
Fix (git diff):
[patch text]
========================
'''