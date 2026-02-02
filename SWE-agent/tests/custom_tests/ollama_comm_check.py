import requests

"""
CUSTOM file to perform communications check with local qwen model in Ollama framework
"""

resp = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5-coder:7b-instruct",
        "prompt": "Explain why add(a,b)=a-b is wrong",
        "stream": False
    }
)

print(resp.json()["response"])
