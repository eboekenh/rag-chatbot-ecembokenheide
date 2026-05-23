import requests

r = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Return this exact JSON: {\"pairs\": [{\"question\": \"What is pandas?\", \"answer\": \"A Python library.\"}]}"}],
    "stream": False,
    "options": {"temperature": 0.1}
}, timeout=300)
print("Status:", r.status_code)
print("Response:", r.text[:500])
