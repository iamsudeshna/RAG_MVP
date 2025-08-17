
import requests

URL = "http://localhost:8001/ingest"
payload = {
    "path": "data/example.txt",
    "chunk_size": 400,
    "chunk_overlap": 50,
    "embed_model": "nomic-embed-text",
    "ollama_base_url": "http://localhost:11434"
}
resp = requests.post(URL, json=payload)
print(resp.json())


question = "Which company made Ozempic?"  
API_URL = "http://localhost:8001/ask"
payload = {"question": question}

response = requests.post(API_URL, json=payload)
if response.ok:
    data = response.json()
    print("\n Answer:", data.get("answer"))
    print("\n Sources:")
    for src in data.get("sources", []):
        print(f" - {src['source']}: {src['snippet']}")
else:
    print("❌ Error:", response.status_code, response.text)
