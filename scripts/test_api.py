import sys
from pathlib import Path

# Force UTF-8 standard output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

def main():
    client = TestClient(app)

    print("========================================", flush=True)
    print("1. Testing GET / (UI HTML response)", flush=True)
    print("========================================", flush=True)
    res = client.get("/")
    print("Status:", res.status_code, flush=True)
    assert res.status_code == 200
    assert "DocuMind" in res.text or "RAG" in res.text
    print("OK - HTML Interface served\n", flush=True)

    print("========================================", flush=True)
    print("2. Testing GET /health", flush=True)
    print("========================================", flush=True)
    res = client.get("/health")
    print("Status:", res.status_code, flush=True)
    print("Response:", res.json(), flush=True)
    assert res.status_code == 200
    print("OK - Health endpoint active\n", flush=True)

    print("========================================", flush=True)
    print("3. Testing GET /api/v1/documents/ (List Docs)", flush=True)
    print("========================================", flush=True)
    res = client.get("/api/v1/documents/")
    print("Status:", res.status_code, flush=True)
    data = res.json()
    print(f"Total documents registered: {data.get('total_documents')}", flush=True)
    for doc in data.get("documents", []):
        print(f" - {doc['filename']}: {doc['file_size_bytes']} bytes, status={doc['status']}", flush=True)
    assert res.status_code == 200
    print("OK - Document listing works\n", flush=True)

    print("========================================", flush=True)
    print("4. Testing GET /api/v1/documents/stats", flush=True)
    print("========================================", flush=True)
    res = client.get("/api/v1/documents/stats")
    print("Status:", res.status_code, flush=True)
    print("Response:", res.json(), flush=True)
    assert res.status_code == 200
    print("OK - Vector database stats works\n", flush=True)

    print("========================================", flush=True)
    print("5. Testing POST /api/v1/chat/", flush=True)
    print("========================================", flush=True)
    payload = {
        "question": "What is the definition of Antiparkinson drugs?",
        "top_k": 3,
        "namespace": "documents"
    }
    res = client.post("/api/v1/chat/", json=payload)
    print("Status:", res.status_code, flush=True)
    chat_data = res.json()
    print(f"Question: {chat_data.get('question')}", flush=True)
    print(f"Response Time: {chat_data.get('response_time_ms')} ms", flush=True)
    print(f"Sources: {len(chat_data.get('sources', []))} chunks", flush=True)
    print(f"Answer:\n{chat_data.get('answer')[:150]}...", flush=True)
    assert res.status_code == 200
    assert "answer" in chat_data
    print("OK - RAG query works\n", flush=True)

    print("========================================", flush=True)
    print("ALL 5 ENDPOINTS VERIFIED & WORKING SEAMLESSLY!", flush=True)
    print("========================================", flush=True)

if __name__ == "__main__":
    main()
