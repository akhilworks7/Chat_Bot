import sys, os
sys.path.insert(0, os.path.abspath("."))
from app.services.rag_service import RAGService

rag = RAGService()
res = rag.query("give the list of ADVISORY BOARD member in a list format")
print("=== ANSWER ===")
print(res["answer"])
print("=== SOURCES COUNT ===", len(res["sources"]))
for i, s in enumerate(res["sources"][:5]):
    print(f"--- Source {i+1} (score: {s.get('score')}, id: {s.get('id')}, chunk: {s.get('chunk_id')}) ---")
    print(s.get("text", "")[:300])
    print()
