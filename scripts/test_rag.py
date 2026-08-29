import sys
from pathlib import Path

# Force UTF-8 standard output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("test_rag")

def main():
    print("========================================", flush=True)
    print("1. Testing Configuration", flush=True)
    print("========================================", flush=True)
    print("Pinecone Index:", settings.PINECONE_INDEX_NAME, flush=True)
    print("Pinecone Namespace:", settings.PINECONE_NAMESPACE, flush=True)
    print("Groq Model:", settings.GROQ_MODEL, flush=True)
    print("Embedding Model:", settings.EMBEDDING_MODEL_NAME, flush=True)
    print("OK\n", flush=True)

    print("========================================", flush=True)
    print("2. Testing Embedding Service", flush=True)
    print("========================================", flush=True)
    from app.services.embedding_service import EmbeddingService
    emb = EmbeddingService()
    vec = emb.embed_query("What is the definition of Antiparkinson drugs?")
    print(f"Generated query vector dimension: {len(vec)}", flush=True)
    assert len(vec) == settings.EMBEDDING_DIMENSION, f"Expected {settings.EMBEDDING_DIMENSION}, got {len(vec)}"
    print("OK\n", flush=True)

    print("========================================", flush=True)
    print("3. Testing Vector Service (Pinecone)", flush=True)
    print("========================================", flush=True)
    from app.services.vector_service import VectorService
    vec_service = VectorService()
    stats = vec_service.get_stats()
    print("Pinecone Stats:", stats, flush=True)
    print("OK\n", flush=True)

    print("========================================", flush=True)
    print("4. Testing RAG End-to-End Query", flush=True)
    print("========================================", flush=True)
    from app.services.rag_service import RAGService
    rag = RAGService()
    test_question = "What is the definition of Antiparkinson drugs?"
    result = rag.query(test_question, top_k=3)
    
    print(f"Question: {result['question']}", flush=True)
    print(f"Response Time: {result['response_time_ms']} ms", flush=True)
    print(f"Retrieved Sources Count: {len(result['sources'])}", flush=True)
    print(f"\n--- Answer --- \n{result['answer']}\n", flush=True)
    print("========================================", flush=True)
    print("ALL TESTS PASSED SUCCESSFULLY!", flush=True)
    print("========================================", flush=True)

if __name__ == "__main__":
    main()
