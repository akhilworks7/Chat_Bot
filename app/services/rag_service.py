import time
from typing import Dict, Any, Optional, List

from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("rag_service")


class RAGService:
    """
    Orchestrates the entire RAG pipeline:
    Query -> Embedding -> Pinecone Retrieval -> Context Assembly -> Groq LLM Generation
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        self.llm_service = LLMService()

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end RAG query.
        Returns:
            {
                "question": str,
                "answer": str,
                "sources": List[Dict],
                "response_time_ms": float
            }
        """
        start_time = time.time()
        logger.info(f"Processing RAG query: '{question}'")

        # 1. Generate query embedding
        query_vector = self.embedding_service.embed_query(question)

        # 2. Retrieve top-k semantic matches from Pinecone
        target_namespace = namespace or settings.PINECONE_NAMESPACE
        documents = self.vector_service.search(
            query_vector=query_vector,
            top_k=top_k or settings.TOP_K,
            namespace=target_namespace
        )

        # 3. Context Window Expansion:
        # For the top semantic matches, fetch adjacent contiguous chunks to keep multi-chunk lists, tables, and sections complete.
        expanded_ids = set()
        existing_ids = {doc.get("id") for doc in documents if doc.get("id")}
        
        for doc in documents[:4]:  # Expand context for top 4 matches
            doc_id = doc.get("id")
            if doc_id and "_chunk_" in doc_id:
                base, num_str = doc_id.rsplit("_chunk_", 1)
                if num_str.isdigit():
                    num = int(num_str)
                    for offset in [1, 2]:
                        next_id = f"{base}_chunk_{num + offset}"
                        if next_id not in existing_ids:
                            expanded_ids.add(next_id)

        if expanded_ids:
            adjacent_docs = self.vector_service.fetch_by_ids(list(expanded_ids), namespace=target_namespace)
            if adjacent_docs:
                logger.info(f"Context Window Expansion: Fetched {len(adjacent_docs)} adjacent chunks from Pinecone.")
                documents.extend(adjacent_docs)

        # Sort documents by source and chunk_id so LLM receives clean, continuous reading order
        documents.sort(key=lambda x: (x.get("source", ""), x.get("chunk_id") if x.get("chunk_id") is not None else 99999))

        # 4. Assemble structured context
        context = self.llm_service.build_context(documents)

        # 5. Generate grounded answer with Groq LLM
        answer = self.llm_service.generate_answer(
            question=question,
            context=context
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"RAG query completed in {elapsed_ms}ms with {len(documents)} retrieved sources.")

        return {
            "question": question,
            "answer": answer,
            "sources": documents,
            "response_time_ms": elapsed_ms
        }
