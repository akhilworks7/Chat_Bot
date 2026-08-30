import time
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.credential_service import CredentialService
from app.db import crud
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("rag_service")


class RAGService:
    """
    Orchestrates the multi-tenant RAG pipeline:
    User Query -> Query Embedding -> User Namespace Pinecone Retrieval ->
    Adjacent Chunk Expansion -> Dynamic Groq LLM Generation -> Usage & History Tracking
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        self.llm_service = LLMService()

    def query(
        self,
        question: str,
        user_id: int,
        db: Session,
        session_id: str = "default",
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end multi-tenant RAG query for a specific user.
        """
        start_time = time.time()
        logger.info(f"Processing RAG query for user #{user_id}: '{question}'")

        # 1. Resolve active user credentials and namespace
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        namespace = creds["namespace"]
        pinecone_key = creds["pinecone_api_key"]
        pinecone_index = creds["pinecone_index"]
        groq_key = creds["groq_api_key"]
        groq_model = creds["groq_model"]

        # 2. Generate query embedding
        query_vector = self.embedding_service.embed_query(question)

        # 3. Retrieve top-k semantic matches from user's isolated namespace
        documents = self.vector_service.search(
            query_vector=query_vector,
            namespace=namespace,
            api_key=pinecone_key,
            index_name=pinecone_index,
            top_k=top_k or settings.TOP_K
        )

        # 4. Context Window Expansion:
        # For the top semantic matches, fetch adjacent contiguous chunks
        if documents:
            expanded_ids = set()
            existing_ids = {doc.get("id") for doc in documents if doc.get("id")}

            for doc in documents[:4]:
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
                adjacent_docs = self.vector_service.fetch_by_ids(
                    vector_ids=list(expanded_ids),
                    namespace=namespace,
                    api_key=pinecone_key,
                    index_name=pinecone_index
                )
                if adjacent_docs:
                    logger.info(f"Context Expansion: Fetched {len(adjacent_docs)} adjacent chunks from Pinecone.")
                    documents.extend(adjacent_docs)

            # Sort documents by source and chunk_id for coherent reading order
            documents.sort(key=lambda x: (x.get("source", ""), x.get("chunk_id") if x.get("chunk_id") is not None else 99999))

        # 5. Assemble structured context
        context = self.llm_service.build_context(documents)

        # 6. Generate grounded answer with Groq LLM
        answer = self.llm_service.generate_answer(
            question=question,
            context=context,
            api_key=groq_key,
            model=groq_model
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"RAG query completed in {elapsed_ms}ms with {len(documents)} retrieved sources.")

        # 7. Record chat messages and usage statistics
        crud.add_chat_message(
            db=db,
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=question
        )
        crud.add_chat_message(
            db=db,
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=documents,
            response_time_ms=elapsed_ms
        )

        return {
            "question": question,
            "answer": answer,
            "sources": documents,
            "response_time_ms": elapsed_ms,
            "credential_mode": creds["mode"],
            "groq_model": groq_model
        }

    def retrieve_context(
        self,
        question: str,
        user_id: int,
        db: Session,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Retrieves context documents and active credentials for streaming generation.
        """
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        namespace = creds["namespace"]
        pinecone_key = creds["pinecone_api_key"]
        pinecone_index = creds["pinecone_index"]

        query_vector = self.embedding_service.embed_query(question)
        documents = self.vector_service.search(
            query_vector=query_vector,
            namespace=namespace,
            api_key=pinecone_key,
            index_name=pinecone_index,
            top_k=top_k or settings.TOP_K
        )

        if documents:
            expanded_ids = set()
            existing_ids = {doc.get("id") for doc in documents if doc.get("id")}
            for doc in documents[:4]:
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
                adjacent_docs = self.vector_service.fetch_by_ids(
                    vector_ids=list(expanded_ids),
                    namespace=namespace,
                    api_key=pinecone_key,
                    index_name=pinecone_index
                )
                if adjacent_docs:
                    documents.extend(adjacent_docs)

            documents.sort(key=lambda x: (x.get("source", ""), x.get("chunk_id") if x.get("chunk_id") is not None else 99999))

        context = self.llm_service.build_context(documents)
        return {
            "creds": creds,
            "documents": documents,
            "context": context
        }

