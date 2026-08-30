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


CASUAL_PATTERNS = {
    "hi", "hello", "hey", "how are you", "how are you doing", "who are you", 
    "what are you", "what can you do", "help", "good morning", "good afternoon",
    "good evening", "thank you", "thanks", "thanks a lot", "bye", "goodbye",
    "tell me a joke", "who made you", "how do i upload documents"
}

def is_casual_query(text: str) -> bool:
    cleaned = "".join(c for c in text.lower().strip() if c.isalnum() or c.isspace()).strip()
    if cleaned in CASUAL_PATTERNS:
        return True
    words = cleaned.split()
    if len(words) <= 3 and any(cleaned.startswith(w) for w in ["hi", "hello", "hey", "thanks", "thank"]):
        return True
    return False


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
        Strictly enforces that answers are grounded only on active documents registered in SQLite.
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

        # 2. Check active registered documents in database
        user_docs = crud.get_user_documents(db=db, user_id=user_id)
        valid_sources = {d.file_name for d in user_docs} if user_docs else set()
        documents = []

        # Instant bypass for casual greetings / chit-chat
        if valid_sources and not is_casual_query(question):
            # 3. Generate query embedding
            query_vector = self.embedding_service.embed_query(question)

            # 4. Retrieve top-k semantic matches from user's isolated namespace
            raw_documents = self.vector_service.search(
                query_vector=query_vector,
                namespace=namespace,
                api_key=pinecone_key,
                index_name=pinecone_index,
                top_k=top_k or settings.TOP_K
            )

            # 5. Filter retrieved chunks strictly to active SQLite documents
            documents = [doc for doc in raw_documents if doc.get("source") in valid_sources]



        # 6. Context Window Expansion:
        # For the top semantic matches, fetch adjacent contiguous chunks conservatively
        if documents:
            expanded_ids = set()
            existing_ids = {doc.get("id") for doc in documents if doc.get("id")}

            for doc in documents[:2]:
                doc_id = doc.get("id")
                if doc_id and "_chunk_" in doc_id:
                    base, num_str = doc_id.rsplit("_chunk_", 1)
                    if num_str.isdigit():
                        num = int(num_str)
                        for offset in [1]:
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
                    valid_adjacent = [d for d in adjacent_docs if d.get("source") in valid_sources]
                    logger.info(f"Context Expansion: Fetched {len(valid_adjacent)} adjacent chunks from Pinecone.")
                    documents.extend(valid_adjacent)

            # Sort documents by source and chunk_id for coherent reading order
            documents.sort(key=lambda x: (x.get("source", ""), x.get("chunk_id") if x.get("chunk_id") is not None else 99999))

        # 7. Assemble structured context
        context = self.llm_service.build_context(documents)


        # 8. Generate grounded answer with Groq LLM
        answer = self.llm_service.generate_answer(
            question=question,
            context=context,
            api_key=groq_key,
            model=groq_model
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"RAG query completed in {elapsed_ms}ms with {len(documents)} retrieved sources.")

        # 9. Record chat messages and usage statistics
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
        Strictly filters out any vectors not belonging to currently registered documents in SQLite.
        """
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        namespace = creds["namespace"]
        pinecone_key = creds["pinecone_api_key"]
        pinecone_index = creds["pinecone_index"]

        # Check active registered documents in database or casual queries
        user_docs = crud.get_user_documents(db=db, user_id=user_id)
        if not user_docs or is_casual_query(question):
            return {
                "creds": creds,
                "documents": [],
                "context": "",
                "has_docs": bool(user_docs)
            }

        valid_sources = {d.file_name for d in user_docs}

        query_vector = self.embedding_service.embed_query(question)

        raw_documents = self.vector_service.search(
            query_vector=query_vector,
            namespace=namespace,
            api_key=pinecone_key,
            index_name=pinecone_index,
            top_k=top_k or settings.TOP_K
        )

        # Filter strictly to active SQLite documents
        documents = [doc for doc in raw_documents if doc.get("source") in valid_sources]

        if documents:
            expanded_ids = set()
            existing_ids = {doc.get("id") for doc in documents if doc.get("id")}
            for doc in documents[:2]:
                doc_id = doc.get("id")
                if doc_id and "_chunk_" in doc_id:
                    base, num_str = doc_id.rsplit("_chunk_", 1)
                    if num_str.isdigit():
                        num = int(num_str)
                        for offset in [1]:
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
                    valid_adjacent = [d for d in adjacent_docs if d.get("source") in valid_sources]
                    documents.extend(valid_adjacent)

            documents.sort(key=lambda x: (x.get("source", ""), x.get("chunk_id") if x.get("chunk_id") is not None else 99999))

        context = self.llm_service.build_context(documents)

        return {
            "creds": creds,
            "documents": documents,
            "context": context,
            "has_docs": True
        }


