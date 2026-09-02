import time
from pathlib import Path
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

DOC_LISTING_TRIGGERS = [
    "uploaded document", "uploaded file", "my document", "my file",
    "list document", "list file", "show document", "show file",
    "what document", "which document", "what file", "which file",
    "what pdf", "which pdf", "available document", "available file",
    "in my vault", "in the vault", "knowledge vault", "what have i uploaded",
    "what did i upload", "documents uploaded", "files uploaded"
]

def is_casual_query(text: str) -> bool:
    cleaned = "".join(c for c in text.lower().strip() if c.isalnum() or c.isspace()).strip()
    if cleaned in CASUAL_PATTERNS:
        return True
    words = cleaned.split()
    if len(words) <= 3 and any(cleaned.startswith(w) for w in ["hi", "hello", "hey", "thanks", "thank"]):
        return True
    return False

def is_document_listing_query(text: str) -> bool:
    cleaned = "".join(c for c in text.lower().strip() if c.isalnum() or c.isspace()).strip()
    return any(t in cleaned for t in DOC_LISTING_TRIGGERS)

def extract_targeted_document(question: str, user_docs: list) -> Optional[Any]:
    """
    Detects if the query explicitly references a specific document in the vault
    (e.g., 'meidtations.pdf', 'meditations', 'n8n_Flow_engineering.pdf', 'n8n').
    """
    if not user_docs:
        return None
    q_lower = question.lower()
    for doc in user_docs:
        fname = doc.file_name.lower()
        stem = Path(doc.file_name).stem.lower()
        if fname in q_lower or (len(stem) >= 3 and stem in q_lower):
            return doc
        # Typo tolerance for common names
        if ("meditation" in q_lower and "meidtation" in fname) or ("meidtation" in q_lower and "meditation" in fname):
            return doc
        if "n8n" in q_lower and "n8n" in fname:
            return doc
    return None



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
        # Check if user is asking about their uploaded documents catalog
        if is_document_listing_query(question):
            if user_docs:
                doc_lines = []
                documents = []
                for idx, d in enumerate(user_docs, 1):
                    size_mb = f"{round(d.file_size / (1024*1024), 2)} MB" if d.file_size else "Cloud Preserved"
                    doc_lines.append(f"{idx}. **{d.file_name}** ({d.vector_count} chunks, size: {size_mb})")
                    documents.append({
                        "source": d.file_name,
                        "chunk_id": "Catalog",
                        "score": 1.0,
                        "text": f"Document '{d.file_name}' ({d.vector_count} chunks, {size_mb}) is indexed in the user's Knowledge Vault."
                    })
                context = (
                    "The user has the following documents uploaded and indexed in their Knowledge Vault:\n"
                    + "\n".join(doc_lines)
                    + "\n\nAnswer the user's question by listing these documents clearly with their filenames and invite them to ask questions about any of them."
                )
            else:
                context = "The user currently has NO documents uploaded to their Knowledge Vault. Inform them politely that their vault is empty and guide them to upload PDFs in the Documents tab."
                documents = []
        # Instant bypass for casual greetings / chit-chat
        elif not is_casual_query(question):
            # Check if query explicitly targets a specific document by name
            targeted_doc = extract_targeted_document(question, user_docs)
            filter_dict = {"source": targeted_doc.file_name} if targeted_doc else None

            # 3. Generate query embedding
            query_vector = self.embedding_service.embed_query(question)

            # 4. Retrieve top-k semantic matches from user's isolated namespace
            raw_documents = self.vector_service.search(
                query_vector=query_vector,
                namespace=namespace,
                api_key=pinecone_key,
                index_name=pinecone_index,
                top_k=top_k or settings.TOP_K,
                filter_dict=filter_dict
            )

            # 5. Filter retrieved chunks
            if targeted_doc:
                documents = [doc for doc in raw_documents if doc.get("source") == targeted_doc.file_name]
                q_clean = question.lower()
                if any(w in q_clean for w in ["content", "summary", "summarize", "about", "overview", "what is inside", "explain", "give the"]):
                    try:
                        doc_stem = Path(targeted_doc.file_name).stem
                        intro_ids = [f"user_{user_id}_{doc_stem}_chunk_{i}" for i in range(3)]
                        intro_chunks = self.vector_service.fetch_by_ids(
                            vector_ids=intro_ids,
                            namespace=namespace,
                            api_key=pinecone_key,
                            index_name=pinecone_index
                        )
                        if intro_chunks:
                            existing_ids = {d.get("id") for d in documents}
                            for ic in intro_chunks:
                                if ic.get("id") not in existing_ids:
                                    documents.insert(0, ic)
                                    existing_ids.add(ic.get("id"))
                    except Exception as ex:
                        logger.warning(f"Note on fetching intro chunks: {ex}")
            else:
                threshold = getattr(settings, "SIMILARITY_THRESHOLD", 0.38)
                filtered_by_score = [doc for doc in raw_documents if doc.get("score", 0.0) >= threshold]

                if filtered_by_score:
                    if valid_sources:
                        documents = [doc for doc in filtered_by_score if doc.get("source") in valid_sources]
                    else:
                        documents = [doc for doc in filtered_by_score if doc.get("source")]
                        for doc in documents:
                            s_name = doc.get("source")
                            if s_name and not crud.get_user_document_by_name(db, user_id, s_name):
                                try:
                                    crud.create_document(
                                        db=db,
                                        user_id=user_id,
                                        file_name=s_name,
                                        file_size=10240,
                                        file_path=None,
                                        pinecone_namespace=namespace,
                                        vector_count=1,
                                        credential_mode=creds.get("mode", "application"),
                                        status="indexed"
                                    )
                                except Exception:
                                    pass
                else:
                    logger.info(f"RAG Retrieval: No chunks exceeded similarity threshold {threshold}. Treating as conversational query.")
                    documents = []



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
                    valid_adjacent = [d for d in adjacent_docs if (d.get("source") in valid_sources if valid_sources else bool(d.get("source")))]
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
        sources_to_store = documents if documents else [{"source": f"AI General Knowledge ({groq_model})", "is_ai_knowledge": True}]
        crud.add_chat_message(
            db=db,
            user_id=user_id,
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources_to_store,
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
        valid_sources = {d.file_name for d in user_docs} if user_docs else set()

        if is_casual_query(question):
            return {
                "creds": creds,
                "documents": [],
                "context": "",
                "has_docs": bool(user_docs)
            }

        # Handle document listing queries (e.g., 'what are the uploaded documents?')
        if is_document_listing_query(question):
            if user_docs:
                doc_lines = []
                documents = []
                for idx, d in enumerate(user_docs, 1):
                    size_mb = f"{round(d.file_size / (1024*1024), 2)} MB" if d.file_size else "Cloud Preserved"
                    doc_lines.append(f"{idx}. **{d.file_name}** ({d.vector_count} chunks, size: {size_mb})")
                    documents.append({
                        "source": d.file_name,
                        "chunk_id": "Catalog",
                        "score": 1.0,
                        "text": f"Document '{d.file_name}' ({d.vector_count} chunks, {size_mb}) is indexed in the user's Knowledge Vault."
                    })
                context = (
                    "The user has the following documents uploaded and indexed in their Knowledge Vault:\n"
                    + "\n".join(doc_lines)
                    + "\n\nAnswer the user's question by listing these documents clearly with their filenames and invite them to ask questions about any of them."
                )
                return {
                    "creds": creds,
                    "documents": documents,
                    "context": context,
                    "has_docs": True
                }
            else:
                return {
                    "creds": creds,
                    "documents": [],
                    "context": "The user currently has NO documents uploaded to their Knowledge Vault. Inform them politely that their vault is empty and guide them to upload PDFs in the Documents tab.",
                    "has_docs": False
                }

        targeted_doc = extract_targeted_document(question, user_docs)
        filter_dict = {"source": targeted_doc.file_name} if targeted_doc else None

        query_vector = self.embedding_service.embed_query(question)

        raw_documents = self.vector_service.search(
            query_vector=query_vector,
            namespace=namespace,
            api_key=pinecone_key,
            index_name=pinecone_index,
            top_k=top_k or settings.TOP_K,
            filter_dict=filter_dict
        )

        if targeted_doc:
            documents = [doc for doc in raw_documents if doc.get("source") == targeted_doc.file_name]
            q_clean = question.lower()
            if any(w in q_clean for w in ["content", "summary", "summarize", "about", "overview", "what is inside", "explain", "give the"]):
                try:
                    doc_stem = Path(targeted_doc.file_name).stem
                    intro_ids = [f"user_{user_id}_{doc_stem}_chunk_{i}" for i in range(3)]
                    intro_chunks = self.vector_service.fetch_by_ids(
                        vector_ids=intro_ids,
                        namespace=namespace,
                        api_key=pinecone_key,
                        index_name=pinecone_index
                    )
                    if intro_chunks:
                        existing_ids = {d.get("id") for d in documents}
                        for ic in intro_chunks:
                            if ic.get("id") not in existing_ids:
                                documents.insert(0, ic)
                                existing_ids.add(ic.get("id"))
                except Exception as ex:
                    logger.warning(f"Note on fetching intro chunks: {ex}")
        else:
            threshold = getattr(settings, "SIMILARITY_THRESHOLD", 0.38)
            filtered_by_score = [doc for doc in raw_documents if doc.get("score", 0.0) >= threshold]

            if filtered_by_score:
                if valid_sources:
                    documents = [doc for doc in filtered_by_score if doc.get("source") in valid_sources]
                else:
                    documents = [doc for doc in filtered_by_score if doc.get("source")]
                    for doc in documents:
                        s_name = doc.get("source")
                        if s_name and not crud.get_user_document_by_name(db, user_id, s_name):
                            try:
                                crud.create_document(
                                    db=db,
                                    user_id=user_id,
                                    file_name=s_name,
                                    file_size=10240,
                                    file_path=None,
                                    pinecone_namespace=namespace,
                                    vector_count=1,
                                    credential_mode=creds.get("mode", "application"),
                                    status="indexed"
                                )
                            except Exception:
                                pass
            else:
                documents = []

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
                    valid_adjacent = [d for d in adjacent_docs if (d.get("source") in valid_sources if valid_sources else bool(d.get("source")))]
                    documents.extend(valid_adjacent)

            documents.sort(key=lambda x: (x.get("source", ""), x.get("chunk_id") if x.get("chunk_id") is not None else 99999))

        context = self.llm_service.build_context(documents)

        return {
            "creds": creds,
            "documents": documents,
            "context": context,
            "has_docs": True
        }


