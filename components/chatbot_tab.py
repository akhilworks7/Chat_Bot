import json
import time
import streamlit as st
from app.db.database import get_db
from app.db import crud
from app.services.rag_service import RAGService
from app.services.vector_service import PineconeQuotaException, PineconeAuthException
from app.services.llm_service import GroqQuotaException, GroqAuthException
from app.services.credential_service import CredentialService


def render_chatbot_tab(user: dict):
    """
    Renders the conversational RAG chatbot tab with a stable fixed input box,
    real-time streaming responses, source citation cards, and friendly quota error handling.
    """
    user_id = user["id"]
    session_id = f"user_{user_id}_session"

    rag_service = RAGService()

    with get_db() as db:
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        user_docs = crud.get_user_documents(db=db, user_id=user_id)
        db_history = crud.get_chat_history(db=db, user_id=user_id, session_id=session_id, limit=50)

    # ----------------------------------------------------
    # HEADER & TOOLBAR
    # ----------------------------------------------------
    col_t, col_b = st.columns([4, 1])
    with col_t:
        st.markdown("### 💬 Document Intelligence Assistant")
        st.caption(f"Semantic search & QA strictly grounded on your **{len(user_docs)}** indexed document(s). (Model: `{creds['groq_model']}`)")
    with col_b:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clear current conversation history"):
            with get_db() as db:
                crud.clear_chat_history(db=db, user_id=user_id, session_id=session_id)
            st.toast("Chat history cleared.")
            st.rerun()

    if not user_docs:
        st.warning("⚠️ You haven't indexed any documents yet! Please upload a PDF in the **📄 Documents** tab to start asking questions.")

    
    # ----------------------------------------------------
    # DEDICATED MESSAGES CONTAINER (Always above input box)
    # ----------------------------------------------------
    chat_container = st.container()

    with chat_container:
        for msg in db_history:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
                if msg.role == "assistant" and msg.sources_json:
                    try:
                        sources = json.loads(msg.sources_json)
                        if sources:
                            with st.expander(f"📚 Retrieved Sources ({len(sources)} chunks)", expanded=False):
                                for s in sources:
                                    score = round(s.get("score", 0.0) * 100, 1) if s.get("score") else "N/A"
                                    chunk_id = s.get("chunk_id", "N/A")
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <span class="source-badge">📄 {s.get('source', 'Unknown')}</span> · <b>Chunk #{chunk_id}</b> · <i>Score: {score}%</i>
                                        <p style="margin-top:6px; color:#cbd5e1; font-size:0.87rem; line-height:1.5;">{s.get('text', '')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    except Exception:
                        pass

    # ----------------------------------------------------
    # CHAT INPUT (Fixed at the bottom)
    # ----------------------------------------------------
    prompt = st.chat_input("Ask a question about your uploaded documents...")
    if not prompt and "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    # ----------------------------------------------------
    # PROCESS NEW USER QUERY (Rendered inside chat_container)
    # ----------------------------------------------------
    if prompt:
        with chat_container:
            # 1. Display user query inside the container
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Assistant response generation inside the container
            with st.chat_message("assistant"):
                start_t = time.time()
                try:
                    with get_db() as db:
                        # Retrieve context & sources
                        retrieval = rag_service.retrieve_context(
                            question=prompt,
                            user_id=user_id,
                            db=db
                        )
                        creds = retrieval["creds"]
                        documents = retrieval["documents"]
                        context = retrieval["context"]

                        # Stream LLM answer tokens in real-time
                        stream_gen = rag_service.llm_service.generate_answer_stream(
                            question=prompt,
                            context=context,
                            api_key=creds["groq_api_key"],
                            model=creds["groq_model"]
                        )
                        full_answer = st.write_stream(stream_gen)

                        elapsed_ms = round((time.time() - start_t) * 1000, 2)
                        st.caption(f"⏱️ Generated in {elapsed_ms}ms via `{creds['groq_model']}` & Pinecone")

                        # Display source citations
                        if documents:
                            with st.expander(f"📚 Retrieved Sources ({len(documents)} chunks)", expanded=False):
                                for s in documents:
                                    score = round(s.get("score", 0.0) * 100, 1) if s.get("score") else "N/A"
                                    chunk_id = s.get("chunk_id", "N/A")
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <span class="source-badge">📄 {s.get('source', 'Unknown')}</span> · <b>Chunk #{chunk_id}</b> · <i>Score: {score}%</i>
                                        <p style="margin-top:6px; color:#cbd5e1; font-size:0.87rem; line-height:1.5;">{s.get('text', '')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # Record in database
                        crud.add_chat_message(
                            db=db,
                            user_id=user_id,
                            session_id=session_id,
                            role="user",
                            content=prompt
                        )
                        crud.add_chat_message(
                            db=db,
                            user_id=user_id,
                            session_id=session_id,
                            role="assistant",
                            content=full_answer,
                            sources=documents,
                            response_time_ms=elapsed_ms
                        )

                except PineconeQuotaException as qe:
                    st.error(str(qe))
                except PineconeAuthException as ae:
                    st.error(str(ae))
                except GroqQuotaException as ge:
                    st.error(str(ge))
                except GroqAuthException as gae:
                    st.error(str(gae))
                except Exception as e:
                    st.error(f"Error answering question: {str(e)}")
