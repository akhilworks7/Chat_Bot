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

    AVAILABLE_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
        "groq/compound-mini"
    ]

    # Initialize selected model in session state
    default_model = creds.get("groq_model", "openai/gpt-oss-120b")
    if "selected_groq_model" not in st.session_state or st.session_state.selected_groq_model not in AVAILABLE_MODELS:
        st.session_state.selected_groq_model = default_model if default_model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]


    # ----------------------------------------------------
    # HEADER & TOOLBAR
    # ----------------------------------------------------
    col_t, col_model, col_b = st.columns([2.5, 1.4, 0.8], vertical_alignment="center")
    with col_t:
        st.markdown("### 💬 Document Intelligence & Chit Chat Assistant")
        if user_docs:
            st.caption(f"Semantic search & QA strictly grounded on **{len(user_docs)}** document(s) + General Chit Chat.")
        else:
            st.caption("🤖 Conversational Chit Chat is ready. Upload PDFs in the **📄 Document Vault** for document QA.")

    with col_model:
        curr_idx = AVAILABLE_MODELS.index(st.session_state.selected_groq_model) if st.session_state.selected_groq_model in AVAILABLE_MODELS else 0
        selected_model = st.selectbox(
            "Select AI Model",
            options=AVAILABLE_MODELS,
            index=curr_idx,
            key="chat_model_selector",
            label_visibility="collapsed",
            help="Select the AI model for answering. If you hit a rate limit or want faster responses, switch models anytime!"
        )
        if selected_model != st.session_state.selected_groq_model:
            st.session_state.selected_groq_model = selected_model
            st.toast(f"Model switched to `{selected_model}`")

    with col_b:
        if st.button("🧹 Clear", use_container_width=True, help="Clear current conversation history"):
            with get_db() as db:
                crud.clear_chat_history(db=db, user_id=user_id, session_id=session_id)
            st.toast("Chat history cleared.")
            st.rerun()

    # Floating Navigation Buttons:
    # 1. Floating Top-Right Scroll-To-Bottom Button (Visible when near top, glides down to latest message)
    # 2. Floating Bottom-Right Scroll-To-Top Button (Visible when scrolled down, glides up to header)
    st.markdown("""
    <a href="#chat-bottom" onclick="
        try {
            var targets = [
                document.querySelector('section.main'),
                document.querySelector('[data-testid=\\'stMain\\']'),
                document.querySelector('.main'),
                document.querySelector('[data-testid=\\'stAppViewContainer\\']'),
                document.documentElement,
                document.body,
                window
            ];
            targets.forEach(function(t) {
                if (t) {
                    t.scrollTop = t.scrollHeight || 9999999;
                    if (t.scrollTo) {
                        t.scrollTo({ top: t.scrollHeight || 9999999, behavior: 'smooth' });
                    }
                }
            });
        } catch(e) {}
    " class="scroll-bottom-btn" title="Scroll to Latest Answer">
        <span>⬇️</span>
        <span>Bottom</span>
    </a>

    <a href="#page-top" onclick="
        try {
            var targets = [
                document.querySelector('section.main'),
                document.querySelector('[data-testid=\\'stMain\\']'),
                document.querySelector('.main'),
                document.querySelector('[data-testid=\\'stAppViewContainer\\']'),
                document.documentElement,
                document.body,
                window
            ];
            targets.forEach(function(t) {
                if (t) {
                    t.scrollTop = 0;
                    if (t.scrollTo) {
                        t.scrollTo({ top: 0, behavior: 'smooth' });
                    }
                }
            });
        } catch(e) {}
    " class="scroll-top-btn" title="Scroll to Top of Page">
        <span>⬆️</span>
        <span>Top</span>
    </a>
    """, unsafe_allow_html=True)


    # ----------------------------------------------------
    # DEDICATED MESSAGES CONTAINER (Always above input box)
    # ----------------------------------------------------
    chat_container = st.container()

    with chat_container:
        if not db_history:
            if user_docs:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 24px; text-align: center; margin: 20px 0;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">✨</div>
                    <h4 style="color: #f8fafc; margin-bottom: 6px;">How can I assist you today?</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem; max-width: 520px; margin: 0 auto 16px auto;">
                        Ask questions about your uploaded documents, extract insights, or have casual chit-chat anytime.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 8px;'>💡 Suggested queries:</div>", unsafe_allow_html=True)
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    if st.button("📋 Summarize all documents", use_container_width=True, key="sug_sum"):
                        st.session_state.pending_prompt = "Provide a comprehensive summary of the main topics in the uploaded documents."
                        st.rerun()
                with col_s2:
                    if st.button("🔍 What are the key takeaways?", use_container_width=True, key="sug_takeaways"):
                        st.session_state.pending_prompt = "What are the most important key findings and takeaways from the documents?"
                        st.rerun()
                with col_s3:
                    if st.button("💬 How are you doing today?", use_container_width=True, key="sug_chitchat"):
                        st.session_state.pending_prompt = "Hello DocuMind AI! How are you doing today?"
                        st.rerun()
            else:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255, 255, 255, 0.12); border-radius: 12px; padding: 24px; text-align: center; margin: 20px 0;">
                    <div style="font-size: 2rem; margin-bottom: 8px;">👋</div>
                    <h4 style="color: #f8fafc; margin-bottom: 6px;">Welcome to DocuMind AI!</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem; max-width: 520px; margin: 0 auto 16px auto;">
                        You can chit-chat with me casually or upload PDF documents in the <b>📄 Document Vault</b> for deep semantic analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='color: #94a3b8; font-size: 0.82rem; font-weight: 600; margin-bottom: 8px;'>💡 Suggested queries:</div>", unsafe_allow_html=True)
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    if st.button("👋 Hello, who are you?", use_container_width=True, key="sug_intro"):
                        st.session_state.pending_prompt = "Hello! Who are you and how can you help me?"
                        st.rerun()
                with col_s2:
                    if st.button("✨ What can you do?", use_container_width=True, key="sug_abilities"):
                        st.session_state.pending_prompt = "What features and capabilities do you offer?"
                        st.rerun()
                with col_s3:
                    if st.button("📄 How do I upload documents?", use_container_width=True, key="sug_upload_help"):
                        st.session_state.pending_prompt = "How do I upload and index my PDF documents in DocuMind?"
                        st.rerun()

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

        # Anchor at bottom of messages
        st.markdown("<div id='chat-bottom' style='height: 1px;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # CHAT INPUT (Fixed at the bottom)
    # ----------------------------------------------------
    prompt = st.chat_input("Ask a question, chit-chat, or explore your uploaded documents...")
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

            # Direct DOM auto-scroll to latest query (No iframe, runs in main window)
            st.markdown("""
            <img src="data:image/svg+xml;utf8,<svg></svg>" style="display:none;" onerror="
                (function() {
                    function glideToBottom() {
                        try {
                            var targets = [
                                document.querySelector('section.main'),
                                document.querySelector('[data-testid=\\'stMain\\']'),
                                document.querySelector('.main'),
                                document.querySelector('[data-testid=\\'stAppViewContainer\\']'),
                                document.documentElement,
                                document.body,
                                window
                            ];
                            targets.forEach(function(t) {
                                if (t) {
                                    t.scrollTop = t.scrollHeight || 9999999;
                                    if (t.scrollTo) {
                                        t.scrollTo({ top: t.scrollHeight || 9999999, behavior: 'smooth' });
                                    }
                                }
                            });
                        } catch(e) {}
                    }
                    glideToBottom();
                    setTimeout(glideToBottom, 100);
                    setTimeout(glideToBottom, 350);
                })();
            " />
            """, unsafe_allow_html=True)

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
                        documents = retrieval.get("documents", [])
                        context = retrieval.get("context", "")

                        active_model = st.session_state.get("selected_groq_model", creds.get("groq_model", "openai/gpt-oss-120b"))

                        # Stream LLM answer tokens in real-time
                        stream_gen = rag_service.llm_service.generate_answer_stream(
                            question=prompt,
                            context=context,
                            api_key=creds["groq_api_key"],
                            model=active_model
                        )
                        full_answer = st.write_stream(stream_gen)

                        elapsed_ms = round((time.time() - start_t) * 1000, 2)
                        
                        if documents:
                            st.caption(f"⏱️ Generated in {elapsed_ms}ms via `{active_model}` & Pinecone ({len(documents)} source chunks)")
                            # Display source citations
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
                        else:
                            st.caption(f"⏱️ Generated in {elapsed_ms}ms via `{active_model}` (Conversational Response)")

                        # Final smooth scroll after response stream completes
                        st.markdown("""
                        <img src="data:image/svg+xml;utf8,<svg></svg>" style="display:none;" onerror="
                            (function() {
                                function glideToBottom() {
                                    try {
                                        var targets = [
                                            document.querySelector('section.main'),
                                            document.querySelector('[data-testid=\\'stMain\\']'),
                                            document.querySelector('.main'),
                                            document.querySelector('[data-testid=\\'stAppViewContainer\\']'),
                                            document.documentElement,
                                            document.body,
                                            window
                                        ];
                                        targets.forEach(function(t) {
                                            if (t) {
                                                t.scrollTop = t.scrollHeight || 9999999;
                                                if (t.scrollTo) {
                                                    t.scrollTo({ top: t.scrollHeight || 9999999, behavior: 'smooth' });
                                                }
                                            }
                                        });
                                    } catch(e) {}
                                }
                                setTimeout(glideToBottom, 100);
                            })();
                        " />
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


