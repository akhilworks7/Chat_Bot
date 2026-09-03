import json
import time
import streamlit as st
from app.db.database import get_db
from app.db import crud

from app.services.rag_service import RAGService
from app.services.vector_service import PineconeQuotaException, PineconeAuthException
from app.services.llm_service import GroqQuotaException, GroqAuthException
from app.services.credential_service import CredentialService


def render_source_citations(sources: list, elapsed_ms: float = None, model_name: str = None):
    """
    Renders clean, prominent source document references directly below the chatbot response.
    Explicitly clarifies whether the answer came from uploaded vault documents or the AI model's general knowledge base.
    """
    model_disp = model_name or "Groq LLM"
    doc_sources = [s for s in sources if s.get("source") and not s.get("is_ai_knowledge")] if sources else []

    if not doc_sources:
        st.markdown(f"""
        <div style="margin-top: 10px; margin-bottom: 6px; padding: 10px 14px; background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(234, 179, 8, 0.35); border-radius: 10px;">
            <div style="font-size: 0.76rem; font-weight: 700; color: #fde047; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                <span>🧠</span> Source: AI General Knowledge Base ({model_disp})
            </div>
            <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.45;">
                <b>Origin of information:</b> This answer was generated from the AI model's internal pre-trained knowledge base. <b>No matching uploaded document was found in your Knowledge Vault</b> (the document is either not uploaded or was deleted).
            </div>
        </div>
        """, unsafe_allow_html=True)
        if elapsed_ms:
            st.caption(f"⏱️ Generated in {elapsed_ms}ms via `{model_disp}` (Conversational / AI Knowledge Base)")
        return

    unique_sources = sorted(list({s.get("source", "Document") for s in doc_sources if s.get("source")}))
    
    badges_html = "".join([
        f'<span style="background: rgba(99, 102, 241, 0.18); color: #c7d2fe; border: 1px solid rgba(99, 102, 241, 0.35); padding: 3px 10px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; display: inline-flex; align-items: center; gap: 5px;">📄 {src}</span>'
        for src in unique_sources
    ])

    st.markdown(f"""
    <div style="margin-top: 10px; margin-bottom: 6px; padding: 10px 14px; background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px;">
        <div style="font-size: 0.76rem; font-weight: 700; color: #a5b4fc; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
            <span>📌</span> Sources & References ({len(unique_sources)} document{'' if len(unique_sources) == 1 else 's'}):
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            {badges_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"🔍 Inspect {len(doc_sources)} Retrieved Context Passages & Scores", expanded=False):
        for s in doc_sources:
            score = round(s.get("score", 0.0) * 100, 1) if s.get("score") else "N/A"
            chunk_id = s.get("chunk_id", "N/A")
            st.markdown(f"""
            <div class="source-card">
                <span class="source-badge">📄 {s.get('source', 'Unknown')}</span> · <b>Chunk #{chunk_id}</b> · <i>Relevance: {score}%</i>
                <p style="margin-top:6px; color:#cbd5e1; font-size:0.85rem; line-height:1.5;">{s.get('text', '')}</p>
            </div>
            """, unsafe_allow_html=True)

    if elapsed_ms and model_name:
        st.caption(f"⏱️ Generated in {elapsed_ms}ms via `{model_name}` · Grounded on Pinecone namespace")


def render_chatbot_tab(user: dict):
    """
    Renders the conversational RAG chatbot tab with a stable fixed input box,
    real-time streaming responses, prominent source citations, and friendly quota error handling.
    Fully responsive across mobile, tablet, and desktop viewports.
    """
    user_id = user["id"]
    session_id = f"user_{user_id}_session"

    rag_service = RAGService()

    with get_db() as db:
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        user_docs = crud.get_user_documents(db=db, user_id=user_id)
        db_history = crud.get_chat_history(db=db, user_id=user_id, session_id=session_id, limit=50)

    AVAILABLE_MODELS = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "qwen/qwen3.6-27b",
        "groq/compound-mini"
    ]

    # Initialize selected model in session state
    default_model = creds.get("groq_model", "openai/gpt-oss-20b")
    if "selected_groq_model" not in st.session_state or st.session_state.selected_groq_model not in AVAILABLE_MODELS:
        st.session_state.selected_groq_model = default_model if default_model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]

    # ----------------------------------------------------
    # HEADER & TOOLBAR
    # ----------------------------------------------------
    col_t, col_model, col_b = st.columns([3.2, 1.6, 0.7], vertical_alignment="center")
    with col_t:
        st.markdown("### 💬 AI Knowledge Assistant")
        if user_docs:
            st.caption(f"Grounded on **{len(user_docs)}** indexed document(s) with multi-turn memory.")
        else:
            st.caption("🤖 Conversational Assistant is ready. Upload PDFs in the **📄 Vault** for document QA.")

    with col_model:
        curr_idx = AVAILABLE_MODELS.index(st.session_state.selected_groq_model) if st.session_state.selected_groq_model in AVAILABLE_MODELS else 0
        selected_model = st.selectbox(
            "Select AI Model",
            options=AVAILABLE_MODELS,
            index=curr_idx,
            key="chat_model_selector",
            label_visibility="collapsed",
            help="Select the AI model for answering. Switch models anytime!"
        )
        if selected_model != st.session_state.selected_groq_model:
            st.session_state.selected_groq_model = selected_model
            st.toast(f"Model switched to `{selected_model}`")

    with col_b:
        if st.button("🧹", use_container_width=True, help="Clear conversation history"):
            with get_db() as db:
                crud.clear_chat_history(db=db, user_id=user_id, session_id=session_id)
            st.toast("Chat history cleared.")
            st.rerun()

    # Floating Navigation Buttons
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
    # DEDICATED MESSAGES CONTAINER
    # ----------------------------------------------------
    chat_container = st.container()

    with chat_container:
        if not db_history:
            if user_docs:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.45); border: 1px dashed rgba(99, 102, 241, 0.25); border-radius: 14px; padding: 22px 16px; text-align: center; margin: 12px 0 16px 0;">
                    <div style="font-size: 2.2rem; margin-bottom: 6px;">✨</div>
                    <h4 style="color: #f8fafc; margin-bottom: 4px; font-size: 1.15rem;">How can I assist you today?</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem; max-width: 500px; margin: 0 auto 12px auto; line-height: 1.4;">
                        Ask questions about your uploaded documents, extract key insights, or have casual chit-chat anytime.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='color: #94a3b8; font-size: 0.78rem; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;'>💡 Suggested queries:</div>", unsafe_allow_html=True)
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    if st.button("📋 Summarize docs", use_container_width=True, key="sug_sum"):
                        st.session_state.pending_prompt = "Provide a comprehensive summary of the main topics in the uploaded documents."
                        st.rerun()
                with col_s2:
                    if st.button("🔍 Key takeaways", use_container_width=True, key="sug_takeaways"):
                        st.session_state.pending_prompt = "What are the most important key findings and takeaways from the documents?"
                        st.rerun()
                with col_s3:
                    if st.button("💬 Chit chat", use_container_width=True, key="sug_chitchat"):
                        st.session_state.pending_prompt = "Hello DocuMind AI! How are you doing today?"
                        st.rerun()
            else:
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.45); border: 1px dashed rgba(99, 102, 241, 0.25); border-radius: 14px; padding: 22px 16px; text-align: center; margin: 12px 0 16px 0;">
                    <div style="font-size: 2.2rem; margin-bottom: 6px;">👋</div>
                    <h4 style="color: #f8fafc; margin-bottom: 4px; font-size: 1.15rem;">Welcome to DocuMind AI!</h4>
                    <p style="color: #94a3b8; font-size: 0.88rem; max-width: 500px; margin: 0 auto 12px auto; line-height: 1.4;">
                        You can chit-chat with me casually or upload PDF documents in the <b>📄 Document Vault</b> for deep semantic analysis.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='color: #94a3b8; font-size: 0.78rem; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.04em;'>💡 Suggested queries:</div>", unsafe_allow_html=True)
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    if st.button("👋 Who are you?", use_container_width=True, key="sug_intro"):
                        st.session_state.pending_prompt = "Hello! Who are you and how can you help me?"
                        st.rerun()
                with col_s2:
                    if st.button("✨ What can you do?", use_container_width=True, key="sug_abilities"):
                        st.session_state.pending_prompt = "What features and capabilities do you offer?"
                        st.rerun()
                with col_s3:
                    if st.button("📄 How to upload?", use_container_width=True, key="sug_upload_help"):
                        st.session_state.pending_prompt = "How do I upload and index my PDF documents in DocuMind?"
                        st.rerun()

        for msg in db_history:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
                if msg.role == "assistant":
                    sources = []
                    if msg.sources_json:
                        try:
                            sources = json.loads(msg.sources_json)
                        except Exception:
                            sources = []
                    render_source_citations(sources, elapsed_ms=msg.response_time_ms)

        # Anchor at bottom of messages
        st.markdown("<div id='chat-bottom' style='height: 1px;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # CHAT INPUT (Fixed at bottom)
    # ----------------------------------------------------
    prompt = st.chat_input("Ask a question, chit-chat, or explore your uploaded documents...")
    if not prompt and "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")

    # ----------------------------------------------------
    # PROCESS NEW USER QUERY
    # ----------------------------------------------------
    if prompt:
        with chat_container:
            # 1. Display user query inside the container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Direct DOM auto-scroll to latest query
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

                        active_model = st.session_state.get("selected_groq_model", creds.get("groq_model", "openai/gpt-oss-20b"))

                        # Stream LLM answer tokens in real-time
                        stream_gen = rag_service.llm_service.generate_answer_stream(
                            question=prompt,
                            context=context,
                            api_key=creds["groq_api_key"],
                            model=active_model
                        )
                        full_answer = st.write_stream(stream_gen)
                        full_answer = rag_service.llm_service.clean_answer_boilerplates(full_answer, prompt)

                        elapsed_ms = round((time.time() - start_t) * 1000, 2)
                        render_source_citations(documents, elapsed_ms=elapsed_ms, model_name=active_model)

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
                        sources_to_save = documents if documents else [{"source": f"AI General Knowledge ({active_model})", "is_ai_knowledge": True}]
                        crud.add_chat_message(
                            db=db,
                            user_id=user_id,
                            session_id=session_id,
                            role="assistant",
                            content=full_answer,
                            sources=sources_to_save,
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
