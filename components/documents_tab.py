import os
import time
from pathlib import Path
import streamlit as st

from app.config import settings
from app.db.database import get_db
from app.db import crud
from app.services.credential_service import CredentialService
from app.services.document_service import DocumentService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService, PineconeQuotaException, PineconeAuthException
from app.services.audit_service import AuditService
from app.utils.logger import get_logger
from components.settings_tab import show_settings_dialog

logger = get_logger("documents_tab")


def render_documents_tab(user: dict):
    """
    Renders the document management tab: PDF upload, initial quota limit enforcement,
    celebratory BYOK prompt, indexing pipeline, and document deletion.
    Fully responsive across mobile, tablet, and desktop viewports.
    """
    user_id = user["id"]
    doc_service = DocumentService()
    chunk_service = ChunkingService()
    embed_service = EmbeddingService()
    vec_service = VectorService()

    with get_db() as db:
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        user_docs = crud.get_user_documents(db=db, user_id=user_id)
        max_size_mb = crud.get_int_setting(db=db, key="MAX_UPLOAD_SIZE_MB", default=settings.MAX_UPLOAD_SIZE_MB)

    is_byok = creds.get("is_byok", False)
    allowed = allowance.get("allowed", True)
    used = allowance.get("used", 0)
    limit = allowance.get("limit", 2)

    # ----------------------------------------------------
    # SECTION 1: HERO METRICS & OVERVIEW
    # ----------------------------------------------------
    total_vectors = sum([d.vector_count for d in user_docs])
    total_storage_mb = round(sum([d.file_size for d in user_docs]) / (1024 * 1024), 2)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(label="📄 Indexed Docs", value=f"{len(user_docs)}", delta="Unlimited" if is_byok else f"{used}/{limit} Free")
    with col_m2:
        st.metric(label="📦 Vectors", value=f"{total_vectors:,}", delta=f"ns: user_{user_id}")
    with col_m3:
        st.metric(label="💾 Storage", value=f"{total_storage_mb} MB", delta="Raw PDFs")
    with col_m4:
        st.metric(label="⚡ Engine", value="BYOK" if is_byok else "Shared", delta=creds.get("groq_model", "LLaMA-3.3")[:12])

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # SECTION 2: UPLOAD & INGESTION ZONE
    # ----------------------------------------------------
    if not allowed and not is_byok:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.45), rgba(88, 28, 135, 0.45)); border: 1px solid rgba(168, 85, 247, 0.5); border-radius: 16px; padding: 20px; margin: 10px 0 18px 0; text-align: center; box-shadow: 0 10px 36px rgba(168, 85, 247, 0.2);">
            <div style="font-size: 2.2rem; margin-bottom: 4px;">🎉</div>
            <h3 style="color: #f8fafc; margin-bottom: 6px; font-size: clamp(1.1rem, 3vw, 1.35rem);">You've reached your free document limit!</h3>
            <p style="color: #cbd5e1; max-width: 600px; margin: 0 auto 14px auto; font-size: 0.90rem; line-height: 1.5;">
                You have indexed <b>{used} of {limit}</b> free documents using our shared application pool.
                Connect your personal <b>Pinecone</b> and <b>Groq</b> API keys in Settings to unlock unlimited document capacity.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1.8, 1])
        with col_btn2:
            if st.button("🔑 Configure BYOK API Keys", use_container_width=True, type="primary"):
                show_settings_dialog(user)

        st.markdown("<hr style='margin: 18px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    else:
        # Ingestion Container
        with st.container():
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.55); border: 1px dashed rgba(99, 102, 241, 0.35); border-radius: 14px; padding: 16px 18px; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px; margin-bottom: 6px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.25rem;">📤</span>
                        <span style="font-size: 1.02rem; font-weight: 700; color: #f8fafc;">Upload & Index Knowledge Documents</span>
                    </div>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #a5b4fc; background: rgba(99, 102, 241, 0.18); border: 1px solid rgba(99, 102, 241, 0.3); padding: 2px 8px; border-radius: 16px;">PDF up to 200 MB</span>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                    Upload PDF files to automatically extract text, generate 384-dimensional dense embeddings, and index into your isolated vector namespace.
                </div>
            </div>
            """, unsafe_allow_html=True)

            uploaded_files = st.file_uploader(
                "Drag and drop PDF files here",
                type=["pdf"],
                accept_multiple_files=True,
                help=f"Supported format: PDF. Maximum file size: {max_size_mb} MB per document.",
                label_visibility="collapsed"
            )

            if uploaded_files:
                st.markdown(f"<div style='font-size: 0.88rem; color: #f8fafc; font-weight: 600; margin: 10px 0 6px 0;'>📁 Selected <b>{len(uploaded_files)}</b> document(s) ready for ingestion:</div>", unsafe_allow_html=True)

                if st.button("🚀 Start Indexing Pipeline", type="primary", use_container_width=True):
                    user_doc_dir = doc_service.get_user_doc_dir(user_id)
                    success_count = 0

                    for uploaded_file in uploaded_files:
                        with get_db() as db:
                            curr_allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
                        
                        if not curr_allowance.get("allowed", True):
                            st.warning(f"⚠️ Document limit reached! Skipped '{uploaded_file.name}'. Please connect personal API keys in Settings.")
                            break

                        file_bytes = uploaded_file.getvalue()
                        file_size_mb = len(file_bytes) / (1024 * 1024)
                        if file_size_mb > max_size_mb:
                            st.error(f"❌ File '{uploaded_file.name}' ({file_size_mb:.1f} MB) exceeds maximum upload limit of {max_size_mb} MB.")
                            continue

                        safe_filename = uploaded_file.name.replace(" ", "_")
                        save_path = os.path.join(user_doc_dir, safe_filename)
                        with open(save_path, "wb") as f:
                            f.write(file_bytes)

                        st.markdown(f"#### ⚙️ Indexing: `{safe_filename}`")
                        progress_bar = st.progress(0, text="1/5 Uploading and validating file...")
                        start_t = time.time()

                        try:
                            # 1. Text Extraction
                            progress_bar.progress(25, text="2/5 Extracting text layer and running OCR if required...")
                            text = doc_service.extract_text(save_path)

                            if not text or not text.strip():
                                st.error(f"Could not extract readable text from '{safe_filename}'.")
                                continue

                            # 2. Chunking
                            progress_bar.progress(50, text="3/5 Splitting document into semantic chunks...")
                            chunks = chunk_service.split_text(text)

                            if not chunks:
                                st.error("No valid text chunks generated.")
                                continue

                            # 3. Embedding and Upserting
                            progress_bar.progress(70, text=f"4/5 Generating vector embeddings for {len(chunks)} chunks...")
                            base_id = Path(safe_filename).stem
                            namespace = creds["namespace"]
                            pinecone_key = creds["pinecone_api_key"]
                            pinecone_index = creds["pinecone_index"]

                            total_upserted = 0
                            all_vector_ids = []
                            batch_chunk_size = 64

                            for start_idx in range(0, len(chunks), batch_chunk_size):
                                sub_chunks = chunks[start_idx : start_idx + batch_chunk_size]
                                sub_embeddings = embed_service.embed_documents(sub_chunks)

                                sub_vectors = []
                                for i, (chunk, embedding) in enumerate(zip(sub_chunks, sub_embeddings)):
                                    global_idx = start_idx + i
                                    v_id = f"user_{user_id}_{base_id}_chunk_{global_idx}"
                                    all_vector_ids.append(v_id)
                                    sub_vectors.append({
                                        "id": v_id,
                                        "values": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                                        "metadata": {
                                            "text": chunk,
                                            "chunk_id": global_idx,
                                            "source": safe_filename,
                                            "user_id": user_id
                                        }
                                    })

                                count = vec_service.upsert_vectors(
                                    vectors=sub_vectors,
                                    namespace=namespace,
                                    api_key=pinecone_key,
                                    index_name=pinecone_index
                                )
                                total_upserted += count
                                pct = min(98, int(50 + (total_upserted / len(chunks)) * 48))
                                progress_bar.progress(pct, text=f"4/5 Indexing: {total_upserted}/{len(chunks)} chunks ({pct}%)...")

                            # 4. Save to Database
                            progress_bar.progress(100, text="5/5 Finalizing document registration...")
                            with get_db() as db:
                                existing_doc = crud.get_user_document_by_name(db, user_id, safe_filename)
                                if existing_doc:
                                    existing_doc.vector_count = total_upserted
                                    existing_doc.file_size = len(file_bytes)
                                    existing_doc.credential_mode = creds["mode"]
                                    existing_doc.status = "indexed"
                                else:
                                    crud.create_document(
                                        db=db,
                                        user_id=user_id,
                                        file_name=safe_filename,
                                        file_size=len(file_bytes),
                                        file_path=save_path,
                                        pinecone_namespace=namespace,
                                        vector_count=total_upserted,
                                        credential_mode=creds["mode"],
                                        status="indexed"
                                    )
                                AuditService.log_event(
                                    db=db,
                                    action="DOCUMENT_INDEXED",
                                    user_id=user_id,
                                    details=f"Indexed {safe_filename} ({total_upserted} vectors) in mode {creds['mode']}"
                                )

                            elapsed = round(time.time() - start_t, 2)
                            st.success(f"✅ Successfully indexed **{safe_filename}** ({len(chunks)} chunks, {total_upserted} vectors) in {elapsed}s!")
                            success_count += 1

                        except PineconeQuotaException as qe:
                            st.error(str(qe))
                            break
                        except PineconeAuthException as ae:
                            st.error(str(ae))
                            break
                        except Exception as e:
                            st.error(f"Ingestion error for '{safe_filename}': {str(e)}")

                    if success_count > 0:
                        time.sleep(1.0)
                        st.rerun()

    # ----------------------------------------------------
    # SECTION 3: INDEXED DOCUMENTS VAULT
    # ----------------------------------------------------
    st.markdown("<hr style='margin: 20px 0 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    
    col_v_title, col_v_purge, col_v_search = st.columns([3.5, 1.8, 2.2], vertical_alignment="center")
    with col_v_title:
        st.markdown(f"### 📚 Knowledge Vault <span style='font-size:0.85rem; color:#94a3b8; font-weight:400;'>({len(user_docs)} files)</span>", unsafe_allow_html=True)
    
    with col_v_purge:
        if st.button("🧹 Purge Vectors", key="btn_purge_ns", use_container_width=True, help="Wipe stale vectors from your Pinecone namespace"):
            try:
                vec_service.delete_namespace(
                    namespace=creds["namespace"],
                    api_key=creds["pinecone_api_key"],
                    index_name=creds["pinecone_index"]
                )
                with get_db() as db:
                    AuditService.log_event(
                        db=db,
                        action="NAMESPACE_PURGED",
                        user_id=user_id,
                        details=f"Purged vector namespace {creds['namespace']}"
                    )
                st.toast("🧹 Vector namespace purged successfully!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Namespace purge error: {e}")

    with col_v_search:
        search_query = st.text_input("🔍 Search files", placeholder="Filter documents...", label_visibility="collapsed")

    filtered_docs = [d for d in user_docs if search_query.lower() in d.file_name.lower()] if search_query else user_docs

    if not user_docs:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.4); border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 14px; padding: 30px 16px; text-align: center; margin: 14px 0;">
            <div style="font-size: 2.4rem; margin-bottom: 6px;">📂</div>
            <h4 style="color: #f8fafc; margin-bottom: 4px; font-size: 1.1rem;">No Documents Indexed Yet</h4>
            <p style="color: #94a3b8; font-size: 0.86rem; max-width: 480px; margin: 0 auto;">
                Upload your first PDF document above to start indexing knowledge for the AI Chatbot.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif search_query and not filtered_docs:
        st.info(f"No documents match '{search_query}'.")
    else:
        for doc in filtered_docs:
            size_mb = round(doc.file_size / (1024 * 1024), 2)
            created_str = doc.created_at.strftime("%b %d, %Y") if doc.created_at else "N/A"
            is_doc_byok = doc.credential_mode == "byok"
            mode_badge = "🚀 BYOK" if is_doc_byok else "🟢 Shared"
            mode_style = "background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35);" if is_doc_byok else "background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35);"

            col_card_info, col_card_dl, col_card_del = st.columns([5.2, 1.1, 1.1], vertical_alignment="center")

            with col_card_info:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px 16px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px;">
                        <div style="font-size: 0.94rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 8px; word-break: break-word;">
                            <span>📄</span> {doc.file_name}
                        </div>
                        <span style="{mode_style} font-size: 0.70rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;">
                            {mode_badge}
                        </span>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; font-size: 0.76rem; color: #94a3b8;">
                        <span>📦 <b>{doc.vector_count}</b> chunks</span>
                        <span>💾 <b>{size_mb}</b> MB</span>
                        <span>📅 {created_str}</span>
                        <span>🏷️ <code style="color: #cbd5e1; font-size: 0.70rem; background: rgba(30, 41, 59, 0.8); padding: 1px 5px; border-radius: 4px;">{doc.pinecone_namespace}</code></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_card_dl:
                file_exists = os.path.exists(doc.file_path) if doc.file_path else False
                if file_exists:
                    with open(doc.file_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="📥 Download",
                        data=file_data,
                        file_name=doc.file_name,
                        mime="application/pdf",
                        key=f"dl_doc_{doc.id}",
                        help=f"Download {doc.file_name}",
                        use_container_width=True
                    )
                else:
                    st.button("📥 N/A", key=f"dl_disabled_{doc.id}", disabled=True, use_container_width=True)

            with col_card_del:
                if st.button("🗑️ Delete", key=f"del_doc_{doc.id}", help=f"Delete {doc.file_name}", use_container_width=True):
                    # 1. Delete vectors from Pinecone using deterministic IDs
                    try:
                        vec_service.delete_document_vectors(
                            user_id=user_id,
                            file_name=doc.file_name,
                            vector_count=doc.vector_count,
                            namespace=doc.pinecone_namespace,
                            api_key=creds["pinecone_api_key"],
                            index_name=creds["pinecone_index"]
                        )
                    except Exception as ex:
                        logger.warning(f"Pinecone vector delete warning: {ex}")

                    # 2. Delete physical files from disk
                    doc_service.delete_document_files(file_path=doc.file_path, filename=doc.file_name)

                    # 3. Delete from DB
                    with get_db() as db:
                        crud.delete_document(db=db, doc_id=doc.id)
                        remaining = crud.get_user_documents(db=db, user_id=user_id)
                        
                        # If user has 0 remaining documents, wipe entire namespace for clean slate
                        if not remaining:
                            try:
                                vec_service.delete_namespace(
                                    namespace=doc.pinecone_namespace,
                                    api_key=creds["pinecone_api_key"],
                                    index_name=creds["pinecone_index"]
                                )
                            except Exception as ex:
                                logger.warning(f"Namespace wipe warning: {ex}")

                        AuditService.log_event(
                            db=db,
                            action="DOCUMENT_DELETED",
                            user_id=user_id,
                            details=f"Deleted {doc.file_name}"
                        )

                    st.toast(f"Deleted {doc.file_name}")
                    time.sleep(0.5)
                    st.rerun()

            st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)
