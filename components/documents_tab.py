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

logger = get_logger("documents_tab")


def render_documents_tab(user: dict):
    """
    Renders the document management tab: PDF upload, initial quota limit enforcement,
    celebratory BYOK prompt, indexing pipeline, and document deletion.
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
    # SECTION 1: UPLOAD & QUOTA MANAGEMENT
    # ----------------------------------------------------
    st.markdown("### 📄 Document Upload & Management")
    st.caption("Upload your PDFs to automatically parse, chunk, embed, and index them into your isolated vector workspace.")

    # If limit reached on Application credentials, show upgrade banner
    if not allowed and not is_byok:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.5), rgba(88, 28, 135, 0.5)); border: 1px solid #a855f7; border-radius: 12px; padding: 24px; margin: 16px 0 24px 0; text-align: center;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🎉</div>
            <h3 style="color: #f8fafc; margin-bottom: 8px;">You've reached the free document limit!</h3>
            <p style="color: #cbd5e1; max-width: 600px; margin: 0 auto 16px auto; font-size: 0.95rem; line-height: 1.6;">
                You have successfully indexed <b>{used} of {limit}</b> free documents using our shared infrastructure.
                To continue uploading and processing more documents without limits, connect your own <b>Pinecone</b> and <b>Groq</b> API keys.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔑 Configure BYOK API Keys", use_container_width=True, type="primary"):
                st.session_state.nav_page = "settings"
                st.rerun()

        st.markdown("---")

    else:
        # User is either BYOK or still has free quota left
        if is_byok:
            st.info("🚀 **BYOK Mode Active:** You have unlimited document uploads powered by your own Pinecone and Groq accounts.")
        else:
            st.markdown(f"""
            <div style="background: rgba(14, 165, 233, 0.1); border-left: 4px solid #0ea5e9; padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 0.9rem;">
                🟢 <b>Shared Application Mode:</b> Using <b>{used} / {limit}</b> free document slots. Connect your own API keys in Settings at any time for unlimited documents.
            </div>
            """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Select PDF documents to index",
            type=["pdf"],
            accept_multiple_files=True,
            help=f"Supported format: PDF. Maximum file size: {max_size_mb} MB."
        )

        if uploaded_files:
            st.write(f"📁 Selected **{len(uploaded_files)}** file(s) for indexing:")

            if st.button("🚀 Upload & Index Selected Documents", type="primary", use_container_width=True):
                user_doc_dir = doc_service.get_user_doc_dir(user_id)
                success_count = 0

                for uploaded_file in uploaded_files:
                    # Check quota before each file in batch if on app credentials
                    with get_db() as db:
                        curr_allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
                    
                    if not curr_allowance.get("allowed", True):
                        st.warning(f"⚠️ Document limit reached! Skipped '{uploaded_file.name}'. Please connect your API keys in Settings.")
                        break

                    # Check file size
                    file_bytes = uploaded_file.getvalue()
                    file_size_mb = len(file_bytes) / (1024 * 1024)
                    if file_size_mb > max_size_mb:
                        st.error(f"❌ File '{uploaded_file.name}' ({file_size_mb:.1f} MB) exceeds maximum upload limit of {max_size_mb} MB.")
                        continue

                    # Save file locally
                    safe_filename = uploaded_file.name.replace(" ", "_")
                    save_path = os.path.join(user_doc_dir, safe_filename)
                    with open(save_path, "wb") as f:
                        f.write(file_bytes)

                    st.markdown(f"#### ⚙️ Processing: `{safe_filename}`")
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

                        # 3. Embedding and Upserting to User Namespace
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
                            # Check if document record already existed
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
                    time.sleep(1.2)
                    st.rerun()

    # ----------------------------------------------------
    # SECTION 2: INDEXED DOCUMENTS TABLE
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader(f"📚 Your Indexed Documents ({len(user_docs)})")

    if not user_docs:
        st.info("No documents indexed in your workspace yet. Upload a PDF above to get started!")
    else:
        for doc in user_docs:
            size_mb = round(doc.file_size / (1024 * 1024), 2)
            created_str = doc.created_at.strftime("%b %d, %Y %H:%M") if doc.created_at else "N/A"
            mode_badge = "🟢 Application" if doc.credential_mode == "application" else "🚀 BYOK"
            mode_style = "color: #38bdf8;" if doc.credential_mode == "application" else "color: #a855f7;"

            col_info, col_mode, col_dl, col_del = st.columns([5, 2, 1, 1])

            with col_info:
                st.markdown(f"**📄 {doc.file_name}**")
                st.caption(f"📦 {doc.vector_count} chunks · {size_mb} MB · Indexed on {created_str}")

            with col_mode:
                st.markdown(f"<span style='{mode_style} font-weight: 600; font-size: 0.85rem;'>{mode_badge}</span>", unsafe_allow_html=True)
                st.caption(f"Namespace: `{doc.pinecone_namespace}`")

            with col_dl:
                file_exists = os.path.exists(doc.file_path) if doc.file_path else False
                if file_exists:
                    with open(doc.file_path, "rb") as f:
                        file_data = f.read()
                    st.download_button(
                        label="📥",
                        data=file_data,
                        file_name=doc.file_name,
                        mime="application/pdf",
                        key=f"dl_doc_{doc.id}",
                        help=f"Download {doc.file_name}"
                    )
                else:
                    st.button("📥", key=f"dl_disabled_{doc.id}", disabled=True, help="File not available on disk")

            with col_del:
                if st.button("🗑️", key=f"del_doc_{doc.id}", help=f"Delete {doc.file_name}"):
                    # 1. Delete vectors from Pinecone
                    try:
                        vec_service.delete_by_source(
                            source_name=doc.file_name,
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
                        AuditService.log_event(
                            db=db,
                            action="DOCUMENT_DELETED",
                            user_id=user_id,
                            details=f"Deleted {doc.file_name}"
                        )

                    st.toast(f"Deleted {doc.file_name}")
                    time.sleep(0.8)
                    st.rerun()

            st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
