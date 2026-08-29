import os
import sys
import time
import shutil
from pathlib import Path
import streamlit as st

# Configure page layout & metadata
st.set_page_config(
    page_title="DocuMind RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load secrets from Streamlit Cloud into environment and settings if available
if hasattr(st, "secrets"):
    for key in ["PINECONE_API_KEY", "GROQ_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_ENVIRONMENT", "PINECONE_CLOUD", "PINECONE_NAMESPACE", "GROQ_MODEL"]:
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])

# Ensure local directories exist
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Import RAG services
from app.config import settings

# Explicitly override settings from st.secrets if present
if hasattr(st, "secrets"):
    if "PINECONE_API_KEY" in st.secrets:
        settings.PINECONE_API_KEY = str(st.secrets["PINECONE_API_KEY"])
    if "GROQ_API_KEY" in st.secrets:
        settings.GROQ_API_KEY = str(st.secrets["GROQ_API_KEY"])
    if "PINECONE_INDEX_NAME" in st.secrets:
        settings.PINECONE_INDEX_NAME = str(st.secrets["PINECONE_INDEX_NAME"])

from app.services.rag_service import RAGService
from app.services.document_service import DocumentService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.registry_service import DocumentRegistryService
from app.utils.logger import get_logger

logger = get_logger("streamlit_app")

# Custom CSS for rich modern aesthetics
st.markdown("""
<style>
    /* Dark glassmorphism & typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }

    .main-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .source-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .source-badge {
        display: inline-block;
        background: #3b82f6;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }

    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize cached services
@st.cache_resource
def get_services():
    return {
        "rag": RAGService(),
        "doc": DocumentService(),
        "chunk": ChunkingService(),
        "embed": EmbeddingService(),
        "vec": VectorService(),
        "reg": DocumentRegistryService()
    }

services = get_services()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# SIDEBAR: DOCUMENT MANAGEMENT & KNOWLEDGE BASE
# ==========================================
with st.sidebar:
    st.markdown("## 🧠 DocuMind **RAG**")
    st.caption("AI-Powered Knowledge Base & Document Chatbot")
    
    st.markdown("---")
    st.subheader("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload PDF for semantic indexing",
        type=["pdf"],
        help="Upload standard or scanned PDFs. Scanned documents will automatically trigger OCR."
    )

    if uploaded_file is not None:
        if st.button("🚀 Upload & Index Document", use_container_width=True, type="primary"):
            save_path = os.path.join("data/documents", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            progress_bar = st.progress(0, text="1/5 Uploading file...")
            start_t = time.time()
            
            try:
                # 1. Extraction
                progress_bar.progress(25, text="2/5 Extracting text (checking searchability & OCR)...")
                text = services["doc"].extract_text(save_path)
                
                # 2. Chunking
                progress_bar.progress(50, text="3/5 Chunking document into semantic blocks...")
                chunks = services["chunk"].split_text(text)
                
                if not chunks:
                    st.error("Could not extract text from document.")
                else:
                    # 3. Embedding & Upserting
                    progress_bar.progress(75, text=f"4/5 Generating vector embeddings for {len(chunks)} chunks...")
                    base_id = Path(uploaded_file.name).stem.replace(" ", "_")
                    
                    total_upserted = 0
                    all_vector_ids = []
                    batch_chunk_size = 40
                    
                    for start_idx in range(0, len(chunks), batch_chunk_size):
                        sub_chunks = chunks[start_idx:start_idx + batch_chunk_size]
                        sub_embeddings = services["embed"].embed_documents(sub_chunks)
                        
                        sub_vectors = []
                        for i, (chunk, embedding) in enumerate(zip(sub_chunks, sub_embeddings)):
                            global_chunk_idx = start_idx + i
                            v_id = f"{base_id}_chunk_{global_chunk_idx}"
                            all_vector_ids.append(v_id)
                            sub_vectors.append({
                                "id": v_id,
                                "values": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                                "metadata": {
                                    "text": chunk,
                                    "chunk_id": global_chunk_idx,
                                    "source": uploaded_file.name
                                }
                            })
                        count = services["vec"].upsert_vectors(sub_vectors, namespace=settings.PINECONE_NAMESPACE)
                        total_upserted += count
                        
                        # Dynamically update progress percentage on screen and in logs
                        pct = min(98, int(50 + (total_upserted / len(chunks)) * 48))
                        progress_bar.progress(pct, text=f"4/5 Embedding & Indexing: {total_upserted}/{len(chunks)} chunks ({pct}%)...")
                        logger.info(f"Indexing progress: {total_upserted}/{len(chunks)} chunks ({pct}%) for '{uploaded_file.name}'")

                    # 4. Registry update
                    progress_bar.progress(100, text="5/5 Indexing completed!")
                    services["reg"].register_document(
                        filename=uploaded_file.name,
                        file_size_bytes=os.path.getsize(save_path),
                        total_chunks=len(chunks),
                        vector_ids=all_vector_ids,
                        namespace=settings.PINECONE_NAMESPACE
                    )
                    
                    elapsed = round(time.time() - start_t, 2)
                    st.success(f"✅ Successfully indexed **{uploaded_file.name}** ({len(chunks)} chunks, {total_upserted} vectors) in {elapsed}s!")
                    time.sleep(1.5)
                    st.rerun()

            except Exception as e:
                st.error(f"Error during ingestion: {str(e)}")

    st.markdown("---")
    st.subheader("📚 Indexed Documents")
    
    docs = services["reg"].get_all_documents()
    if docs:
        for d in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                size_mb = round(d.get("file_size_bytes", 0) / (1024 * 1024), 2)
                st.markdown(f"**{d.get('filename')}**")
                st.caption(f"📦 {d.get('total_chunks', 0)} chunks · {size_mb} MB")
            with col2:
                if st.button("🗑️", key=f"del_{d.get('filename')}", help=f"Delete {d.get('filename')}"):
                    # Delete from Pinecone & local
                    services["vec"].delete_by_source(
                        source_name=d.get("filename"),
                        vector_ids=d.get("vector_ids", []),
                        namespace=settings.PINECONE_NAMESPACE
                    )
                    services["doc"].delete_document_files(d.get("filename"))
                    services["reg"].remove_document(d.get("filename"))
                    st.toast(f"Deleted {d.get('filename')}")
                    st.rerun()
    else:
        st.info("No documents indexed yet. Upload a PDF above.")

    st.markdown("---")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("🟢 Pinecone: " + settings.PINECONE_INDEX_NAME)
    st.caption("⚡ Model: " + settings.GROQ_MODEL)


# ==========================================
# MAIN PAGE: CONVERSATIONAL CHATBOT
# ==========================================
st.markdown("## 💬 Document Intelligence Assistant")
st.markdown("Ask any question regarding your indexed documents. Answers are strictly grounded with semantic source citations.")

# Suggested quick questions if conversation is empty
if len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions:")
    colA, colB = st.columns(2)
    with colA:
        if st.button("🔍 What is Bronchoscopy?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What is Bronchoscopy?"})
            st.rerun()
        if st.button("💊 What are the precautions for Barbiturates?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are the precautions for Barbiturates?"})
            st.rerun()
    with colB:
        if st.button("📑 Summarize the main topics in the document", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Summarize the main topics in the document"})
            st.rerun()
        if st.button("🔬 What are Antiparkinson drugs?", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are Antiparkinson drugs?"})
            st.rerun()

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])} chunks)", expanded=False):
                for s in msg["sources"]:
                    score = round(s.get("score", 0.0) * 100, 1) if s.get("score") else "N/A"
                    chunk_id = s.get("chunk_id", "N/A")
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-badge">📄 {s.get('source', 'Unknown')}</span> · <b>Chunk #{chunk_id}</b> · <i>Score: {score}%</i>
                        <p style="margin-top:6px; color:#cbd5e1;">{s.get('text', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input Handler
if prompt := st.chat_input("Ask anything about your documents..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching Pinecone knowledge base and generating answer..."):
            try:
                res = services["rag"].query(question=prompt)
                answer_text = res.get("answer", "No response generated.")
                sources = res.get("sources", [])
                elapsed_ms = res.get("response_time_ms", 0.0)

                st.markdown(answer_text)
                st.caption(f"⏱️ Generated in {elapsed_ms}ms via Groq LLM & Pinecone")

                if sources:
                    with st.expander(f"📚 Retrieved Sources ({len(sources)} chunks)", expanded=False):
                        for s in sources:
                            score = round(s.get("score", 0.0) * 100, 1) if s.get("score") else "N/A"
                            chunk_id = s.get("chunk_id", "N/A")
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="source-badge">📄 {s.get('source', 'Unknown')}</span> · <b>Chunk #{chunk_id}</b> · <i>Score: {score}%</i>
                                <p style="margin-top:6px; color:#cbd5e1;">{s.get('text', '')}</p>
                            </div>
                            """, unsafe_allow_html=True)

                # Save assistant response in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
