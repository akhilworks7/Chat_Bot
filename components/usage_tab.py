import streamlit as st
from app.db.database import get_db
from app.db import crud
from app.services.credential_service import CredentialService


def render_usage_tab(user: dict):
    """
    Renders user's personal usage statistics and infrastructure consumption details.
    """
    user_id = user["id"]

    with get_db() as db:
        usage = crud.get_user_usage_statistics(db=db, user_id=user_id)
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)

    is_byok = creds.get("is_byok", False)
    docs_count = usage.documents_uploaded if usage else 0
    storage_mb = round((usage.storage_used or 0) / (1024 * 1024), 2) if usage else 0.0
    vecs_count = usage.vector_count if usage else 0
    queries_count = usage.query_count if usage else 0
    last_act = usage.last_activity.strftime("%b %d, %Y %H:%M") if (usage and usage.last_activity) else "Never"

    st.markdown("### 📊 My Workspace Usage & Analytics")
    st.caption("Track your indexed document volume, vector representations, and query activity.")

    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Indexed Documents",
            value=f"{docs_count} Docs",
            delta="Unlimited BYOK" if is_byok else f"{allowance.get('used', 0)}/{allowance.get('limit', 2)} Free"
        )
    with col2:
        st.metric(
            label="Vectors in Namespace",
            value=f"{vecs_count:,}",
            delta=f"ns: user_{user_id}"
        )
    with col3:
        st.metric(
            label="RAG Queries Executed",
            value=f"{queries_count:,}",
            delta="Completed"
        )
    with col4:
        st.metric(
            label="Storage Consumed",
            value=f"{storage_mb} MB",
            delta="Raw Files"
        )

    st.markdown("<hr style='margin: 20px 0 24px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    col_details, col_info = st.columns([1, 1])

    with col_details:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 22px; height: 100%; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
            <h4 style="color: #f8fafc; margin-top: 0; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; font-size: 1.15rem;">
                ⚙️ Workspace Configuration
            </h4>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 2.0;">
                <div><b>User ID:</b> <code style="background: rgba(30, 41, 59, 0.8); padding: 2px 8px; border-radius: 6px; color: #a5b4fc;">user_{user_id}</code></div>
                <div><b>Email:</b> <code style="background: rgba(30, 41, 59, 0.8); padding: 2px 8px; border-radius: 6px; color: #cbd5e1;">{user.get('email', '')}</code></div>
                <div><b>Active Mode:</b> <span style="color: {'#34d399' if is_byok else '#38bdf8'}; font-weight: 700;">{'🚀 User Credentials (BYOK)' if is_byok else '🟢 Application Shared'}</span></div>
                <div><b>Pinecone Index:</b> <code style="background: rgba(30, 41, 59, 0.8); padding: 2px 8px; border-radius: 6px; color: #cbd5e1;">{creds.get('pinecone_index', '')}</code></div>
                <div><b>Isolated Namespace:</b> <code style="background: rgba(30, 41, 59, 0.8); padding: 2px 8px; border-radius: 6px; color: #cbd5e1;">{creds.get('namespace', '')}</code></div>
                <div><b>Groq LLM Model:</b> <code style="background: rgba(30, 41, 59, 0.8); padding: 2px 8px; border-radius: 6px; color: #818cf8;">{creds.get('groq_model', '')}</code></div>
                <div><b>Last Workspace Activity:</b> <span style="color: #94a3b8;">{last_act}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 22px; height: 100%; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
            <h4 style="color: #f8fafc; margin-top: 0; margin-bottom: 14px; display: flex; align-items: center; gap: 8px; font-size: 1.15rem;">
                🛡️ Data Isolation & Privacy
            </h4>
            <div style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.9;">
                <div style="margin-bottom: 8px;">🔒 <b>Multi-Tenant Isolation:</b> Your document vectors are stored in an isolated Pinecone namespace strictly segregated from other users.</div>
                <div style="margin-bottom: 8px;">🎯 <b>Grounded Responses:</b> AI answers are derived strictly from your uploaded files with zero hallucination.</div>
                <div>🔑 <b>Encrypted Secrets:</b> Personal API keys are protected at rest using <b>AES-256 GCM</b> encryption.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
