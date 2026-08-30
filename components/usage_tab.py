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
            delta=None if is_byok else f"{allowance.get('used', 0)}/{allowance.get('limit', 2)} free limit"
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

    st.markdown("---")

    col_details, col_info = st.columns([1, 1])

    with col_details:
        st.markdown("#### ⚙️ Workspace Configuration")
        st.markdown(f"""
        - **User ID:** `user_{user_id}`
        - **Account Email:** `{user.get('email', '')}`
        - **Active Mode:** `{'🚀 User Credentials (BYOK)' if is_byok else '🟢 Application Credentials (Shared)'}`
        - **Pinecone Index:** `{creds.get('pinecone_index', '')}`
        - **Isolated Namespace:** `{creds.get('namespace', '')}`
        - **Groq LLM Model:** `{creds.get('groq_model', '')}`
        - **Last Workspace Activity:** `{last_act}`
        """)

    with col_info:
        st.markdown("#### 🛡️ Data Isolation & Privacy")
        st.markdown("""
        - **Multi-Tenant Isolation:** Your document vectors are stored exclusively in an isolated namespace inaccessible to other accounts.
        - **Grounded Responses:** Answers are retrieved strictly from your uploaded files with zero halluncinations.
        - **Encrypted Secrets:** Personal API keys are encrypted at rest using AES-256 and are never exposed.
        """)
