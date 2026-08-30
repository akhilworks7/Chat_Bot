import os
import sys
import time
from pathlib import Path
import streamlit as st

# Configure page layout & metadata
st.set_page_config(
    page_title="DocuMind Multi-User RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load secrets from Streamlit Cloud into environment if available
try:
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        for key in [
            "PINECONE_API_KEY", "GROQ_API_KEY", "PINECONE_INDEX_NAME",
            "PINECONE_ENVIRONMENT", "PINECONE_CLOUD", "DATABASE_URL",
            "ENCRYPTION_KEY", "GROQ_MODEL"
        ]:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
except Exception:
    # No secrets.toml found locally; environment variables and .env are used
    pass

# Ensure local directories exist
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Initialize Database and Seed initial data (cached so it runs only once)
from app.config import settings
from app.db.database import init_db, get_db
from app.db import crud

@st.cache_resource(show_spinner=False)
def setup_database():
    init_db()
    with get_db() as db:
        crud.seed_initial_data(db)
    return True

setup_database()

# Import UI Components
from components import (
    render_auth_ui,
    render_header_ui,
    render_documents_tab,
    render_chatbot_tab,
    render_settings_tab,
    render_usage_tab,
    render_admin_dashboard
)
from app.services.credential_service import CredentialService
from app.services.audit_service import AuditService

# ==========================================
# MODERN CSS THEME & GLASSMORPHISM
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
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


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.nav_page = "workspace"


# ==========================================
# MAIN ROUTER (AUTHENTICATION GUARD)
# ==========================================
if not st.session_state.authenticated or not st.session_state.user:
    render_auth_ui()
else:
    # Current active user
    user = st.session_state.user
    user_id = user["id"]

    # ==========================================
    # SIDEBAR: NAVIGATION & USER PROFILE
    # ==========================================
    with st.sidebar:
        st.markdown("## 🧠 DocuMind **RAG**")
        st.caption("Multi-User AI Knowledge Platform")
        
        st.markdown("---")

        # User Profile card in sidebar
        with get_db() as db:
            user_obj = crud.get_user_by_id(db, user_id)
            if user_obj:
                user = user_obj.to_dict()
                st.session_state.user = user

            allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
            creds_info = CredentialService.get_credentials(user_id=user_id, db=db)

        is_byok = creds_info.get("is_byok", False)

        st.markdown(f"**👤 {user['name']}**")
        st.caption(f"✉️ {user['email']}")

        # Credential Mode & Progress
        if is_byok:
            st.success("🚀 BYOK Mode Active")
        else:
            used_docs = allowance.get("used", 0)
            limit_docs = allowance.get("limit", 2)
            pct = min(1.0, used_docs / limit_docs) if limit_docs > 0 else 0.0
            st.progress(pct, text=f"📄 Free Docs: {used_docs} / {limit_docs}")

        st.markdown("---")
        st.markdown("### 🧭 Navigation")

        # Navigation buttons
        btn_ws = st.button("📂 RAG Workspace", use_container_width=True, type="primary" if st.session_state.nav_page == "workspace" else "secondary")
        if btn_ws:
            st.session_state.nav_page = "workspace"
            st.rerun()

        btn_usage = st.button("📊 My Usage", use_container_width=True, type="primary" if st.session_state.nav_page == "usage" else "secondary")
        if btn_usage:
            st.session_state.nav_page = "usage"
            st.rerun()

        btn_settings = st.button("⚙️ API Settings", use_container_width=True, type="primary" if st.session_state.nav_page == "settings" else "secondary")
        if btn_settings:
            st.session_state.nav_page = "settings"
            st.rerun()

        if user.get("role") == "admin":
            st.markdown("---")
            st.markdown("### 🛡️ Admin Controls")
            btn_admin = st.button("🛡️ Admin Dashboard", use_container_width=True, type="primary" if st.session_state.nav_page == "admin" else "secondary")
            if btn_admin:
                st.session_state.nav_page = "admin"
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Log Out", use_container_width=True):
            with get_db() as db:
                AuditService.log_event(db, action="USER_LOGOUT", user_id=user_id, details=f"User {user['email']} logged out")
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.nav_page = "workspace"
            st.toast("Logged out successfully.")
            st.rerun()

    # ==========================================
    # MAIN APPLICATION VIEW
    # ==========================================
    # Render Global Top Header
    render_header_ui(user)

    # Page Router
    if st.session_state.nav_page == "workspace":
        tab_docs, tab_chat = st.tabs(["📄 Documents", "💬 Chatbot"])
        with tab_docs:
            render_documents_tab(user)
        with tab_chat:
            render_chatbot_tab(user)

    elif st.session_state.nav_page == "settings":
        render_settings_tab(user)

    elif st.session_state.nav_page == "usage":
        render_usage_tab(user)

    elif st.session_state.nav_page == "admin":
        render_admin_dashboard(user)

