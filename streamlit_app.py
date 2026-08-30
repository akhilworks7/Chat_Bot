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
    initial_sidebar_state="collapsed"
)


# Load secrets from Streamlit Cloud into environment if available
try:
    if hasattr(st, "secrets") and len(st.secrets) > 0:
        for k, v in st.secrets.items():
            if isinstance(v, (str, int, float, bool)):
                os.environ[k.upper()] = str(v)
            elif hasattr(v, "items"):
                for sub_k, sub_v in v.items():
                    os.environ[sub_k.upper()] = str(sub_v)
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
def setup_system():
    init_db()
    with get_db() as db:
        crud.seed_initial_data(db)
    
    # Non-blocking background warmup so initial page load renders instantly (< 20ms)
    try:
        from app.services.embedding_service import EmbeddingService
        EmbeddingService.start_background_warmup()
    except Exception:
        pass
    return True

setup_system()



# Import UI Components
from components import (
    render_auth_ui,
    render_header_ui,
    render_documents_tab,
    render_chatbot_tab,
    render_settings_tab,
    show_settings_dialog,
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
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Streamlit header transparency and pointer-events handling */
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
        z-index: 90 !important;
    }

    /* Hide Streamlit dev toolbar from covering top buttons */
    [data-testid="stToolbar"] {
        display: none !important;
        pointer-events: none !important;
    }

    /* Hide the unused sidebar and its toggle controls completely */
    section[data-testid="stSidebar"],
    button[kind="header"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Popover dropdown menu styling */
    div[data-testid="stPopoverBody"] {
        background: #0f172a !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
        padding: 12px !important;
    }

    /* Main Content Container Spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }


    /* Ensure all Streamlit buttons have a uniform, solid, full-surface click area */
    div.stButton > button {
        position: relative !important;
        z-index: 20 !important;
        pointer-events: auto !important;
        cursor: pointer !important;
        width: 100% !important;
        min-height: 2.5rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Modern Card & Container Styling */
    .stCard, .source-card {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
    }

    .source-badge {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 6px;
        letter-spacing: 0.01em;
    }

    /* Chat Messages Polish */
    .stChatMessage {
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    /* Metric cards glow */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4) !important;
    }

    /* Floating Scroll-to-Top and Scroll-to-Bottom Buttons */
    .scroll-top-btn, .scroll-bottom-btn {
        position: fixed !important;
        right: 28px !important;
        z-index: 99999 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        background: rgba(30, 27, 75, 0.92) !important;
        border: 1px solid rgba(129, 140, 248, 0.5) !important;
        color: #e0e7ff !important;
        text-decoration: none !important;
        padding: 8px 16px !important;
        border-radius: 30px !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45), 0 0 16px rgba(99, 102, 241, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    .scroll-top-btn {
        bottom: 82px !important;
    }

    .scroll-bottom-btn {
        top: 130px !important;
    }


    .scroll-top-btn:hover, .scroll-bottom-btn:hover {
        background: linear-gradient(135deg, #4338ca, #6366f1) !important;
        color: #ffffff !important;
        border-color: #a5b4fc !important;
        transform: scale(1.05) !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.55) !important;
    }

    .scroll-top-btn:active, .scroll-bottom-btn:active {
        transform: scale(0.98) !important;
    }


    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)




# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.nav_page = "documents"




# ==========================================
# MAIN ROUTER (AUTHENTICATION GUARD)
# ==========================================
if not st.session_state.authenticated or not st.session_state.user:
    render_auth_ui()
else:
    # Current active user
    user = st.session_state.user
    user_id = user["id"]

    # Refresh user data from database
    with get_db() as db:
        user_obj = crud.get_user_by_id(db, user_id)
        if user_obj:
            user = user_obj.to_dict()
            st.session_state.user = user

    # ==========================================
    # MAIN APPLICATION VIEW
    # ==========================================
    # Anchor at the absolute top of the page for scroll-to-top actions
    st.markdown("<div id='page-top' style='position: relative; top: -30px;'></div>", unsafe_allow_html=True)

    # Render Global Top Header (with Docs, Chat, Analytics, Admin, Settings, and Mode Pill)
    render_header_ui(user)


    # Page Router
    curr_page = st.session_state.get("nav_page", "documents")

    if curr_page in ("documents", "workspace"):
        render_documents_tab(user)

    elif curr_page == "chatbot":
        render_chatbot_tab(user)

    elif curr_page == "usage":
        render_usage_tab(user)

    elif curr_page == "admin":
        render_admin_dashboard(user)

    elif curr_page == "settings":
        render_settings_tab(user)

    else:
        render_documents_tab(user)




