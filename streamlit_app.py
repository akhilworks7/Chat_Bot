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
                    # Set both compound key (e.g. SMTP_USER) and sub_key (e.g. USER)
                    compound = f"{k}_{sub_k}".upper()
                    os.environ[compound] = str(sub_v)
                    if sub_k.upper() not in ("USER", "PATH", "HOME"):  # Avoid overriding OS system variables
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
    # 1. Automatically restore complete database from Pinecone Cloud if running fresh after reboot/sleep
    try:
        from app.services.cloud_sync_service import CloudSyncService
        CloudSyncService.restore_database_from_cloud()
    except Exception:
        pass

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

# ========================================================
# RESPONSIVE CROSS-DEVICE DESIGN SYSTEM (Mobile, Tablet, Desktop)
# ========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Base Reset & Fonts */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: #0b0f19 !important;
        color: #f1f5f9 !important;
        -webkit-tap-highlight-color: transparent;
    }

    /* Ambient Background Mesh */
    .stApp {
        background-image: 
            radial-gradient(at 12% 15%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 88% 10%, rgba(168, 85, 247, 0.10) 0px, transparent 45%),
            radial-gradient(at 50% 85%, rgba(14, 165, 233, 0.08) 0px, transparent 50%),
            linear-gradient(180deg, #0b0f19 0%, #0d111e 100%) !important;
        background-attachment: fixed !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
        color: #f8fafc !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
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

    /* Desktop Content Container Spacing */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1400px !important;
        margin: 0 auto !important;
    }

    /* Modern Tabs Design */
    div[data-testid="stTabs"] {
        margin-top: 0.5rem !important;
    }
    
    div[data-baseweb="tab-list"] {
        gap: 6px !important;
        background: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        overflow-x: auto !important;
        scrollbar-width: none !important;
    }

    div[data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }

    button[data-baseweb="tab"] {
        height: auto !important;
        padding: 8px 16px !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        background: transparent !important;
        border: none !important;
        white-space: nowrap !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #f8fafc !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25)) !important;
        color: #ffffff !important;
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25) !important;
    }

    div[data-baseweb="tab-border"], div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Modern Primary & Secondary Buttons */
    div.stButton > button {
        position: relative !important;
        z-index: 10 !important;
        pointer-events: auto !important;
        cursor: pointer !important;
        width: 100% !important;
        min-height: 2.5rem !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.01em !important;
        border-radius: 10px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* Primary Button */
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #7c3aed 50%, #9333ea 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    div.stButton > button[kind="primary"]:active,
    div.stButton > button[data-testid="stBaseButton-primary"]:active {
        transform: translateY(0) !important;
    }

    /* Secondary Button */
    div.stButton > button[kind="secondary"],
    div.stButton > button[data-testid="stBaseButton-secondary"] {
        background: rgba(15, 23, 42, 0.7) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(8px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }

    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(30, 41, 59, 0.85) !important;
        color: #ffffff !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* Header Logout Button Accent */
    button:has(p:contains("Log Out")),
    button:has(div:contains("Log Out")) {
        color: #cbd5e1 !important;
        border-color: rgba(239, 68, 68, 0.25) !important;
    }
    button:has(p:contains("Log Out")):hover,
    button:has(div:contains("Log Out")):hover {
        background: rgba(239, 68, 68, 0.15) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
        color: #f87171 !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.2) !important;
    }

    /* Download Buttons */
    div[data-testid="stDownloadButton"] > button {
        background: rgba(15, 23, 42, 0.7) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
        min-height: 2.3rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: rgba(56, 189, 248, 0.15) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* Form Inputs, Selectboxes & TextAreas */
    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"],
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background-color: rgba(15, 23, 42, 0.75) !important;
        border-color: rgba(99, 102, 241, 0.25) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stTextInput"] input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
    }

    div[data-testid="stSelectbox"] svg {
        fill: #94a3b8 !important;
    }

    /* Popover & Dropdown Menu Styling */
    div[data-testid="stPopoverBody"],
    ul[data-baseweb="menu"] {
        background: #0f172a !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.6) !important;
        padding: 8px !important;
        color: #f8fafc !important;
    }

    li[data-baseweb="menu-item"] {
        border-radius: 8px !important;
        color: #cbd5e1 !important;
        transition: all 0.15s ease !important;
    }

    li[data-baseweb="menu-item"]:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        color: #ffffff !important;
    }

    /* Modern Card & Container Styling */
    .stCard, .source-card, .glass-panel {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(14px);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .source-card:hover {
        border-color: rgba(99, 102, 241, 0.3) !important;
        box-shadow: 0 6px 28px rgba(0, 0, 0, 0.35);
    }

    .source-badge {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: #ffffff;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        margin-bottom: 6px;
        letter-spacing: 0.02em;
    }

    /* Chat Messages Polish */
    div[data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 16px !important;
        padding: 14px 18px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
    }

    /* User Chat Message Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.5), rgba(15, 23, 42, 0.7)) !important;
        border: 1px solid rgba(129, 140, 248, 0.25) !important;
    }

    /* Metric Cards Styling */
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 14px 16px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6366f1, #a855f7, #38bdf8);
        opacity: 0.8;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.45) !important;
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.65rem !important;
        color: #f8fafc !important;
        letter-spacing: -0.02em !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
    }

    /* File Uploader Dropzone Styling */
    section[data-testid="stFileUploadDropzone"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 2px dashed rgba(99, 102, 241, 0.35) !important;
        border-radius: 16px !important;
        padding: 20px 14px !important;
        transition: all 0.25s ease !important;
    }

    section[data-testid="stFileUploadDropzone"]:hover {
        border-color: #818cf8 !important;
        background: rgba(30, 41, 59, 0.65) !important;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.2) !important;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"] {
        border-radius: 14px !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        background: rgba(15, 23, 42, 0.9) !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.4), 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        margin-top: 8px !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stExpander"] summary {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    /* Modal Dialog Window */
    div[data-testid="stDialog"] div[role="dialog"] {
        background: #0d111e !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        border-radius: 18px !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7), 0 0 40px rgba(99, 102, 241, 0.15) !important;
        padding: 20px !important;
        max-width: 95vw !important;
    }

    /* Floating Scroll Buttons */
    .scroll-top-btn, .scroll-bottom-btn {
        position: fixed !important;
        right: 20px !important;
        z-index: 99999 !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        background: rgba(15, 23, 42, 0.88) !important;
        border: 1px solid rgba(129, 140, 248, 0.4) !important;
        color: #e0e7ff !important;
        text-decoration: none !important;
        padding: 7px 14px !important;
        border-radius: 30px !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45), 0 0 16px rgba(99, 102, 241, 0.25) !important;
        backdrop-filter: blur(12px) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    .scroll-top-btn {
        bottom: 82px !important;
    }

    .scroll-bottom-btn {
        top: 90px !important;
    }

    .scroll-top-btn:hover, .scroll-bottom-btn:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: #ffffff !important;
        border-color: #c7d2fe !important;
        transform: scale(1.05) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
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

    /* ========================================================
       MOBILE & TABLET RESPONSIVE MEDIA QUERIES
       ======================================================== */
    
    /* Tablet Viewports (max-width: 992px) */
    @media screen and (max-width: 992px) {
        .block-container {
            padding-top: 1rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
            padding-bottom: 3rem !important;
        }

        /* Allow metrics to wrap into 2 columns on tablet */
        div[data-testid="stMetric"] {
            margin-bottom: 8px !important;
        }
    }

    /* Mobile Phones Viewports (max-width: 768px) */
    @media screen and (max-width: 768px) {
        .block-container {
            padding-top: 0.75rem !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-bottom: 2.5rem !important;
        }

        /* Headings scale down gracefully on mobile */
        h1 {
            font-size: 1.75rem !important;
        }
        h2 {
            font-size: 1.45rem !important;
        }
        h3 {
            font-size: 1.2rem !important;
        }
        h4 {
            font-size: 1.05rem !important;
        }

        /* Metric values smaller on mobile */
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }

        /* Streamlit columns flex-wrap on mobile so they don't crush */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 6px !important;
        }

        /* Buttons on mobile take full width & comfortable touch targets */
        div.stButton > button {
            min-height: 2.6rem !important;
            font-size: 0.85rem !important;
            padding: 8px 12px !important;
        }

        /* Floating buttons position on mobile */
        .scroll-top-btn, .scroll-bottom-btn {
            right: 12px !important;
            padding: 5px 10px !important;
            font-size: 0.72rem !important;
        }
        .scroll-top-btn {
            bottom: 74px !important;
        }
        .scroll-bottom-btn {
            top: 70px !important;
        }

        /* Chat messages padding on mobile */
        div[data-testid="stChatMessage"] {
            padding: 10px 12px !important;
            border-radius: 12px !important;
            margin-bottom: 10px !important;
            font-size: 0.92rem !important;
        }

        /* Card padding on mobile */
        .stCard, .source-card, .glass-panel {
            padding: 12px 14px !important;
            border-radius: 12px !important;
        }
    }

    /* Small Mobile Phones (max-width: 480px) */
    @media screen and (max-width: 480px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        h1 {
            font-size: 1.5rem !important;
        }

        /* Hide floating buttons on very small screens to avoid obstructing view */
        .scroll-top-btn, .scroll-bottom-btn {
            display: none !important;
        }
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
    auth_container = st.empty()
    with auth_container.container():
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
