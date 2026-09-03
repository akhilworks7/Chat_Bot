import streamlit as st
from app.db.database import get_db
from app.db import crud
from app.services.crypto_service import CryptoService
from app.services.credential_service import CredentialService
from app.services.audit_service import AuditService
from app.config import settings

AVAILABLE_GROQ_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
    "groq/compound",
    "allam-2-7b"
]


def render_settings_content(user: dict, in_dialog: bool = False):
    """
    Renders the API & workspace settings interface with tabbed layout.
    """
    user_id = user["id"]
    crypto = CryptoService()

    with get_db() as db:
        current_status = CredentialService.get_credentials(user_id=user_id, db=db)
        creds = crud.get_user_credentials(db=db, user_id=user_id)
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)

    is_byok = current_status.get("is_byok", False)

    # Credentials are ONLY displayed if the user/admin has explicitly added and saved their own personal keys.
    # System infrastructure keys are never exposed or pre-filled.
    existing_pinecone_key = (
        crypto.decrypt(creds.pinecone_api_key_encrypted)
        if (creds and creds.pinecone_api_key_encrypted)
        else ""
    )
    existing_pinecone_index = (
        creds.pinecone_index
        if (creds and creds.pinecone_index)
        else ""
    )
    existing_groq_key = (
        crypto.decrypt(creds.groq_api_key_encrypted)
        if (creds and creds.groq_api_key_encrypted)
        else ""
    )
    existing_groq_model = (
        creds.groq_model
        if (creds and creds.groq_model)
        else "openai/gpt-oss-20b"
    )

    is_admin = (user.get("role") == "admin")

    # Status Banner
    if is_byok:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
            <span style="color: #34d399; font-weight: 700; font-size: 0.95rem;">🚀 User Credentials (BYOK) Active:</span>
            <span style="color: #cbd5e1; font-size: 0.88rem; margin-left: 6px;">Unlimited document capacity & customized Groq LLM inference enabled.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        used = allowance.get("used", 0)
        limit = allowance.get("limit", 2)
        quota_text = f"Using <b>{used}/{limit}</b> free document slots." if not is_admin else "Using system shared infrastructure."
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
            <span style="color: #38bdf8; font-weight: 700; font-size: 0.95rem;">🟢 Application Shared Mode:</span>
            <span style="color: #cbd5e1; font-size: 0.88rem; margin-left: 6px;">{quota_text} Connect your personal API keys below to unlock unlimited custom indexing.</span>
        </div>
        """, unsafe_allow_html=True)

    tab_pinecone, tab_groq, tab_account = st.tabs([
        "🌲 Pinecone Vector DB",
        "⚡ Groq LLM Provider",
        "👤 Account & Mode"
    ])

    # ----------------------------------------------------
    # TAB 1: PINECONE
    # ----------------------------------------------------
    with tab_pinecone:
        st.markdown("#### 🌲 Pinecone Vector Database")
        st.caption("Vectors are stored in your dedicated serverless index.")
        
        p_key_input = st.text_input(
            "Pinecone API Key",
            value=existing_pinecone_key,
            type="password",
            placeholder="pcsk_...",
            help="Your Pinecone API Key from app.pinecone.io",
            key="cfg_pinecone_key"
        )
        p_index_input = st.text_input(
            "Pinecone Index Name",
            value=existing_pinecone_index,
            placeholder="e.g. pdf-rag1-index",
            help="Enter your Pinecone index name (e.g. pdf-rag1-index).",
            key="cfg_pinecone_index"
        )

        col_p_test, col_p_save = st.columns([1, 1])
        with col_p_test:
            if st.button("🔌 Test Pinecone Connection", key="test_p_btn", use_container_width=True):
                if not p_key_input or not p_index_input:
                    st.warning("Please enter both Pinecone API key and Index name to test.")
                else:
                    with st.spinner("Connecting to Pinecone (auto-creates index if needed)..."):
                        p_ok, p_msg, _ = CredentialService.test_pinecone(p_key_input, p_index_input)
                        if p_ok:
                            st.success(f"✅ {p_msg}")
                        else:
                            st.error(f"❌ {p_msg}")

        with st.expander("📖 Pinecone Setup Guide", expanded=False):
            st.markdown("""
            1. Visit [Pinecone Console](https://app.pinecone.io/) and create an account.
            2. Go to **API Keys** and generate a new API key.
            3. Enter any **Index Name** (e.g., `my-rag-index`).
            4. Click **Test Pinecone Connection** to verify.
            """)

    # ----------------------------------------------------
    # TAB 2: GROQ
    # ----------------------------------------------------
    with tab_groq:
        st.markdown("#### ⚡ Groq High-Speed LLM")
        st.caption("Ultra-fast inference for question answering and context synthesis.")
        
        g_key_input = st.text_input(
            "Groq API Key",
            value=existing_groq_key,
            type="password",
            placeholder="gsk_...",
            help="Your Groq API Key from console.groq.com",
            key="cfg_groq_key"
        )
        
        default_model_idx = 0
        if existing_groq_model in AVAILABLE_GROQ_MODELS:
            default_model_idx = AVAILABLE_GROQ_MODELS.index(existing_groq_model)

        g_model_input = st.selectbox(
            "Select Groq Model",
            options=AVAILABLE_GROQ_MODELS,
            index=default_model_idx,
            help="Choose model for RAG response generation",
            key="cfg_groq_model"
        )

        col_g_test, col_g_save = st.columns([1, 1])
        with col_g_test:
            if st.button("🔌 Test Groq Connection", key="test_g_btn", use_container_width=True):
                if not g_key_input:
                    st.warning("Please enter your Groq API key to test.")
                else:
                    with st.spinner("Connecting to Groq..."):
                        g_ok, g_msg = CredentialService.test_groq(g_key_input, g_model_input)
                        if g_ok:
                            st.success(f"✅ {g_msg}")
                        else:
                            st.error(f"❌ {g_msg}")

        with st.expander("📖 Groq Setup Guide", expanded=False):
            st.markdown("""
            1. Visit [Groq Console](https://console.groq.com/keys) and log in.
            2. Click **Create API Key** and copy the `gsk_...` key.
            3. Paste it above and choose your preferred model (e.g. `openai/gpt-oss-20b`).
            """)

    # ----------------------------------------------------
    # TAB 3: ACCOUNT & MODE
    # ----------------------------------------------------
    with tab_account:
        st.markdown("#### 👤 Account & Workspace Details")
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px; font-size: 0.88rem; line-height: 2.0; color: #cbd5e1;">
            <div><b>Name:</b> <span style="color: #f8fafc;">{user.get('name', 'User')}</span></div>
            <div><b>Email:</b> <span style="color: #f8fafc;">{user.get('email', '')}</span></div>
            <div><b>Role:</b> <span style="color: #c084fc; font-weight: 700;">{user.get('role', 'user').upper()}</span></div>
            <div><b>User Namespace:</b> <code style="background: rgba(30, 41, 59, 0.8); padding: 2px 6px; border-radius: 4px; color: #a5b4fc;">user_{user_id}</code></div>
            <div><b>Active Credentials:</b> <span style="color: {'#34d399' if is_byok else '#38bdf8'}; font-weight: 700;">{'🚀 BYOK (Custom Keys)' if is_byok else '🟢 Application Shared'}</span></div>
        </div>
        """, unsafe_allow_html=True)

        if is_byok:
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Revert to Application Shared Credentials", key="btn_revert_creds", use_container_width=True):
                with get_db() as db:
                    crud.delete_user_credentials(db=db, user_id=user_id)
                    AuditService.log_event(
                        db=db,
                        action="REVERT_TO_APP_CREDENTIALS",
                        user_id=user_id,
                        details="Removed personal BYOK API keys"
                    )
                # 1. Remove from cloud user credentials namespace
                try:
                    from app.services.vector_service import VectorService
                    VectorService().delete_user_credentials_from_cloud(user_email=user["email"])
                except Exception:
                    pass
                # 2. Immediately sync removal across cloud database snapshot
                try:
                    from app.services.cloud_sync_service import CloudSyncService
                    CloudSyncService.backup_database_to_cloud()
                except Exception:
                    pass
                st.toast("Reverted to shared application credentials.")
                st.rerun()

    st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # Global Save Button
    if st.button("💾 Save & Apply Workspace Credentials", type="primary", use_container_width=True, key="btn_save_all_settings"):
        p_val = st.session_state.get("cfg_pinecone_key", "").strip() or existing_pinecone_key
        p_idx = st.session_state.get("cfg_pinecone_index", "").strip() or existing_pinecone_index
        g_val = st.session_state.get("cfg_groq_key", "").strip() or existing_groq_key
        g_mod = st.session_state.get("cfg_groq_model", "openai/gpt-oss-20b") or existing_groq_model

        if not p_val or not g_val or not p_idx:
            st.error("Please provide both Pinecone and Groq API keys along with index name.")
        else:
            with get_db() as db:
                p_encrypted = crypto.encrypt(p_val)
                g_encrypted = crypto.encrypt(g_val)

                crud.upsert_user_credentials(
                    db=db,
                    user_id=user_id,
                    pinecone_api_key_encrypted=p_encrypted,
                    pinecone_index=p_idx,
                    groq_api_key_encrypted=g_encrypted,
                    groq_model=g_mod
                )
                AuditService.log_event(
                    db=db,
                    action="BYOK_CREDENTIALS_SAVED",
                    user_id=user_id,
                    details="Configured personal Pinecone and Groq API keys"
                )

            # 1. Persist BYOK credentials directly into Pinecone Cloud __system_config__
            try:
                from app.services.vector_service import VectorService
                VectorService().save_user_credentials_to_cloud(
                    user_email=user["email"],
                    creds_dict={
                        "pinecone_key": p_val,
                        "pinecone_index": p_idx,
                        "groq_key": g_val,
                        "groq_model": g_mod
                    }
                )
            except Exception:
                pass

            # 2. Persist entire database snapshot to central persistence index
            try:
                from app.services.cloud_sync_service import CloudSyncService
                CloudSyncService.backup_database_to_cloud()
            except Exception:
                pass

            st.toast("Credentials securely encrypted and saved! BYOK mode active.")
            st.rerun()


@st.dialog("⚙️ Workspace Settings & Credentials", width="large")
def show_settings_dialog(user: dict):
    """
    Renders settings inside a modal dialog window.
    """
    render_settings_content(user, in_dialog=True)


def render_settings_tab(user: dict):
    """
    Standard full-page rendering for settings.
    """
    render_settings_content(user, in_dialog=False)
