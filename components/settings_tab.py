import streamlit as st
from app.db.database import get_db
from app.db import crud
from app.services.crypto_service import CryptoService
from app.services.credential_service import CredentialService
from app.services.audit_service import AuditService
from app.config import settings

AVAILABLE_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
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
        creds = crud.get_user_credentials(db=db, user_id=user_id)
        current_status = CredentialService.get_credentials(user_id=user_id, db=db)
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)

    is_byok = current_status.get("is_byok", False)

    existing_pinecone_key = crypto.decrypt(creds.pinecone_api_key_encrypted) if (creds and creds.pinecone_api_key_encrypted) else ""
    existing_pinecone_index = creds.pinecone_index if (creds and creds.pinecone_index) else ""
    existing_groq_key = crypto.decrypt(creds.groq_api_key_encrypted) if (creds and creds.groq_api_key_encrypted) else ""
    existing_groq_model = creds.groq_model if (creds and creds.groq_model) else "llama-3.3-70b-versatile"

    # Status Banner
    if is_byok:
        st.success("🚀 **User Credentials (BYOK) Active:** Unlimited documents & custom Groq model.")
    else:
        used = allowance.get("used", 0)
        limit = allowance.get("limit", 2)
        st.info(f"🟢 **Application Shared Mode:** Using {used}/{limit} free document slots. Connect your personal API keys below to unlock unlimited document indexing.")

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
            value=existing_pinecone_index or "my-rag-index",
            placeholder="e.g. my-rag-index",
            help="Enter any index name. If it doesn't exist, we auto-create a 384-dimension serverless index.",
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
            3. Paste it above and choose your preferred model (e.g. `llama-3.3-70b-versatile`).
            """)

    # ----------------------------------------------------
    # TAB 3: ACCOUNT & MODE
    # ----------------------------------------------------
    with tab_account:
        st.markdown("#### 👤 Account & Workspace Details")
        st.markdown(f"""
        - **Name:** `{user.get('name', 'User')}`
        - **Email:** `{user.get('email', '')}`
        - **Role:** `{user.get('role', 'user').upper()}`
        - **User Namespace:** `user_{user_id}`
        - **Active Credentials:** `{'🚀 BYOK (Custom Keys)' if is_byok else '🟢 Application Shared'}`
        """)

        if is_byok:
            st.markdown("---")
            if st.button("🔄 Revert to Application Shared Credentials", key="btn_revert_creds", use_container_width=True):
                with get_db() as db:
                    crud.delete_user_credentials(db=db, user_id=user_id)
                    AuditService.log_event(
                        db=db,
                        action="REVERT_TO_APP_CREDENTIALS",
                        user_id=user_id,
                        details="Removed personal BYOK API keys"
                    )
                st.toast("Reverted to shared application credentials.")
                st.rerun()

    st.markdown("---")

    # Global Save Button
    if st.button("💾 Save & Apply Workspace Credentials", type="primary", use_container_width=True, key="btn_save_all_settings"):
        p_val = st.session_state.get("cfg_pinecone_key", "").strip()
        p_idx = st.session_state.get("cfg_pinecone_index", "").strip()
        g_val = st.session_state.get("cfg_groq_key", "").strip()
        g_mod = st.session_state.get("cfg_groq_model", "llama-3.3-70b-versatile")

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

