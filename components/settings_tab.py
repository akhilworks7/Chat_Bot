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
    "allam-2-7b",
    "groq/compound",
    "groq/compound-mini"
]


def render_settings_tab(user: dict):
    """
    Renders the BYOK API Settings tab with live connection testers, encrypted storage, and setup guides.
    """
    user_id = user["id"]
    crypto = CryptoService()

    with get_db() as db:
        creds = crud.get_user_credentials(db=db, user_id=user_id)
        current_status = CredentialService.get_credentials(user_id=user_id, db=db)

    is_byok = current_status.get("is_byok", False)

    existing_pinecone_key = crypto.decrypt(creds.pinecone_api_key_encrypted) if (creds and creds.pinecone_api_key_encrypted) else ""
    existing_pinecone_index = creds.pinecone_index if (creds and creds.pinecone_index) else ""
    existing_groq_key = crypto.decrypt(creds.groq_api_key_encrypted) if (creds and creds.groq_api_key_encrypted) else ""
    existing_groq_model = creds.groq_model if (creds and creds.groq_model) else "llama-3.3-70b-versatile"

    st.markdown("### 🔑 API Credentials & Hybrid Configuration")
    st.markdown("""
    Connect your own **Pinecone** and **Groq** accounts to unlock unlimited document indexing and custom LLM model selection.
    Your API keys are encrypted at rest with **AES-256** and are strictly isolated to your user workspace.
    """)

    # Current Status Banner
    if is_byok:
        st.success("🚀 **Status: User Credentials (BYOK) Active** — Your documents and queries are powered by your own accounts with unlimited document capacity.")
    else:
        st.info("🟢 **Status: Application Credentials (Shared) Active** — You are currently using our shared infrastructure with an onboarding limit.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ----------------------------------------------------
    # PINECONE CONFIGURATION
    # ----------------------------------------------------
    with col1:
        st.markdown("#### 🌲 Pinecone Vector Database")
        p_key_input = st.text_input(
            "Pinecone API Key",
            value=existing_pinecone_key,
            type="password",
            placeholder="pcsk_...",
            help="Your Pinecone API Key from the Pinecone console",
            key="cfg_pinecone_key"
        )
        p_index_input = st.text_input(
            "Pinecone Index Name",
            value=existing_pinecone_index or "my-rag-index",
            placeholder="e.g. my-rag-index",
            help="Enter any index name. If it does not exist in your Pinecone account, we will automatically create it for you.",
            key="cfg_pinecone_index"
        )

        if st.button("🔌 Test Pinecone Connection", key="test_p_btn", use_container_width=True):
            if not p_key_input or not p_index_input:
                st.warning("Please enter both Pinecone API key and Index name to test.")
            else:
                with st.spinner("Connecting to Pinecone (will auto-create index if needed)..."):
                    p_ok, p_msg, p_stats = CredentialService.test_pinecone(p_key_input, p_index_input)
                    if p_ok:
                        st.success(f"✅ {p_msg}")
                    else:
                        st.error(f"❌ {p_msg}")

        with st.expander("📖 Step-by-Step Pinecone Setup Guide", expanded=False):
            st.markdown("""
            1. Go to [Pinecone Console](https://app.pinecone.io/) and log in.
            2. Navigate to **API Keys** in the sidebar.
            3. Click **Create API Key**, copy the key, and paste it above.
            4. Choose any **Index Name** (e.g. `my-rag-index`).
               *💡 If the index doesn't exist yet, our system will automatically create the 384-dimension serverless index for you!*
            5. Click **Test Pinecone Connection** and click **Save User Credentials**.
            """)

    # ----------------------------------------------------
    # GROQ CONFIGURATION
    # ----------------------------------------------------
    with col2:
        st.markdown("#### ⚡ Groq LLM Provider")
        g_key_input = st.text_input(
            "Groq API Key",
            value=existing_groq_key,
            type="password",
            placeholder="gsk_...",
            help="Your Groq API Key from console.groq.com",
            key="cfg_groq_key"
        )
        
        # Determine model default index
        default_model_idx = 0
        if existing_groq_model in AVAILABLE_GROQ_MODELS:
            default_model_idx = AVAILABLE_GROQ_MODELS.index(existing_groq_model)

        g_model_input = st.selectbox(
            "Groq Model",
            options=AVAILABLE_GROQ_MODELS,
            index=default_model_idx,
            help="Select the Groq high-speed LLM model",
            key="cfg_groq_model"
        )

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

        with st.expander("📖 Step-by-Step Groq Setup Guide", expanded=False):
            st.markdown("""
            1. Go to [Groq Console](https://console.groq.com/keys) and log in.
            2. Click **Create API Key**.
            3. Copy the key (starts with `gsk_`) and paste it above.
            4. Choose your preferred model (e.g. `llama-3.3-70b-versatile`).
            5. Click **Test Connection** and save your settings.
            """)

    st.markdown("---")

    # ----------------------------------------------------
    # SAVE & REVERT ACTIONS
    # ----------------------------------------------------
    col_save, col_rev = st.columns([1, 1])

    with col_save:
        if st.button("💾 Save User Credentials", type="primary", use_container_width=True):
            if not p_key_input or not g_key_input or not p_index_input:
                st.error("Please provide both Pinecone and Groq API keys along with index name.")
            else:
                with get_db() as db:
                    p_encrypted = crypto.encrypt(p_key_input.strip())
                    g_encrypted = crypto.encrypt(g_key_input.strip())

                    crud.upsert_user_credentials(
                        db=db,
                        user_id=user_id,
                        pinecone_api_key_encrypted=p_encrypted,
                        pinecone_index=p_index_input.strip(),
                        groq_api_key_encrypted=g_encrypted,
                        groq_model=g_model_input
                    )
                    AuditService.log_event(
                        db=db,
                        action="BYOK_CREDENTIALS_SAVED",
                        user_id=user_id,
                        details="Configured personal Pinecone and Groq API keys"
                    )

                st.success("🎉 Credentials securely encrypted and saved! BYOK mode is now active.")
                st.rerun()

    with col_rev:
        if is_byok:
            if st.button("🔄 Revert to Application Shared Credentials", use_container_width=True):
                with get_db() as db:
                    crud.delete_user_credentials(db=db, user_id=user_id)
                    AuditService.log_event(
                        db=db,
                        action="REVERT_TO_APP_CREDENTIALS",
                        user_id=user_id,
                        details="Removed personal BYOK API keys"
                    )
                st.warning("Reverted to shared application credentials.")
                st.rerun()
