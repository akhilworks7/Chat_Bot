import streamlit as st
from app.db.database import get_db
from app.db import crud
from app.services.credential_service import CredentialService
from app.services.audit_service import AuditService
from components.settings_tab import show_settings_dialog



def render_header_ui(user: dict):
    """
    Renders top navigation header with active section title, credential mode pill,
    live document quota counter, and top-right Settings icon/modal trigger.
    """
    user_id = user.get("id")
    with get_db() as db:
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
        creds = CredentialService.get_credentials(user_id=user_id, db=db)

    is_byok = creds.get("is_byok", False)
    mode_pill = "🚀 BYOK Mode" if is_byok else "🟢 Shared Mode"
    mode_color = "#10b981" if is_byok else "#38bdf8"

    if is_byok:
        quota_pill = "♾️ Unlimited Docs"
        quota_color = "#a855f7"
    else:
        used = allowance.get("used", 0)
        limit = allowance.get("limit", 2)
        quota_pill = f"📄 {used}/{limit} Docs"
        quota_color = "#ef4444" if used >= limit else "#0ea5e9"

    role_badge = user.get("role", "user").upper()

    # Determine current view label
    nav_page = st.session_state.get("nav_page", "documents")
    page_labels = {
        "documents": "📄 Document Vault",
        "chatbot": "💬 AI Chatbot",
        "usage": "📊 Analytics & Usage",
        "admin": "🛡️ Admin Portal",
        "settings": "⚙️ Settings"
    }
    current_label = page_labels.get(nav_page, "Document Vault")

    is_admin = user.get("role") == "admin"

    # Header Row using Streamlit Columns based on role
    if is_admin:
        col_brand, col_btn_docs, col_btn_chat, col_btn_usage, col_btn_admin, col_status, col_btn_settings, col_btn_logout = st.columns(
            [3.0, 0.85, 0.85, 1.0, 0.85, 1.7, 0.85, 0.85],
            vertical_alignment="center"
        )
    else:
        col_brand, col_btn_docs, col_btn_chat, col_btn_usage, col_status, col_btn_settings, col_btn_logout = st.columns(
            [3.3, 0.95, 0.95, 1.1, 1.9, 0.9, 0.9],
            vertical_alignment="center"
        )

    with col_brand:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="font-size: 2.6rem; line-height: 1; filter: drop-shadow(0 4px 14px rgba(99, 102, 241, 0.5));">🧠</div>
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.85rem; font-weight: 900; background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #f43f5e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.03em; filter: drop-shadow(0 2px 10px rgba(99, 102, 241, 0.35));">
                        DocuMind AI
                    </span>
                    <span style="font-size: 0.72rem; font-weight: 600; color: #818cf8; background: rgba(99, 102, 241, 0.16); border: 1px solid rgba(99, 102, 241, 0.32); padding: 2px 8px; border-radius: 6px; letter-spacing: 0.02em;">
                        {current_label}
                    </span>
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 1px;">
                    {user.get('name', 'User')} · <span style="color: #64748b;">{user.get('email', '')}</span> · <span style="color: #c084fc; font-weight: 700; font-size: 0.72rem;">{role_badge}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


    with col_btn_docs:
        if st.button("📄 Docs", key="qn_docs_btn", use_container_width=True, type="primary" if nav_page in ("documents", "workspace") else "secondary", help="Switch to Document Vault"):
            st.session_state.nav_page = "documents"
            st.rerun()

    with col_btn_chat:
        if st.button("💬 Chat", key="qn_chat_btn", use_container_width=True, type="primary" if nav_page == "chatbot" else "secondary", help="Switch to AI Chatbot"):
            st.session_state.nav_page = "chatbot"
            st.rerun()

    with col_btn_usage:
        if st.button("📊 Analytics", key="qn_usage_btn", use_container_width=True, type="primary" if nav_page == "usage" else "secondary", help="View Token Analytics & System Activity"):
            st.session_state.nav_page = "usage"
            st.rerun()

    if is_admin:
        with col_btn_admin:
            if st.button("🛡️ Admin", key="qn_admin_btn", use_container_width=True, type="primary" if nav_page == "admin" else "secondary", help="User Management & Shared Quotas"):
                st.session_state.nav_page = "admin"
                st.rerun()

    with col_status:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid {mode_color}; color: {mode_color}; padding: 4px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; white-space: nowrap;">
                {mode_pill}
            </div>
            <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid {quota_color}; color: {quota_color}; padding: 4px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; white-space: nowrap;">
                {quota_pill}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_btn_settings:
        if st.button("⚙️ Settings", key="btn_header_settings", use_container_width=True, help="Configure Pinecone and Groq API keys"):
            show_settings_dialog(user)

    with col_btn_logout:
        if st.button("🚪 Logout", key="btn_header_logout", use_container_width=True, help="Log out of DocuMind"):
            with get_db() as db:
                AuditService.log_event(db, action="USER_LOGOUT", user_id=user_id, details=f"User {user['email']} logged out")
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.nav_page = "documents"
            st.toast("Logged out successfully.")
            st.rerun()

    st.markdown("<hr style='margin: 8px 0 18px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)







