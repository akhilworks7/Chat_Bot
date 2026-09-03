import streamlit as st
from app.db.database import get_db
from app.db import crud
from app.services.credential_service import CredentialService
from app.services.audit_service import AuditService
from components.settings_tab import show_settings_dialog


def render_header_ui(user: dict):
    """
    Renders top navigation header with active section title, credential mode pill,
    live document quota counter, and responsive controls for all screen sizes.
    """
    user_id = user.get("id")
    with get_db() as db:
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
        creds = CredentialService.get_credentials(user_id=user_id, db=db)

    is_byok = creds.get("is_byok", False)
    is_admin = user.get("role") == "admin"
    is_unlimited = is_byok or is_admin

    if is_byok:
        mode_pill = "🚀 BYOK Mode"
        mode_bg = "rgba(16, 185, 129, 0.15)"
        mode_border = "rgba(16, 185, 129, 0.35)"
        mode_color = "#34d399"
    elif is_admin:
        mode_pill = "👑 Admin Workspace"
        mode_bg = "rgba(168, 85, 247, 0.15)"
        mode_border = "rgba(168, 85, 247, 0.35)"
        mode_color = "#c084fc"
    else:
        mode_pill = "🟢 Shared Mode"
        mode_bg = "rgba(56, 189, 248, 0.15)"
        mode_border = "rgba(56, 189, 248, 0.35)"
        mode_color = "#38bdf8"

    if is_unlimited:
        quota_pill = "♾️ Unlimited Docs"
        quota_bg = "rgba(168, 85, 247, 0.15)"
        quota_border = "rgba(168, 85, 247, 0.35)"
        quota_color = "#c084fc"
    else:
        used = allowance.get("used", 0)
        limit = allowance.get("limit", 2)
        quota_pill = f"📄 {used}/{limit} Docs"
        if used >= limit:
            quota_bg = "rgba(239, 68, 68, 0.15)"
            quota_border = "rgba(239, 68, 68, 0.35)"
            quota_color = "#f87171"
        else:
            quota_bg = "rgba(14, 165, 233, 0.15)"
            quota_border = "rgba(14, 165, 233, 0.35)"
            quota_color = "#38bdf8"

    role_badge = user.get("role", "user").upper()
    is_admin = user.get("role") == "admin"
    nav_page = st.session_state.get("nav_page", "documents")

    # ========================================================
    # TOP ROW: BRAND IDENTITY & USER PROFILE / STATUS PILLS
    # ========================================================
    col_brand, col_actions = st.columns([5, 5], vertical_alignment="center")

    with col_brand:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; padding: 2px 0;">
            <div style="font-size: 2.2rem; line-height: 1; filter: drop-shadow(0 4px 14px rgba(99, 102, 241, 0.55));">🧠</div>
            <div style="min-width: 0;">
                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span style="font-size: 1.65rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #f43f5e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.03em;">
                        DocuMind AI
                    </span>
                    <span style="font-size: 0.70rem; font-weight: 700; color: #a5b4fc; background: rgba(99, 102, 241, 0.18); border: 1px solid rgba(129, 140, 248, 0.3); padding: 1px 7px; border-radius: 6px; text-transform: uppercase;">
                        {role_badge}
                    </span>
                </div>
                <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {user.get('name', 'User')} <span style="color: #475569;">•</span> <span style="color: #64748b;">{user.get('email', '')}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_actions:
        col_pills, col_btns = st.columns([1, 1.8], vertical_alignment="center")
        with col_pills:
            st.markdown(f"""
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px;">
                <div style="background: {mode_bg}; border: 1px solid {mode_border}; color: {mode_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 700; white-space: nowrap;">
                    {mode_pill}
                </div>
                <div style="background: {quota_bg}; border: 1px solid {quota_border}; color: {quota_color}; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem; font-weight: 700; white-space: nowrap;">
                    {quota_pill}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_btns:
            btn_col1, btn_col2 = st.columns([1, 1], vertical_alignment="center")
            with btn_col1:
                if st.button("⚙️ Settings", key="btn_header_settings", use_container_width=True, help="Configure Pinecone & Groq API Keys"):
                    show_settings_dialog(user)
            with btn_col2:
                if st.button("🚪 Logout", key="btn_header_logout", use_container_width=True, help="Log out of DocuMind"):
                    with get_db() as db:
                        AuditService.log_event(db, action="USER_LOGOUT", user_id=user_id, details=f"User {user['email']} logged out")
                    st.session_state.authenticated = False
                    st.session_state.user = None
                    st.session_state.nav_page = "documents"
                    st.toast("Logged out successfully.")
                    st.rerun()

    # ========================================================
    # NAVIGATION BAR (Responsive Button Group)
    # ========================================================
    st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)

    if is_admin:
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
        with col_nav1:
            if st.button("📄 Vault", key="nav_docs", use_container_width=True, type="primary" if nav_page in ("documents", "workspace") else "secondary", help="Document Vault"):
                st.session_state.nav_page = "documents"
                st.rerun()
        with col_nav2:
            if st.button("💬 Chat", key="nav_chat", use_container_width=True, type="primary" if nav_page == "chatbot" else "secondary", help="AI Chatbot"):
                st.session_state.nav_page = "chatbot"
                st.rerun()
        with col_nav3:
            if st.button("📊 Analytics", key="nav_usage", use_container_width=True, type="primary" if nav_page == "usage" else "secondary", help="Workspace Analytics"):
                st.session_state.nav_page = "usage"
                st.rerun()
        with col_nav4:
            if st.button("🛡️ Admin", key="nav_admin", use_container_width=True, type="primary" if nav_page == "admin" else "secondary", help="Admin Portal"):
                st.session_state.nav_page = "admin"
                st.rerun()
    else:
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        with col_nav1:
            if st.button("📄 Document Vault", key="nav_docs", use_container_width=True, type="primary" if nav_page in ("documents", "workspace") else "secondary"):
                st.session_state.nav_page = "documents"
                st.rerun()
        with col_nav2:
            if st.button("💬 AI Chatbot", key="nav_chat", use_container_width=True, type="primary" if nav_page == "chatbot" else "secondary"):
                st.session_state.nav_page = "chatbot"
                st.rerun()
        with col_nav3:
            if st.button("📊 Analytics", key="nav_usage", use_container_width=True, type="primary" if nav_page == "usage" else "secondary"):
                st.session_state.nav_page = "usage"
                st.rerun()

    st.markdown("<hr style='margin: 10px 0 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
