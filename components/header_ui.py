import streamlit as st
from app.db.database import get_db
from app.services.credential_service import CredentialService


def render_header_ui(user: dict):
    """
    Renders top navigation header with user identity, credential mode pill, and document quota counter.
    """
    user_id = user.get("id")
    with get_db() as db:
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
        creds = CredentialService.get_credentials(user_id=user_id, db=db)

    is_byok = creds.get("is_byok", False)
    mode_pill = "🚀 User Credentials (BYOK)" if is_byok else "🟢 Application Credentials"
    mode_color = "#10b981" if is_byok else "#3b82f6"

    if is_byok:
        quota_pill = "♾️ Unlimited Documents (BYOK)"
        quota_color = "#8b5cf6"
    else:
        used = allowance.get("used", 0)
        limit = allowance.get("limit", 2)
        quota_pill = f"📄 {used} / {limit} Free Docs Used"
        quota_color = "#ef4444" if used >= limit else "#0ea5e9"

    role_badge = f"<span style='background: #475569; color: #fff; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; text-transform: uppercase;'>{user.get('role', 'user')}</span>"

    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="font-size: 1.6rem;">🧠</div>
            <div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                    DocuMind RAG <span style="font-size: 0.85rem; font-weight: 400; color: #94a3b8;">| Workspace</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748b;">
                    👤 {user.get('name', 'User')} ({user.get('email', '')}) {role_badge}
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid {mode_color}; color: {mode_color}; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">
                {mode_pill}
            </div>
            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid {quota_color}; color: {quota_color}; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">
                {quota_pill}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
