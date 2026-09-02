import streamlit as st
from app.db.database import get_db
from app.db import crud
from app.services.vector_service import VectorService
from app.services.audit_service import AuditService
from app.services.auth_service import AuthService
from app.config import settings


def render_admin_dashboard(admin_user: dict):
    """
    Renders the Administrator Dashboard:
    - High-level KPIs & Shared Infrastructure usage
    - User Management Table (search, filter, activate/deactivate, promote, delete)
    - Dynamic System Settings configuration
    - Audit Logs viewer
    """
    if admin_user.get("role") != "admin":
        st.error("⛔ Access Denied. Administrator privileges required.")
        return

    st.markdown("## 🛡️ Enterprise Administration & Infrastructure Dashboard")
    st.caption("Monitor multi-tenant resource consumption, manage accounts, and configure global application policies.")

    with get_db() as db:
        metrics = crud.get_admin_dashboard_metrics(db)
        app_doc_limit = crud.get_int_setting(db, "APPLICATION_CREDENTIAL_DOCUMENT_LIMIT", default=settings.APPLICATION_CREDENTIAL_DOCUMENT_LIMIT)
        max_upload_size = crud.get_int_setting(db, "MAX_UPLOAD_SIZE_MB", default=settings.MAX_UPLOAD_SIZE_MB)
        auto_verify = crud.get_bool_setting(db, "AUTO_VERIFY_EMAIL", default=settings.AUTO_VERIFY_EMAIL)

    admin_tab1, admin_tab2, admin_tab3, admin_tab4 = st.tabs([
        "📊 Infrastructure & Metrics", "👥 User Management", "⚙️ System Configuration", "📜 Audit Logs"
    ])

    # Fetch live Pinecone infrastructure stats
    vec_service = VectorService()
    pinecone_stats = vec_service.get_namespace_stats(settings.PINECONE_NAMESPACE)
    live_total_pinecone = pinecone_stats.get("total_index_vectors", 0)
    ns_map = pinecone_stats.get("namespaces", {})

    # ----------------------------------------------------
    # TAB 1: INFRASTRUCTURE & METRICS
    # ----------------------------------------------------
    with admin_tab1:
        st.markdown("### 📈 Overall System KPIs")
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)

        with kpi1:
            st.metric("Total Users", metrics["total_users"])
        with kpi2:
            st.metric("Active Users", metrics["active_users"])
        with kpi3:
            st.metric("Verified Users", metrics["verified_users"])
        with kpi4:
            st.metric("Total Documents", metrics["total_documents"])
        with kpi5:
            st.metric("App Shared Users", metrics["app_credential_users"])
        with kpi6:
            st.metric("BYOK Users", metrics["byok_users"])
        with kpi7:
            st.metric("Live Pinecone Vectors", f"{live_total_pinecone:,}", delta="Cloud Index")

        st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        st.markdown("### 🏢 Shared Infrastructure Resource Consumption")
        st.caption("Track resources consumed by users relying on your application's Pinecone and Groq keys.")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 14px; padding: 22px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
                <h4 style="margin-top:0; color:#38bdf8; display:flex; align-items:center; gap:8px;">🟢 Application Shared Keys Usage</h4>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 2.0; margin-top: 10px;">
                    <div>Users utilizing shared keys: <b>{app_users}</b></div>
                    <div>Documents processed: <b>{app_docs}</b></div>
                    <div>Live Pinecone Total Vectors: <b>{live_pinecone:,}</b></div>
                    <div>Database-tracked vectors: <b>{app_vecs:,}</b></div>
                    <div>Total queries routed: <b>{queries:,}</b></div>
                </div>
            </div>
            """.format(
                app_users=metrics["app_credential_users"],
                app_docs=metrics["app_docs_processed"],
                live_pinecone=live_total_pinecone,
                app_vecs=metrics["app_vectors"],
                queries=metrics["total_queries"]
            ), unsafe_allow_html=True)

        with col_res2:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 14px; padding: 22px; margin-bottom: 16px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
                <h4 style="margin-top:0; color:#a855f7; display:flex; align-items:center; gap:8px;">🚀 BYOK User Usage (Zero App Cost)</h4>
                <div style="font-size: 0.9rem; color: #cbd5e1; line-height: 2.0; margin-top: 10px;">
                    <div>Users with personal keys: <b>{byok_users}</b></div>
                    <div>Pinecone & Groq Costs: <b>$0.00 (External accounts)</b></div>
                    <div>Document Capacity: <b>Unlimited</b></div>
                    <div>Isolation Status: <b>Dedicated user namespaces</b></div>
                </div>
            </div>
            """.format(
                byok_users=metrics["byok_users"]
            ), unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 2: USER MANAGEMENT
    # ----------------------------------------------------
    with admin_tab2:
        st.markdown("### 👥 User Directory & Workspace Controls")

        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            search_query = st.text_input("🔍 Search Users", placeholder="Search by name or email...", key="admin_user_search")
        with col_s2:
            role_filter = st.selectbox("Filter Role", ["All", "user", "admin"], key="admin_role_filter")

        actual_role = None if role_filter == "All" else role_filter

        with get_db() as db:
            users_list = crud.get_all_users(db, search=search_query, role=actual_role, limit=100)

        if not users_list:
            st.info("No matching users found.")
        else:
            for u in users_list:
                with get_db() as db:
                    u_usage = crud.get_user_usage_statistics(db, u.id)
                    u_creds = crud.get_user_credentials(db, u.id)

                has_byok = bool(u_creds and u_creds.pinecone_api_key_encrypted and u_creds.groq_api_key_encrypted)
                mode_label = "🚀 BYOK" if has_byok else "🟢 Application"
                mode_color = "#a855f7" if has_byok else "#38bdf8"
                docs_count = u_usage.documents_uploaded if u_usage else 0
                live_user_vecs = ns_map.get(f"user_{u.id}", 0)
                vec_count = live_user_vecs if live_user_vecs > 0 else (u_usage.vector_count if u_usage else 0)
                q_count = u_usage.query_count if u_usage else 0
                last_act = u_usage.last_activity.strftime("%b %d, %Y") if (u_usage and u_usage.last_activity) else "Never"
                status_pill = "🟢 Active" if u.is_active else "🔴 Deactivated"
                verify_pill = "✅ Verified" if u.email_verified else "⏳ Unverified"

                col_u_info, col_u_metrics, col_u_actions = st.columns([4, 4, 3], vertical_alignment="center")

                with col_u_info:
                    st.markdown(f"**{u.name}** (`{u.role}`)")
                    st.caption(f"✉️ {u.email} · {verify_pill} · {status_pill}")

                with col_u_metrics:
                    st.markdown(f"<span style='color:{mode_color}; font-weight:700;'>{mode_label}</span> · <b>{docs_count}</b> Docs", unsafe_allow_html=True)
                    st.caption(f"📊 {vec_count:,} Vectors · {q_count} Queries · Active: {last_act}")

                with col_u_actions:
                    act_col1, act_col2, act_col3 = st.columns(3)
                    with act_col1:
                        toggle_btn_label = "Deactivate" if u.is_active else "Activate"
                        if st.button(toggle_btn_label, key=f"tog_{u.id}", help="Toggle user active status"):
                            with get_db() as db:
                                crud.update_user_status(db, u.id, not u.is_active)
                                AuditService.log_event(
                                    db=db,
                                    action="ADMIN_TOGGLE_USER_STATUS",
                                    user_id=admin_user["id"],
                                    details=f"Changed active status of user #{u.id} ({u.email}) to {not u.is_active}"
                                )
                            st.rerun()

                    with act_col2:
                        new_role = "user" if u.role == "admin" else "admin"
                        role_btn_label = "Demote" if u.role == "admin" else "Make Admin"
                        if u.id != admin_user["id"]:
                            if st.button(role_btn_label, key=f"role_{u.id}", help="Change role"):
                                with get_db() as db:
                                    crud.update_user_role(db, u.id, new_role)
                                    AuditService.log_event(
                                        db=db,
                                        action="ADMIN_CHANGE_USER_ROLE",
                                        user_id=admin_user["id"],
                                        details=f"Changed role of user #{u.id} ({u.email}) to {new_role}"
                                    )
                                st.rerun()

                    with act_col3:
                        if u.id != admin_user["id"]:
                            if st.button("🗑️", key=f"del_user_{u.id}", help=f"Purge and delete user #{u.id} completely"):
                                with get_db() as db:
                                    AuditService.log_event(
                                        db=db,
                                        action="ADMIN_DELETE_USER",
                                        user_id=admin_user["id"],
                                        details=f"Permanently purged all data for user #{u.id} ({u.email})"
                                    )
                                    purge_ok, purge_msg = AuthService.purge_user(db, u.id)
                                
                                if purge_ok:
                                    st.toast(f"✅ User #{u.id} and all related files & vectors purged completely.")
                                else:
                                    st.error(purge_msg)
                                st.rerun()

                st.markdown("<hr style='margin: 10px 0; border-color: rgba(255,255,255,0.06);'>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 3: SYSTEM CONFIGURATION
    # ----------------------------------------------------
    with admin_tab3:
        st.markdown("### ⚙️ Global Policy & Quota Settings")
        st.caption("Changes take effect immediately across all active user sessions without requiring app restart.")

        with get_db() as db:
            smtp_host = crud.get_system_setting(db, "SMTP_HOST") or settings.SMTP_HOST or "smtp.gmail.com"
            smtp_port = crud.get_int_setting(db, "SMTP_PORT", default=settings.SMTP_PORT)
            smtp_user = crud.get_system_setting(db, "SMTP_USER") or settings.SMTP_USER or ""
            smtp_pass = crud.get_system_setting(db, "SMTP_PASSWORD") or settings.SMTP_PASSWORD or ""
            smtp_from = crud.get_system_setting(db, "SMTP_FROM_EMAIL") or settings.SMTP_FROM_EMAIL or "no-reply@documind.ai"
            smtp_tls = crud.get_bool_setting(db, "SMTP_USE_TLS", default=settings.SMTP_USE_TLS)

        with st.form("sys_config_form"):
            st.markdown("#### 📋 General Policies")
            new_doc_limit = st.number_input(
                "Shared Application Credential Free Document Limit",
                min_value=1,
                max_value=50,
                value=app_doc_limit,
                help="The maximum number of documents free-tier users can upload before needing to add their own Pinecone & Groq keys."
            )

            new_max_size = st.number_input(
                "Maximum Upload Size per Document (MB)",
                min_value=5,
                max_value=200,
                value=max_upload_size,
                help="Applies to all users (both shared and BYOK) to protect server RAM and processing time."
            )

            new_auto_verify = st.checkbox(
                "Auto-Verify Email on Registration (Demo Bypass)",
                value=auto_verify,
                help="When enabled, newly registered users are automatically verified without requiring SMTP delivery."
            )

            st.markdown("<hr style='margin: 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
            st.markdown("#### ✉️ Transactional Email & SMTP Server Settings")
            st.caption("Configure SMTP to send real verification and password reset emails to user inboxes (e.g. Gmail, Outlook, SendGrid).")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                new_smtp_host = st.text_input("SMTP Host", value=smtp_host, placeholder="smtp.gmail.com")
                new_smtp_user = st.text_input("SMTP Username / Email", value=smtp_user, placeholder="your_email@gmail.com")
                new_smtp_from = st.text_input("From Email Address", value=smtp_from, placeholder="your_email@gmail.com")
            with col_m2:
                new_smtp_port = st.number_input("SMTP Port", min_value=25, max_value=65535, value=smtp_port, help="Use 587 for TLS or 465 for SSL")
                new_smtp_pass = st.text_input("SMTP Password / App Password", value=smtp_pass, type="password", help="For Gmail, use a 16-character Google App Password (not your normal password)")
                new_smtp_tls = st.checkbox("Enable STARTTLS (Recommended for Port 587)", value=smtp_tls)

            submit_settings = st.form_submit_button("💾 Save Global System & SMTP Settings", type="primary", use_container_width=True)

            if submit_settings:
                with get_db() as db:
                    crud.set_system_setting(
                        db=db,
                        key="APPLICATION_CREDENTIAL_DOCUMENT_LIMIT",
                        value=str(new_doc_limit),
                        description="Max documents allowed for users on shared application credentials"
                    )
                    crud.set_system_setting(
                        db=db,
                        key="MAX_UPLOAD_SIZE_MB",
                        value=str(new_max_size),
                        description="Maximum file size allowed per uploaded document in MB"
                    )
                    crud.set_system_setting(
                        db=db,
                        key="AUTO_VERIFY_EMAIL",
                        value=str(new_auto_verify).lower(),
                        description="Auto verify user emails on registration"
                    )
                    crud.set_system_setting(db=db, key="SMTP_HOST", value=new_smtp_host.strip(), description="SMTP Host")
                    crud.set_system_setting(db=db, key="SMTP_PORT", value=str(new_smtp_port), description="SMTP Port")
                    crud.set_system_setting(db=db, key="SMTP_USER", value=new_smtp_user.strip(), description="SMTP Username")
                    crud.set_system_setting(db=db, key="SMTP_PASSWORD", value=new_smtp_pass.strip(), description="SMTP Password")
                    crud.set_system_setting(db=db, key="SMTP_FROM_EMAIL", value=new_smtp_from.strip(), description="SMTP From Email")
                    crud.set_system_setting(db=db, key="SMTP_USE_TLS", value=str(new_smtp_tls).lower(), description="SMTP TLS")

                    AuditService.log_event(
                        db=db,
                        action="ADMIN_UPDATE_SETTINGS",
                        user_id=admin_user["id"],
                        details="Updated system policies and SMTP configuration"
                    )
                st.success("✅ System and SMTP settings updated successfully!")
                st.rerun()

        # SMTP Test Card
        st.markdown("##### 🧪 Test SMTP Email Delivery")
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            test_target_email = st.text_input("Send Test Email To:", value=admin_user["email"], key="test_smtp_target_email")
        with col_t2:
            st.write("")
            st.write("")
            if st.button("📤 Send Test Email", use_container_width=True):
                if not test_target_email:
                    st.warning("Please provide a destination email address.")
                else:
                    from app.services.email_service import EmailService
                    with st.spinner("Connecting to SMTP server and sending test email..."):
                        with get_db() as db:
                            ok, msg = EmailService.send_email(
                                to_email=test_target_email,
                                subject="DocuMind RAG — SMTP Test Email",
                                html_body="<h3>✅ SMTP Connection Successful!</h3><p>Your DocuMind RAG email service is configured correctly and ready to deliver real verification codes.</p>",
                                text_body="SMTP Connection Successful! Your DocuMind RAG email service is configured correctly.",
                                db=db
                            )
                            if ok:
                                st.success(f"✅ Email delivered successfully to {test_target_email}!")
                            else:
                                st.error(f"❌ {msg}")

    # ----------------------------------------------------
    # TAB 4: AUDIT LOGS
    # ----------------------------------------------------
    with admin_tab4:
        st.markdown("### 📜 System Audit Trail")
        st.caption("Immutable record of user registrations, logins, document modifications, and security actions.")

        with get_db() as db:
            logs = crud.get_audit_logs(db, limit=100)

        if not logs:
            st.info("No audit logs recorded yet.")
        else:
            for l in logs:
                t_str = l.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border-left: 3px solid #6366f1; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.86rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);">
                    <span style="color: #94a3b8; font-family: monospace;">[{t_str}]</span> <b style="color: #f8fafc;">{l.action}</b> — <span style="color: #cbd5e1;">{l.details or ''}</span> <span style="color: #818cf8;">(User #{l.user_id or 'N/A'})</span>
                </div>
                """, unsafe_allow_html=True)
