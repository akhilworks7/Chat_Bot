import streamlit as st
from app.db.database import get_db
from app.services.auth_service import AuthService
from app.db import crud


def render_auth_ui():
    """
    Renders the authentication interface with Login, Register, Dedicated Email Verification,
    and Password Reset flows.
    """
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🧠 DocuMind Multi-User RAG
        </h1>
        <p style="color: #94a3b8; font-size: 1.05rem;">
            Enterprise AI Document Intelligence with Isolated Vector Workspaces
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "auth_view" not in st.session_state:
        st.session_state.auth_view = "tabs"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # ====================================================
        # DEDICATED VERIFICATION SCREEN (After Registration)
        # ====================================================
        if st.session_state.auth_view == "verify_pending":
            pending_email = st.session_state.get("pending_reg_email", "")

            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;">
                <div style="font-size: 2.5rem; margin-bottom: 8px;">✉️</div>
                <h3 style="margin: 0; color: #f8fafc;">Verify Your Email Address</h3>
                <p style="color: #94a3b8; margin-top: 8px; font-size: 0.95rem;">
                    Please enter the 6-digit verification code sent to your email inbox.
                </p>
                <div style="display: inline-block; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 6px; padding: 4px 12px; color: #f87171; font-size: 0.85rem; margin-top: 6px;">
                    ⏱️ Code expires in 10 minutes
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.form("dedicated_verify_form", clear_on_submit=False):
                email_input = st.text_input(
                    "Your Account Email",
                    value=pending_email,
                    placeholder="name@company.com",
                    key="dedicated_email_val"
                ).strip()

                otp_input = st.text_input(
                    "Enter 6-Digit Verification Code",
                    placeholder="e.g. 123456",
                    key="dedicated_otp_val",
                    max_chars=10
                ).strip()

                submit_otp = st.form_submit_button("✅ Verify & Activate Account", use_container_width=True, type="primary")

            if submit_otp:
                if not otp_input:
                    st.error("Please enter the 6-digit verification code.")
                else:
                    success, msg, user = False, "", None
                    with get_db() as db:
                        success, msg, user = AuthService.complete_registration(
                            db=db,
                            email=email_input or pending_email,
                            otp_code=otp_input
                        )
                    if success and user:
                        st.session_state.authenticated = True
                        st.session_state.user = user.to_dict()
                        st.session_state.user_id = user.id
                        st.session_state.user_name = user.name
                        st.session_state.user_email = user.email
                        st.session_state.user_role = user.role
                        st.session_state.nav_page = "documents"
                        st.session_state.auth_view = "tabs"
                        st.toast(f"Welcome, {user.name}! Your account is now active.")
                        st.rerun()
                    else:
                        st.error(msg)

            # Secondary Action Bar (Resend & Change Email)
            col_resend, col_back = st.columns(2)
            with col_resend:
                if st.button("🔄 Resend Code", use_container_width=True, help="Invalidate old code and generate a new one"):
                    target_email = st.session_state.get("dedicated_email_val", "").strip() or pending_email
                    if not target_email:
                        st.warning("Please enter your account email above to resend the code.")
                    else:
                        res_ok, res_msg = False, ""
                        with get_db() as db:
                            res_ok, res_msg, _ = AuthService.resend_registration_otp(db=db, email=target_email)
                        if res_ok:
                            st.success("✅ A new verification code has been dispatched to your email.")
                        else:
                            st.error(res_msg)

            with col_back:
                if st.button("← Back to Register", use_container_width=True, help="Return to registration form"):
                    st.session_state.auth_view = "tabs"
                    st.rerun()

            return

        # ====================================================
        # STANDARD AUTHENTICATION TABS
        # ====================================================
        tab_login, tab_register, tab_verify, tab_reset = st.tabs([
            "🔑 Log In", "✨ Register", "✉️ Verify Email", "🔄 Reset Password"
        ])

        # ----------------------------------------------------
        # TAB 1: LOGIN
        # ----------------------------------------------------
        with tab_login:
            st.markdown("### Welcome Back")
            st.caption("Enter your credentials to access your document workspace.")

            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email Address", placeholder="name@company.com", key="login_email").strip()
                password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
                submit_login = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submit_login:
                if not email or not password:
                    st.error("Please provide both email and password.")
                else:
                    success, msg, user = False, "", None
                    with get_db() as db:
                        success, msg, user = AuthService.authenticate(db=db, email=email, password=password)
                    if success and user:
                        st.session_state.authenticated = True
                        st.session_state.user = user.to_dict()
                        st.session_state.user_id = user.id
                        st.session_state.user_name = user.name
                        st.session_state.user_email = user.email
                        st.session_state.user_role = user.role
                        st.session_state.nav_page = "documents"
                        st.success(f"Welcome back, {user.name}!")
                        st.rerun()
                    else:
                        st.error(msg)

        # ----------------------------------------------------
        # TAB 2: REGISTER
        # ----------------------------------------------------
        with tab_register:
            st.markdown("### Create an Account")
            st.caption("Get started immediately with 2 free documents on our shared infrastructure.")

            with st.form("register_form", clear_on_submit=False):
                reg_name = st.text_input("Full Name", placeholder="Jane Doe", key="reg_name").strip()
                reg_email = st.text_input("Email Address", placeholder="jane@example.com", key="reg_email").strip()
                reg_pass = st.text_input("Password", type="password", placeholder="At least 8 characters (A-Z, a-z, 0-9)", key="reg_pass")
                reg_pass_conf = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_pass_conf")
                submit_reg = st.form_submit_button("Create Account", use_container_width=True, type="primary")

            if submit_reg:
                if not reg_name or not reg_email or not reg_pass:
                    st.error("Please fill in all required fields.")
                elif reg_pass != reg_pass_conf:
                    st.error("Passwords do not match.")
                else:
                    success, msg, dev_token = False, "", None
                    with get_db() as db:
                        success, msg, dev_token = AuthService.initiate_registration(
                            db=db,
                            name=reg_name,
                            email=reg_email,
                            password=reg_pass
                        )
                    if success:
                        # Switch directly to dedicated verification page
                        st.session_state.pending_reg_email = reg_email.lower().strip()
                        st.session_state.pending_reg_name = reg_name.strip()
                        st.session_state.auth_view = "verify_pending"
                        st.rerun()
                    else:
                        st.error(msg)

        # ----------------------------------------------------
        # TAB 3: VERIFY EMAIL (For direct activation)
        # ----------------------------------------------------
        with tab_verify:
            st.markdown("### Verify Email Address")
            st.caption("Enter your email and the 6-digit code received in your inbox.")

            with st.form("tab_verify_form", clear_on_submit=False):
                v_email = st.text_input("Your Account Email", placeholder="name@company.com", key="tab_v_email").strip()
                v_code = st.text_input("6-Digit Verification Code", placeholder="e.g. 123456", key="tab_v_code", max_chars=10).strip()
                submit_tab_verify = st.form_submit_button("Verify Account", use_container_width=True, type="primary")

            if submit_tab_verify:
                if not v_code:
                    st.error("Please enter the verification code.")
                else:
                    success, msg, user = False, "", None
                    with get_db() as db:
                        success, msg, user = AuthService.complete_registration(
                            db=db,
                            email=v_email,
                            otp_code=v_code
                        )
                    if success and user:
                        st.session_state.authenticated = True
                        st.session_state.user = user.to_dict()
                        st.session_state.user_id = user.id
                        st.session_state.user_name = user.name
                        st.session_state.user_email = user.email
                        st.session_state.user_role = user.role
                        st.session_state.nav_page = "documents"
                        st.toast(f"Welcome, {user.name}! Your account is now active.")
                        st.rerun()
                    else:
                        st.error(msg)

        # ----------------------------------------------------
        # TAB 4: PASSWORD RESET
        # ----------------------------------------------------
        with tab_reset:
            st.markdown("### Reset Password")
            st.caption("Request a reset code sent to your email and set a new password.")

            step = st.radio("Step", ["1. Request Reset Code", "2. Submit New Password"], horizontal=True, label_visibility="collapsed")

            if step == "1. Request Reset Code":
                with st.form("req_reset_form"):
                    reset_email = st.text_input("Your Account Email", placeholder="user@example.com", key="req_reset_email").strip()
                    submit_req = st.form_submit_button("Send Reset Code", use_container_width=True)

                if submit_req:
                    if not reset_email:
                        st.error("Please provide your email.")
                    else:
                        success, msg = False, ""
                        with get_db() as db:
                            success, msg, _ = AuthService.request_password_reset(db=db, email=reset_email)
                        if success:
                            st.success(f"📩 If an account exists with **{reset_email}**, a reset code has been sent to your inbox.")
                        else:
                            st.error(msg)

            else:
                with st.form("exec_reset_form"):
                    reset_token = st.text_input("Reset Code", placeholder="e.g. 654321", key="exec_reset_token").strip()
                    new_password = st.text_input("New Password", type="password", placeholder="At least 8 characters (A-Z, a-z, 0-9)", key="exec_new_pass")
                    submit_exec = st.form_submit_button("Update Password", use_container_width=True, type="primary")

                if submit_exec:
                    if not reset_token or not new_password:
                        st.error("Please fill in both the code and new password.")
                    else:
                        success, msg = False, ""
                        with get_db() as db:
                            success, msg = AuthService.reset_password(db=db, token=reset_token, new_password=new_password)
                        if success:
                            st.success(f"✅ {msg} You can now log in with your new password.")
                        else:
                            st.error(msg)
