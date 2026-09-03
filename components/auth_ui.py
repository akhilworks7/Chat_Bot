import streamlit as st
from app.db.database import get_db
from app.services.auth_service import AuthService
from app.db import crud


def render_auth_ui():
    """
    Renders the modern SaaS authentication interface with Login, Register,
    and Password Reset flows inside a centered glassmorphic card responsive on all device screens.
    - Login: Direct login with registered credentials (no verification required on login page).
    - Register: OTP-based registration flow (enter details -> receive OTP -> verify OTP -> activate account).
    - Reset Password: Unified single flow with automatic navigation to Login upon success.
    """
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
        <div style="font-size: 2.8rem; line-height: 1; filter: drop-shadow(0 6px 20px rgba(99, 102, 241, 0.6)); margin-bottom: 8px;">🧠</div>
        <h1 style="font-size: clamp(1.8rem, 5vw, 2.5rem); font-weight: 900; background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #f43f5e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.03em; margin: 0;">
            DocuMind AI
        </h1>
        <p style="color: #94a3b8; font-size: clamp(0.85rem, 2.5vw, 1.02rem); margin-top: 6px; font-weight: 500;">
            Enterprise Multi-Tenant RAG with Isolated Vector Workspaces
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "auth_tab" not in st.session_state:
        st.session_state.auth_tab = "🔑 Log In"

    col_pad_left, col_card, col_pad_right = st.columns([1, 2.2, 1])

    with col_card:
        # ====================================================
        # PROGRAMMATICALLY CONTROLLABLE NAVIGATION BAR
        # ====================================================
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            btn_t1 = "primary" if st.session_state.auth_tab == "🔑 Log In" else "secondary"
            if st.button("🔑 Log In", use_container_width=True, type=btn_t1, key="nav_btn_login"):
                st.session_state.auth_tab = "🔑 Log In"
                st.rerun()
        with col_t2:
            btn_t2 = "primary" if st.session_state.auth_tab == "✨ Register" else "secondary"
            if st.button("✨ Register", use_container_width=True, type=btn_t2, key="nav_btn_reg"):
                st.session_state.auth_tab = "✨ Register"
                st.rerun()
        with col_t3:
            btn_t3 = "primary" if st.session_state.auth_tab == "🔄 Reset" else "secondary"
            if st.button("🔄 Reset", use_container_width=True, type=btn_t3, key="nav_btn_reset"):
                st.session_state.auth_tab = "🔄 Reset"
                st.rerun()

        st.markdown("<hr style='margin: 12px 0 16px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        # ----------------------------------------------------
        # TAB 1: LOGIN (Direct Login - No Verification Barrier)
        # ----------------------------------------------------
        if st.session_state.auth_tab == "🔑 Log In":
            st.markdown(
                "<div style='padding: 2px 0 10px 0;'>"
                "<h3 style='margin:0; font-size:1.15rem; color:#f8fafc;'>Welcome Back</h3>"
                "<p style='color:#94a3b8; font-size:0.84rem; margin:3px 0 0 0;'>Enter your registered credentials to access your workspace.</p>"
                "</div>",
                unsafe_allow_html=True
            )

            flash_msg = st.session_state.pop("login_flash_success", None)
            if flash_msg:
                st.success(f"✅ {flash_msg}")

            default_login_email = st.session_state.get("login_email", "")

            with st.form("login_form", clear_on_submit=False):
                email = st.text_input("Email Address", value=default_login_email, placeholder="name@company.com", key="login_email_input").strip()
                password = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass_input")
                submit_login = st.form_submit_button("Sign In to Workspace", use_container_width=True, type="primary")

            if submit_login:
                if not email or not password:
                    st.error("Please provide both email and password.")
                else:
                    success, msg, user = False, "", None
                    with st.spinner("Signing in to workspace..."):
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
                        st.toast(f"Welcome back, {user.name}!")
                        st.rerun()
                    else:
                        st.error(msg)

            # Secondary navigation links
            col_forgot, col_need_acc = st.columns(2)
            with col_forgot:
                if st.button("Forgot Password?", use_container_width=True, help="Reset your password"):
                    st.session_state.auth_tab = "🔄 Reset"
                    st.rerun()
            with col_need_acc:
                if st.button("Create Account", use_container_width=True, help="Register a new account"):
                    st.session_state.auth_tab = "✨ Register"
                    st.rerun()

        # ----------------------------------------------------
        # TAB 2: REGISTER (OTP-Based Registration Flow)
        # ----------------------------------------------------
        elif st.session_state.auth_tab == "✨ Register":
            st.markdown(
                "<div style='padding: 2px 0 10px 0;'>"
                "<h3 style='margin:0; font-size:1.15rem; color:#f8fafc;'>Create an Account</h3>"
                "<p style='color:#94a3b8; font-size:0.84rem; margin:3px 0 0 0;'>Enter your details to receive a 6-digit OTP code to verify and activate your account.</p>"
                "</div>",
                unsafe_allow_html=True
            )

            reg_pending_email = st.session_state.get("reg_otp_email", "")
            reg_manual_otp = st.session_state.get("reg_manual_otp", False)

            # Stage 1: Registration Form (Name, Email, Password) -> Dispatches OTP
            if not reg_pending_email and not reg_manual_otp:
                with st.form("register_form", clear_on_submit=False):
                    reg_name = st.text_input("Full Name", placeholder="Jane Doe", key="reg_name").strip()
                    reg_email = st.text_input("Email Address", placeholder="jane@example.com", key="reg_email").strip()
                    reg_pass = st.text_input("Password", type="password", placeholder="At least 8 chars (A-Z, 0-9)", key="reg_pass")
                    reg_pass_conf = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_pass_conf")
                    submit_reg = st.form_submit_button("📩 Send Verification OTP", use_container_width=True, type="primary")

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
                            st.session_state.reg_otp_email = reg_email.lower().strip()
                            st.session_state.reg_otp_name = reg_name.strip()
                            st.rerun()
                        else:
                            st.error(msg)

                # Option for user who already holds an OTP
                col_already, col_login_switch = st.columns(2)
                with col_already:
                    if st.button("Have an OTP Code?", use_container_width=True):
                        st.session_state.reg_manual_otp = True
                        st.rerun()
                with col_login_switch:
                    if st.button("Already Registered? Sign In", use_container_width=True):
                        st.session_state.auth_tab = "🔑 Log In"
                        st.rerun()

            # Stage 2: OTP Verification & Account Activation in the same flow
            else:
                active_reg_email = reg_pending_email or st.session_state.get("reg_verify_email", "")

                if reg_pending_email:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 12px 16px; margin-bottom: 14px;">
                        <span style="color: #34d399; font-weight: 600; font-size: 0.92rem;">✉️ OTP Sent to Your Email!</span>
                        <p style="color: #cbd5e1; font-size: 0.84rem; margin: 4px 0 0 0; line-height: 1.4;">
                            A 6-digit verification code has been dispatched to <b>{reg_pending_email}</b>.
                            Please enter it below to activate your account.
                        </p>
                        <div style="display: inline-block; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 20px; padding: 2px 10px; color: #f87171; font-size: 0.74rem; font-weight: 600; margin-top: 8px;">
                            ⏱️ Code expires in 10 minutes
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.form("reg_otp_verify_form", clear_on_submit=False):
                    verify_email = st.text_input(
                        "Account Email",
                        value=active_reg_email,
                        placeholder="name@company.com",
                        key="reg_verify_email"
                    ).strip()

                    verify_otp = st.text_input(
                        "6-Digit Verification OTP",
                        placeholder="e.g. 123456",
                        key="reg_verify_otp",
                        max_chars=10
                    ).strip()

                    submit_verify_otp = st.form_submit_button("✅ Verify OTP & Activate Account", use_container_width=True, type="primary")

                if submit_verify_otp:
                    if not verify_otp:
                        st.error("Please enter the 6-digit verification OTP.")
                    else:
                        success, msg, user = False, "", None
                        with get_db() as db:
                            success, msg, user = AuthService.complete_registration(
                                db=db,
                                email=verify_email or reg_pending_email,
                                otp_code=verify_otp
                            )
                        if success and user:
                            st.session_state.pop("reg_otp_email", None)
                            st.session_state.pop("reg_otp_name", None)
                            st.session_state.pop("reg_manual_otp", None)

                            st.session_state.authenticated = True
                            st.session_state.user = user.to_dict()
                            st.session_state.user_id = user.id
                            st.session_state.user_name = user.name
                            st.session_state.user_email = user.email
                            st.session_state.user_role = user.role
                            st.session_state.nav_page = "documents"
                            st.toast(f"🎉 Welcome, {user.name}! Your account has been verified and activated.")
                            st.rerun()
                        else:
                            st.error(msg)

                # Secondary row: Resend & Edit Details
                col_resend, col_back = st.columns(2)
                with col_resend:
                    if st.button("🔄 Resend OTP", use_container_width=True, help="Invalidate old OTP and send a new one"):
                        target_mail = verify_email or reg_pending_email
                        if not target_mail:
                            st.warning("Please enter your account email above to resend.")
                        else:
                            r_ok, r_msg, _ = False, "", None
                            with get_db() as db:
                                r_ok, r_msg, _ = AuthService.resend_registration_otp(db=db, email=target_mail)
                            if r_ok:
                                st.session_state.reg_otp_email = target_mail.lower().strip()
                                st.success("✅ A fresh 6-digit OTP has been sent to your email.")
                                st.rerun()
                            else:
                                st.error(r_msg)

                with col_back:
                    if st.button("← Edit Details", use_container_width=True):
                        st.session_state.pop("reg_otp_email", None)
                        st.session_state.pop("reg_otp_name", None)
                        st.session_state.pop("reg_manual_otp", None)
                        st.rerun()

        # ----------------------------------------------------
        # TAB 3: PASSWORD RESET (Unified Single Flow -> Auto Navigate to Login)
        # ----------------------------------------------------
        else:
            st.markdown(
                "<div style='padding: 2px 0 10px 0;'>"
                "<h3 style='margin:0; font-size:1.15rem; color:#f8fafc;'>Reset Password</h3>"
                "<p style='color:#94a3b8; font-size:0.84rem; margin:3px 0 0 0;'>"
                "Receive a 6-digit reset code at your registered email address, verify it, and set your new password."
                "</p>"
                "</div>",
                unsafe_allow_html=True
            )

            code_sent_email = st.session_state.get("reset_code_sent_to", "")
            manual_entry = st.session_state.get("manual_code_entry", False)

            # Stage 1: Request Code
            if not code_sent_email and not manual_entry:
                with st.form("single_reset_req_form", clear_on_submit=False):
                    reset_email = st.text_input(
                        "Registered Account Email",
                        placeholder="name@company.com",
                        key="reset_flow_email"
                    ).strip()
                    submit_req = st.form_submit_button("Send Reset Code", use_container_width=True, type="primary")

                if submit_req:
                    if not reset_email:
                        st.error("Please enter your registered email address.")
                    else:
                        success, msg, _ = False, "", None
                        with get_db() as db:
                            success, msg, _ = AuthService.request_password_reset(db=db, email=reset_email)
                        if success:
                            st.session_state.reset_code_sent_to = reset_email.lower().strip()
                            st.rerun()
                        else:
                            st.error(msg)

                # Helper navigation options
                col_has_code, col_back_login = st.columns(2)
                with col_has_code:
                    if st.button("Have a Reset Code?", use_container_width=True, key="btn_already_have_code"):
                        st.session_state.manual_code_entry = True
                        st.rerun()
                with col_back_login:
                    if st.button("Remember Password? Sign In", use_container_width=True, key="btn_back_to_login_from_req"):
                        st.session_state.auth_tab = "🔑 Log In"
                        st.rerun()

            # Stage 2: Code Verification and Set Password in Same Flow
            else:
                active_email = code_sent_email or st.session_state.get("reset_exec_email", "")

                if code_sent_email:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 12px; padding: 12px 16px; margin-bottom: 14px;">
                        <span style="color: #34d399; font-weight: 600; font-size: 0.92rem;">📩 Reset code sent!</span>
                        <p style="color: #cbd5e1; font-size: 0.84rem; margin: 4px 0 0 0; line-height: 1.4;">
                            A 6-digit verification code has been dispatched to <b>{code_sent_email}</b>.
                            Please enter it below along with your new password.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with st.form("single_reset_exec_form", clear_on_submit=False):
                    email_for_reset = st.text_input(
                        "Registered Account Email",
                        value=active_email,
                        placeholder="name@company.com",
                        key="reset_exec_email"
                    ).strip()

                    reset_token = st.text_input(
                        "6-Digit Reset Code",
                        placeholder="e.g. 123456",
                        key="reset_token_input",
                        max_chars=10
                    ).strip()

                    new_pass = st.text_input(
                        "New Password",
                        type="password",
                        placeholder="At least 8 characters (A-Z, 0-9)",
                        key="reset_new_pass_input"
                    )

                    new_pass_conf = st.text_input(
                        "Confirm New Password",
                        type="password",
                        placeholder="Repeat new password",
                        key="reset_new_pass_conf_input"
                    )

                    submit_exec = st.form_submit_button("✅ Update Password", use_container_width=True, type="primary")

                if submit_exec:
                    if not reset_token or not new_pass or not new_pass_conf:
                        st.error("Please fill in all required fields.")
                    elif new_pass != new_pass_conf:
                        st.error("New passwords do not match.")
                    else:
                        success, msg = False, ""
                        with get_db() as db:
                            success, msg = AuthService.reset_password(
                                db=db,
                                token=reset_token,
                                new_password=new_pass,
                                email=email_for_reset or None
                            )
                        if success:
                            # Clean up reset state
                            st.session_state.pop("reset_code_sent_to", None)
                            st.session_state.pop("manual_code_entry", None)

                            # Automatically sign in and navigate directly to workspace dashboard
                            with get_db() as db:
                                auth_ok, auth_msg, auth_user = AuthService.authenticate(
                                    db=db,
                                    email=email_for_reset,
                                    password=new_pass
                                )
                            if auth_ok and auth_user:
                                st.session_state.authenticated = True
                                st.session_state.user = auth_user.to_dict()
                                st.session_state.user_id = auth_user.id
                                st.session_state.user_name = auth_user.name
                                st.session_state.user_email = auth_user.email
                                st.session_state.user_role = auth_user.role
                                st.session_state.nav_page = "documents"
                                st.toast(f"🎉 Password reset successfully! Welcome, {auth_user.name}!")
                                st.rerun()
                            else:
                                # Fallback: redirect to login tab if authentication fails
                                st.session_state.auth_tab = "🔑 Log In"
                                st.session_state.login_email = email_for_reset
                                st.session_state.login_flash_success = "Password updated successfully! Please sign in with your new password."
                                st.toast("✅ Password updated successfully!")
                                st.rerun()
                        else:
                            st.error(msg)

                # Resend & Return controls
                col_res, col_back = st.columns(2)
                with col_res:
                    if st.button("🔄 Resend Code", use_container_width=True, help="Send a fresh reset code to your email"):
                        target_mail = email_for_reset or code_sent_email
                        if not target_mail:
                            st.warning("Please enter your registered email above to resend.")
                        else:
                            r_ok, r_msg, _ = False, "", None
                            with get_db() as db:
                                r_ok, r_msg, _ = AuthService.request_password_reset(db=db, email=target_mail)
                            if r_ok:
                                st.session_state.reset_code_sent_to = target_mail.lower().strip()
                                st.success("✅ A new reset code has been sent.")
                                st.rerun()
                            else:
                                st.error(r_msg)

                with col_back:
                    if st.button("← Enter Different Email", use_container_width=True):
                        st.session_state.pop("reset_code_sent_to", None)
                        st.session_state.pop("manual_code_entry", None)
                        st.rerun()
