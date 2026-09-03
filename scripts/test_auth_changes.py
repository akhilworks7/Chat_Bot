import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.db.database import init_db, get_db
from app.db import crud
from app.services.auth_service import AuthService

def run_auth_verification():
    print("==================================================")
    print("Testing Updated Authentication & Password Reset Flow")
    print("==================================================")

    init_db()

    with get_db() as db:
        # 1. Test OTP-based Registration Flow
        print("\n[1/4] Testing OTP-based Registration (Enter Details -> OTP -> Activate Account)...")
        reg_user_email = "otp_reg_user@documind.ai"
        existing = crud.get_user_by_email(db, reg_user_email)
        if existing:
            crud.delete_user(db, existing.id)
        crud.delete_pending_registration(db, reg_user_email)

        # 1a. User submits registration details -> OTP generated
        init_ok, init_msg, dev_token = AuthService.initiate_registration(
            db=db,
            name="OTP Reg User",
            email=reg_user_email,
            password="SecurePassword123!"
        )
        assert init_ok is True, f"Initiate registration failed: {init_msg}"
        pending_reg = crud.get_pending_registration(db, reg_user_email)
        assert pending_reg is not None, "Pending registration was not created in DB"
        otp_code = pending_reg.otp_code
        print(f"✅ Registration OTP dispatched for {reg_user_email}: {otp_code}")

        # Ensure user does NOT exist in users table yet (cannot log in before OTP verification)
        unverified_user = crud.get_user_by_email(db, reg_user_email)
        assert unverified_user is None, "User should not exist in users table before OTP verification!"
        
        login_before_otp_ok, _, _ = AuthService.authenticate(db, reg_user_email, "SecurePassword123!")
        assert login_before_otp_ok is False, "Login must NOT succeed before OTP verification!"
        print("✅ User cannot log in prior to OTP verification.")

        # 1b. Complete registration with valid OTP
        comp_ok, comp_msg, new_user = AuthService.complete_registration(
            db=db,
            email=reg_user_email,
            otp_code=otp_code
        )
        assert comp_ok is True, f"OTP verification failed: {comp_msg}"
        assert new_user is not None
        assert new_user.email_verified is True
        print(f"✅ Account verified and activated via OTP: {new_user.email}")

        # 2. Test Direct Login with Registered Credentials (No email verification on login page)
        print("\n[2/4] Testing Direct Login with Registered Credentials...")
        auth_ok, auth_msg, auth_user = AuthService.authenticate(
            db=db,
            email=reg_user_email,
            password="SecurePassword123!"
        )
        assert auth_ok is True, f"Direct authentication failed: {auth_msg}"
        assert auth_user is not None
        print(f"✅ Direct login succeeded with registered credentials for {auth_user.email}")

        # 3. Test Reset Password with Unregistered Email
        print("\n[3/4] Testing Reset Password with Unregistered Email...")
        unreg_email = "nonexistent_user_999@example.com"
        req_ok, req_msg, req_tok = AuthService.request_password_reset(db=db, email=unreg_email)
        assert req_ok is False, "Expected request_password_reset to return False for unregistered email"
        assert req_msg == "User not found. Please enter a registered email address.", f"Unexpected error message: {req_msg}"
        print(f"✅ Unregistered email correctly rejected with: '{req_msg}'")

        # 4. Test Unified Password Reset Flow
        print("\n[4/4] Testing Unified Reset Flow (Request Code -> Verify & Set Password -> Login)...")
        reset_ok, reset_msg, dev_tok = AuthService.request_password_reset(db=db, email=reg_user_email)
        assert reset_ok is True, f"Password reset request failed: {reset_msg}"
        from app.db.models import PasswordResetToken
        tok_record = db.query(PasswordResetToken).filter(
            PasswordResetToken.user_id == new_user.id,
            PasswordResetToken.used == False
        ).order_by(PasswordResetToken.id.desc()).first()
        assert tok_record is not None, "No reset token found in database"
        reset_tok = tok_record.token
        print(f"✅ Password reset code generated: {reset_tok}")

        # Test setting password with incorrect token
        bad_token_ok, bad_token_msg = AuthService.reset_password(
            db=db,
            token="999999",
            new_password="NewSecurePassword456!",
            email=reg_user_email
        )
        assert bad_token_ok is False
        print("✅ Invalid reset token rejected properly.")

        # Test setting password with valid token and email matching
        set_ok, set_msg = AuthService.reset_password(
            db=db,
            token=reset_tok,
            new_password="NewSecurePassword456!",
            email=reg_user_email
        )
        assert set_ok is True, f"Password reset execution failed: {set_msg}"
        print(f"✅ Password successfully updated: {set_msg}")

        # Authenticate with OLD password (should fail)
        old_auth_ok, old_auth_msg, _ = AuthService.authenticate(
            db=db,
            email=reg_user_email,
            password="SecurePassword123!"
        )
        assert old_auth_ok is False, "Old password should no longer work"
        print("✅ Old password rejected after reset.")

        # Authenticate with NEW password (should succeed)
        new_auth_ok, new_auth_msg, reset_login_user = AuthService.authenticate(
            db=db,
            email=reg_user_email,
            password="NewSecurePassword456!"
        )
        assert new_auth_ok is True, f"New password login failed: {new_auth_msg}"
        print(f"✅ Login with new password succeeded for {reset_login_user.email}")

        # Clean up test user
        crud.delete_user(db, new_user.id)
        print("✅ Cleaned up test user.")

    print("\n==================================================")
    print("🎉 ALL 4 AUTHENTICATION FLOW TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_auth_verification()
