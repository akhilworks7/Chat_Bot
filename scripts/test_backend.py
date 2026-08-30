import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


from app.db.database import init_db, get_db
from app.db import crud
from app.services.auth_service import AuthService
from app.services.crypto_service import CryptoService
from app.services.credential_service import CredentialService
from app.config import settings

def run_tests():
    print("==================================================")
    print("Running Multi-User RAG Backend Verification Suite")
    print("==================================================")

    # 1. Initialize Database & Seed data
    print("\n[1/7] Initializing Database & Seed Data...")
    init_db()
    with get_db() as db:
        crud.seed_initial_data(db)
        admin = crud.get_user_by_email(db, "admin@documind.ai")
        assert admin is not None, "Admin user not found!"
        assert admin.role == "admin", "Admin role mismatch!"
        print(f"✅ Admin user initialized: {admin.email} (Role: {admin.role})")

    # 2. Test User Registration, Email Validation & Dedicated Pending Verification
    print("\n[2/7] Testing Email Validation, Pending Registration & OTP Verification...")
    with get_db() as db:
        # 2a. Test Invalid Email Formats
        for invalid_email in ["notanemail", "user@", "@domain.com", "user@domain", "user..name@domain.com"]:
            ok, err, _ = AuthService.initiate_registration(db, "Bad User", invalid_email, "SecurePassword123!")
            assert not ok, f"Expected invalid email rejection for '{invalid_email}'"
        print("✅ Strict email format validation rejected invalid email patterns successfully.")

        # 2b. Initiate Registration (Staged in pending_registrations, not users table)
        test_email = "testuser@documind.ai"
        existing = crud.get_user_by_email(db, test_email)
        if existing:
            crud.delete_user(db, existing.id)
        crud.delete_pending_registration(db, test_email)

        success, msg, dev_token = AuthService.initiate_registration(
            db=db,
            name="Test User",
            email=test_email,
            password="SecurePassword123!"
        )
        assert success, f"Initiate registration failed: {msg}"

        # Verify NO record exists in users table yet
        unverified_user = crud.get_user_by_email(db, test_email)
        assert unverified_user is None, "User should NOT be stored in users table before email verification!"

        pending_reg = crud.get_pending_registration(db, test_email)
        assert pending_reg is not None, "Pending registration must exist!"
        otp_to_test = pending_reg.otp_code
        print(f"✅ User staged in pending table with OTP ({otp_to_test}). Users table remains clean.")

        # 2c. Test Resending Verification Code
        res_ok, res_msg, _ = AuthService.resend_registration_otp(db, test_email)
        assert res_ok is True
        pending_after_resend = crud.get_pending_registration(db, test_email)
        new_otp_to_test = pending_after_resend.otp_code
        print(f"✅ Resend OTP succeeded. New OTP: {new_otp_to_test}")

        # 2d. Test Invalid Code Rejection
        v_bad_ok, v_bad_msg, _ = AuthService.complete_registration(db, test_email, "000000")
        assert v_bad_ok is False
        print("✅ Incorrect verification code rejected properly.")

        # 2e. Complete Registration with Valid Code
        v_ok, v_msg, verified_user = AuthService.complete_registration(db, test_email, new_otp_to_test)
        assert v_ok is True, f"Verification failed: {v_msg}"
        assert verified_user is not None
        assert verified_user.email_verified is True
        user_id = verified_user.id
        print(f"✅ Email verified and User activated: {verified_user.email}")

        # 2f. Test Authentication Allowed After Activation
        auth_ok, auth_msg, auth_user = AuthService.authenticate(db, test_email, "SecurePassword123!")
        assert auth_ok is True, f"Authentication failed: {auth_msg}"
        print(f"✅ Authentication successful for {auth_user.email}")

    # 3. Test Symmetric Encryption for BYOK Keys
    print("\n[3/7] Testing API Key Encryption/Decryption (Fernet)...")
    crypto = CryptoService()
    test_key = "pcsk_secret_test_pinecone_key_12345"
    encrypted = crypto.encrypt(test_key)
    decrypted = crypto.decrypt(encrypted)
    assert decrypted == test_key, "Encryption/Decryption roundtrip mismatch!"
    masked = crypto.mask_api_key(test_key)
    print(f"✅ Plaintext: {test_key[:10]}... -> Encrypted: {encrypted[:20]}... -> Decrypted: {decrypted[:10]}...")
    print(f"✅ Masked UI output: {masked}")

    # 4. Test Hybrid Credential Resolution (Application Mode)
    print("\n[4/7] Testing Hybrid Credential Resolution (Application Mode)...")
    with get_db() as db:
        creds = CredentialService.get_credentials(user_id=user_id, db=db)
        assert creds["mode"] == "application", f"Expected application mode, got {creds['mode']}"
        assert creds["namespace"] == f"user_{user_id}", f"Namespace mismatch: {creds['namespace']}"
        print(f"✅ Resolved mode: {creds['mode']} | Namespace: {creds['namespace']} | Pinecone index: {creds['pinecone_index']}")

    # 5. Test Document Limit Enforcement for Application Mode
    print("\n[5/7] Testing Free Document Limit Enforcement...")
    with get_db() as db:
        allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
        assert allowance["allowed"] is True, "Expected allowed for new user"
        assert allowance["used"] == 0
        assert allowance["limit"] == 2
        print(f"✅ Initial quota check: {allowance['used']}/{allowance['limit']} used (Allowed: {allowance['allowed']})")

        # Simulate uploading 2 documents
        doc1 = crud.create_document(db, user_id=user_id, file_name="doc1.pdf", file_size=1024, file_path="data/doc1.pdf", pinecone_namespace=f"user_{user_id}", vector_count=10, credential_mode="application")
        doc2 = crud.create_document(db, user_id=user_id, file_name="doc2.pdf", file_size=2048, file_path="data/doc2.pdf", pinecone_namespace=f"user_{user_id}", vector_count=15, credential_mode="application")

        allowance_after_2 = CredentialService.check_upload_allowance(user_id=user_id, db=db)
        assert allowance_after_2["used"] == 2
        assert allowance_after_2["allowed"] is False, "Expected allowed=False after reaching limit of 2"
        print(f"✅ Limit reached check: {allowance_after_2['used']}/{allowance_after_2['limit']} used (Allowed: {allowance_after_2['allowed']})")

    # 6. Test Switching to BYOK Mode & Limit Removal
    print("\n[6/7] Testing BYOK Mode Upgrade & Limit Bypass...")
    with get_db() as db:
        crud.upsert_user_credentials(
            db=db,
            user_id=user_id,
            pinecone_api_key_encrypted=crypto.encrypt("pcsk_user_custom_pinecone_key"),
            pinecone_index="custom-user-index",
            groq_api_key_encrypted=crypto.encrypt("gsk_user_custom_groq_key"),
            groq_model="llama-3.3-70b-versatile"
        )
        byok_creds = CredentialService.get_credentials(user_id=user_id, db=db)
        assert byok_creds["mode"] == "user", f"Expected user mode, got {byok_creds['mode']}"
        assert byok_creds["is_byok"] is True
        print(f"✅ Credential switched to: {byok_creds['mode']} ({byok_creds['mode_label']})")

        byok_allowance = CredentialService.check_upload_allowance(user_id=user_id, db=db)
        assert byok_allowance["allowed"] is True, "BYOK user must be allowed unlimited uploads"
        assert byok_allowance["is_unlimited"] is True
        print(f"✅ BYOK quota check: Allowed={byok_allowance['allowed']} (Unlimited={byok_allowance['is_unlimited']})")

    # 7. Test Admin Dashboard Analytics & User Management
    print("\n[7/7] Testing Admin Dashboard Analytics...")
    with get_db() as db:
        metrics = crud.get_admin_dashboard_metrics(db)
        assert metrics["total_users"] >= 2, "Expected at least 2 users (admin + testuser)"
        print(f"✅ Admin Metrics: Total Users={metrics['total_users']}, Active={metrics['active_users']}, App Users={metrics['app_credential_users']}, BYOK Users={metrics['byok_users']}")

    print("\n==================================================")
    print("🎉 ALL 7 TEST SUITES PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
