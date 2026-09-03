import secrets
import string
import re
from typing import Optional, Tuple, Dict, Any
import bcrypt
from sqlalchemy.orm import Session
from app.db import crud
from app.db.models import User
from app.services.email_service import EmailService
from app.utils.logger import get_logger

logger = get_logger("auth_service")


class AuthService:
    """
    Handles user authentication, password hashing, token generation, and account validation.
    """

    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str]:
        """
        Validates email format strictly against RFC standards.
        Requires valid username, domain, and top-level domain (e.g. .com, .org, .ai).
        """
        if not email or not email.strip():
            return False, "Email address cannot be empty."

        email = email.strip().lower()
        if len(email) > 254:
            return False, "Email address is too long (maximum 254 characters)."

        # Must have exactly one @
        if email.count("@") != 1:
            return False, "Invalid email format. Must contain exactly one '@' symbol."

        # Disallow consecutive dots or misplaced dots anywhere
        if ".." in email or email.startswith(".") or email.endswith("."):
            return False, "Invalid email format with consecutive or misplaced dots."

        parts = email.split("@")
        local_part, domain_part = parts[0], parts[1]

        if not local_part or not domain_part:
            return False, "Please provide a complete email address (e.g. user@example.com)."

        if local_part.startswith(".") or local_part.endswith("."):
            return False, "Username in email cannot start or end with a dot."

        # Domain must contain at least one dot
        if "." not in domain_part:
            return False, "Invalid domain name. Please include a domain extension such as .com, .org, or .ai."

        domain_pieces = domain_part.split(".")
        for piece in domain_pieces:
            if not piece or len(piece) < 1:
                return False, "Invalid domain format with consecutive or misplaced dots."

        # Top-level domain (last piece) must be at least 2 alphabetic characters
        tld = domain_pieces[-1]
        if len(tld) < 2 or not tld.isalpha():
            return False, "Invalid domain extension. Domain extension must be at least 2 letters (e.g. .com, .org, .ai)."

        # Strict RFC-compliant regex pattern
        pattern = r"^[a-zA-Z0-9_+-]+(?:\.[a-zA-Z0-9_+-]+)*@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.[a-zA-Z]{2,}$"
        if not re.match(pattern, email):
            return False, "Please provide a valid email address (e.g. user@example.com)."

        return True, "Email format is valid."

    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, str]:
        """
        Validates password strength: at least 8 characters, containing uppercase, lowercase, and digits.
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long."
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter (A-Z)."
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter (a-z)."
        if not re.search(r"[0-9]", password):
            return False, "Password must contain at least one numeric digit (0-9)."
        return True, "Password meets strength requirements."

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hashes password using bcrypt with salt.
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """
        Verifies plaintext password against stored bcrypt hash.
        """
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

    @staticmethod
    def generate_token(length: int = 6) -> str:
        """
        Generates a secure numeric or alphanumeric code.
        """
        digits = string.digits
        return "".join(secrets.choice(digits) for _ in range(length))

    @classmethod
    def initiate_registration(
        cls,
        db: Session,
        name: str,
        email: str,
        password: str,
        role: str = "user"
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Validates details, creates a pending registration with a 10-minute OTP, and sends verification email.
        NOTE: Does NOT create an unverified record in the users table!
        Returns: (success: bool, message: str, dev_token: Optional[str])
        """
        email = email.lower().strip()
        name = name.strip()

        is_valid_email, email_err = cls.validate_email(email)
        if not is_valid_email:
            return False, email_err, None

        if not name:
            return False, "Please enter your full name.", None

        # Check existing active user
        existing_user = crud.get_user_by_email(db, email)
        if existing_user:
            return False, "An account with this email address already exists. Please log in.", None

        # Check password strength
        valid_pwd, pwd_msg = cls.validate_password_strength(password)
        if not valid_pwd:
            return False, pwd_msg, None

        # Generate 6-digit OTP code and hash password
        otp_code = cls.generate_token(6)
        hashed_password = cls.hash_password(password)

        # Store in pending_registrations table (valid for 10 minutes)
        crud.create_or_update_pending_registration(
            db=db,
            name=name,
            email=email,
            password_hash=hashed_password,
            otp_code=otp_code,
            expires_minutes=10
        )
        try:
            db.commit()
        except Exception:
            db.flush()

        # Dispatch email
        email_ok, email_msg = EmailService.send_verification_email(
            to_email=email,
            name=name,
            token=otp_code,
            db=db
        )

        if not email_ok and EmailService.is_smtp_configured(db):
            return False, f"Could not send email: {email_msg}", None

        dev_token = otp_code if not EmailService.is_smtp_configured(db) else None
        return True, "Verification code sent to your email.", dev_token

    @classmethod
    def resend_registration_otp(cls, db: Session, email: str) -> Tuple[bool, str, Optional[str]]:
        """
        Invalidates the old verification code, generates a fresh code, and sends a new email.
        """
        clean_email = email.lower().strip() if email else ""
        if not clean_email:
            return False, "Please provide your email address to resend the code.", None

        pending = crud.get_pending_registration(db, clean_email)
        if not pending:
            existing = crud.get_user_by_email(db, clean_email)
            if existing:
                return False, "This account is already registered and verified! Please log in.", None
            return False, f"No pending registration found for '{clean_email}'. Please create your account.", None

        new_otp = cls.generate_token(6)
        crud.create_or_update_pending_registration(
            db=db,
            name=pending.name,
            email=pending.email,
            password_hash=pending.password_hash,
            otp_code=new_otp,
            expires_minutes=10
        )
        try:
            db.commit()
        except Exception:
            db.flush()

        email_ok, email_msg = EmailService.send_verification_email(
            to_email=pending.email,
            name=pending.name,
            token=new_otp,
            db=db
        )

        if not email_ok and EmailService.is_smtp_configured(db):
            return False, f"Could not send email: {email_msg}", None

        dev_token = new_otp if not EmailService.is_smtp_configured(db) else None
        return True, "A new 6-digit verification code has been sent to your email.", dev_token

    @classmethod
    def complete_registration(cls, db: Session, email: str, otp_code: str) -> Tuple[bool, str, Optional[User]]:
        """
        Validates the OTP from pending_registrations. If valid, creates the User in the database.
        Resilient to browser refresh: if email is empty in session, attempts lookup by unique OTP.
        """
        clean_email = email.lower().strip() if email else ""
        clean_code = otp_code.strip()

        if not clean_code:
            return False, "Please enter the 6-digit verification code.", None

        # 1. Look up pending registration by email
        pending = None
        if clean_email:
            pending = crud.get_pending_registration(db, clean_email)

        # 2. Fallback: if email wasn't provided or session was reset, find by OTP code
        if not pending:
            from app.db.models import PendingRegistration
            pending = db.query(PendingRegistration).filter(PendingRegistration.otp_code == clean_code).first()
            if pending:
                clean_email = pending.email

        # 3. Check legacy or existing unverified user tokens in EmailVerificationToken table
        if not pending and clean_email:
            user = crud.verify_email_token(db, clean_code)
            if user and user.email.lower() == clean_email:
                return True, "Email verified successfully! You can now log in.", user

            existing_user = crud.get_user_by_email(db, clean_email)
            if existing_user and existing_user.email_verified:
                return True, "Your account is already active and verified! You can log in directly.", existing_user

        if not pending:
            if clean_email:
                return False, f"No pending registration found for '{clean_email}'. Please click '← Back to Register' to create your account.", None
            return False, "No pending registration found. Please check your email or click '← Back to Register'.", None

        # Check maximum attempts (brute force protection)
        if pending.attempts >= 5:
            return False, "Too many failed attempts. Please click 'Resend Code' to request a new code.", None

        # Check expiry
        import datetime
        if datetime.datetime.utcnow() > pending.expires_at:
            return False, "⚠️ Verification code has expired (codes are valid for 10 minutes). Please click 'Resend Code'.", None

        # Check code
        if pending.otp_code != clean_code:
            attempts_left = max(0, 5 - crud.increment_pending_attempts(db, pending.email))
            return False, f"Incorrect verification code. ({attempts_left} attempts remaining)", None

        # Check if user was already created in the meantime
        existing = crud.get_user_by_email(db, pending.email)
        if existing:
            crud.delete_pending_registration(db, pending.email)
            return True, "Account is already active. You can now log in.", existing

        # Create verified active user
        user = crud.create_user(
            db=db,
            name=pending.name,
            email=pending.email,
            password_hash=pending.password_hash,
            role="user",
            email_verified=True
        )

        # Remove from pending table
        crud.delete_pending_registration(db, pending.email)

        # Audit log
        crud.create_audit_log(
            db=db,
            action="USER_REGISTER_AND_VERIFIED",
            user_id=user.id,
            details=f"User {user.email} completed email verification and account activation"
        )

        db.commit()
        try:
            from app.services.cloud_sync_service import CloudSyncService
            CloudSyncService.backup_database_to_cloud()
        except Exception:
            pass

        return True, "Email verified and account activated successfully!", user

    @classmethod
    def register_direct(
        cls,
        db: Session,
        name: str,
        email: str,
        password: str,
        role: str = "user"
    ) -> Tuple[bool, str, Optional[User]]:
        """
        Directly creates an active registered user account without email verification gating.
        """
        email = email.lower().strip()
        name = name.strip()

        is_valid_email, email_err = cls.validate_email(email)
        if not is_valid_email:
            return False, email_err, None

        if not name:
            return False, "Please enter your full name.", None

        existing_user = crud.get_user_by_email(db, email)
        if existing_user:
            return False, "An account with this email address already exists. Please log in.", None

        valid_pwd, pwd_msg = cls.validate_password_strength(password)
        if not valid_pwd:
            return False, pwd_msg, None

        hashed_password = cls.hash_password(password)
        user = crud.create_user(
            db=db,
            name=name,
            email=email,
            password_hash=hashed_password,
            role=role,
            email_verified=True
        )

        crud.create_audit_log(
            db=db,
            action="USER_REGISTER_DIRECT",
            user_id=user.id,
            details=f"User {user.email} registered account directly"
        )

        return True, "Account created successfully!", user

    @classmethod
    def register(
        cls,
        db: Session,
        name: str,
        email: str,
        password: str,
        role: str = "user"
    ) -> Tuple[bool, str, Optional[User], Optional[str]]:
        """
        Backwards-compatible registration helper.
        """
        ok, msg, dev_tok = cls.initiate_registration(db, name, email, password, role)
        if not ok:
            return False, msg, None, None
        return True, msg, None, dev_tok

    @classmethod
    def verify_email(cls, db: Session, token: str) -> Tuple[bool, str, Optional[User]]:
        """
        Verifies legacy or password reset tokens.
        """
        token = token.strip()
        user = crud.verify_email_token(db, token)
        if not user:
            return False, "Invalid or expired verification code.", None

        crud.create_audit_log(
            db=db,
            action="EMAIL_VERIFIED",
            user_id=user.id,
            details=f"Email verified for {user.email}"
        )
        return True, "Email successfully verified! You can now log in.", user

    @classmethod
    def authenticate(
        cls,
        db: Session,
        email: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, str, Optional[User]]:
        """
        Authenticates user with email and password directly using registered credentials.
        No email verification requirement on login.
        """
        email = email.lower().strip()
        is_valid_email, email_err = cls.validate_email(email)
        if not is_valid_email:
            return False, email_err, None

        user = crud.get_user_by_email(db, email)
        if not user:
            return False, "Invalid email or password.", None

        if not user.is_active:
            return False, "Your account has been deactivated. Please contact an administrator.", None

        if not cls.verify_password(password, user.password_hash):
            crud.create_audit_log(
                db=db,
                action="LOGIN_FAILED",
                user_id=user.id,
                details=f"Failed login attempt for {email}",
                ip_address=ip_address
            )
            return False, "Invalid email or password.", None

        # Update last login
        crud.update_user_last_login(db, user.id)

        # Audit log
        crud.create_audit_log(
            db=db,
            action="LOGIN_SUCCESS",
            user_id=user.id,
            details=f"Successful login for {email}",
            ip_address=ip_address
        )

        return True, "Login successful.", user

    @classmethod
    def request_password_reset(cls, db: Session, email: str) -> Tuple[bool, str, Optional[str]]:
        """
        Creates a password reset token and sends it.
        If user is not found, clearly informs the caller.
        """
        email = email.lower().strip()
        is_valid_email, email_err = cls.validate_email(email)
        if not is_valid_email:
            return False, email_err, None

        user = crud.get_user_by_email(db, email)
        if not user:
            return False, "User not found. Please enter a registered email address.", None

        token_str = cls.generate_token(6)
        crud.create_password_reset_token(db, user_id=user.id, token=token_str)
        try:
            db.commit()
        except Exception:
            db.flush()
        EmailService.send_password_reset_email(to_email=user.email, name=user.name, token=token_str, db=db)

        crud.create_audit_log(
            db=db,
            action="PASSWORD_RESET_REQUESTED",
            user_id=user.id,
            details=f"Password reset requested for {email}"
        )
        dev_token = token_str if not EmailService.is_smtp_configured(db) else None
        return True, f"A password reset code has been sent to {user.email}.", dev_token

    @classmethod
    def reset_password(cls, db: Session, token: str, new_password: str, email: Optional[str] = None) -> Tuple[bool, str]:
        """
        Resets user password using a valid token, optionally matching email.
        """
        token = token.strip()
        user = crud.verify_password_reset_token(db, token)
        if not user:
            return False, "Invalid or expired reset code."

        if email and user.email.lower().strip() != email.lower().strip():
            return False, "This reset code does not match the provided email address."

        valid, msg = cls.validate_password_strength(new_password)
        if not valid:
            return False, msg

        user.password_hash = cls.hash_password(new_password)
        crud.mark_password_reset_token_used(db, token)

        crud.create_audit_log(
            db=db,
            action="PASSWORD_RESET_COMPLETED",
            user_id=user.id,
            details=f"Password successfully reset for {user.email}"
        )
        db.commit()
        try:
            from app.services.cloud_sync_service import CloudSyncService
            CloudSyncService.backup_database_to_cloud()
        except Exception:
            pass

        return True, "Password has been reset successfully! You can now log in."

    @classmethod
    def purge_user(cls, db: Session, user_id: int) -> Tuple[bool, str]:
        """
        Permanently and completely purges ALL data belonging to a user across:
        1. Physical disk storage (data/documents/user_<id>/)
        2. Pinecone Vector database (user_<id> namespace in App index and BYOK index)
        3. All database tables (documents, credentials, history, statistics, tokens, audit logs, user record)
        """
        import os
        import shutil
        from app.config import settings

        user = crud.get_user_by_id(db, user_id)
        if not user:
            return False, f"User #{user_id} not found."

        user_email = user.email
        namespace = f"user_{user_id}"

        # 1. Purge Pinecone Vector Namespaces (Application index)
        try:
            from app.services.vector_service import VectorService
            vec_svc = VectorService()
            vec_svc.delete_namespace(namespace=namespace)
            logger.info(f"Purged application Pinecone namespace '{namespace}'")
        except Exception as e:
            logger.warning(f"Note on app Pinecone namespace cleanup for {namespace}: {e}")

        # 2. Purge BYOK Pinecone Vector Namespace if user had custom keys
        try:
            creds = crud.get_user_credentials(db, user_id)
            if creds and creds.pinecone_api_key_encrypted:
                from app.services.credential_service import CredentialService
                byok_key = CredentialService.decrypt_key(creds.pinecone_api_key_encrypted)
                byok_index = creds.pinecone_index or settings.PINECONE_INDEX
                if byok_key:
                    byok_vec = VectorService(api_key=byok_key, index_name=byok_index)
                    byok_vec.delete_namespace(namespace=namespace)
                    logger.info(f"Purged BYOK Pinecone namespace '{namespace}' on index '{byok_index}'")
        except Exception as e:
            logger.warning(f"Note on BYOK Pinecone namespace cleanup for {namespace}: {e}")

        # 3. Purge Physical Files on Disk
        user_folder = os.path.join(settings.DOCUMENTS_DIR, namespace)
        if os.path.exists(user_folder):
            try:
                shutil.rmtree(user_folder, ignore_errors=True)
                logger.info(f"Purged local document folder on disk: {user_folder}")
            except Exception as e:
                logger.error(f"Error removing user directory {user_folder}: {e}")

        # 4. Cascade Delete from all Database Tables
        crud.delete_user(db, user_id)
        db.commit()

        # 5. Immediately sync deletion across cloud storage so it is removed everywhere
        try:
            from app.services.cloud_sync_service import CloudSyncService
            CloudSyncService.backup_database_to_cloud()
        except Exception:
            pass

        logger.info(f"Successfully purged all data for user #{user_id} ({user_email})")
        return True, f"User #{user_id} ({user_email}) and all associated data permanently deleted."

