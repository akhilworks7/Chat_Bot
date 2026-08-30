import json
import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, or_
from app.db.models import (
    User, UserCredential, Document, ChatHistory,
    UsageStatistic, EmailVerificationToken, PasswordResetToken,
    AuditLog, SystemSetting, PendingRegistration
)
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("crud")


# ==========================================
# USERS CRUD
# ==========================================
def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(func.lower(User.email) == email.lower().strip()).first()


def create_user(
    db: Session,
    name: str,
    email: str,
    password_hash: str,
    role: str = "user",
    email_verified: bool = False
) -> User:
    user = User(
        name=name.strip(),
        email=email.lower().strip(),
        password_hash=password_hash,
        role=role,
        email_verified=email_verified,
        is_active=True
    )
    db.add(user)
    db.flush()

    # Initialize empty usage statistics
    usage = UsageStatistic(
        user_id=user.id,
        documents_uploaded=0,
        storage_used=0,
        vector_count=0,
        query_count=0,
        credential_mode="application"
    )
    db.add(usage)
    db.flush()
    return user


def update_user_last_login(db: Session, user_id: int):
    user = get_user_by_id(db, user_id)
    if user:
        user.last_login = datetime.datetime.utcnow()
        db.flush()


def update_user_status(db: Session, user_id: int, is_active: bool) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if user:
        user.is_active = is_active
        db.flush()
    return user


def update_user_role(db: Session, user_id: int, role: str) -> Optional[User]:
    user = get_user_by_id(db, user_id)
    if user:
        user.role = role
        db.flush()
    return user


def delete_user(db: Session, user_id: int) -> bool:
    """
    Deletes user and cascades deletion to all child tables.
    """
    user = get_user_by_id(db, user_id)
    if user:
        # Explicit child record cleanup to ensure zero orphaned data
        clean_email = user.email.lower().strip()
        db.query(ChatHistory).filter(ChatHistory.user_id == user_id).delete(synchronize_session=False)
        db.query(Document).filter(Document.user_id == user_id).delete(synchronize_session=False)
        db.query(UserCredential).filter(UserCredential.user_id == user_id).delete(synchronize_session=False)
        db.query(UsageStatistic).filter(UsageStatistic.user_id == user_id).delete(synchronize_session=False)
        db.query(EmailVerificationToken).filter(EmailVerificationToken.user_id == user_id).delete(synchronize_session=False)
        db.query(PasswordResetToken).filter(PasswordResetToken.user_id == user_id).delete(synchronize_session=False)
        db.query(PendingRegistration).filter(func.lower(PendingRegistration.email) == clean_email).delete(synchronize_session=False)
        db.query(AuditLog).filter(AuditLog.user_id == user_id).delete(synchronize_session=False)
        
        db.delete(user)
        db.flush()
        return True
    return False


def get_all_users(
    db: Session,
    search: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[User]:
    query = db.query(User)
    if search:
        s = f"%{search.lower()}%"
        query = query.filter(or_(func.lower(User.name).like(s), func.lower(User.email).like(s)))
    if role:
        query = query.filter(User.role == role)
    return query.order_by(desc(User.created_at)).offset(offset).limit(limit).all()


def count_all_users(db: Session) -> int:
    return db.query(func.count(User.id)).scalar() or 0


# ==========================================
# USER CREDENTIALS CRUD
# ==========================================
def get_user_credentials(db: Session, user_id: int) -> Optional[UserCredential]:
    return db.query(UserCredential).filter(UserCredential.user_id == user_id).first()


def upsert_user_credentials(
    db: Session,
    user_id: int,
    pinecone_api_key_encrypted: Optional[str],
    pinecone_index: Optional[str],
    groq_api_key_encrypted: Optional[str],
    groq_model: Optional[str]
) -> UserCredential:
    creds = get_user_credentials(db, user_id)
    if not creds:
        creds = UserCredential(
            user_id=user_id,
            pinecone_api_key_encrypted=pinecone_api_key_encrypted,
            pinecone_index=pinecone_index,
            groq_api_key_encrypted=groq_api_key_encrypted,
            groq_model=groq_model or "openai/gpt-oss-120b"
        )
        db.add(creds)
    else:
        if pinecone_api_key_encrypted is not None:
            creds.pinecone_api_key_encrypted = pinecone_api_key_encrypted
        if pinecone_index is not None:
            creds.pinecone_index = pinecone_index
        if groq_api_key_encrypted is not None:
            creds.groq_api_key_encrypted = groq_api_key_encrypted
        if groq_model is not None:
            creds.groq_model = groq_model
        creds.updated_at = datetime.datetime.utcnow()

    # Update credential mode in usage statistics
    has_keys = bool(creds.pinecone_api_key_encrypted and creds.groq_api_key_encrypted)
    usage = get_or_create_usage_statistics(db, user_id)
    usage.credential_mode = "user" if has_keys else "application"

    db.flush()
    return creds


def delete_user_credentials(db: Session, user_id: int) -> bool:
    creds = get_user_credentials(db, user_id)
    if creds:
        db.delete(creds)
        usage = get_or_create_usage_statistics(db, user_id)
        usage.credential_mode = "application"
        db.flush()
        return True
    return False


# ==========================================
# DOCUMENTS CRUD
# ==========================================
def create_document(
    db: Session,
    user_id: int,
    file_name: str,
    file_size: int,
    file_path: str,
    pinecone_namespace: str,
    vector_count: int,
    credential_mode: str = "application",
    status: str = "indexed"
) -> Document:
    doc = Document(
        user_id=user_id,
        file_name=file_name,
        file_size=file_size,
        file_path=file_path,
        pinecone_namespace=pinecone_namespace,
        vector_count=vector_count,
        credential_mode=credential_mode,
        status=status
    )
    db.add(doc)
    db.flush()

    # Update user usage statistics
    usage = get_or_create_usage_statistics(db, user_id)
    usage.documents_uploaded = db.query(func.count(Document.id)).filter(Document.user_id == user_id, Document.status == "indexed").scalar() or 0
    usage.storage_used = db.query(func.sum(Document.file_size)).filter(Document.user_id == user_id, Document.status == "indexed").scalar() or 0
    usage.vector_count = db.query(func.sum(Document.vector_count)).filter(Document.user_id == user_id, Document.status == "indexed").scalar() or 0
    usage.last_activity = datetime.datetime.utcnow()
    db.flush()
    return doc


def get_user_documents(db: Session, user_id: int) -> List[Document]:
    return db.query(Document).filter(Document.user_id == user_id, Document.status != "deleted").order_by(desc(Document.created_at)).all()


def get_user_document_by_name(db: Session, user_id: int, file_name: str) -> Optional[Document]:
    return db.query(Document).filter(Document.user_id == user_id, Document.file_name == file_name, Document.status != "deleted").first()


def get_document_by_id(db: Session, doc_id: int) -> Optional[Document]:
    return db.query(Document).filter(Document.id == doc_id).first()


def delete_document(db: Session, doc_id: int) -> Optional[Document]:
    doc = get_document_by_id(db, doc_id)
    if doc:
        user_id = doc.user_id
        db.delete(doc)
        db.flush()

        # Update usage statistics
        usage = get_or_create_usage_statistics(db, user_id)
        usage.documents_uploaded = db.query(func.count(Document.id)).filter(Document.user_id == user_id, Document.status == "indexed").scalar() or 0
        usage.storage_used = db.query(func.sum(Document.file_size)).filter(Document.user_id == user_id, Document.status == "indexed").scalar() or 0
        usage.vector_count = db.query(func.sum(Document.vector_count)).filter(Document.user_id == user_id, Document.status == "indexed").scalar() or 0
        usage.last_activity = datetime.datetime.utcnow()
        db.flush()
        return doc
    return None


def count_user_app_credential_documents(db: Session, user_id: int) -> int:
    """
    Returns count of documents uploaded under application shared credentials.
    """
    return db.query(func.count(Document.id)).filter(
        Document.user_id == user_id,
        Document.credential_mode == "application",
        Document.status == "indexed"
    ).scalar() or 0


# ==========================================
# CHAT HISTORY CRUD
# ==========================================
def add_chat_message(
    db: Session,
    user_id: int,
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    response_time_ms: float = 0.0
) -> ChatHistory:
    sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
    msg = ChatHistory(
        user_id=user_id,
        session_id=session_id,
        role=role,
        content=content,
        sources_json=sources_json,
        response_time_ms=response_time_ms
    )
    db.add(msg)

    # Increment query count in usage if it is a user message or response
    if role == "assistant":
        usage = get_or_create_usage_statistics(db, user_id)
        usage.query_count = (usage.query_count or 0) + 1
        usage.last_activity = datetime.datetime.utcnow()

    db.flush()
    return msg


def get_chat_history(db: Session, user_id: int, session_id: Optional[str] = None, limit: int = 50) -> List[ChatHistory]:
    query = db.query(ChatHistory).filter(ChatHistory.user_id == user_id)
    if session_id:
        query = query.filter(ChatHistory.session_id == session_id)
    return query.order_by(ChatHistory.created_at.asc()).limit(limit).all()


def clear_chat_history(db: Session, user_id: int, session_id: Optional[str] = None):
    query = db.query(ChatHistory).filter(ChatHistory.user_id == user_id)
    if session_id:
        query = query.filter(ChatHistory.session_id == session_id)
    query.delete()
    db.flush()


# ==========================================
# USAGE STATISTICS CRUD
# ==========================================
def get_or_create_usage_statistics(db: Session, user_id: int) -> UsageStatistic:
    usage = db.query(UsageStatistic).filter(UsageStatistic.user_id == user_id).first()
    if not usage:
        usage = UsageStatistic(
            user_id=user_id,
            documents_uploaded=0,
            storage_used=0,
            vector_count=0,
            query_count=0,
            credential_mode="application"
        )
        db.add(usage)
        db.flush()
    return usage


def get_user_usage_statistics(db: Session, user_id: int) -> Optional[UsageStatistic]:
    return db.query(UsageStatistic).filter(UsageStatistic.user_id == user_id).first()


# ==========================================
# EMAIL & PASSWORD RESET TOKENS
# ==========================================
def create_email_verification_token(db: Session, user_id: int, token: str, expires_in_hours: int = 24) -> EmailVerificationToken:
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=expires_in_hours)
    tok = EmailVerificationToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at,
        used=False
    )
    db.add(tok)
    db.flush()
    return tok


def verify_email_token(db: Session, token: str) -> Optional[User]:
    tok = db.query(EmailVerificationToken).filter(
        EmailVerificationToken.token == token,
        EmailVerificationToken.used == False,
        EmailVerificationToken.expires_at > datetime.datetime.utcnow()
    ).first()
    if tok:
        tok.used = True
        user = get_user_by_id(db, tok.user_id)
        if user:
            user.email_verified = True
            db.flush()
            return user
    return None


def create_password_reset_token(db: Session, user_id: int, token: str, expires_in_hours: int = 2) -> PasswordResetToken:
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=expires_in_hours)
    tok = PasswordResetToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at,
        used=False
    )
    db.add(tok)
    db.flush()
    return tok


def verify_password_reset_token(db: Session, token: str) -> Optional[User]:
    tok = db.query(PasswordResetToken).filter(
        PasswordResetToken.token == token,
        PasswordResetToken.used == False,
        PasswordResetToken.expires_at > datetime.datetime.utcnow()
    ).first()
    if tok:
        return get_user_by_id(db, tok.user_id)
    return None


def mark_password_reset_token_used(db: Session, token: str):
    tok = db.query(PasswordResetToken).filter(PasswordResetToken.token == token).first()
    if tok:
        tok.used = True
        db.flush()


# ==========================================
# PENDING REGISTRATIONS CRUD
# ==========================================
def create_or_update_pending_registration(
    db: Session,
    name: str,
    email: str,
    password_hash: str,
    otp_code: str,
    expires_minutes: int = 10
) -> PendingRegistration:
    clean_email = email.lower().strip()
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=expires_minutes)

    existing = db.query(PendingRegistration).filter(
        func.lower(PendingRegistration.email) == clean_email
    ).first()

    if existing:
        existing.name = name.strip()
        existing.password_hash = password_hash
        existing.otp_code = otp_code.strip()
        existing.expires_at = expires_at
        existing.attempts = 0
        existing.created_at = datetime.datetime.utcnow()
        db.flush()
        return existing
    else:
        pending = PendingRegistration(
            name=name.strip(),
            email=clean_email,
            password_hash=password_hash,
            otp_code=otp_code.strip(),
            expires_at=expires_at,
            attempts=0
        )
        db.add(pending)
        db.flush()
        return pending


def get_pending_registration(db: Session, email: str) -> Optional[PendingRegistration]:
    return db.query(PendingRegistration).filter(
        func.lower(PendingRegistration.email) == email.lower().strip()
    ).first()


def delete_pending_registration(db: Session, email: str):
    pending = get_pending_registration(db, email)
    if pending:
        db.delete(pending)
        db.flush()


def increment_pending_attempts(db: Session, email: str) -> int:
    pending = get_pending_registration(db, email)
    if pending:
        pending.attempts += 1
        db.flush()
        return pending.attempts
    return 0


# ==========================================
# AUDIT LOGS CRUD
# ==========================================
def create_audit_log(
    db: Session,
    action: str,
    user_id: Optional[int] = None,
    details: Optional[str] = None,
    ip_address: Optional[str] = None
) -> AuditLog:
    log_entry = AuditLog(
        user_id=user_id,
        action=action,
        details=details,
        ip_address=ip_address,
        timestamp=datetime.datetime.utcnow()
    )
    db.add(log_entry)
    db.flush()
    return log_entry


def get_audit_logs(db: Session, limit: int = 100, offset: int = 0) -> List[AuditLog]:
    return db.query(AuditLog).order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit).all()


# ==========================================
# SYSTEM SETTINGS CRUD
# ==========================================
def get_system_setting(db: Session, key: str, default: Optional[str] = None) -> Optional[str]:
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    return setting.value if setting else default


def set_system_setting(db: Session, key: str, value: str, description: Optional[str] = None) -> SystemSetting:
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if not setting:
        setting = SystemSetting(key=key, value=str(value), description=description)
        db.add(setting)
    else:
        setting.value = str(value)
        if description:
            setting.description = description
        setting.updated_at = datetime.datetime.utcnow()
    db.flush()
    return setting


def get_int_setting(db: Session, key: str, default: int = 2) -> int:
    val = get_system_setting(db, key, str(default))
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_bool_setting(db: Session, key: str, default: bool = True) -> bool:
    val = get_system_setting(db, key, str(default).lower())
    return str(val).lower() in ("true", "1", "yes")


# ==========================================
# ADMIN ANALYTICS
# ==========================================
def get_admin_dashboard_metrics(db: Session) -> Dict[str, Any]:
    total_users = db.query(func.count(User.id)).scalar() or 0
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar() or 0
    verified_users = db.query(func.count(User.id)).filter(User.email_verified == True).scalar() or 0
    
    total_docs = db.query(func.count(Document.id)).filter(Document.status == "indexed").scalar() or 0
    
    # Mode breakdown
    app_credential_users = db.query(func.count(UsageStatistic.id)).filter(UsageStatistic.credential_mode == "application").scalar() or 0
    byok_users = db.query(func.count(UsageStatistic.id)).filter(UsageStatistic.credential_mode == "user").scalar() or 0

    # Application infrastructure resource consumption
    app_docs_processed = db.query(func.count(Document.id)).filter(Document.credential_mode == "application", Document.status == "indexed").scalar() or 0
    app_vectors = db.query(func.sum(Document.vector_count)).filter(Document.credential_mode == "application", Document.status == "indexed").scalar() or 0
    
    # Query count summary
    total_queries = db.query(func.sum(UsageStatistic.query_count)).scalar() or 0

    return {
        "total_users": total_users,
        "active_users": active_users,
        "verified_users": verified_users,
        "total_documents": total_docs,
        "app_credential_users": app_credential_users,
        "byok_users": byok_users,
        "app_docs_processed": app_docs_processed,
        "app_vectors": int(app_vectors) if app_vectors else 0,
        "total_queries": int(total_queries) if total_queries else 0
    }


def seed_initial_data(db: Session):
    """
    Seeds default system settings and root admin user if not present.
    """
    # 1. Default system settings
    if not db.query(SystemSetting).filter(SystemSetting.key == "APPLICATION_CREDENTIAL_DOCUMENT_LIMIT").first():
        set_system_setting(
            db,
            "APPLICATION_CREDENTIAL_DOCUMENT_LIMIT",
            str(settings.APPLICATION_CREDENTIAL_DOCUMENT_LIMIT),
            "Max documents allowed for users on shared application credentials"
        )
    if not db.query(SystemSetting).filter(SystemSetting.key == "MAX_UPLOAD_SIZE_MB").first():
        set_system_setting(
            db,
            "MAX_UPLOAD_SIZE_MB",
            str(settings.MAX_UPLOAD_SIZE_MB),
            "Maximum file size allowed per uploaded document in MB"
        )
    if not db.query(SystemSetting).filter(SystemSetting.key == "AUTO_VERIFY_EMAIL").first():
        set_system_setting(
            db,
            "AUTO_VERIFY_EMAIL",
            str(settings.AUTO_VERIFY_EMAIL).lower(),
            "Mandatory email verification on registration"
        )
    else:
        # Update existing setting with configured default if needed
        setting_rec = db.query(SystemSetting).filter(SystemSetting.key == "AUTO_VERIFY_EMAIL").first()
        if setting_rec and settings.AUTO_VERIFY_EMAIL is False:
            setting_rec.value = "false"
            db.flush()

    # 2. Seed Default Admin User if no admin exists
    admin = db.query(User).filter(User.role == "admin").first()
    if not admin:
        import bcrypt
        hashed = bcrypt.hashpw(b"Admin@123456", bcrypt.gensalt()).decode("utf-8")
        admin = User(
            name="System Administrator",
            email="admin@documind.ai",
            password_hash=hashed,
            role="admin",
            email_verified=True,
            is_active=True
        )
        db.add(admin)
        db.flush()
        usage = UsageStatistic(
            user_id=admin.id,
            documents_uploaded=0,
            storage_used=0,
            vector_count=0,
            query_count=0,
            credential_mode="application"
        )
        db.add(usage)
        db.flush()
        logger.info("Created default system administrator: admin@documind.ai")
