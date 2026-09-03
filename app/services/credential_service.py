import time
from typing import Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from pinecone import Pinecone
from groq import Groq

from app.config import settings
from app.db import crud
from app.services.crypto_service import CryptoService
from app.utils.logger import get_logger

logger = get_logger("credential_service")


class CredentialService:
    """
    Manages hybrid credential resolution (Application shared keys vs BYOK user keys),
    document quota calculations, and live connection test validations.
    """

    @classmethod
    def get_credentials(cls, user_id: int, db: Session) -> Dict[str, Any]:
        """
        Determines whether the user is on 'user' (BYOK) or 'application' credential mode,
        and retrieves the corresponding active API keys and namespace.
        """
        crypto = CryptoService()
        creds = crud.get_user_credentials(db, user_id)

        # If credentials are not in local SQLite (e.g. after a reboot), attempt cloud hydration
        if not (creds and creds.pinecone_api_key_encrypted and creds.groq_api_key_encrypted):
            try:
                user_obj = crud.get_user_by_id(db, user_id)
                if user_obj and user_obj.email:
                    from app.services.vector_service import VectorService
                    cloud_c = VectorService().load_user_credentials_from_cloud(user_obj.email)
                    if cloud_c and cloud_c.get("pinecone_key") and cloud_c.get("groq_key"):
                        creds = crud.upsert_user_credentials(
                            db=db,
                            user_id=user_id,
                            pinecone_api_key_encrypted=crypto.encrypt(cloud_c["pinecone_key"]),
                            pinecone_index=cloud_c.get("pinecone_index", settings.PINECONE_INDEX_NAME),
                            groq_api_key_encrypted=crypto.encrypt(cloud_c["groq_key"]),
                            groq_model=cloud_c.get("groq_model", settings.GROQ_MODEL)
                        )
                        logger.info(f"Auto-hydrated BYOK credentials from Pinecone Cloud for user #{user_id}")
            except Exception as e:
                logger.warning(f"Note on auto-hydrating credentials from cloud: {e}")

        if creds and creds.pinecone_api_key_encrypted and creds.groq_api_key_encrypted:
            p_key = crypto.decrypt(creds.pinecone_api_key_encrypted)
            g_key = crypto.decrypt(creds.groq_api_key_encrypted)

            if p_key and g_key:
                return {
                    "mode": "user",
                    "mode_label": "User Credentials (BYOK)",
                    "pinecone_api_key": p_key,
                    "pinecone_index": creds.pinecone_index or settings.PINECONE_INDEX_NAME,
                    "groq_api_key": g_key,
                    "groq_model": creds.groq_model or settings.GROQ_MODEL,
                    "namespace": f"user_{user_id}",
                    "is_byok": True
                }

        # Fallback to shared application credentials
        return {
            "mode": "application",
            "mode_label": "Application Credentials (Shared)",
            "pinecone_api_key": settings.PINECONE_API_KEY,
            "pinecone_index": settings.PINECONE_INDEX_NAME,
            "groq_api_key": settings.GROQ_API_KEY,
            "groq_model": settings.GROQ_MODEL,
            "namespace": f"user_{user_id}",
            "is_byok": False
        }

    @classmethod
    def check_upload_allowance(cls, user_id: int, db: Session) -> Dict[str, Any]:
        """
        Calculates document quota status for the user.
        - Admin or BYOK users: Unlimited document capacity.
        - Shared mode regular users: Restricted by APPLICATION_CREDENTIAL_DOCUMENT_LIMIT (default 2).
        """
        user = crud.get_user_by_id(db, user_id)
        if user and user.role == "admin":
            user_docs = crud.get_user_documents(db, user_id)
            return {
                "allowed": True,
                "mode": "admin",
                "used": len(user_docs),
                "limit": None,
                "is_unlimited": True,
                "reason": "Admin Unlimited"
            }

        creds_info = cls.get_credentials(user_id, db)
        mode = creds_info["mode"]

        if mode == "user":
            # BYOK users are exempt from application-imposed document limits
            user_docs = crud.get_user_documents(db, user_id)
            return {
                "allowed": True,
                "mode": "user",
                "used": len(user_docs),
                "limit": None,
                "is_unlimited": True,
                "reason": "BYOK Unlimited"
            }

        # Application credentials mode
        app_docs_count = crud.count_user_app_credential_documents(db, user_id)
        limit = crud.get_int_setting(db, "APPLICATION_CREDENTIAL_DOCUMENT_LIMIT", default=settings.APPLICATION_CREDENTIAL_DOCUMENT_LIMIT)

        allowed = (app_docs_count < limit)
        return {
            "allowed": allowed,
            "mode": "application",
            "used": app_docs_count,
            "limit": limit,
            "is_unlimited": False,
            "reason": "OK" if allowed else "Document limit reached on shared application infrastructure"
        }

    @classmethod
    def test_pinecone(cls, api_key: str, index_name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Tests Pinecone API Key and index accessibility.
        Returns: (success: bool, message: str, stats: dict)
        """
        api_key = api_key.strip()
        index_name = index_name.strip()

        if not api_key:
            return False, "Pinecone API Key is required.", {}
        if not index_name:
            return False, "Pinecone Index Name is required.", {}

        try:
            pc = Pinecone(api_key=api_key)
            indexes = pc.list_indexes()
            index_names = [idx.name for idx in indexes]

            if index_name not in index_names:
                logger.info(f"Index '{index_name}' not found. Automatically creating serverless index in user's Pinecone account...")
                from pinecone import ServerlessSpec
                pc.create_index(
                    name=index_name,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.PINECONE_CLOUD,
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                time.sleep(2)
                return True, f"Index '{index_name}' was not found, so it was automatically created in your Pinecone account and is ready to use!", {"total_vectors": 0}

            idx = pc.Index(index_name)
            stats = idx.describe_index_stats()
            total_vecs = getattr(stats, "total_vector_count", 0) or (stats.get("total_vector_count", 0) if isinstance(stats, dict) else 0)

            return True, f"Successfully connected to Pinecone! Index '{index_name}' is online ({total_vecs} vectors).", {"total_vectors": total_vecs}
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Pinecone connection test failed: {err_msg}")
            if "401" in err_msg or "Unauthorized" in err_msg:
                return False, "Authentication Failed: Invalid Pinecone API Key.", {}
            elif "403" in err_msg or "Forbidden" in err_msg:
                return False, "Pinecone Access Denied / Quota exceeded on account.", {}
            return False, f"Pinecone connection error: {err_msg}", {}

    @classmethod
    def test_groq(cls, api_key: str, model_name: str) -> Tuple[bool, str]:
        """
        Tests Groq API Key and model response.
        Returns: (success: bool, message: str)
        """
        api_key = api_key.strip()
        model_name = (model_name or settings.GROQ_MODEL).strip()

        if not api_key:
            return False, "Groq API Key is required."

        try:
            client = Groq(api_key=api_key)
            # Lightweight test prompt
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a test ping responder."},
                    {"role": "user", "content": "Respond with 'OK'."}
                ],
                max_tokens=5,
                temperature=0.0
            )
            return True, f"Successfully connected to Groq! Model '{model_name}' is ready."
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Groq connection test failed: {err_msg}")
            if "401" in err_msg or "AuthenticationError" in err_msg or "invalid_api_key" in err_msg:
                return False, "Authentication Failed: Invalid Groq API Key."
            elif "429" in err_msg or "rate_limit_exceeded" in err_msg:
                return False, "Groq Rate Limit Reached: Your Groq key hit its quota or rate limit."
            elif "model_not_found" in err_msg:
                return False, f"Model '{model_name}' not available on Groq."
            return False, f"Groq connection error: {err_msg}"
