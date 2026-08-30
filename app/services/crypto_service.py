import os
import base64
from typing import Optional
from cryptography.fernet import Fernet
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("crypto_service")

KEY_FILE = os.path.join("data", ".encryption_key")


class CryptoService:
    """
    Provides symmetric encryption and decryption for user API keys using Fernet (AES-128-CBC with HMAC-SHA256).
    """

    _instance = None
    _cipher = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CryptoService, cls).__new__(cls)
            cls._instance._init_cipher()
        return cls._instance

    def _init_cipher(self):
        key = settings.ENCRYPTION_KEY
        if not key:
            # Check if persistent key file exists
            if os.path.exists(KEY_FILE):
                try:
                    with open(KEY_FILE, "r", encoding="utf-8") as f:
                        key = f.read().strip()
                except Exception as e:
                    logger.warning(f"Failed to read encryption key file: {e}")

            if not key:
                # Generate new key and persist to data/.encryption_key
                key = Fernet.generate_key().decode("utf-8")
                os.makedirs("data", exist_ok=True)
                try:
                    with open(KEY_FILE, "w", encoding="utf-8") as f:
                        f.write(key)
                    logger.info("Generated new persistent encryption key in data/.encryption_key")
                except Exception as e:
                    logger.warning(f"Could not persist encryption key to file: {e}")

        # Ensure key is in bytes format for Fernet
        key_bytes = key.encode("utf-8") if isinstance(key, str) else key
        try:
            self._cipher = Fernet(key_bytes)
        except Exception as e:
            # Fallback safe key if key format is invalid
            logger.error(f"Invalid encryption key ({e}). Generating fallback key.")
            new_key = Fernet.generate_key()
            self._cipher = Fernet(new_key)

    def encrypt(self, plain_text: str) -> str:
        """
        Encrypts plaintext string to encrypted token string.
        """
        if not plain_text:
            return ""
        return self._cipher.encrypt(plain_text.encode("utf-8")).decode("utf-8")

    def decrypt(self, cipher_text: str) -> str:
        """
        Decrypts encrypted token string to original plaintext string.
        """
        if not cipher_text:
            return ""
        try:
            return self._cipher.decrypt(cipher_text.encode("utf-8")).decode("utf-8")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return ""

    @staticmethod
    def mask_api_key(key: Optional[str]) -> str:
        """
        Masks an API key for safe UI display (e.g., pcsk_...a1b2 -> pcsk_••••••••••••a1b2).
        """
        if not key or len(key) < 8:
            return "••••••••••••"
        prefix = key[:5]
        suffix = key[-4:]
        return f"{prefix}••••••••••••{suffix}"
