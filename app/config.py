import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Credentials (Shared Infrastructure)
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "pdf-rag1-index"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_CLOUD: str = "aws"
    PINECONE_NAMESPACE: str = "documents"

    # Groq LLM Configuration (Shared Infrastructure)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TEMPERATURE: float = 0.0


    # Database Configuration (PostgreSQL with SQLite fallback)
    DATABASE_URL: str = "sqlite:///./data/rag_app.db"

    # Security & Encryption
    ENCRYPTION_KEY: Optional[str] = None
    JWT_SECRET: str = "super-secret-documind-jwt-key-change-in-prod"
    PASSWORD_SALT: str = "documind-secure-salt-2026"

    # Hybrid Onboarding & Document Limits
    APPLICATION_CREDENTIAL_DOCUMENT_LIMIT: int = 2
    MAX_UPLOAD_SIZE_MB: int = 50
    AUTO_VERIFY_EMAIL: bool = False  # Mandatory verification by default

    # SMTP / Email Configuration
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "no-reply@documind.ai"
    SMTP_USE_TLS: bool = True

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Chunking Configuration
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # RAG Retrieval Configuration
    TOP_K: int = 4


    # Directory Paths
    DOCUMENTS_DIR: str = "data/documents"
    PROCESSED_DIR: str = "data/processed"
    OUTPUT_DIR: str = "output"

    # API Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True


settings = Settings()