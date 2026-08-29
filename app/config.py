from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Pinecone Vector Database
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "pdf-rag1-index"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_CLOUD: str = "aws"
    PINECONE_NAMESPACE: str = "documents"

    # Groq LLM Configuration
    GROQ_API_KEY: str
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_TEMPERATURE: float = 0.0

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 32

    # Chunking Configuration
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # RAG Retrieval Configuration
    TOP_K: int = 8

    # Directory Paths
    DOCUMENTS_DIR: str = "data/documents"
    PROCESSED_DIR: str = "data/processed"
    OUTPUT_DIR: str = "output"

    # API Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True


settings = Settings()