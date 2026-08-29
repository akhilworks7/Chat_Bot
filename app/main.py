import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.utils.logger import get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    logger.info("==================================================")
    logger.info(f"Starting RAG Chatbot API (v1.0.0)...")
    logger.info(f"Pinecone Index: {settings.PINECONE_INDEX_NAME} (ns: {settings.PINECONE_NAMESPACE})")
    logger.info(f"Groq Model: {settings.GROQ_MODEL}")
    logger.info("==================================================")

    yield

    logger.info("Shutting down RAG Chatbot API...")


app = FastAPI(
    title="RAG Document Chatbot API",
    description="Production-ready Retrieval-Augmented Generation (RAG) backend utilizing Pinecone vector database, HuggingFace embeddings, and Groq LLMs.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers with v1 prefix
app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")

# Mount Static Assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", tags=["General"])
def serve_ui():
    """
    Serves the primary two-tab web interface.
    """
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "message": "RAG Chatbot API is online and operational",
        "docs_url": "/docs",
        "version": "1.0.0"
    }


@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "pinecone_index": settings.PINECONE_INDEX_NAME,
        "groq_model": settings.GROQ_MODEL
    }