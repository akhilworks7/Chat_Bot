from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService
from app.utils.logger import get_logger

logger = get_logger("chat_route")

router = APIRouter(
    prefix="/chat",
    tags=["Chat & Retrieval"]
)

rag_service = RAGService()


@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG knowledge base",
    description="Embeds the input question, retrieves the most relevant context snippets from Pinecone, and generates a grounded response using Groq LLM."
)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        result = rag_service.query(
            question=request.question,
            top_k=request.top_k,
            namespace=request.namespace
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Error during RAG query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat query: {str(e)}"
        )


@router.get(
    "/health",
    summary="Check Chat service health",
    status_code=status.HTTP_200_OK
)
def chat_health():
    return {
        "status": "healthy",
        "service": "RAG Chat API"
    }