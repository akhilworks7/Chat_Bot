from typing import List, Optional
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    id: str = Field(..., description="Unique vector ID in Pinecone")
    score: float = Field(..., description="Cosine similarity score")
    text: str = Field(..., description="Text content of the retrieved chunk")
    source: str = Field(..., description="Document source name or path")
    chunk_id: Optional[int] = Field(None, description="Index of the chunk in the document")
    page: Optional[int] = Field(None, description="Page number if available")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask the RAG system", example="What is the definition of Antiparkinson drugs?")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of document chunks to retrieve")
    namespace: Optional[str] = Field("documents", description="Pinecone namespace to query")


class ChatResponse(BaseModel):
    question: str = Field(..., description="User's submitted question")
    answer: str = Field(..., description="Generated answer based on retrieved documents")
    sources: List[SourceDocument] = Field(default_factory=list, description="Retrieved source chunks used as context")
    response_time_ms: float = Field(..., description="Total execution time in milliseconds")