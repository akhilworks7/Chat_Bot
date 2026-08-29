from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class IngestLocalRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Relative or absolute path to the PDF document to ingest",
        example="data/documents/Medical_book.pdf"
    )
    namespace: Optional[str] = Field(
        "documents",
        description="Target Pinecone namespace"
    )


class IngestResponse(BaseModel):
    status: str = Field("success", description="Status of the ingestion operation")
    source: str = Field(..., description="Path or filename of the processed document")
    chunks_created: int = Field(..., description="Number of text chunks created")
    vectors_upserted: int = Field(..., description="Number of vector embeddings uploaded to Pinecone")
    namespace: str = Field(..., description="Pinecone namespace where vectors were saved")
    time_taken_seconds: float = Field(..., description="Time taken to process and index the document")


class IndexStatsResponse(BaseModel):
    dimension: int = Field(..., description="Dimension of vector embeddings")
    total_vector_count: int = Field(..., description="Total vector count across all namespaces")
    namespaces: Dict[str, Any] = Field(default_factory=dict, description="Stats per namespace")


class DocumentItem(BaseModel):
    filename: str = Field(..., description="Name of the PDF file")
    source: str = Field(..., description="Source identifier")
    file_size_bytes: int = Field(0, description="File size in bytes")
    total_chunks: int = Field(0, description="Total chunks extracted")
    vector_ids: List[str] = Field(default_factory=list, description="Vector IDs generated for this document")
    upload_time: str = Field(..., description="ISO formatted upload/creation timestamp")
    status: str = Field("ready", description="Current status of the document")
    namespace: str = Field("documents", description="Pinecone namespace")


class DocumentListResponse(BaseModel):
    total_documents: int = Field(..., description="Count of registered documents")
    documents: List[DocumentItem] = Field(default_factory=list, description="List of document items")


class DeleteResponse(BaseModel):
    status: str = Field("success", description="Status of the deletion")
    filename: str = Field(..., description="Name of the deleted file")
    vectors_deleted: int = Field(0, description="Number of vectors removed from Pinecone")
    files_deleted: Dict[str, bool] = Field(default_factory=dict, description="Files removed from storage")
    message: str = Field(..., description="Human-readable summary of deletion")
