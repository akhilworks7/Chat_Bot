import os
import time
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.schemas.document import (
    IngestLocalRequest,
    IngestResponse,
    IndexStatsResponse,
    DocumentItem,
    DocumentListResponse,
    DeleteResponse
)
from app.services.document_service import DocumentService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.registry_service import DocumentRegistryService
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("documents_route")

router = APIRouter(
    prefix="/documents",
    tags=["Documents & Ingestion"]
)

document_service = DocumentService()
chunking_service = ChunkingService()
embedding_service = EmbeddingService()
vector_service = VectorService()
registry_service = DocumentRegistryService()


def _run_ingestion_pipeline(file_path: str, namespace: str) -> IngestResponse:
    """
    Executes the ingestion pipeline for a given PDF path and registers it in the document registry.
    """
    start_time = time.time()
    filename = Path(file_path).name
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

    logger.info(f"Starting ingestion pipeline for {file_path} into namespace '{namespace}'")

    # 1. Text extraction (with searchability check / OCR)
    text = document_service.extract_text(file_path)

    # 2. Chunking
    chunks = chunking_service.split_text(text)
    if not chunks:
        raise ValueError("No text content could be extracted from the document.")

    # 3. Generate embeddings
    embeddings = embedding_service.embed_documents(chunks, show_progress_bar=False)

    # 4. Prepare vector records
    vectors = []
    vector_ids = []
    base_id = Path(filename).stem.replace(" ", "_")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        v_id = f"{base_id}_chunk_{i}"
        vector_ids.append(v_id)
        vectors.append({
            "id": v_id,
            "values": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            "metadata": {
                "text": chunk,
                "chunk_id": i,
                "source": filename
            }
        })

    # 5. Batch upsert to Pinecone
    upserted_count = vector_service.upsert_vectors(vectors, namespace=namespace)
    elapsed = round(time.time() - start_time, 2)

    # 6. Save in document registry
    registry_service.register_document(
        filename=filename,
        file_size_bytes=file_size,
        total_chunks=len(chunks),
        vector_ids=vector_ids,
        namespace=namespace
    )

    logger.info(
        f"Ingestion successful for {filename}: {len(chunks)} chunks, {upserted_count} vectors indexed in {elapsed}s"
    )

    return IngestResponse(
        status="success",
        source=filename,
        chunks_created=len(chunks),
        vectors_upserted=upserted_count,
        namespace=namespace,
        time_taken_seconds=elapsed
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List all uploaded/indexed documents",
    description="Returns a list of all documents currently tracked in the system."
)
def list_documents() -> DocumentListResponse:
    docs = registry_service.get_all_documents()
    return DocumentListResponse(
        total_documents=len(docs),
        documents=[DocumentItem(**d) for d in docs]
    )


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
    description="Uploads a PDF document, runs OCR if necessary, extracts text, generates embeddings, and uploads vectors to Pinecone."
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to upload and index"),
    namespace: str = Form(settings.PINECONE_NAMESPACE, description="Target Pinecone namespace")
) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are currently supported."
        )

    temp_dir = os.path.join(settings.DOCUMENTS_DIR)
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return _run_ingestion_pipeline(temp_path, namespace=namespace)
    except Exception as e:
        logger.error(f"Failed to ingest uploaded document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )


@router.post(
    "/ingest-local",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a local document already on the server",
    description="Ingests a file from the server's local filesystem (e.g. data/documents/Medical_book.pdf)."
)
def ingest_local_document(request: IngestLocalRequest) -> IngestResponse:
    if not os.path.exists(request.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Local file not found at path: {request.file_path}"
        )

    try:
        namespace = request.namespace or settings.PINECONE_NAMESPACE
        return _run_ingestion_pipeline(request.file_path, namespace=namespace)
    except Exception as e:
        logger.error(f"Failed to ingest local document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local document ingestion failed: {str(e)}"
        )


@router.delete(
    "/{document_name}",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document and purge all its vectors from Pinecone",
    description="Completely removes a document, including its embeddings in Pinecone, raw PDF, OCR PDF, and extracted text."
)
def delete_document(document_name: str, namespace: str = settings.PINECONE_NAMESPACE) -> DeleteResponse:
    logger.info(f"Initiating deletion for document: '{document_name}' in namespace '{namespace}'")

    # 1. Retrieve registered vector IDs if tracked
    doc_info = registry_service.get_document(document_name)
    vector_ids = doc_info.get("vector_ids", []) if doc_info else []

    # 2. Delete vectors from Pinecone (by vector IDs and metadata filter)
    deleted_vectors_count = vector_service.delete_by_source(
        source_name=document_name,
        vector_ids=vector_ids,
        namespace=namespace
    )

    # 3. Delete physical files from disk
    files_deleted = document_service.delete_document_files(document_name)

    # 4. Remove from registry
    registry_service.remove_document(document_name)

    logger.info(
        f"Document '{document_name}' deleted successfully. Vectors purged: {deleted_vectors_count}. Files removed: {files_deleted}"
    )

    return DeleteResponse(
        status="success",
        filename=document_name,
        vectors_deleted=deleted_vectors_count,
        files_deleted=files_deleted,
        message=f"Document '{document_name}' and all associated vector embeddings were permanently deleted."
    )


@router.get(
    "/stats",
    response_model=IndexStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Pinecone index statistics",
    description="Returns vector count and namespace information from Pinecone."
)
def get_index_statistics() -> IndexStatsResponse:
    try:
        stats = vector_service.get_stats()
        return IndexStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to fetch index stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve vector database stats: {str(e)}"
        )
