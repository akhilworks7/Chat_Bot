import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.document_service import DocumentService
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.utils.logger import get_logger

logger = get_logger("ingest_cli")


def ingest_file(pdf_path: str, namespace: str = "documents", batch_size: int = 100):
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return False

    filename = Path(pdf_path).name
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"STARTING INGESTION: {pdf_path}")
    logger.info("=" * 60)

    # 1. Document Extraction
    doc_service = DocumentService()
    logger.info("Step 1/4: Extracting text from PDF (checking searchability & OCR if needed)...")
    text = doc_service.extract_text(pdf_path)
    logger.info(f"Extracted {len(text)} characters.")

    # 2. Chunking
    chunk_service = ChunkingService()
    logger.info("Step 2/4: Chunking text...")
    chunks = chunk_service.split_text(text)
    logger.info(f"Created {len(chunks)} chunks.")

    if not chunks:
        logger.warning(f"No chunks created for {pdf_path}. Skipping.")
        return False

    # 3. Embedding
    embed_service = EmbeddingService()
    logger.info("Step 3/4: Generating vector embeddings...")
    embeddings = embed_service.embed_documents(chunks, show_progress_bar=True)

    # 4. Uploading to Pinecone
    vector_service = VectorService()
    logger.info(f"Step 4/4: Uploading {len(embeddings)} vectors to Pinecone (namespace: '{namespace}')...")

    vectors = []
    base_id = Path(filename).stem.replace(" ", "_")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{base_id}_chunk_{i}",
            "values": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
            "metadata": {
                "text": chunk,
                "chunk_id": i,
                "source": filename
            }
        })

    upserted = vector_service.upsert_vectors(vectors, namespace=namespace, batch_size=batch_size)
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"✅ INGESTION COMPLETED: {upserted} vectors indexed in {elapsed:.2f} seconds.")
    logger.info("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents into Pinecone Vector Database")
    parser.add_argument(
        "--file",
        type=str,
        default="data/documents/Medical_book.pdf",
        help="Path to the PDF file to ingest"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing multiple PDF files to ingest"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=settings.PINECONE_NAMESPACE,
        help="Pinecone namespace to save vectors into"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for Pinecone vector upsert"
    )

    args = parser.parse_args()

    if args.dir:
        pdf_dir = Path(args.dir)
        if not pdf_dir.exists():
            logger.error(f"Directory not found: {args.dir}")
            return
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {args.dir}")
        for pdf in pdf_files:
            ingest_file(str(pdf), namespace=args.namespace, batch_size=args.batch_size)
    else:
        ingest_file(args.file, namespace=args.namespace, batch_size=args.batch_size)


if __name__ == "__main__":
    main()