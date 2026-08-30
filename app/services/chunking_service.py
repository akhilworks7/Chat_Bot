from typing import List, Dict, Any, Optional

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("chunking_service")


class ChunkingService:
    """
    Handles chunking of document text using RecursiveCharacterTextSplitter with metadata tagging.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self._splitter = None

    @property
    def splitter(self):
        if self._splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",   # Paragraph
                    "\n",     # Line
                    ". ",     # Sentence
                    " ",      # Word
                    ""        # Character
                ]
            )
        return self._splitter

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into plain chunks of text.
        """
        chunks = self.splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} text chunks (size: {self.chunk_size}, overlap: {self.chunk_overlap})")
        return chunks

    def create_chunks_with_metadata(
        self,
        text: str,
        source: str = "document.pdf",
        doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Splits text and associates each chunk with metadata (chunk_id, source, char_count).
        """
        raw_chunks = self.split_text(text)
        chunks_with_metadata = []

        for i, chunk in enumerate(raw_chunks):
            chunk_record = {
                "id": f"{doc_id or 'doc'}_{i}",
                "chunk_id": i,
                "text": chunk,
                "source": source,
                "char_count": len(chunk)
            }
            chunks_with_metadata.append(chunk_record)

        return chunks_with_metadata


# Convenience function for compatibility
def create_chunks(text: str) -> List[str]:
    service = ChunkingService()
    return service.split_text(text)