import torch
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("embedding_service")


class EmbeddingService:
    """
    Generates dense vector embeddings using SentenceTransformers.
    Optimized for fast batch encoding on CUDA or CPU with normalized vectors.
    """

    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._model is None:
            self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading embedding model '{self.model_name}' on device: {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded successfully.")

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIMENSION

    def embed_documents(
        self,
        chunks: List[str],
        batch_size: int = None,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Embeds a list of document chunks into normalized vectors.
        """
        if not chunks:
            return np.empty((0, self.dimension))

        batch_size = batch_size or (128 if self.device == "cuda" else settings.EMBEDDING_BATCH_SIZE)
        embeddings = self._model.encode(
            chunks,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar
        )
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string into a normalized vector list.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        embedding = self._model.encode(
            query.strip(),
            normalize_embeddings=True
        )

        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)