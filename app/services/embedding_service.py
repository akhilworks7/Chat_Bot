from typing import List, Union
import numpy as np

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("embedding_service")


import threading

class EmbeddingService:
    """
    Generates dense vector embeddings using SentenceTransformers.
    Optimized for fast batch encoding on CUDA or CPU with normalized vectors.
    """

    _instance = None
    _model = None
    _lock = threading.Lock()
    _warmup_thread = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer

                    import os
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    if self.device == "cpu":
                        torch.set_num_threads(min(4, os.cpu_count() or 2))
                    logger.info(f"Loading embedding model '{self.model_name}' on device: {self.device}")
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    logger.info("Embedding model loaded successfully.")
        return self._model

    @classmethod
    def start_background_warmup(cls):
        """Spawns a background daemon thread to load model weights without blocking the page load."""
        if cls._model is None and (cls._warmup_thread is None or not cls._warmup_thread.is_alive()):
            def _runner():
                try:
                    inst = cls()
                    inst.embed_query("warmup")
                    logger.info("Background embedding warmup complete.")
                except Exception as e:
                    logger.warning(f"Background embedding warmup notice: {e}")

            cls._warmup_thread = threading.Thread(target=_runner, daemon=True)
            cls._warmup_thread.start()

    @classmethod
    def warmup(cls):
        """Backward compatible warmup method."""
        cls.start_background_warmup()

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
        import gc
        import torch
        if not chunks:
            return np.empty((0, self.dimension))

        model = self._get_model()
        device = getattr(self, "device", "cpu")
        batch_size = batch_size or (128 if device == "cuda" else 16)
        with torch.inference_mode():
            embeddings = model.encode(
                chunks,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar
            )
        gc.collect()
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string into a normalized vector list.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        import torch
        model = self._get_model()
        with torch.inference_mode():
            embedding = model.encode(
                query.strip(),
                normalize_embeddings=True
            )

        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)