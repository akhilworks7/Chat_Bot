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
                    import os
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"
                    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

                    import torch
                    from sentence_transformers import SentenceTransformer

                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    if self.device == "cpu":
                        torch.set_num_threads(min(4, os.cpu_count() or 2))

                    logger.info(f"Loading embedding model '{self.model_name}' on device: {self.device}")
                    
                    local_model_path = os.path.join(os.getcwd(), "data", "models", "all-MiniLM-L6-v2")

                    # Tier 1: Load directly from local project directory (instant offline load ~0.2s)
                    if os.path.isdir(local_model_path):
                        try:
                            self._model = SentenceTransformer(local_model_path, device=self.device)
                            logger.info("Embedding model loaded successfully from local directory (offline mode).")
                            return self._model
                        except Exception as e:
                            logger.warning(f"Could not load from local directory {local_model_path}: {e}")

                    # Tier 2: Try HuggingFace cache with local_files_only=True (bypasses HF Hub rate limits)
                    try:
                        self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
                        logger.info("Embedding model loaded successfully from HuggingFace local cache.")
                        return self._model
                    except Exception:
                        pass

                    # Tier 3: Fallback to remote download (first run on clean cloud container)
                    logger.info(f"Model not found in local cache; downloading from HuggingFace Hub...")
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    logger.info("Embedding model loaded successfully.")

                    # Cache to local directory for future instantaneous reloads
                    try:
                        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                        self._model.save(local_model_path)
                    except Exception:
                        pass

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