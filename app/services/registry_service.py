import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("registry_service")

REGISTRY_FILE = os.path.join("data", "documents_registry.json")


class DocumentRegistryService:
    """
    Manages document metadata, tracking uploaded files, chunk counts, vector IDs, and indexing timestamps.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DocumentRegistryService, cls).__new__(cls)
            cls._instance._init_registry()
        return cls._instance

    def _init_registry(self):
        self.registry_path = REGISTRY_FILE
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        if not os.path.exists(self.registry_path):
            self._save_data({})
            self._auto_discover_existing_files()

    def _load_data(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.registry_path):
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading document registry: {e}")
        return {}

    def _save_data(self, data: Dict[str, Any]):
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving document registry: {e}")

    def _auto_discover_existing_files(self):
        """
        Discovers existing PDF files in data/documents and creates initial registry entries.
        """
        data_dir = Path(settings.DOCUMENTS_DIR)
        if not data_dir.exists():
            return

        data = self._load_data()
        pdf_files = list(data_dir.glob("*.pdf"))

        for pdf in pdf_files:
            filename = pdf.name
            if filename not in data:
                stat = pdf.stat()
                data[filename] = {
                    "filename": filename,
                    "source": filename,
                    "file_size_bytes": stat.st_size,
                    "total_chunks": 0,
                    "vector_ids": [],
                    "upload_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "status": "ready",
                    "namespace": settings.PINECONE_NAMESPACE
                }
        self._save_data(data)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        self._auto_discover_existing_files()
        data = self._load_data()
        return list(data.values())

    def get_document(self, filename: str) -> Optional[Dict[str, Any]]:
        data = self._load_data()
        return data.get(filename)

    def register_document(
        self,
        filename: str,
        file_size_bytes: int,
        total_chunks: int,
        vector_ids: List[str],
        namespace: str = "documents"
    ) -> Dict[str, Any]:
        data = self._load_data()
        entry = {
            "filename": filename,
            "source": filename,
            "file_size_bytes": file_size_bytes,
            "total_chunks": total_chunks,
            "vector_ids": vector_ids,
            "upload_time": datetime.now().isoformat(),
            "status": "ready",
            "namespace": namespace
        }
        data[filename] = entry
        self._save_data(data)
        logger.info(f"Registered document in registry: {filename} ({total_chunks} chunks, {len(vector_ids)} vectors)")
        return entry

    def remove_document(self, filename: str) -> Optional[Dict[str, Any]]:
        data = self._load_data()
        if filename in data:
            removed = data.pop(filename)
            self._save_data(data)
            logger.info(f"Removed document from registry: {filename}")
            return removed
        return None
