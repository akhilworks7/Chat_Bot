from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException, PineconeException

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("vector_service")


class PineconeQuotaException(Exception):
    """Raised when Pinecone storage/rate quota is exceeded."""
    pass


class PineconeAuthException(Exception):
    """Raised when Pinecone credentials are unauthorized."""
    pass


class VectorService:
    """
    Manages dynamic Pinecone client instances, index initialization,
    isolated namespace upserts, semantic search, and friendly error handling.
    """

    _instance = None
    _clients: Dict[str, Pinecone] = {}
    _indexes: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
        return cls._instance

    def _get_client(self, api_key: Optional[str] = None) -> Pinecone:
        key = api_key or settings.PINECONE_API_KEY
        if not key:
            raise PineconeAuthException("No Pinecone API Key configured.")

        if key not in self._clients:
            logger.info("Initializing new Pinecone client...")
            self._clients[key] = Pinecone(api_key=key)
        return self._clients[key]

    def _get_index(self, api_key: Optional[str] = None, index_name: Optional[str] = None) -> Any:
        key = api_key or settings.PINECONE_API_KEY
        idx_name = index_name or settings.PINECONE_INDEX_NAME
        cache_key = f"{key}_{idx_name}"

        if cache_key not in self._indexes:
            client = self._get_client(key)
            self._ensure_index_exists(client, idx_name)
            try:
                self._indexes[cache_key] = client.Index(idx_name)
                logger.info(f"Connected to Pinecone index: '{idx_name}'")
            except Exception as e:
                self._handle_pinecone_error(e)
        return self._indexes[cache_key]

    def _ensure_index_exists(self, client: Pinecone, index_name: str):
        """
        Creates serverless index if it doesn't already exist in Pinecone.
        """
        try:
            if not client.has_index(index_name):
                import time
                logger.info(f"Index '{index_name}' does not exist in account. Automatically creating serverless index...")
                client.create_index(
                    name=index_name,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.PINECONE_CLOUD,
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"Created Pinecone index '{index_name}' successfully. Waiting for index initialization...")
                time.sleep(2)
        except Exception as e:
            logger.warning(f"Index existence check note: {e}")

    def _handle_pinecone_error(self, e: Exception):
        err_msg = str(e)
        logger.error(f"Pinecone API Error: {err_msg}")

        if "401" in err_msg or "Unauthorized" in err_msg:
            raise PineconeAuthException("Invalid Pinecone API Key. Please verify your credentials in Settings.")
        elif "403" in err_msg or "quota" in err_msg.lower() or "storage" in err_msg.lower() or "limit" in err_msg.lower():
            raise PineconeQuotaException(
                "⚠️ Pinecone Storage Limit Reached\n\n"
                "Your Pinecone account has reached its available storage/quota.\n\n"
                "Please either:\n"
                "• Upgrade your Pinecone plan, or\n"
                "• Delete some existing documents/vectors.\n\n"
                "After freeing storage, try again."
            )
        else:
            raise Exception(f"Pinecone Error: {err_msg}")

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """
        Upserts vector dictionaries into Pinecone in batches within the isolated user namespace.
        """
        try:
            index = self._get_index(api_key, index_name)
            total_upserted = 0

            for start in range(0, len(vectors), batch_size):
                batch = vectors[start : start + batch_size]
                index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                total_upserted += len(batch)
                logger.info(f"Upserted {len(batch)} vectors into namespace '{namespace}' (total: {total_upserted}/{len(vectors)})")

            return total_upserted
        except Exception as e:
            self._handle_pinecone_error(e)
            return 0

    def search(
        self,
        query_vector: List[float],
        namespace: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches Pinecone index within the specified isolated user namespace.
        """
        try:
            index = self._get_index(api_key, index_name)
            target_k = top_k or settings.TOP_K

            query_params = {
                "vector": query_vector,
                "top_k": target_k,
                "namespace": namespace,
                "include_metadata": True
            }

            if filter_dict:
                query_params["filter"] = filter_dict

            raw_results = index.query(**query_params)

            documents = []
            for match in raw_results.get("matches", []):
                meta = match.get("metadata", {})
                documents.append({
                    "id": match["id"],
                    "score": float(match.get("score", 0.0)),
                    "text": meta.get("text", ""),
                    "source": meta.get("source", ""),
                    "chunk_id": meta.get("chunk_id", None),
                    "page": meta.get("page", None)
                })

            logger.info(f"Retrieved {len(documents)} document matches from namespace '{namespace}'")
            return documents
        except Exception as e:
            self._handle_pinecone_error(e)
            return []

    def fetch_by_ids(
        self,
        vector_ids: List[str],
        namespace: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches vector records by IDs within the isolated user namespace.
        """
        if not vector_ids:
            return []

        try:
            index = self._get_index(api_key, index_name)
            fetch_res = index.fetch(ids=vector_ids, namespace=namespace)
            raw_vectors = getattr(fetch_res, "vectors", {}) or (fetch_res.get("vectors", {}) if isinstance(fetch_res, dict) else {})

            records = []
            for vid, vdata in raw_vectors.items():
                meta = getattr(vdata, "metadata", None) or (vdata.get("metadata", {}) if isinstance(vdata, dict) else {})
                records.append({
                    "id": vid,
                    "score": 1.0,
                    "text": meta.get("text", "") if isinstance(meta, dict) else getattr(meta, "text", ""),
                    "source": meta.get("source", "") if isinstance(meta, dict) else getattr(meta, "source", ""),
                    "chunk_id": meta.get("chunk_id", None) if isinstance(meta, dict) else getattr(meta, "chunk_id", None),
                    "page": meta.get("page", None) if isinstance(meta, dict) else getattr(meta, "page", None)
                })
            return records
        except Exception as e:
            logger.warning(f"Error fetching vectors by ID: {e}")
            return []

    def delete_by_source(
        self,
        source_name: str,
        namespace: str,
        vector_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> int:
        """
        Deletes all vectors belonging to a source document within user's isolated namespace.
        """
        try:
            index = self._get_index(api_key, index_name)
            deleted_count = 0

            # 1. Delete by vector IDs if provided
            if vector_ids and len(vector_ids) > 0:
                for start in range(0, len(vector_ids), 1000):
                    batch_ids = vector_ids[start : start + 1000]
                    try:
                        index.delete(ids=batch_ids, namespace=namespace)
                        deleted_count += len(batch_ids)
                    except Exception as e:
                        logger.warning(f"Batch vector ID delete warning: {e}")
                logger.info(f"Deleted {deleted_count} vectors by ID for '{source_name}' in namespace '{namespace}'")

            # 2. Metadata filter deletion
            try:
                index.delete(
                    filter={"source": {"$eq": source_name}},
                    namespace=namespace
                )
            except Exception:
                try:
                    index.delete(
                        filter={"source": source_name},
                        namespace=namespace
                    )
                except Exception as ex:
                    logger.warning(f"Metadata filter deletion note: {ex}")

            return deleted_count
        except Exception as e:
            self._handle_pinecone_error(e)
            return 0

    def delete_namespace(self, namespace: str, api_key: Optional[str] = None, index_name: Optional[str] = None):
        """
        Completely deletes an entire namespace (e.g. when a user is deleted).
        """
        try:
            index = self._get_index(api_key, index_name)
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted all vectors in namespace '{namespace}'")
        except Exception as e:
            logger.warning(f"Delete namespace note: {e}")