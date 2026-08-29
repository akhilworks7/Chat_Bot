from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("vector_service")


class VectorService:
    """
    Manages Pinecone connection, index initialization, batched upserts, and vector similarity search.
    """

    _instance = None
    _pc = None
    _index = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def _get_index(self):
        if self._index is None:
            if self._pc is None:
                logger.info("Initializing Pinecone client...")
                self._pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                self._ensure_index_exists()
            self._index = self._pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index: '{settings.PINECONE_INDEX_NAME}'")
        return self._index

    def _ensure_index_exists(self):
        """
        Creates the serverless index if it doesn't already exist.
        """
        try:
            if not self._pc.has_index(settings.PINECONE_INDEX_NAME):
                logger.info(
                    f"Index '{settings.PINECONE_INDEX_NAME}' does not exist. Creating serverless index..."
                )
                self._pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.PINECONE_CLOUD,
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                logger.info(f"Created Pinecone index '{settings.PINECONE_INDEX_NAME}' successfully.")
        except Exception as e:
            logger.warning(f"Index existence check/creation note: {e}")

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        batch_size: int = 100
    ) -> int:
        """
        Upserts vector dictionaries into Pinecone in batches.
        Each vector format: {"id": str, "values": List[float], "metadata": Dict}
        """
        index = self._get_index()
        target_namespace = namespace or settings.PINECONE_NAMESPACE
        total_upserted = 0

        for start in range(0, len(vectors), batch_size):
            batch = vectors[start : start + batch_size]
            index.upsert(
                vectors=batch,
                namespace=target_namespace
            )
            total_upserted += len(batch)
            logger.info(f"Upserted {total_upserted}/{len(vectors)} vectors into namespace '{target_namespace}'")

        return total_upserted

    def upsert(self, vectors: List[Dict[str, Any]], namespace: Optional[str] = None):
        """
        Alias for upsert_vectors for backward compatibility.
        """
        return self.upsert_vectors(vectors, namespace=namespace)

    def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches Pinecone index and returns normalized document records.
        """
        index = self._get_index()
        target_k = top_k or settings.TOP_K
        target_namespace = namespace or settings.PINECONE_NAMESPACE

        query_params = {
            "vector": query_vector,
            "top_k": target_k,
            "namespace": target_namespace,
            "include_metadata": True
        }

        if filter_dict:
            query_params["filter"] = filter_dict

        raw_results = index.query(**query_params)

        documents = []
        for match in raw_results.get("matches", []):
            documents.append({
                "id": match["id"],
                "score": float(match.get("score", 0.0)),
                "text": match.get("metadata", {}).get("text", ""),
                "source": match.get("metadata", {}).get("source", ""),
                "chunk_id": match.get("metadata", {}).get("chunk_id", None),
                "page": match.get("metadata", {}).get("page", None)
            })

        logger.info(f"Retrieved {len(documents)} document matches from namespace '{target_namespace}'")
        return documents

    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves Pinecone index description and vector statistics.
        """
        index = self._get_index()
        stats = index.describe_index_stats()
        
        namespaces_dict = {}
        raw_namespaces = getattr(stats, "namespaces", None) or (stats.get("namespaces") if isinstance(stats, dict) else {})
        if raw_namespaces:
            for ns, ns_info in raw_namespaces.items():
                if hasattr(ns_info, "vector_count"):
                    namespaces_dict[ns] = {"vector_count": ns_info.vector_count}
                elif isinstance(ns_info, dict):
                    namespaces_dict[ns] = ns_info
                else:
                    namespaces_dict[ns] = {"vector_count": getattr(ns_info, "vector_count", 0)}

        return {
            "dimension": getattr(stats, "dimension", None) or (stats.get("dimension") if isinstance(stats, dict) else settings.EMBEDDING_DIMENSION),
            "total_vector_count": getattr(stats, "total_vector_count", None) or (stats.get("total_vector_count") if isinstance(stats, dict) else 0),
            "namespaces": namespaces_dict
        }

    def fetch_by_ids(self, vector_ids: List[str], namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches vector records and metadata by their vector IDs from Pinecone.
        """
        if not vector_ids:
            return []

        index = self._get_index()
        target_namespace = namespace or settings.PINECONE_NAMESPACE
        try:
            fetch_res = index.fetch(ids=vector_ids, namespace=target_namespace)
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
            logger.warning(f"Error fetching vector IDs from Pinecone: {e}")
            return []

    def delete_by_source(
        self,
        source_name: str,
        vector_ids: Optional[List[str]] = None,
        namespace: Optional[str] = None
    ) -> int:
        """
        Deletes all vectors belonging to a source document from Pinecone.
        Executes ID deletion and metadata filter deletion to guarantee complete removal.
        """
        index = self._get_index()
        target_namespace = namespace or settings.PINECONE_NAMESPACE
        deleted_count = 0

        # 1. Delete by vector IDs if provided
        if vector_ids and len(vector_ids) > 0:
            for start in range(0, len(vector_ids), 1000):
                batch_ids = vector_ids[start : start + 1000]
                try:
                    index.delete(ids=batch_ids, namespace=target_namespace)
                    deleted_count += len(batch_ids)
                except Exception as e:
                    logger.warning(f"Error deleting batch IDs from Pinecone: {e}")
            logger.info(f"Deleted {deleted_count} vectors by IDs for source '{source_name}'")

        # 2. Also execute metadata filter delete to catch all vectors for this source
        try:
            index.delete(
                filter={"source": {"$eq": source_name}},
                namespace=target_namespace
            )
            logger.info(f"Executed metadata filter deletion for source '{source_name}' in namespace '{target_namespace}'")
        except Exception as e:
            # Fallback filter format for some Pinecone index configurations
            try:
                index.delete(
                    filter={"source": source_name},
                    namespace=target_namespace
                )
                logger.info(f"Executed direct metadata filter deletion for source '{source_name}'")
            except Exception as ex:
                logger.warning(f"Metadata filter deletion note (IDs deletion performed: {deleted_count}): {ex}")

        return deleted_count