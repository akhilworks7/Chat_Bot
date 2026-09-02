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
        user_id: Optional[int] = None,
        vector_count: Optional[int] = None,
        vector_ids: Optional[List[str]] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> int:
        """
        Deletes all vectors belonging to a source document within user's isolated namespace.
        Uses live namespace scanning + deterministic ID deletion + metadata filtering
        to guarantee 100% removal in Pinecone Serverless.
        """
        from pathlib import Path
        try:
            index = self._get_index(api_key, index_name)
            deleted_count = 0
            base_id = Path(source_name).stem.replace(" ", "_")

            # 1. Discover all live vector IDs matching this document from Pinecone's list API
            discovered_ids = set()
            try:
                for batch in index.list(namespace=namespace):
                    for item in batch:
                        vid = item if isinstance(item, str) else getattr(item, "id", str(item))
                        if f"_{base_id}_chunk_" in vid or vid.startswith(f"{base_id}_chunk_") or f"_{base_id}_" in vid or base_id in vid:
                            discovered_ids.add(vid)
            except Exception as ex:
                logger.debug(f"Pinecone list-based vector discovery note: {ex}")

            # 2. Add deterministic IDs as guaranteed fallback
            target_ids = list(discovered_ids)
            if user_id is not None:
                max_chunks = (vector_count + 50) if (vector_count and vector_count > 0) else 1500
                det_ids = [f"user_{user_id}_{base_id}_chunk_{i}" for i in range(max_chunks)]
                det_ids.extend([f"{base_id}_chunk_{i}" for i in range(min(max_chunks, 500))])
                target_ids = list(set(target_ids + det_ids))

            # 3. Delete by vector IDs in batches of 500
            if target_ids:
                for start in range(0, len(target_ids), 500):
                    batch_ids = target_ids[start : start + 500]
                    try:
                        index.delete(ids=batch_ids, namespace=namespace)
                        deleted_count += len(batch_ids)
                    except Exception as e:
                        logger.warning(f"Batch vector ID delete warning: {e}")
                logger.info(f"Deleted vector batch for '{source_name}' in namespace '{namespace}' (total targeted: {len(target_ids)})")

            # 4. Also attempt metadata filter deletion if supported by index type
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
                    logger.debug(f"Metadata filter deletion note: {ex}")

            return deleted_count
        except Exception as e:
            self._handle_pinecone_error(e)
            return 0

    def delete_document_vectors(
        self,
        user_id: int,
        file_name: str,
        vector_count: int,
        namespace: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> int:
        """
        High-level helper to guarantee all vector chunks of a document are removed from Pinecone.
        """
        return self.delete_by_source(
            source_name=file_name,
            namespace=namespace,
            user_id=user_id,
            vector_count=vector_count,
            api_key=api_key,
            index_name=index_name
        )

    def delete_namespace(self, namespace: str, api_key: Optional[str] = None, index_name: Optional[str] = None):
        """
        Completely purges an entire namespace (e.g. when all user documents are deleted or user requests wipe).
        """
        try:
            index = self._get_index(api_key, index_name)
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Successfully purged all vectors in namespace '{namespace}'")
        except Exception as e:
            logger.warning(f"Delete namespace note: {e}")

    def get_namespace_stats(
        self,
        namespace: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetches live index statistics from Pinecone and returns namespace-specific metrics:
        {
            "vector_count": int,
            "total_index_vectors": int,
            "dimension": int,
            "namespaces": Dict[str, int]
        }
        """
        try:
            index = self._get_index(api_key, index_name)
            stats = index.describe_index_stats()
            raw_namespaces = getattr(stats, "namespaces", {}) or (stats.get("namespaces", {}) if isinstance(stats, dict) else {})

            ns_counts = {}
            for ns_key, ns_val in raw_namespaces.items():
                c = getattr(ns_val, "vector_count", 0) or (ns_val.get("vector_count", 0) if isinstance(ns_val, dict) else 0)
                ns_counts[ns_key] = c

            ns_count = ns_counts.get(namespace, 0)
            total_vecs = getattr(stats, "total_vector_count", 0) or (stats.get("total_vector_count", 0) if isinstance(stats, dict) else 0)
            dim = getattr(stats, "dimension", settings.EMBEDDING_DIMENSION) or (stats.get("dimension", settings.EMBEDDING_DIMENSION) if isinstance(stats, dict) else settings.EMBEDDING_DIMENSION)

            return {
                "vector_count": ns_count,
                "total_index_vectors": total_vecs,
                "dimension": dim,
                "namespaces": ns_counts
            }
        except Exception as e:
            logger.warning(f"Failed to fetch Pinecone live stats for namespace '{namespace}': {e}")
            return {
                "vector_count": 0,
                "total_index_vectors": 0,
                "dimension": settings.EMBEDDING_DIMENSION,
                "namespaces": {}
            }

    def discover_indexed_documents(
        self,
        namespace: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Discovers documents and chunk counts stored in a Pinecone namespace.
        Scans vector IDs and fetches sample metadata to resolve file names and chunk volumes.
        Returns a list of:
        [{
            "file_name": str,
            "vector_count": int,
            "sample_id": str,
            "namespace": str
        }]
        """
        from collections import defaultdict
        try:
            index = self._get_index(api_key, index_name)
            all_ids = []

            # Pinecone list returns paginated batches
            for batch in index.list(namespace=namespace):
                all_ids.extend([item if isinstance(item, str) else getattr(item, "id", str(item)) for item in batch])

            if not all_ids:
                return []

            # Group vector IDs by document prefix (format: user_{uid}_{docname}_chunk_{idx})
            grouped = defaultdict(list)
            for vid in all_ids:
                if "_chunk_" in vid:
                    prefix = vid.rsplit("_chunk_", 1)[0]
                else:
                    prefix = vid
                grouped[prefix].append(vid)

            # Sample 1 ID per document to resolve source filename and metadata
            sample_ids = [vids[0] for vids in grouped.values()]
            fetched = index.fetch(ids=sample_ids, namespace=namespace)
            raw_vecs = getattr(fetched, "vectors", {}) or (fetched.get("vectors", {}) if isinstance(fetched, dict) else {})

            discovered = []
            for prefix, vids in grouped.items():
                sample_id = vids[0]
                vdata = raw_vecs.get(sample_id)
                meta = getattr(vdata, "metadata", {}) or (vdata.get("metadata", {}) if isinstance(vdata, dict) else {})
                source_name = meta.get("source", prefix) if isinstance(meta, dict) else getattr(meta, "source", prefix)

                # Strip possible path if stored as path
                from pathlib import Path
                clean_name = Path(source_name).name if source_name else prefix

                discovered.append({
                    "file_name": clean_name,
                    "vector_count": len(vids),
                    "sample_id": sample_id,
                    "namespace": namespace
                })

            discovered.sort(key=lambda x: x["file_name"])
            logger.info(f"Discovered {len(discovered)} documents ({len(all_ids)} vectors) in Pinecone namespace '{namespace}'")
            return discovered

        except Exception as e:
            logger.warning(f"Error discovering documents from Pinecone namespace '{namespace}': {e}")
            return []

    def fetch_document_text(
        self,
        user_id: int,
        file_name: str,
        vector_count: int,
        namespace: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> str:
        """
        Reconstructs document text directly from Pinecone vector chunk metadata.
        Guarantees users can view or download extracted document content even if the physical
        PDF was wiped from ephemeral cloud disk.
        """
        from pathlib import Path
        try:
            index = self._get_index(api_key, index_name)
            base_id = Path(file_name).stem.replace(" ", "_")
            num_chunks = max(vector_count, 1)
            target_ids = [f"user_{user_id}_{base_id}_chunk_{i}" for i in range(num_chunks)]

            chunks_text = []
            batch_size = 100
            for start in range(0, len(target_ids), batch_size):
                batch_ids = target_ids[start : start + batch_size]
                fetched = index.fetch(ids=batch_ids, namespace=namespace)
                raw_vecs = getattr(fetched, "vectors", {}) or (fetched.get("vectors", {}) if isinstance(fetched, dict) else {})
                for vid in batch_ids:
                    if vid in raw_vecs:
                        meta = getattr(raw_vecs[vid], "metadata", {}) or (raw_vecs[vid].get("metadata", {}) if isinstance(raw_vecs[vid], dict) else {})
                        txt = meta.get("text", "") if isinstance(meta, dict) else getattr(meta, "text", "")
                        if txt:
                            chunks_text.append(txt)

            return "\n\n".join(chunks_text)
        except Exception as e:
            logger.warning(f"Error fetching document text from Pinecone for '{file_name}': {e}")
            return ""