import os
import time
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

    def _resolve_pinecone_key(self, api_key: Optional[str] = None) -> str:
        if api_key and str(api_key).strip():
            return str(api_key).strip()
        try:
            import streamlit as st
            if hasattr(st, "secrets") and len(st.secrets) > 0:
                if "PINECONE_API_KEY" in st.secrets:
                    return str(st.secrets["PINECONE_API_KEY"]).strip()
                if "pinecone_api_key" in st.secrets:
                    return str(st.secrets["pinecone_api_key"]).strip()
                if "pinecone" in st.secrets and "api_key" in st.secrets["pinecone"]:
                    return str(st.secrets["pinecone"]["api_key"]).strip()
        except Exception:
            pass
        return os.environ.get("PINECONE_API_KEY") or getattr(settings, "PINECONE_API_KEY", "")

    def _resolve_pinecone_index(self, index_name: Optional[str] = None) -> str:
        if index_name and str(index_name).strip():
            return str(index_name).strip()
        try:
            import streamlit as st
            if hasattr(st, "secrets") and len(st.secrets) > 0:
                if "PINECONE_INDEX_NAME" in st.secrets:
                    return str(st.secrets["PINECONE_INDEX_NAME"]).strip()
                if "PINECONE_INDEX" in st.secrets:
                    return str(st.secrets["PINECONE_INDEX"]).strip()
                if "pinecone_index" in st.secrets:
                    return str(st.secrets["pinecone_index"]).strip()
        except Exception:
            pass
        return os.environ.get("PINECONE_INDEX_NAME") or getattr(settings, "PINECONE_INDEX_NAME", "pdf-rag1-index")

    def _get_client(self, api_key: Optional[str] = None) -> Pinecone:
        key = self._resolve_pinecone_key(api_key)
        if not key:
            raise PineconeAuthException("No Pinecone API Key configured.")

        if key not in self._clients:
            logger.info("Initializing new Pinecone client...")
            self._clients[key] = Pinecone(api_key=key)
        return self._clients[key]

    def _get_index(self, api_key: Optional[str] = None, index_name: Optional[str] = None) -> Any:
        key = self._resolve_pinecone_key(api_key)
        idx_name = self._resolve_pinecone_index(index_name)
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

    def save_system_config_to_cloud(self, config_dict: Dict[str, Any], api_key: Optional[str] = None, index_name: Optional[str] = None) -> bool:
        """
        Persists global system settings (such as SMTP credentials and system policies)
        directly into cloud Pinecone storage under '__system_config__' namespace.
        Guarantees 100% persistence across Streamlit Cloud sleeps, restarts, and redeploys.
        """
        try:
            index = self._get_index(api_key, index_name)
            dim = getattr(settings, "EMBEDDING_DIMENSION", 384)
            dummy_vector = [0.001] * dim
            clean_meta = {}
            for k, v in config_dict.items():
                if v is not None:
                    clean_meta[str(k)] = v

            index.upsert(
                vectors=[{
                    "id": "documind_global_system_config",
                    "values": dummy_vector,
                    "metadata": clean_meta
                }],
                namespace="__system_config__"
            )
            logger.info("Successfully persisted global system & SMTP settings to Pinecone Cloud.")
            return True
        except Exception as e:
            logger.warning(f"Unable to persist system config to Pinecone Cloud: {e}")
            return False

    def load_system_config_from_cloud(self, api_key: Optional[str] = None, index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves global system settings from cloud Pinecone storage.
        """
        try:
            index = self._get_index(api_key, index_name)
            fetched = index.fetch(ids=["documind_global_system_config"], namespace="__system_config__")
            raw_vecs = getattr(fetched, "vectors", {}) or (fetched.get("vectors", {}) if isinstance(fetched, dict) else {})
            if "documind_global_system_config" in raw_vecs:
                target = raw_vecs["documind_global_system_config"]
                meta = getattr(target, "metadata", {}) or (target.get("metadata", {}) if isinstance(target, dict) else {})
                if isinstance(meta, dict):
                    logger.info("Successfully loaded persisted system & SMTP settings from Pinecone Cloud.")
                    return meta
            return {}
        except Exception as e:
            logger.warning(f"Note on loading system config from Pinecone Cloud: {e}")
            return {}

    def save_user_credentials_to_cloud(
        self,
        user_email: str,
        creds_dict: Dict[str, Any],
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> bool:
        """
        Persists personal BYOK API keys (Pinecone & Groq) for a specific user into Pinecone Cloud.
        """
        try:
            index = self._get_index(api_key, index_name)
            dim = getattr(settings, "EMBEDDING_DIMENSION", 384)
            dummy_vector = [0.001] * dim
            clean_email = user_email.lower().strip()
            clean_meta = {
                "user_email": clean_email,
                "pinecone_key": creds_dict.get("pinecone_key", ""),
                "pinecone_index": creds_dict.get("pinecone_index", ""),
                "groq_key": creds_dict.get("groq_key", ""),
                "groq_model": creds_dict.get("groq_model", "openai/gpt-oss-20b"),
                "updated_at": str(time.time())
            }
            index.upsert(
                vectors=[{
                    "id": f"user_creds_{clean_email}",
                    "values": dummy_vector,
                    "metadata": clean_meta
                }],
                namespace="__system_config__"
            )
            self._register_cloud_user_email(clean_email, api_key, index_name)
            logger.info(f"Persisted BYOK credentials for '{clean_email}' to Pinecone Cloud.")
            return True
        except Exception as e:
            logger.warning(f"Error persisting user credentials to cloud: {e}")
            return False

    def load_user_credentials_from_cloud(
        self,
        user_email: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieves personal BYOK API keys (Pinecone & Groq) for a specific user from Pinecone Cloud.
        """
        try:
            index = self._get_index(api_key, index_name)
            clean_email = user_email.lower().strip()
            target_id = f"user_creds_{clean_email}"
            fetched = index.fetch(ids=[target_id], namespace="__system_config__")
            raw_vecs = getattr(fetched, "vectors", {}) or (fetched.get("vectors", {}) if isinstance(fetched, dict) else {})
            if target_id in raw_vecs:
                target = raw_vecs[target_id]
                meta = getattr(target, "metadata", {}) or (target.get("metadata", {}) if isinstance(target, dict) else {})
                if isinstance(meta, dict) and meta.get("pinecone_key") and meta.get("groq_key"):
                    logger.info(f"Loaded persisted BYOK credentials for '{clean_email}' from Pinecone Cloud.")
                    return meta
            return {}
        except Exception as e:
            logger.warning(f"Note on loading user credentials from cloud for '{user_email}': {e}")
            return {}

    def delete_user_credentials_from_cloud(
        self,
        user_email: str,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> bool:
        """
        Removes personal BYOK API keys from Pinecone Cloud when user reverts to shared credentials or deletes them.
        """
        try:
            index = self._get_index(api_key, index_name)
            clean_email = user_email.lower().strip()
            target_id = f"user_creds_{clean_email}"
            index.delete(ids=[target_id], namespace="__system_config__")
            logger.info(f"Deleted BYOK credentials for '{clean_email}' from Pinecone Cloud.")
            return True
        except Exception as e:
            logger.warning(f"Note on deleting user credentials from cloud: {e}")
            return False

    def _register_cloud_user_email(self, email: str, api_key: Optional[str] = None, index_name: Optional[str] = None):
        """Maintains the active list of user emails in the cloud catalog."""
        try:
            index = self._get_index(api_key, index_name)
            fetched = index.fetch(ids=["documind_users_registry"], namespace="__system_config__")
            raw_vecs = getattr(fetched, "vectors", {}) or (fetched.get("vectors", {}) if isinstance(fetched, dict) else {})
            emails = []
            if "documind_users_registry" in raw_vecs:
                meta = getattr(raw_vecs["documind_users_registry"], "metadata", {}) or {}
                raw_list = meta.get("emails", [])
                if isinstance(raw_list, list):
                    emails = list(raw_list)
                elif isinstance(raw_list, str):
                    emails = [e.strip() for e in raw_list.split(",") if e.strip()]
            if email not in emails:
                emails.append(email)
                dim = getattr(settings, "EMBEDDING_DIMENSION", 384)
                index.upsert(
                    vectors=[{
                        "id": "documind_users_registry",
                        "values": [0.001] * dim,
                        "metadata": {"emails": emails}
                    }],
                    namespace="__system_config__"
                )
        except Exception as e:
            logger.warning(f"Note on registering cloud user email: {e}")

    def save_user_account_to_cloud(
        self,
        email: str,
        name: str,
        password_hash: str,
        role: str = "user",
        api_key: Optional[str] = None,
        index_name: Optional[str] = None
    ) -> bool:
        """
        Persists a registered user account to Pinecone Cloud so that accounts survive reboots.
        """
        try:
            import base64
            index = self._get_index(api_key, index_name)
            dim = getattr(settings, "EMBEDDING_DIMENSION", 384)
            clean_email = email.lower().strip()
            b64_hash = base64.b64encode(password_hash.encode("utf-8")).decode("utf-8") if password_hash else ""
            meta = {
                "email": clean_email,
                "name": name or "User",
                "password_hash_b64": b64_hash,
                "role": role or "user"
            }
            index.upsert(
                vectors=[{
                    "id": f"user_account_{clean_email}",
                    "values": [0.001] * dim,
                    "metadata": meta
                }],
                namespace="__system_config__"
            )
            self._register_cloud_user_email(clean_email, api_key, index_name)
            logger.info(f"Persisted user account '{clean_email}' to Pinecone Cloud.")
            return True
        except Exception as e:
            logger.warning(f"Error persisting user account to cloud: {e}")
            return False

    def sync_all_users_from_cloud(self, db, api_key: Optional[str] = None, index_name: Optional[str] = None):
        """
        Hydrates all registered users and their BYOK credentials from Pinecone Cloud into SQLite.
        """
        try:
            import base64
            from app.db import crud
            from app.services.crypto_service import CryptoService
            crypto = CryptoService()

            index = self._get_index(api_key, index_name)
            fetched = index.fetch(ids=["documind_users_registry"], namespace="__system_config__")
            raw_vecs = getattr(fetched, "vectors", {}) or (fetched.get("vectors", {}) if isinstance(fetched, dict) else {})
            if "documind_users_registry" not in raw_vecs:
                return

            meta = getattr(raw_vecs["documind_users_registry"], "metadata", {}) or {}
            raw_emails = meta.get("emails", [])
            emails = raw_emails if isinstance(raw_emails, list) else [e.strip() for e in str(raw_emails).split(",") if e.strip()]

            if not emails:
                return

            target_ids = [f"user_account_{e}" for e in emails] + [f"user_creds_{e}" for e in emails]
            batch_fetch = index.fetch(ids=target_ids, namespace="__system_config__")
            batch_vecs = getattr(batch_fetch, "vectors", {}) or (batch_fetch.get("vectors", {}) if isinstance(batch_fetch, dict) else {})

            for email in emails:
                acc_id = f"user_account_{email}"
                user_obj = crud.get_user_by_email(db, email)
                if acc_id in batch_vecs:
                    acc_meta = getattr(batch_vecs[acc_id], "metadata", {}) or (batch_vecs[acc_id].get("metadata", {}) if isinstance(batch_vecs[acc_id], dict) else {})
                    b64_hash = acc_meta.get("password_hash_b64", "")
                    raw_hash = base64.b64decode(b64_hash.encode("utf-8")).decode("utf-8") if b64_hash else ""
                    user_name = acc_meta.get("name", "User")
                    user_role = acc_meta.get("role", "user")

                    if not user_obj and raw_hash:
                        user_obj = crud.create_user(
                            db=db,
                            email=email,
                            password_hash=raw_hash,
                            name=user_name,
                            role=user_role,
                            email_verified=True
                        )
                        logger.info(f"Restored user account '{email}' from Pinecone Cloud.")

                # Restore credentials if present
                creds_id = f"user_creds_{email}"
                if user_obj and creds_id in batch_vecs:
                    c_meta = getattr(batch_vecs[creds_id], "metadata", {}) or (batch_vecs[creds_id].get("metadata", {}) if isinstance(batch_vecs[creds_id], dict) else {})
                    p_key = c_meta.get("pinecone_key", "")
                    g_key = c_meta.get("groq_key", "")
                    p_idx = c_meta.get("pinecone_index", settings.PINECONE_INDEX_NAME)
                    g_mod = c_meta.get("groq_model", settings.GROQ_MODEL)
                    if p_key and g_key:
                        crud.upsert_user_credentials(
                            db=db,
                            user_id=user_obj.id,
                            pinecone_api_key_encrypted=crypto.encrypt(p_key),
                            pinecone_index=p_idx,
                            groq_api_key_encrypted=crypto.encrypt(g_key),
                            groq_model=g_mod
                        )
                        logger.info(f"Restored BYOK credentials for user '{email}' from Pinecone Cloud.")
        except Exception as e:
            logger.warning(f"Note on syncing users from cloud storage: {e}")