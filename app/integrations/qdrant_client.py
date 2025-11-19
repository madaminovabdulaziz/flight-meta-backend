"""
Qdrant Vector Database Client - Production Version
==================================================
Layer 3 of the memory system — Semantic memory storage.

Responsibilities:
- Connect to Qdrant (lazy, reusable async client)
- Ensure collection exists with correct schema
- Upsert user memories (with embeddings)
- Semantic search with user + memory_type filters
- Read / update / delete individual memories
- Bulk delete all memories for a user

Design Principles:
- Single shared async client (connection pooling)
- Defensive guard rails (no client → safe fallbacks)
- Clear logging for observability
- Strong typing and payload normalization
- Explicit error propagation for upstream handling
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from uuid import uuid4
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Production-ready Qdrant service wrapper.

    Public API (intentionally small and stable):
        - connect() / disconnect()
        - upsert_memory(...)
        - search_memories(...)
        - get_memory_by_id(...)
        - update_memory_metadata(...)
        - delete_memory(...)
        - delete_user_memories(...)

    Notes:
    - `connect()` is idempotent and lazy: called automatically by get_qdrant_client()
    - All methods assume an initialized client; if not, they log and raise
    - Caller is responsible for handling exceptions at the boundary (e.g., node level)
    """

    def __init__(self) -> None:
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name: str = settings.QDRANT_COLLECTION_NAME
        self.vector_size: int = settings.QDRANT_VECTOR_SIZE
        self._is_collection_ensured: bool = False

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS
    # -------------------------------------------------------------------------

    def _ensure_client(self) -> AsyncQdrantClient:
        """
        Ensure that the Qdrant client is initialized.

        Returns:
            AsyncQdrantClient

        Raises:
            RuntimeError if client is not connected.
        """
        if not self.client:
            raise RuntimeError("Qdrant client is not connected. Call connect() first.")
        return self.client

    async def _ensure_collection_once(self) -> None:
        """
        Ensure collection exists exactly once per process lifetime.

        This prevents doing `get_collections` on every call.
        """
        if self._is_collection_ensured:
            return

        client = self._ensure_client()

        try:
            collections = await client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self.collection_name not in existing_names:
                logger.info(f"[Qdrant] Creating collection '{self.collection_name}'")

                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

                logger.info(f"[Qdrant] Collection '{self.collection_name}' created")
            else:
                logger.info(f"[Qdrant] Collection '{self.collection_name}' already exists")

            self._is_collection_ensured = True

        except Exception as e:
            logger.error(f"[Qdrant] Failed to ensure collection: {e}", exc_info=True)
            # Do not silently ignore: caller should know semantic memory is unavailable
            raise

    # -------------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Initialize Qdrant connection and ensure collection exists.

        Idempotent: safe to call multiple times.
        """
        if self.client is not None:
            # Already connected
            return

        try:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
                timeout=10.0,
            )
            logger.info(f"[Qdrant] Connected to {settings.QDRANT_URL}")

            await self._ensure_collection_once()

        except Exception as e:
            # If connection fails, clear client so we don't use a half-initialized object
            self.client = None
            logger.error(f"[Qdrant] Failed to connect: {e}", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """
        Close Qdrant connection.

        Safe to call multiple times.
        """
        if self.client:
            try:
                await self.client.close()
                logger.info("[Qdrant] Disconnected")
            except Exception as e:
                logger.warning(f"[Qdrant] Error while disconnecting: {e}", exc_info=True)
            finally:
                self.client = None
                self._is_collection_ensured = False

    # -------------------------------------------------------------------------
    # MEMORY UPSERT
    # -------------------------------------------------------------------------

    async def upsert_memory(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        memory_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create or update a memory entry.

        Args:
            user_id: Application-level user identifier (stringified)
            content: Raw text content of memory
            embedding: Vector embedding (must match configured vector_size)
            memory_type: Semantic category label
            metadata: Arbitrary JSON-serializable dict

        Returns:
            memory_id (UUID string)

        Raises:
            RuntimeError if client not connected
            Exception from Qdrant if upsert fails
        """
        client = self._ensure_client()
        await self._ensure_collection_once()

        if not isinstance(embedding, list) or len(embedding) != self.vector_size:
            raise ValueError(
                f"Embedding size mismatch: expected {self.vector_size}, "
                f"got {len(embedding)}"
            )

        memory_id = str(uuid4())
        metadata = metadata or {}

        point = PointStruct(
            id=memory_id,
            vector=embedding,
            payload={
                "user_id": user_id,
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata,
            },
        )

        try:
            await client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.info(
                f"[Qdrant] Upserted memory {memory_id} "
                f"for user={user_id}, type={memory_type}"
            )
            return memory_id

        except Exception as e:
            logger.error(f"[Qdrant] Failed to upsert memory: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # SEMANTIC SEARCH
    # -------------------------------------------------------------------------

    async def search_memories(
        self,
        user_id: str,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across a user's stored memories.

        Args:
            user_id: Application-level user identifier (stringified)
            query_embedding: Query embedding vector
            limit: Max number of results
            score_threshold: Minimum similarity score (0.0-1.0)
            memory_type: Optional filter for memory_type payload

        Returns:
            List of memory dicts:
            [
                {
                    "id": str,
                    "score": float,
                    "content": str,
                    "memory_type": str,
                    "metadata": dict,
                },
                ...
            ]

        Raises:
            RuntimeError if client not connected
            Exception from Qdrant if search fails
        """
        client = self._ensure_client()
        await self._ensure_collection_once()

        if not query_embedding or not isinstance(query_embedding, list):
            logger.warning("[Qdrant] Empty or invalid query embedding, returning []")
            return []

        conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]

        if memory_type:
            conditions.append(
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type))
            )

        try:
            results = await client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=conditions),
                limit=limit,
                score_threshold=score_threshold,
            )

            formatted: List[Dict[str, Any]] = []
            for r in results:
                payload = r.payload or {}
                formatted.append(
                    {
                        "id": r.id,
                        "score": r.score,
                        "content": payload.get("content"),
                        "memory_type": payload.get("memory_type"),
                        "metadata": payload.get("metadata") or {},
                    }
                )

            logger.info(
                f"[Qdrant] Search for user={user_id}, type={memory_type or 'any'} "
                f"→ {len(formatted)} results (limit={limit}, threshold={score_threshold})"
            )

            return formatted

        except Exception as e:
            logger.error(f"[Qdrant] Failed to search memories: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # READ / UPDATE
    # -------------------------------------------------------------------------

    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve single memory by its Qdrant ID.

        Returns:
            Memory dict or None if not found.
        """
        client = self._ensure_client()
        await self._ensure_collection_once()

        try:
            result = await client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
            )

            if not result:
                logger.info(f"[Qdrant] Memory not found: {memory_id}")
                return None

            point = result[0]
            payload = point.payload or {}

            return {
                "id": point.id,
                "content": payload.get("content"),
                "memory_type": payload.get("memory_type"),
                "metadata": payload.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"[Qdrant] Failed to retrieve memory {memory_id}: {e}", exc_info=True)
            raise

    async def update_memory_metadata(
        self,
        memory_id: str,
        metadata_updates: Dict[str, Any],
    ) -> None:
        """
        Merge new metadata into existing memory metadata.

        Raises:
            ValueError if memory not found
        """
        if not metadata_updates:
            # Nothing to do
            return

        client = self._ensure_client()
        await self._ensure_collection_once()

        try:
            memory = await self.get_memory_by_id(memory_id)
            if not memory:
                raise ValueError(f"Memory not found: {memory_id}")

            existing_metadata = memory.get("metadata") or {}
            updated_metadata = {**existing_metadata, **metadata_updates}

            await client.set_payload(
                collection_name=self.collection_name,
                points=[memory_id],
                payload={"metadata": updated_metadata},
            )

            logger.info(f"[Qdrant] Updated metadata for memory {memory_id}")

        except Exception as e:
            logger.error(f"[Qdrant] Failed to update metadata: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # DELETE OPERATIONS
    # -------------------------------------------------------------------------

    async def delete_memory(self, memory_id: str) -> None:
        """
        Delete a single memory by ID.
        """
        client = self._ensure_client()
        await self._ensure_collection_once()

        try:
            await client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[memory_id]),
            )
            logger.info(f"[Qdrant] Deleted memory {memory_id}")

        except Exception as e:
            logger.error(f"[Qdrant] Failed to delete memory {memory_id}: {e}", exc_info=True)
            raise

    async def delete_user_memories(self, user_id: str) -> None:
        """
        Delete ALL memories for a given user (by payload filter).
        """
        client = self._ensure_client()
        await self._ensure_collection_once()

        try:
            await client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )
            logger.info(f"[Qdrant] Deleted all memories for user={user_id}")

        except Exception as e:
            logger.error(f"[Qdrant] Failed to delete user memories for {user_id}: {e}", exc_info=True)
            raise


# -------------------------------------------------------------------------
# GLOBAL SINGLETON + ACCESSOR
# -------------------------------------------------------------------------

qdrant_service = QdrantService()


async def get_qdrant_client() -> QdrantService:
    """
    Get global QdrantService instance, ensuring connection is established.

    Usage:
        qdrant = await get_qdrant_client()
        await qdrant.search_memories(...)
    """
    if not qdrant_service.client:
        await qdrant_service.connect()
    return qdrant_service
