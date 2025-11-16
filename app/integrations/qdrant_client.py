"""
Qdrant Vector Database Client
Layer 3 of the memory system — Semantic memory storage
"""

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
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Handles all Qdrant operations:
    - Connecting
    - Upserting user memories
    - Semantic search
    - Updating metadata
    - Deleting memories
    """

    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.vector_size = settings.QDRANT_VECTOR_SIZE

    async def connect(self) -> None:
        """Initialize Qdrant connection + ensure the collection exists."""
        try:
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=10.0,
            )
            logger.info(f"Connected to Qdrant at {settings.QDRANT_URL}")

            await self.ensure_collection()

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Qdrant connection."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from Qdrant")

    async def ensure_collection(self) -> None:
        """Create collection if not exist."""
        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection '{self.collection_name}'")

                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    )
                )

                logger.info(f"Collection '{self.collection_name}' created")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    # ------------------------------------------------------
    # MEMORY UPSERT
    # ------------------------------------------------------

    async def upsert_memory(
        self,
        user_id: str,
        content: str,
        embedding: List[float],
        memory_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create or update a memory entry.
        Returns: UUID string of the memory.
        """

        memory_id = str(uuid4())
        metadata = metadata or {}

        try:
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

            await self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.info(f"Upserted memory {memory_id} for user {user_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to upsert memory: {e}")
            raise

    # ------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------

    async def search_memories(
        self,
        user_id: str,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.5,
        memory_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        Semantic search across a user's stored memories.
        """

        try:
            conditions = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]

            if memory_type:
                conditions.append(
                    FieldCondition(key="memory_type", match=MatchValue(value=memory_type))
                )

            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=Filter(must=conditions),
                limit=limit,
                score_threshold=score_threshold,
            )

            formatted = [
                {
                    "id": r.id,
                    "score": r.score,
                    "content": r.payload.get("content"),
                    "memory_type": r.payload.get("memory_type"),
                    "metadata": r.payload.get("metadata") or {},
                }
                for r in results
            ]

            return formatted

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise

    # ------------------------------------------------------
    # GET MEMORY BY ID
    # ------------------------------------------------------

    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict]:
        try:
            result = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id],
            )

            if not result:
                return None

            point = result[0]
            return {
                "id": point.id,
                "content": point.payload.get("content"),
                "memory_type": point.payload.get("memory_type"),
                "metadata": point.payload.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            raise

    # ------------------------------------------------------
    # UPDATE METADATA
    # ------------------------------------------------------

    async def update_memory_metadata(self, memory_id: str, metadata_updates: Dict) -> None:
        try:
            memory = await self.get_memory_by_id(memory_id)
            if not memory:
                raise ValueError(f"Memory not found: {memory_id}")

            updated_metadata = {**memory["metadata"], **metadata_updates}

            await self.client.set_payload(
                collection_name=self.collection_name,
                points=[memory_id],
                payload={"metadata": updated_metadata},
            )

        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            raise

    # ------------------------------------------------------
    # DELETE MEMORY
    # ------------------------------------------------------

    async def delete_memory(self, memory_id: str) -> None:
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id],
            )
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise

    # ------------------------------------------------------
    # DELETE ALL MEMORIES FOR USER
    # ------------------------------------------------------

    async def delete_user_memories(self, user_id: str) -> None:
        try:
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
            )

        except Exception as e:
            logger.error(f"Failed to delete user memories: {e}")
            raise


# Global instance
qdrant_service = QdrantService()


async def get_qdrant_client() -> QdrantService:
    if not qdrant_service.client:
        await qdrant_service.connect()
    return qdrant_service
