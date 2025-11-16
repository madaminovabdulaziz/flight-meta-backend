import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from services.embedding_service import generate_embedding
from app.integrations.qdrant_client import get_qdrant_client
from app.models.models import MemoryEvent, MemoryType

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Hybrid memory system:
    - SQL stores structured memories (audit trail)
    - Qdrant stores semantic embeddings for search
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ------------------------------------------------------
    # STORE MEMORY
    # ------------------------------------------------------
    async def store_memory(
        self,
        user_id: int,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[str] = None,
    ) -> int:
        """
        Saves the memory in SQL + Qdrant
        """

        try:
            # 1) Embed the natural language content
            embedding = await generate_embedding(content)

            # 2) Save structured fields into SQL
            db_memory = MemoryEvent(
                user_id=user_id,
                content=content,
                type=memory_type,              # <-- CORRECT FIELD
                metadata_json=metadata or {},  # <-- CORRECT FIELD
            )

            self.session.add(db_memory)
            await self.session.flush()  # ensures db_memory.id is available

            # 3) Save vector to Qdrant
            qdrant = await get_qdrant_client()
            vector_id = await qdrant.upsert_memory(
                user_id=str(user_id),
                content=content,
                embedding=embedding,
                memory_type=memory_type.value,  # string for Qdrant payload
                metadata=metadata or {},
            )

            # 4) Store Qdrant ID in SQL
            db_memory.vector_id = vector_id

            await self.session.commit()
            return db_memory.id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            await self.session.rollback()
            raise

    # ------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # ------------------------------------------------------
    async def retrieve_memories(
        self,
        user_id: int,
        query_text: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Semantic search using Qdrant vectors
        """

        try:
            embedding = await generate_embedding(query_text)

            qdrant = await get_qdrant_client()
            results = await qdrant.search_memories(
                user_id=str(user_id),
                query_embedding=embedding,
                limit=limit,
            )

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve semantic memories: {e}")
            raise

    # ------------------------------------------------------
    # RAW SQL LIST
    # ------------------------------------------------------
    async def list_user_memories(self, user_id: int):
        stmt = select(MemoryEvent).where(MemoryEvent.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    # ------------------------------------------------------
    # DELETE MEMORY
    # ------------------------------------------------------
    async def delete_memory(self, memory_id: int):
        try:
            stmt = select(MemoryEvent).where(MemoryEvent.id == memory_id)
            result = await self.session.execute(stmt)
            memory = result.scalar_one_or_none()

            if memory is None:
                raise ValueError("Memory not found")

            # Delete from Qdrant layer
            if memory.vector_id:
                qdrant = await get_qdrant_client()
                await qdrant.delete_memory(memory.vector_id)

            # Delete from SQL layer
            await self.session.delete(memory)
            await self.session.commit()

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            await self.session.rollback()
            raise

    # ------------------------------------------------------
    # DELETE ALL USER MEMORIES
    # ------------------------------------------------------
    async def delete_user_memories(self, user_id: int):
        try:
            # Delete Qdrant vectors
            qdrant = await get_qdrant_client()
            await qdrant.delete_user_memories(str(user_id))

            # Delete SQL memory events
            stmt = select(MemoryEvent).where(MemoryEvent.user_id == user_id)
            result = await self.session.execute(stmt)
            memories = result.scalars().all()

            for m in memories:
                await self.session.delete(m)

            await self.session.commit()

        except Exception as e:
            logger.error(f"Failed to delete user memories: {e}")
            await self.session.rollback()
            raise
