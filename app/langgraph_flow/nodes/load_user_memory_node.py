import logging
from datetime import datetime
from app.langgraph_flow.state import ConversationState, update_state

from services.memory_service import MemoryService
from services.preference_service import PreferenceService

from app.integrations.qdrant_client import get_qdrant_client
from services.embedding_service import generate_embedding

logger = logging.getLogger(__name__)

async def load_user_memory_node(state: ConversationState) -> ConversationState:
    """
    Loads all 3 memory layers:
      - short-term memory (already in state)
      - long-term structured preferences (MySQL)
      - semantic memories (Qdrant)
    """

    user_id = state.get("user_id")
    latest_user_message = state.get("latest_user_message", "")

    if not user_id:
        logger.info("[LoadUserMemoryNode] Anonymous user → skipping memory loading")
        return state

    logger.info(f"[LoadUserMemoryNode] Loading memory for user {user_id}")

    # 1. Load structured preferences (MySQL)
    pref_service = PreferenceService()
    preferences = await pref_service.get_user_preferences(user_id)

    # 2. Load semantic memories using user message embedding
    qdrant = await get_qdrant_client()
    embed = await generate_embedding(latest_user_message)

    semantic_results = await qdrant.search_memories(
        user_id=str(user_id),
        query_embedding=embed,
        limit=5
    )

    # Normalize structure
    semantic_memories = [
        {
            "id": r["id"],
            "content": r["content"],
            "score": r["score"],
            "metadata": r.get("metadata", {}),
        }
        for r in semantic_results
    ]

    merged_updates = {
        "long_term_preferences": preferences or {},
        "semantic_memories": semantic_memories,
        "updated_at": datetime.utcnow(),
    }

    return update_state(state, merged_updates)
