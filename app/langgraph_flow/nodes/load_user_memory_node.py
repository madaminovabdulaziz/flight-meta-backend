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
    Loads long-term preferences + semantic memory (Qdrant) with fallbacks.
    - Skips anonymous users
    - Skips semantic memory when routing confidence is high
    - Never blocks the pipeline on errors
    """

    # --------------------------------------
    # 1. Safe dataclass attribute access
    # --------------------------------------
    user_id = getattr(state, "user_id", None)
    latest_user_message = getattr(state, "latest_user_message", "")

    # --------------------------------------
    # Anonymous users → skip
    # --------------------------------------
    if not user_id:
        logger.info("[LoadUserMemoryNode] Anonymous user → skipping")
        return state

    logger.info(f"[LoadUserMemoryNode] Loading memory for user {user_id}")

    updates = {}
    preferences = {}

    # --------------------------------------
    # 2. Load structured preferences (DB)
    # --------------------------------------
    try:
        pref_service = PreferenceService()
        preferences = await pref_service.get_user_preferences(user_id)

        updates["long_term_preferences"] = preferences or {}
        updates["preferences"] = preferences or {}   # alias for convenience

    except Exception as e:
        logger.error(f"[LoadUserMemoryNode] Failed to load preferences: {e}")
        updates["long_term_preferences"] = {}
        updates["preferences"] = {}

    # --------------------------------------
    # 3. Skip semantic memory if high routing confidence
    # --------------------------------------
    confidence = getattr(state, "routing_confidence", 0.0)

    if confidence >= 0.90:
        logger.info(
            f"[LoadUserMemoryNode] Skipping semantic memory (confidence={confidence:.2f})"
        )
        updates["semantic_memories"] = []
        return update_state(state, {**updates, "updated_at": datetime.utcnow()})

    # --------------------------------------
    # 4. Semantic memory via Qdrant + Embeddings
    # --------------------------------------
    try:
        if not latest_user_message:
            raise ValueError("No latest user message to embed")

        embedding = await generate_embedding(latest_user_message)
        qdrant = await get_qdrant_client()

        results = await qdrant.search_memories(
            user_id=str(user_id),
            query_embedding=embedding,
            limit=5,
        )

        # Normalize results
        semantic_memories = []
        for r in results:
            try:
                id_ = getattr(r, "id", None) or r.get("id")
                payload = getattr(r, "payload", None) or r.get("content")
                score = getattr(r, "score", None) or r.get("score", 0)

                if isinstance(payload, dict):
                    content = payload.get("content", "")
                    metadata = payload.get("metadata", {})
                else:
                    content, metadata = "", {}

                semantic_memories.append({
                    "id": id_,
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                })
            except Exception:
                continue

        updates["semantic_memories"] = semantic_memories

    except Exception as e:
        logger.error(f"[LoadUserMemoryNode] Semantic memory failed: {e}")
        updates["semantic_memories"] = []

    # --------------------------------------
    # 5. Final updates
    # --------------------------------------
    updates["updated_at"] = datetime.utcnow()

    return update_state(state, updates)
