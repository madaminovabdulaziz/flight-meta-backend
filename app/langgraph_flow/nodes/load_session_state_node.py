import logging
from datetime import datetime
from app.langgraph_flow.state import ConversationState, update_state

from services.session_store import get_session_state_from_redis
from app.db.database import AsyncSessionLocal
from app.db.crud.session import load_session_state

logger = logging.getLogger(__name__)


async def load_session_state_node(state: ConversationState) -> ConversationState:
    """
    Load and merge:
    - Redis short-term session state
    - MySQL long-term session record
    """

    session_id = state["session_id"]
    user_id = state.get("user_id")

    logger.info(f"[LoadSessionStateNode] Loading session {session_id}")

    # --------------------------------------------------------
    # 1. Load short-term Redis state
    # --------------------------------------------------------
    redis_state = await get_session_state_from_redis(session_id)
    redis_state = redis_state or {}

    # --------------------------------------------------------
    # 2. Load DB session safely (requires DB session!)
    # --------------------------------------------------------
    db_session = None
    if user_id:
        async with AsyncSessionLocal() as db:
            db_session = await load_session_state(
                db=db,
                session_id=session_id,
                user_id=user_id
            )

    # --------------------------------------------------------
    # 3. Merge updates into current state
    # --------------------------------------------------------
    merged_updates = {
        # Redis short-term memory restored
        "short_term_memory": redis_state.get(
            "short_term_memory",
            state.get("short_term_memory", {})
        ),

        # Session token from DB (if exists)
        "session_token": (
            db_session.session_token
            if db_session else state.get("session_token", "")
        ),

        # Debug tracking
        "debug_info": {
            **state.get("debug_info", {}),
            "redis_loaded": redis_state != {},
            "db_session_loaded": db_session is not None,
        },

        # Timestamp
        "updated_at": datetime.utcnow(),
    }

    logger.info(
        f"[LoadSessionStateNode] Loaded Redis={redis_state != {}}, "
        f"DB={'YES' if db_session else 'NO'}"
    )

    return update_state(state, merged_updates)
