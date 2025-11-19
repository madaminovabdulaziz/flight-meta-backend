import logging
from datetime import datetime

from app.langgraph_flow.state import ConversationState, update_state
from services.session_store import get_session_state_from_redis
from app.db.database import AsyncSessionLocal
from app.db.crud.session import load_session_state

logger = logging.getLogger(__name__)

# ======================================================
# KEYS THAT MUST NEVER BE RESTORED FROM REDIS/DB
# ======================================================
ROUTING_KEYS = {
    "use_rule_based_path",
    "routing_confidence",
    "confidence_breakdown",
    "intent",
    "extracted_params",
    "missing_parameter",
    "flow_stage",
}


async def load_session_state_node(state: ConversationState) -> ConversationState:
    """
    Safely load session state from Redis and DB,
    while protecting routing-related fields.
    """

    # -------------------------------------------------
    # SAFE DATACLASS ACCESS
    # -------------------------------------------------
    session_id = state.session_id
    logger.info(f"[LoadSessionStateNode] Loading session: {session_id}")

    # ====================================================
    # 1. Load Redis
    # ====================================================
    try:
        redis_data = await get_session_state_from_redis(session_id)
    except Exception as e:
        logger.error(f"[LoadSessionStateNode] Redis error: {e}")
        redis_data = {}

    redis_data = redis_data or {}
    redis_loaded = redis_data != {}

    # ====================================================
    # 2. Load DB
    # ====================================================
    try:
        async with AsyncSessionLocal() as db:
            db_record = await load_session_state(db, session_id=session_id)
    except Exception as e:
        logger.error(f"[LoadSessionStateNode] DB load error: {e}")
        db_record = None

    db_loaded = db_record is not None

    # Convert DB object → dict
    db_data = {}
    if db_record:
        if hasattr(db_record, "__table__"):  # ORM object
            for col in db_record.__table__.columns:
                db_data[col.name] = getattr(db_record, col.name)
        elif isinstance(db_record, dict):
            db_data = db_record.copy()
        else:
            logger.warning(f"[LoadSessionStateNode] Unexpected DB record type: {type(db_record)}")

    # ====================================================
    # 3. SAFE MERGE
    #    Priority:
    #      Redis > DB > Current Memory
    # ====================================================

    merged = {}

    # Collect all keys across sources
    all_keys = (
        set(redis_data.keys())
        | set(db_data.keys())
        | set(vars(state).keys())
    )

    for key in all_keys:

        # ------------------------------------------------
        # ROUTING KEYS — NEVER RESTORE
        # ------------------------------------------------
        if key in ROUTING_KEYS:
            merged[key] = getattr(state, key, None)
            continue

        # ------------------------------------------------
        # Priority 1: Redis
        # ------------------------------------------------
        if key in redis_data and redis_data[key] not in (None, "", {}):
            merged[key] = redis_data[key]
            continue

        # ------------------------------------------------
        # Priority 2: DB
        # ------------------------------------------------
        if key in db_data and db_data[key] not in (None, ""):
            merged[key] = db_data[key]
            continue

        # ------------------------------------------------
        # Priority 3: In-memory state
        # ------------------------------------------------
        merged[key] = getattr(state, key, None)

    # ====================================================
    # 4. Debug information
    # ====================================================
    debug_info = getattr(state, "debug_info", {}) or {}

    merged["debug_info"] = {
        **debug_info,
        "redis_state_loaded": redis_loaded,
        "db_session_loaded": db_loaded,
        "redis_keys": list(redis_data.keys()),
        "db_keys": list(db_data.keys()),
    }

    merged["updated_at"] = datetime.utcnow()

    logger.info(
        f"[LoadSessionStateNode] Loaded Redis={redis_loaded}, "
        f"DB={db_loaded}, merged_keys={len(merged)}"
    )

    return update_state(state, merged)
