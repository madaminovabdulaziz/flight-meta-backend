import json
from typing import Optional, Dict, Any
from datetime import date, datetime
from redis import asyncio as aioredis


REDIS_URL = "redis://localhost:6379/0"

# Global singleton client
_redis_client: Optional[aioredis.Redis] = None


def json_serializer(obj):
    """
    Custom JSON serializer for objects not serializable by default json code.
    Handles datetime, date, and other common types.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Type {type(obj).__name__} not serializable")


async def get_redis_client() -> aioredis.Redis:
    """Lazy init Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.Redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


# -----------------------------------------
# SESSION GET / SET HELPERS
# -----------------------------------------

async def get_session_state_from_redis(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve stored session state JSON."""
    client = await get_redis_client()
    data = await client.get(f"session:{session_id}")

    if not data:
        return None

    try:
        return json.loads(data)
    except Exception as e:
        # Log error but don't crash
        import logging
        logging.getLogger(__name__).warning(f"Failed to deserialize session {session_id}: {e}")
        return None


async def save_session_state_to_redis(session_id: str, state: Dict[str, Any]):
    """
    Store session state JSON with proper serialization of dates and datetimes.
    """
    client = await get_redis_client()
    
    try:
        # Use custom serializer to handle date/datetime objects
        serialized_state = json.dumps(state, default=json_serializer)
        
        await client.set(
            f"session:{session_id}",
            serialized_state,
            ex=3600  # 1 hour expiry
        )
    except Exception as e:
        # Log error but don't crash the flow
        import logging
        logging.getLogger(__name__).error(
            f"Failed to save session {session_id} to Redis: {e}",
            exc_info=True
        )
        raise