import json
from typing import Optional, Dict, Any, Union
from datetime import date, datetime
from redis import asyncio as aioredis
import redis.asyncio as redis
from dataclasses import asdict, is_dataclass

from app.core.config import settings


# Global singleton client

_redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)


def json_serializer(obj):
    """
    Custom JSON serializer for objects not serializable by default json code.
    Handles datetime, date, dataclasses, and other common types.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Type {type(obj).__name__} not serializable")


async def get_redis_client() -> aioredis.Redis:
    """Lazy init Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.Redis.from_url(
            settings.REDIS_URL,
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


async def save_session_state_to_redis(
    session_id: str, 
    state: Union[Dict[str, Any], Any]  # Accept both dict and ConversationState
):
    """
    Store session state JSON with proper serialization of dates, datetimes, and dataclasses.
    
    Args:
        session_id: Unique session identifier
        state: Either a dict or a ConversationState dataclass
    """
    import logging
    logger = logging.getLogger(__name__)
    
    client = await get_redis_client()
    
    try:
        # Convert ConversationState to dict if needed
        if is_dataclass(state):
            state_dict = asdict(state)
            logger.debug(f"[Redis] Converted dataclass to dict with {len(state_dict)} keys")
        elif isinstance(state, dict):
            state_dict = state
        else:
            # Try to get __dict__ attribute
            state_dict = getattr(state, '__dict__', state)
        
        # Use custom serializer to handle date/datetime objects
        serialized_state = json.dumps(state_dict, default=json_serializer)
        
        await client.set(
            f"session:{session_id}",
            serialized_state,
            ex=86400  # 24 hour expiry (increased from 1 hour for better persistence)
        )
        
        logger.debug(f"[Redis] Saved session {session_id} ({len(serialized_state)} bytes)")
        
    except Exception as e:
        # Log error but don't crash the flow
        logger.error(
            f"[Redis] Failed to save session {session_id}: {e}",
            exc_info=True
        )
        raise


async def delete_session_from_redis(session_id: str) -> bool:
    """Delete session from Redis."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        client = await get_redis_client()
        result = await client.delete(f"session:{session_id}")
        
        if result > 0:
            logger.info(f"[Redis] Deleted session {session_id}")
            return True
        else:
            logger.debug(f"[Redis] Session {session_id} not found")
            return False
            
    except Exception as e:
        logger.error(f"[Redis] Error deleting session {session_id}: {e}")
        return False


async def check_redis_connection() -> bool:
    """Health check for Redis connection."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        client = await get_redis_client()
        await client.ping()
        logger.debug("[Redis] Connection healthy")
        return True
    except Exception as e:
        logger.error(f"[Redis] Connection failed: {e}")
        return False