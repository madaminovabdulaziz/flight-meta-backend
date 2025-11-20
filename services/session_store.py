"""
Redis Session Store - Fixed Serialization
==========================================
Handles session state caching with proper datetime serialization.
"""

import json
import asyncio
from typing import Optional, Dict, Any
from datetime import date, datetime
from redis import asyncio as aioredis

import logging

logger = logging.getLogger(__name__)

REDIS_URL = "redis://localhost:6379/0"

# Global singleton client
_redis_client: Optional[aioredis.Redis] = None
_redis_lock: asyncio.Lock = asyncio.Lock()  # Thread-safety fix


class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for datetime and date objects.
    Handles nested datetime objects correctly.
    """
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


async def get_redis_client() -> aioredis.Redis:
    """
    Lazy init Redis client with thread-safety.
    
    FIXED: Now uses asyncio.Lock to prevent race conditions.
    """
    global _redis_client
    
    async with _redis_lock:
        if _redis_client is None:
            _redis_client = aioredis.Redis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,  # Connection pooling
            )
            logger.info("[Redis] Client initialized")
    
    return _redis_client


# -----------------------------------------
# SESSION GET / SET HELPERS
# -----------------------------------------

async def get_session_state_from_redis(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve stored session state JSON.
    
    Returns:
        Dict with session state, or None if not found/invalid
    """
    try:
        client = await get_redis_client()
        data = await client.get(f"session:{session_id}")

        if not data:
            return None

        return json.loads(data)
        
    except json.JSONDecodeError as e:
        logger.warning(f"[Redis] Failed to deserialize session {session_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"[Redis] Error retrieving session {session_id}: {e}", exc_info=True)
        return None


async def save_session_state_to_redis(
    session_id: str,
    state: Any  # Can be dict or ConversationState
) -> bool:
    """
    Store session state JSON with proper serialization of dates and datetimes.
    
    Returns:
        True if save succeeded, False otherwise
    """
    try:
        client = await get_redis_client()
        
        # Convert state to dict if it's a dataclass
        if hasattr(state, 'to_dict'):
            state_dict = state.to_dict()
        elif isinstance(state, dict):
            state_dict = state
        else:
            logger.error(f"[Redis] Invalid state type: {type(state)}")
            return False
        
        # Serialize with custom encoder
        serialized_state = json.dumps(state_dict, cls=DateTimeEncoder)
        
        # Save with 24-hour expiry (increased from 1 hour)
        await client.set(
            f"session:{session_id}",
            serialized_state,
            ex=86400  # 24 hours
        )
        
        logger.debug(f"[Redis] Saved session {session_id}")
        return True
        
    except Exception as e:
        logger.error(
            f"[Redis] Failed to save session {session_id}: {e}",
            exc_info=True
        )
        return False