"""
Redis Session Store - Production Version
========================================
Handles session state caching with proper configuration loading.
"""

import json
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import date, datetime
from redis import asyncio as aioredis

# 1. Import settings to get the REAL Redis URL from env vars
from app.core.config import settings

logger = logging.getLogger(__name__)

# Global singleton client
_redis_client: Optional[aioredis.Redis] = None
_redis_lock: asyncio.Lock = asyncio.Lock()


class DateTimeEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for datetime and date objects.
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
    Uses the URL from settings (which loads from Railway env vars).
    """
    global _redis_client
    
    async with _redis_lock:
        if _redis_client is None:
            # 2. Use settings.REDIS_URL instead of hardcoded string
            # Fallback to localhost only if settings fail (for local dev)
            redis_url = getattr(settings, "REDIS_URL", None) or "redis://localhost:6379/0"
            
            logger.info(f"[Redis] Connecting to: {redis_url.split('@')[-1]}") # Log host only for safety
            
            try:
                _redis_client = aioredis.Redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20,
                    socket_connect_timeout=5,  # Fail fast if connection is bad
                    socket_keepalive=True,
                )
                
                # Test connection immediately
                await _redis_client.ping()
                logger.info("[Redis] ✅ Client initialized and connected")
                
            except Exception as e:
                logger.error(f"[Redis] ❌ Connection failed: {e}")
                _redis_client = None
                raise
    
    return _redis_client


# -----------------------------------------
# SESSION GET / SET HELPERS
# -----------------------------------------

async def get_session_state_from_redis(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve stored session state JSON."""
    try:
        client = await get_redis_client()
        data = await client.get(f"session:{session_id}")

        if not data:
            return None

        return json.loads(data)
        
    except Exception as e:
        # Don't crash app if Redis is down, just log and return None (fallback to DB)
        logger.warning(f"[Redis] Read failed for {session_id}: {e}")
        return None


async def save_session_state_to_redis(
    session_id: str,
    state: Any
) -> bool:
    """Store session state JSON."""
    try:
        client = await get_redis_client()
        
        if hasattr(state, 'to_dict'):
            state_dict = state.to_dict()
        elif isinstance(state, dict):
            state_dict = state
        else:
            logger.error(f"[Redis] Invalid state type: {type(state)}")
            return False
        
        serialized_state = json.dumps(state_dict, cls=DateTimeEncoder)
        
        await client.set(
            f"session:{session_id}",
            serialized_state,
            ex=86400  # 24 hours
        )
        return True
        
    except Exception as e:
        logger.error(f"[Redis] Write failed for {session_id}: {e}")
        return False
