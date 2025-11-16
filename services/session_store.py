import json
from typing import Optional, Dict, Any
from redis import asyncio as aioredis


REDIS_URL = "redis://localhost:6379/0"

# Global singleton client
_redis_client: Optional[aioredis.Redis] = None


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
    except Exception:
        return None


async def save_session_state_to_redis(session_id: str, state: Dict[str, Any]):
    """Store session state JSON."""
    client = await get_redis_client()
    await client.set(
        f"session:{session_id}",
        json.dumps(state),
        ex=3600  # 1 hour expiry
    )
