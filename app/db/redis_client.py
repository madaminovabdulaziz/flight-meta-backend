# app/db/redis_client.py
"""
🔴 Redis Connection Management
================================

Singleton Redis client with connection pooling for SkySearch AI.
Handles caching, rate limiting, and circuit breaker state.
"""

import logging
from typing import Optional
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global Redis client instance
_redis_client: Optional[redis.Redis] = None
_redis_pool: Optional[ConnectionPool] = None


async def init_redis() -> redis.Redis:
    """
    Initialize Redis connection pool and client.
    
    Call this during app startup.
    """
    global _redis_client, _redis_pool
    
    try:
        # Create connection pool
        _redis_pool = ConnectionPool.from_url(
            settings.REDIS_URI,
            encoding="utf-8",
            decode_responses=False,  # Keep as bytes for compression
            max_connections=50,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        
        # Create Redis client
        _redis_client = redis.Redis(connection_pool=_redis_pool)
        
        # Test connection
        await _redis_client.ping()
        
        logger.info("✅ Redis connection established successfully")
        return _redis_client
    
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        raise


async def get_redis() -> redis.Redis:
    """
    Get Redis client instance.
    
    Use this as a dependency in FastAPI routes:
    ```python
    async def my_route(redis: Redis = Depends(get_redis)):
        await redis.set("key", "value")
    ```
    """
    global _redis_client
    
    if _redis_client is None:
        _redis_client = await init_redis()
    
    return _redis_client


async def close_redis():
    """
    Close Redis connection pool.
    
    Call this during app shutdown.
    """
    global _redis_client, _redis_pool
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("✅ Redis connection closed")
    
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None
        logger.info("✅ Redis connection pool closed")


async def health_check() -> bool:
    """
    Check if Redis is healthy.
    
    Returns True if Redis is accessible, False otherwise.
    """
    try:
        client = await get_redis()
        await client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


# ==================== HELPER FUNCTIONS ====================

async def cache_get(key: str) -> Optional[bytes]:
    """
    Get value from cache.
    
    Returns bytes or None if not found.
    """
    try:
        client = await get_redis()
        return await client.get(key)
    except Exception as e:
        logger.error(f"Cache get error for key {key}: {e}")
        return None


async def cache_set(key: str, value: bytes, ttl: int = 300):
    """
    Set value in cache with TTL.
    
    Args:
        key: Cache key
        value: Value to cache (bytes)
        ttl: Time to live in seconds (default 5 minutes)
    """
    try:
        client = await get_redis()
        await client.setex(key, ttl, value)
    except Exception as e:
        logger.error(f"Cache set error for key {key}: {e}")


async def cache_delete(key: str):
    """Delete key from cache"""
    try:
        client = await get_redis()
        await client.delete(key)
    except Exception as e:
        logger.error(f"Cache delete error for key {key}: {e}")


async def cache_exists(key: str) -> bool:
    """Check if key exists in cache"""
    try:
        client = await get_redis()
        return await client.exists(key) > 0
    except Exception as e:
        logger.error(f"Cache exists error for key {key}: {e}")
        return False


async def cache_ttl(key: str) -> int:
    """Get remaining TTL for key in seconds"""
    try:
        client = await get_redis()
        return await client.ttl(key)
    except Exception as e:
        logger.error(f"Cache TTL error for key {key}: {e}")
        return -1


async def cache_flush_pattern(pattern: str):
    """
    Delete all keys matching pattern.
    
    Example: cache_flush_pattern("duffel:offer:*")
    """
    try:
        client = await get_redis()
        cursor = 0
        
        while True:
            cursor, keys = await client.scan(cursor, match=pattern, count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break
        
        logger.info(f"Flushed cache keys matching: {pattern}")
    except Exception as e:
        logger.error(f"Cache flush pattern error: {e}")


async def get_cache_stats() -> dict:
    """
    Get Redis cache statistics.
    
    Returns info about memory usage, keys, etc.
    """
    try:
        client = await get_redis()
        info = await client.info("stats")
        memory = await client.info("memory")
        
        return {
            "total_keys": await client.dbsize(),
            "memory_used_mb": round(memory["used_memory"] / 1024 / 1024, 2),
            "memory_peak_mb": round(memory["used_memory_peak"] / 1024 / 1024, 2),
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "hit_rate": round(
                info.get("keyspace_hits", 0) / 
                max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1) * 100,
                2
            )
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        return {}