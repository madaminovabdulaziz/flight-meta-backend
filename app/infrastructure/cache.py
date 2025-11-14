# app/infrastructure/cache.py
"""
Cache Adapter for Conversation System
Provides clean interface for caching operations.
Integrates with your existing Redis client.
"""

import json
import logging
from typing import Optional, Any, List
from datetime import datetime

from app.db.redis_client import get_redis

logger = logging.getLogger(__name__)


class CacheAdapter:
    """
    Abstract interface for cache operations.
    Allows easy swapping of cache backends.
    """
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        raise NotImplementedError
    
    async def set(self, key: str, value: str, ttl: int) -> bool:
        """Set value with TTL"""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        raise NotImplementedError


class RedisCache(CacheAdapter):
    """
    Redis-based cache implementation.
    Uses your existing redis_client for connection management.
    """
    
    def __init__(self):
        """
        Initialize with Redis client from your existing infrastructure.
        No need to pass redis_client - uses get_redis() singleton.
        """
        self.redis = get_redis()
        logger.debug("✓ RedisCache initialized with shared Redis client")
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value as string, or None if not found
        """
        try:
            value = await self.redis.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
            else:
                logger.debug(f"Cache MISS: {key}")
            return value
        except Exception as e:
            logger.error(f"[RedisCache] Error getting key '{key}': {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int) -> bool:
        """
        Set value with TTL.
        
        Args:
            key: Cache key
            value: Value to cache (string)
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        try:
            await self.redis.setex(key, ttl, value)
            logger.debug(f"Cache SET: {key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.error(f"[RedisCache] Error setting key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful
        """
        try:
            result = await self.redis.delete(key)
            if result > 0:
                logger.debug(f"Cache DELETE: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"[RedisCache] Error deleting key '{key}': {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists
        """
        try:
            result = await self.redis.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"[RedisCache] Error checking key '{key}': {e}")
            return False
    
    # ============================================================
    # LIST OPERATIONS (For conversation history)
    # ============================================================
    
    async def lpush(self, key: str, value: str) -> bool:
        """
        List push operation (prepend to list).
        
        Args:
            key: List key
            value: Value to prepend
            
        Returns:
            True if successful
        """
        try:
            await self.redis.lpush(key, value)
            logger.debug(f"List LPUSH: {key}")
            return True
        except Exception as e:
            logger.error(f"[RedisCache] Error lpush to '{key}': {e}")
            return False
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """
        List trim operation (keep only elements in range).
        
        Args:
            key: List key
            start: Start index (0-based)
            end: End index (inclusive)
            
        Returns:
            True if successful
        """
        try:
            await self.redis.ltrim(key, start, end)
            logger.debug(f"List LTRIM: {key} [{start}:{end}]")
            return True
        except Exception as e:
            logger.error(f"[RedisCache] Error ltrim '{key}': {e}")
            return False
    
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """
        List range operation (get elements in range).
        
        Args:
            key: List key
            start: Start index (0-based)
            end: End index (inclusive, -1 for end of list)
            
        Returns:
            List of values
        """
        try:
            values = await self.redis.lrange(key, start, end)
            logger.debug(f"List LRANGE: {key} [{start}:{end}] -> {len(values)} items")
            return values
        except Exception as e:
            logger.error(f"[RedisCache] Error lrange '{key}': {e}")
            return []
    
    # ============================================================
    # PATTERN OPERATIONS
    # ============================================================
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Redis pattern (e.g., "conversation:*")
            
        Returns:
            Number of keys deleted
        """
        try:
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted = await self.redis.delete(*keys)
                    deleted_count += deleted
                if cursor == 0:
                    break
            
            logger.info(f"Deleted {deleted_count} keys matching pattern: {pattern}")
            return deleted_count
        except Exception as e:
            logger.error(f"[RedisCache] Error deleting pattern '{pattern}': {e}")
            return 0
    
    # ============================================================
    # TTL OPERATIONS
    # ============================================================
    
    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        try:
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"[RedisCache] Error getting TTL for '{key}': {e}")
            return -2
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set/update TTL for existing key.
        
        Args:
            key: Cache key
            ttl: New TTL in seconds
            
        Returns:
            True if successful
        """
        try:
            result = await self.redis.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"[RedisCache] Error setting expire for '{key}': {e}")
            return False


class InMemoryCache(CacheAdapter):
    """
    In-memory cache for testing/development.
    NOT for production use.
    """
    
    def __init__(self, ttl_seconds: int = 1800, max_size: int = 1000):
        import time
        from collections import OrderedDict
        
        self._store: OrderedDict[str, dict] = OrderedDict()
        self.ttl = ttl_seconds
        self.max_size = max_size
        logger.warning("⚠️  Using InMemoryCache - NOT RECOMMENDED FOR PRODUCTION")
    
    def _evict_expired(self):
        """Remove expired items"""
        import time
        now = time.time()
        expired = [k for k, v in self._store.items() if v['expires_at'] < now]
        for k in expired:
            self._store.pop(k, None)
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        self._evict_expired()
        item = self._store.get(key)
        if not item:
            return None
        self._store.move_to_end(key)
        return item['data']
    
    async def set(self, key: str, value: str, ttl: int) -> bool:
        """Set value (ignores TTL in this simple implementation)"""
        import time
        self._evict_expired()
        
        if len(self._store) >= self.max_size and key not in self._store:
            self._store.popitem(last=False)
        
        self._store[key] = {
            'data': value,
            'expires_at': time.time() + ttl
        }
        self._store.move_to_end(key)
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        if key in self._store:
            del self._store[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        self._evict_expired()
        return key in self._store
    
    # List operations (basic implementation)
    async def lpush(self, key: str, value: str) -> bool:
        """List push (not implemented in memory cache)"""
        logger.warning("lpush not implemented in InMemoryCache")
        return False
    
    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """List trim (not implemented in memory cache)"""
        logger.warning("ltrim not implemented in InMemoryCache")
        return False
    
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        """List range (not implemented in memory cache)"""
        logger.warning("lrange not implemented in InMemoryCache")
        return []


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_cache_adapter(use_redis: bool = True) -> CacheAdapter:
    """
    Factory function to get appropriate cache adapter.
    
    Args:
        use_redis: If True, use Redis; otherwise use in-memory cache
        
    Returns:
        CacheAdapter instance
    """
    if use_redis:
        return RedisCache()
    else:
        return InMemoryCache()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'CacheAdapter',
    'RedisCache',
    'InMemoryCache',
    'get_cache_adapter'
]