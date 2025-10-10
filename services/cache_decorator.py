#services/cache_decorator.py


from __future__ import annotations
import functools
import json
import logging
from typing import Any, Callable


# Reuse your existing Redis helpers to keep compatibility
from services.amadeus_service import _zget, _zset # type: ignore


_cache_logger = logging.getLogger("redis_cache")




def _json_key(obj: Any) -> str:
# Avoid serializing non-primitive objects by converting to string
    def default(o):
        try:
            return dict(o)
        except Exception:
            return str(o)


    return json.dumps(obj, sort_keys=True, default=default)




def redis_cache(ttl: int) -> Callable:
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
        # Build a namespaced key (skip self)
            cls_name = args[0].__class__.__name__ if args else ""
            key_ns = f"{func.__module__}:{cls_name}.{func.__name__}"
            key_payload = _json_key(kwargs)
            cache_key = f"{key_ns}:{key_payload}"


            cached = await _zget(cache_key)
            if cached is not None:
                _cache_logger.debug("CACHE HIT %s", cache_key)
                return cached


            _cache_logger.debug("CACHE MISS %s", cache_key)
            result = await func(*args, **kwargs)


        # Only cache JSON-serializable results; let _zset handle lists/dicts/models
            try:
                await _zset(cache_key, result, ttl=ttl)
            except Exception as e:
                _cache_logger.warning("Cache set failed for %s: %s", cache_key, e)


            return result


        return wrapper


    return decorator