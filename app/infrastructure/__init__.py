# app/infrastructure/__init__.py
"""
Infrastructure Module
Contains adapters for external services and infrastructure concerns.
"""

from app.infrastructure.cache import CacheAdapter, RedisCache, InMemoryCache
from app.infrastructure.geoip import IPGeolocationService, GeoIPService

__all__ = [
    "CacheAdapter",
    "RedisCache",
    "InMemoryCache",
    "IPGeolocationService",
    "GeoIPService",  # Alias for compatibility
]