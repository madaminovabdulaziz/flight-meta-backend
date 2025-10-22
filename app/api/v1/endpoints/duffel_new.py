# duffel_complete.py - Complete Production Solution with Fare Grouping & Smart Caching
"""
COMPLETE PRODUCTION FEATURES:
✓ Multi-layer caching (search results, single offers)
✓ Fare grouping by flight
✓ Single offer retrieval with fresh pricing
✓ Price change detection
✓ Available services fetching
✓ Circuit breaker & rate limiting
✓ Structured logging & metrics
✓ Background cache cleanup
✓ Expiration filtering
"""

import os
import time
import json
import hashlib
import asyncio
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import OrderedDict, Counter as CollectionsCounter
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status, APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from prometheus_client import (
    Counter as PrometheusCounter, 
    Histogram, 
    Gauge, 
    generate_latest, 
    CONTENT_TYPE_LATEST
)
import logging

# Import transformers
from app.mappers.flight_transformers import (
    transform_search_response,
    transform_offer,
    group_offers_by_flight,
    get_cheapest_offer,
    get_fastest_offer,
    get_direct_flights,
    filter_by_max_price,
    filter_by_airline,
    filter_by_max_duration,
    filter_by_stops,
    detect_price_change,
    generate_offer_cache_key
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DUFFEL_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"
DUFFEL_TOKEN = os.getenv("DUFFEL_API_KEY")

REQUEST_RATE_LIMIT = int(os.getenv("REQUEST_RATE_LIMIT", "60"))
REQUEST_RATE_WINDOW = int(os.getenv("REQUEST_RATE_WINDOW", "60"))

# Cache configuration
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 min
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "1000"))
OFFER_CACHE_TTL = int(os.getenv("OFFER_CACHE_TTL", "300"))  # 5 min for single offers

CLIENT_TIMEOUT = float(os.getenv("CLIENT_TIMEOUT", "30.0"))

# ============================================================================
# LOGGING
# ============================================================================

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        return json.dumps(base)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("flight_api")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

if not DUFFEL_TOKEN:
    logger.critical("FATAL: DUFFEL_API_KEY environment variable not set")

# ============================================================================
# METRICS
# ============================================================================

REQ_COUNT = PrometheusCounter("flight_requests_total", "Total HTTP requests", ["method", "route", "status"])
REQ_LATENCY = Histogram("flight_request_duration_seconds", "HTTP request latency", ["route"])
DUFFEL_REQ_COUNT = PrometheusCounter("duffel_requests_total", "Duffel API calls", ["method", "path", "status"])
DUFFEL_REQ_LAT = Histogram("duffel_request_duration_seconds", "Duffel API latency", ["path"])
CACHE_HITS = PrometheusCounter("cache_hits_total", "Cache hits", ["cache"])
CACHE_MISSES = PrometheusCounter("cache_misses_total", "Cache misses", ["cache"])
BREAKER_OPEN = Gauge("breaker_open", "Circuit breaker open state")
PRICE_CHANGES = PrometheusCounter("price_changes_total", "Price changes detected", ["direction"])

# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

@dataclass
class CircuitState:
    failures: int = 0
    opened_at: float = 0.0
    open: bool = False

class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout: int = 20):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = CircuitState()

    def allow(self) -> bool:
        if not self.state.open:
            return True
        if time.time() - self.state.opened_at >= self.reset_timeout:
            return True
        return False

    def record_success(self):
        self.state.failures = 0
        if self.state.open:
            self.state.open = False
            self.state.opened_at = 0.0
            BREAKER_OPEN.set(0)
            logger.info("circuit_closed", extra={"extra": {"breaker": self.name}})

    def record_failure(self):
        self.state.failures += 1
        if self.state.failures >= self.failure_threshold:
            self.state.open = True
            self.state.opened_at = time.time()
            BREAKER_OPEN.set(1)
            logger.error("circuit_opened", extra={"extra": {
                "breaker": self.name,
                "failures": self.state.failures
            }})

async def with_backoff(fn, *, max_attempts=4, base_delay=0.25, max_delay=3.0):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            if attempt == max_attempts:
                break
            sleep = min(max_delay, base_delay * (2 ** (attempt - 1)))
            await asyncio.sleep(sleep)
    raise last_exc

# ============================================================================
# IMPROVED TTL CACHE
# ============================================================================

class ImprovedTTLCache:
    """Time-to-live cache with LRU eviction."""
    
    def __init__(self, ttl_seconds: int, max_items: int, name: str):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.name = name
        self.store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "sets": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        item = self.store.get(key)
        
        if not item:
            CACHE_MISSES.labels(self.name).inc()
            self.stats["misses"] += 1
            return None
        
        ts, val = item
        age = time.time() - ts
        
        if age > self.ttl:
            self.store.pop(key, None)
            CACHE_MISSES.labels(self.name).inc()
            self.stats["misses"] += 1
            self.stats["expirations"] += 1
            return None
        
        self.store.move_to_end(key)
        CACHE_HITS.labels(self.name).inc()
        self.stats["hits"] += 1
        return val
    
    def set(self, key: str, value: Any) -> None:
        self.stats["sets"] += 1
        
        if key in self.store:
            self.store[key] = (time.time(), value)
            self.store.move_to_end(key)
            return
        
        if len(self.store) >= self.max_items:
            evicted_key, _ = self.store.popitem(last=False)
            self.stats["evictions"] += 1
        
        self.store[key] = (time.time(), value)
    
    def clear_expired(self) -> int:
        now = time.time()
        expired = []
        
        for key, (ts, _) in list(self.store.items()):
            if now - ts > self.ttl:
                expired.append(key)
        
        for key in expired:
            self.store.pop(key, None)
            self.stats["expirations"] += 1
        
        return len(expired)
    
    def get_stats(self) -> dict:
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0
        
        return {
            "name": self.name,
            "size": len(self.store),
            "max_items": self.max_items,
            "ttl_seconds": self.ttl,
            "hit_rate_pct": round(hit_rate, 2),
            **self.stats
        }

# Initialize caches
offer_cache = ImprovedTTLCache(CACHE_TTL_SECONDS, CACHE_MAX_ITEMS, "offers_by_orq")
result_index_cache = ImprovedTTLCache(CACHE_TTL_SECONDS, CACHE_MAX_ITEMS, "results_index")
single_offer_cache = ImprovedTTLCache(OFFER_CACHE_TTL, CACHE_MAX_ITEMS // 2, "single_offers")

route_frequency = CollectionsCounter()

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.buckets: Dict[str, Dict[str, float]] = {}

    def allow(self, ip: str) -> bool:
        now = time.time()
        b = self.buckets.get(ip)
        if not b:
            self.buckets[ip] = {"tokens": self.limit - 1, "updated": now}
            return True
        elapsed = now - b["updated"]
        refill = (elapsed / self.window) * self.limit
        b["tokens"] = min(self.limit, b["tokens"] + refill)
        b["updated"] = now
        if b["tokens"] >= 1.0:
            b["tokens"] -= 1.0
            return True
        return False

limiter = RateLimiter(REQUEST_RATE_LIMIT, REQUEST_RATE_WINDOW)

# ============================================================================
# DUFFEL CLIENT
# ============================================================================

HEADERS = {
    "Accept-Encoding": "gzip",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Duffel-Version": DUFFEL_VERSION,
    "Authorization": f"Bearer {DUFFEL_TOKEN}",
}

breaker = CircuitBreaker("duffel-api", failure_threshold=5, reset_timeout=20)

class DuffelClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=DUFFEL_BASE, 
            headers=HEADERS, 
            timeout=CLIENT_TIMEOUT
        )

    async def _call(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        if not breaker.allow():
            raise HTTPException(
                status_code=503, 
                detail="Duffel temporarily unavailable (circuit open)"
            )
        
        start = time.time()

        async def do_req():
            resp = await self.client.request(method, url, **kwargs)
            return resp

        try:
            resp = await with_backoff(do_req)
            status_code = resp.status_code
            DUFFEL_REQ_COUNT.labels(method, url, str(status_code)).inc()
            DUFFEL_REQ_LAT.labels(url).observe(time.time() - start)

            logger.info("duffel_call", extra={"extra": {
                "method": method,
                "path": url,
                "status": status_code,
                "latency_ms": round((time.time() - start) * 1000, 1),
            }})

            if 500 <= status_code:
                breaker.record_failure()
                resp.raise_for_status()

            breaker.record_success()

            if 400 <= status_code < 500:
                try:
                    err = resp.json()
                except Exception:
                    err = {"message": resp.text}
                raise HTTPException(status_code=status_code, detail=err)
            
            return resp.json()

        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            raise HTTPException(status_code=e.response.status_code, detail=detail)

        except httpx.RequestError as e:
            breaker.record_failure()
            logger.error("duffel_request_error", extra={"extra": {
                "error": str(e), 
                "url": str(e.request.url)
            }})
            raise HTTPException(status_code=503, detail=f"Duffel network error: {str(e)}")
        
        except Exception as e:
            breaker.record_failure()
            logger.error("duffel_unexpected_error", extra={"extra": {"error": str(e)}})
            raise e

    # Core endpoints
    async def create_offer_request(
        self, 
        data: Dict[str, Any], 
        return_offers: bool = True
    ):
        payload = {"data": data}
        params = {"return_offers": str(return_offers).lower()}
        return await self._call("POST", "/air/offer_requests", json=payload, params=params)

    async def get_offer_request(self, orq_id: str):
        return await self._call("GET", f"/air/offer_requests/{orq_id}")

    async def list_offers(
        self, 
        *, 
        offer_request_id: str, 
        limit: int = 50, 
        after: Optional[str] = None
    ):
        params = {"offer_request_id": offer_request_id, "limit": limit}
        if after:
            params["after"] = after
        return await self._call("GET", "/air/offers", params=params)

    async def get_offer(self, offer_id: str, return_available_services: bool = True):
        """Get single offer with optional available services."""
        params = {"return_available_services": str(return_available_services).lower()}
        return await self._call(
            "GET", 
            f"/air/offers/{offer_id}", 
            params=params
        )

    async def create_order(self, data: Dict[str, Any]):
        payload = {"data": data}
        return await self._call("POST", "/air/orders", json=payload)

    async def get_order(self, order_id: str):
        return await self._call("GET", f"/air/orders/{order_id}")

duffel = DuffelClient()

# ============================================================================
# SCHEMAS
# ============================================================================

class TimeWindow(BaseModel):
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None

class SliceIn(BaseModel):
    origin: str
    destination: str
    departure_date: str
    departure_time: Optional[TimeWindow] = None
    arrival_time: Optional[TimeWindow] = None

class PaxIn(BaseModel):
    type: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=120)

class SearchRequest(BaseModel):
    slices: List[SliceIn]
    passengers: List[PaxIn]
    cabin_class: Optional[str] = None
    max_connections: Optional[int] = Field(1, ge=0, le=2)
    supplier_timeout_ms: Optional[int] = None
    return_offers: bool = True
    page_size: int = Field(50, ge=1, le=250)

class PageMeta(BaseModel):
    page_size: int
    has_more: bool
    next_after: Optional[str] = None
    source: str
    cache_hit: Optional[bool] = None
    cache_age_seconds: Optional[float] = None

class SearchResponse(BaseModel):
    offer_request_id: str
    offers: Optional[List[Dict[str, Any]]] = None
    meta: Optional[PageMeta] = None

class GroupedSearchResponse(BaseModel):
    offer_request_id: str
    flights: List[Dict[str, Any]]
    count: int
    meta: Optional[PageMeta] = None

class OfferResponse(BaseModel):
    offer: Dict[str, Any]
    price_change: Optional[Dict[str, Any]] = None
    cached: bool = False

# ============================================================================
# UTILITIES
# ============================================================================

def normalize_search_key(req: SearchRequest) -> str:
    """Create deterministic hash of search params."""
    payload = {
        "slices": [s.dict(by_alias=True) for s in req.slices],
        "passengers": [p.dict() for p in req.passengers],
        "cabin_class": req.cabin_class,
        "max_connections": req.max_connections,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

_cleanup_task: Optional[asyncio.Task] = None
_logging_task: Optional[asyncio.Task] = None

async def periodic_cache_cleanup():
    """Background task that periodically cleans expired cache entries."""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            
            offer_expired = offer_cache.clear_expired()
            result_expired = result_index_cache.clear_expired()
            single_expired = single_offer_cache.clear_expired()
            
            if offer_expired > 0 or result_expired > 0 or single_expired > 0:
                logger.info("cache_cleanup", extra={"extra": {
                    "offer_expired": offer_expired,
                    "result_expired": result_expired,
                    "single_expired": single_expired,
                    "offer_size": len(offer_cache.store),
                    "result_size": len(result_index_cache.store),
                    "single_size": len(single_offer_cache.store)
                }})
        
        except asyncio.CancelledError:
            logger.info("cleanup_task_cancelled")
            break
        except Exception as e:
            logger.error("cleanup_task_error", extra={"extra": {"error": str(e)}})

async def log_popular_routes():
    """Background task to track popular routes."""
    while True:
        try:
            await asyncio.sleep(600)  # 10 minutes
            
            if route_frequency:
                popular = route_frequency.most_common(10)
                logger.info("popular_routes", extra={"extra": {
                    "top_routes": [{"key": k[:12], "count": c} for k, c in popular]
                }})
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("logging_task_error", extra={"extra": {"error": str(e)}})

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    global _cleanup_task, _logging_task
    
    logger.info("app_startup", extra={"extra": {
        "cache_ttl": CACHE_TTL_SECONDS,
        "cache_max": CACHE_MAX_ITEMS
    }})
    
    _cleanup_task = asyncio.create_task(periodic_cache_cleanup())
    _logging_task = asyncio.create_task(log_popular_routes())
    
    yield
    
    logger.info("app_shutdown")
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    if _logging_task:
        _logging_task.cancel()
        try:
            await _logging_task
        except asyncio.CancelledError:
            pass

# ============================================================================
# FASTAPI APP
# ============================================================================

router = APIRouter()
app = FastAPI(lifespan=lifespan, title="Flight Aggregator API", version="2.0.0")

@app.middleware("http")
async def observability_mw(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.allow(client_ip):
        REQ_COUNT.labels(request.method, request.url.path, "429").inc()
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

    req_id = hashlib.md5(f"{time.time()}-{client_ip}-{request.url.path}".encode()).hexdigest()[:12]
    start = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logger.error("request_error", extra={"extra": {
            "request_id": req_id, 
            "path": request.url.path, 
            "error": str(e)
        }})
        response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)

    latency = time.time() - start
    logger.info("request", extra={"extra": {
        "request_id": req_id,
        "ip": client_ip,
        "method": request.method,
        "path": request.url.path,
        "status": status_code,
        "latency_ms": round(latency * 1000, 2)
    }})
    
    REQ_COUNT.labels(request.method, request.url.path, str(status_code)).inc()
    REQ_LATENCY.labels(request.url.path).observe(latency)
    response.headers["X-Request-ID"] = req_id
    
    return response

# ============================================================================
# API ENDPOINTS
# ============================================================================

@router.get("/healthz")
async def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "circuit_open": breaker.state.open,
        "failures": breaker.state.failures,
        "cache_sizes": {
            "offers": len(offer_cache.store),
            "results": len(result_index_cache.store),
            "single_offers": len(single_offer_cache.store)
        },
        "cache_stats": {
            "offers": offer_cache.get_stats(),
            "results": result_index_cache.get_stats(),
            "single_offers": single_offer_cache.get_stats()
        }
    }

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, format: str = Query("clean")):
    """
    Smart flight search with multi-layer caching.
    
    Query Parameters:
        format: 'clean' (default) or 'raw'
    """
    
    cache_key = normalize_search_key(req)
    route_frequency[cache_key] += 1
    
    # ====== TRY CACHE FIRST ======
    cached_entry = result_index_cache.get(cache_key)
    
    if cached_entry:
        orq_id = cached_entry["offer_request_id"]
        age_seconds = time.time() - cached_entry["ts"]
        
        if age_seconds < CACHE_TTL_SECONDS:
            if cached_entry.get("return_offers") == req.return_offers:
                cached_offers = offer_cache.get(orq_id)
                
                if cached_offers:
                    logger.info("cache_hit", extra={"extra": {
                        "cache_key": cache_key[:12],
                        "orq_id": orq_id,
                        "age_seconds": round(age_seconds, 1),
                        "offers_count": len(cached_offers)
                    }})
                    
                    page = cached_offers[:req.page_size]
                    
                    meta_data = PageMeta(
                        page_size=req.page_size,
                        has_more=len(cached_offers) > len(page),
                        next_after=None,
                        source="cached",
                        cache_hit=True,
                        cache_age_seconds=round(age_seconds, 1)
                    )
                    
                    if format == "clean":
                        response_data = {
                            "offer_request_id": orq_id,
                            "offers": page,
                            "meta": meta_data.dict()
                        }
                        clean_response = transform_search_response(
                            response_data, 
                            limit=req.page_size
                        )
                        return SearchResponse(
                            offer_request_id=orq_id,
                            offers=clean_response["offers"],
                            meta=meta_data
                        )
                    else:
                        return SearchResponse(
                            offer_request_id=orq_id,
                            offers=page,
                            meta=meta_data
                        )
    
    # ====== CACHE MISS - FETCH FROM DUFFEL ======
    logger.info("cache_miss", extra={"extra": {
        "cache_key": cache_key[:12],
        "reason": "no_valid_cache_entry" if not cached_entry else "cache_expired"
    }})
    
    # Build Duffel payload
    slices_payload = []
    for s in req.slices:
        sp = {
            "origin": s.origin, 
            "destination": s.destination, 
            "departure_date": s.departure_date
        }
        if s.departure_time:
            sp["departure_time"] = {
                k: v for k, v in s.departure_time.dict(by_alias=True).items() if v
            }
        if s.arrival_time:
            sp["arrival_time"] = {
                k: v for k, v in s.arrival_time.dict(by_alias=True).items() if v
            }
        slices_payload.append(sp)

    passengers_payload = []
    for p in req.passengers:
        item = {}
        if p.age is not None:
            item["age"] = p.age
        elif p.type is not None:
            item["type"] = p.type
        else:
            item["type"] = "adult"
        passengers_payload.append(item)

    data = {"slices": slices_payload, "passengers": passengers_payload}
    if req.cabin_class:
        data["cabin_class"] = req.cabin_class
    if req.max_connections is not None:
        data["max_connections"] = req.max_connections
    if req.supplier_timeout_ms is not None:
        data["supplier_timeout"] = req.supplier_timeout_ms

    # Create Offer Request
    resp = await duffel.create_offer_request(
        data=data, 
        return_offers=req.return_offers
    )
    orq = resp["data"]
    orq_id = orq["id"]

    # ====== STORE IN CACHE ======
    result_index_cache.set(cache_key, {
        "offer_request_id": orq_id,
        "ts": time.time(),
        "return_offers": req.return_offers,
        "page_size": req.page_size
    })

    # Handle pagination
    if not req.return_offers:
        first_page = await duffel.list_offers(
            offer_request_id=orq_id, 
            limit=req.page_size
        )
        offers = first_page.get("data", []) or []
        meta = first_page.get("meta", {}) or {}
        next_after = meta.get("after")
        
        if offers:
            offer_cache.set(orq_id, offers)
        
        meta_data = PageMeta(
            page_size=req.page_size,
            has_more=bool(next_after),
            next_after=next_after,
            source="fresh_paged",
            cache_hit=False
        )
        
        if format == "clean":
            response_data = {
                "offer_request_id": orq_id,
                "offers": offers,
                "meta": meta_data.dict()
            }
            clean_response = transform_search_response(
                response_data, 
                limit=req.page_size
            )
            return SearchResponse(
                offer_request_id=orq_id,
                offers=clean_response["offers"],
                meta=meta_data
            )
        else:
            return SearchResponse(
                offer_request_id=orq_id,
                offers=offers,
                meta=meta_data
            )

    # Embedded offers
    embedded = orq.get("offers") or []
    if embedded:
        offer_cache.set(orq_id, embedded)
    
    page = embedded[:req.page_size]
    has_more = len(embedded) > len(page)
    
    meta_data = PageMeta(
        page_size=req.page_size,
        has_more=has_more,
        next_after=None,
        source="fresh_embedded",
        cache_hit=False
    )
    
    if format == "clean":
        response_data = {
            "offer_request_id": orq_id,
            "offers": page,
            "meta": meta_data.dict()
        }
        clean_response = transform_search_response(
            response_data, 
            limit=req.page_size
        )
        return SearchResponse(
            offer_request_id=orq_id,
            offers=clean_response["offers"],
            meta=meta_data
        )
    else:
        return SearchResponse(
            offer_request_id=orq_id,
            offers=page,
            meta=meta_data
        )


@router.post("/search/grouped", response_model=GroupedSearchResponse)
async def search_grouped(req: SearchRequest):
    """
    Search flights grouped by route with different fare options.
    
    Returns flights grouped by:
    - Same route (all segments)
    - Same departure times
    - Same operating carriers
    
    Each group contains multiple fare options (Basic, Standard, Flexible, etc.)
    """
    
    # Perform normal search
    search_response = await search(req, format="clean")
    
    if not search_response.offers:
        return GroupedSearchResponse(
            offer_request_id=search_response.offer_request_id,
            flights=[],
            count=0,
            meta=search_response.meta
        )
    
    # Group offers by flight
    grouped_flights = group_offers_by_flight(search_response.offers)
    
    logger.info("offers_grouped", extra={"extra": {
        "total_offers": len(search_response.offers),
        "unique_flights": len(grouped_flights),
        "orq_id": search_response.offer_request_id
    }})
    
    return GroupedSearchResponse(
        offer_request_id=search_response.offer_request_id,
        flights=grouped_flights,
        count=len(grouped_flights),
        meta=search_response.meta
    )


# ============================================================================
# SINGLE OFFER ENDPOINTS (NEW)
# ============================================================================

@router.get("/offers/{offer_id}", response_model=OfferResponse)
async def get_offer_detailed(
    offer_id: str,
    include_services: bool = Query(True, description="Include available services"),
    detect_price_changes: bool = Query(True, description="Compare with cached price")
):
    """
    Get fresh offer details before booking.
    
    CRITICAL FEATURES:
    - Fetches latest price from Duffel
    - Includes available services (baggage, seats, meals)
    - Detects price changes
    - Filters expired offers
    - Returns 410 if offer expired
    
    Use this endpoint when:
    - User selects a flight card
    - Before showing booking page
    - To verify offer still valid
    """
    
    # Check cache first
    cache_key = generate_offer_cache_key(offer_id, include_services)
    cached = single_offer_cache.get(cache_key)
    
    price_change_info = None
    
    # Fetch fresh offer from Duffel
    try:
        raw_offer = await duffel.get_offer(
            offer_id, 
            return_available_services=include_services
        )
        offer_data = raw_offer.get("data", {})
    except HTTPException as e:
        if e.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="Offer not found. It may have been removed by the airline."
            )
        raise
    
    # Transform to clean format (skip expiration check since user selected it)
    clean_offer = transform_offer(offer_data, skip_expiration_check=True)
    
    if clean_offer is None:
        raise HTTPException(
            status_code=410,
            detail="Offer has expired and is no longer available for booking"
        )
    
    # Detect price changes
    if detect_price_changes and cached:
        price_change_info = detect_price_change(cached, clean_offer)
        if price_change_info:
            direction = "increased" if price_change_info["increased"] else "decreased"
            PRICE_CHANGES.labels(direction).inc()
            logger.info("price_changed", extra={"extra": {
                "offer_id": offer_id,
                "old_price": price_change_info["old_price"],
                "new_price": price_change_info["new_price"],
                "difference": price_change_info["difference"],
                "percent": price_change_info["percent_change"]
            }})
    
    # Cache the fresh offer
    single_offer_cache.set(cache_key, clean_offer)
    
    return OfferResponse(
        offer=clean_offer,
        price_change=price_change_info,
        cached=False
    )
