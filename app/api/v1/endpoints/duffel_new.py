# app.py - SkySearch AI with Fixed Caching System
import os
import time
import math
import json
import hashlib
import asyncio
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict, Counter as CollectionsCounter
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status, APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr, validator

# Prometheus - alias Counter to avoid conflict with collections.Counter
from prometheus_client import Counter as PrometheusCounter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging


from app.mappers.flight_transformers import (
    transform_search_response,
    transform_offer,
    get_cheapest_offer,
    get_fastest_offer,
    get_direct_flights,
    filter_by_max_price,
    filter_by_airline
)

# ---------------------------
# Config
# ---------------------------
DUFFEL_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"

# --- SECURITY FIX ---
# Load token from environment variables. DO NOT hardcode here.
DUFFEL_TOKEN = os.getenv("DUFFEL_API_KEY")
# --- END FIX ---

REQUEST_RATE_LIMIT = int(os.getenv("REQUEST_RATE_LIMIT", "60"))   # tokens per window
REQUEST_RATE_WINDOW = int(os.getenv("REQUEST_RATE_WINDOW", "60")) # seconds

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))    # 15 min
CACHE_MAX_ITEMS = int(os.getenv("CACHE_MAX_ITEMS", "1000"))

CLIENT_TIMEOUT = float(os.getenv("CLIENT_TIMEOUT", "30.0"))

# ---------------------------
# Logging (structured JSON)
# ---------------------------

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
logger = logging.getLogger("ota")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

# --- SECURITY FIX ---
# Add check for missing token
if not DUFFEL_TOKEN:
    logger.critical("FATAL: DUFFEL_TOKEN environment variable not set. Application will not work.")
    # In a real app, you might want to raise ValueError here to stop startup
    # raise ValueError("DUFFEL_TOKEN environment variable not set.")
# --- END FIX ---


# ---------------------------
# Metrics (using PrometheusCounter to avoid conflict with collections.Counter)
# ---------------------------
REQ_COUNT = PrometheusCounter("ota_requests_total", "Total HTTP requests", ["method", "route", "status"])
REQ_LATENCY = Histogram("ota_request_duration_seconds", "HTTP request latency", ["route"])
DUFFEL_REQ_COUNT = PrometheusCounter("ota_duffel_requests_total", "Duffel API calls", ["method", "path", "status"])
DUFFEL_REQ_LAT = Histogram("ota_duffel_request_duration_seconds", "Duffel API latency", ["path"])
CACHE_HITS = PrometheusCounter("ota_cache_hits_total", "Cache hits", ["cache"])
CACHE_MISSES = PrometheusCounter("ota_cache_misses_total", "Cache misses", ["cache"])
BREAKER_OPEN = Gauge("ota_breaker_open", "Circuit breaker open state (1=open,0=closed)")

# ---------------------------
# Resiliency: Circuit Breaker
# ---------------------------
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
            # half-open
            return True
        return False

    def record_success(self):
        self.state.failures = 0
        if self.state.open:
            self.state.open = False
            self.state.opened_at = 0.0
            BREAKER_OPEN.set(0)

    def record_failure(self):
        self.state.failures += 1
        if self.state.failures >= self.failure_threshold:
            self.state.open = True
            self.state.opened_at = time.time()
            BREAKER_OPEN.set(1)

# Simple exponential backoff helper
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

# ---------------------------
# Improved TTL Cache with LRU
# ---------------------------
class ImprovedTTLCache:
    """
    Time-to-live cache with LRU eviction policy.
    """
    
    def __init__(self, ttl_seconds: int, max_items: int, name: str):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.name = name
        # OrderedDict maintains insertion/access order
        self.store: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "sets": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking."""
        item = self.store.get(key)
        
        if not item:
            CACHE_MISSES.labels(self.name).inc()
            self.stats["misses"] += 1
            return None
        
        ts, val = item
        age = time.time() - ts
        
        # Check expiration
        if age > self.ttl:
            self.store.pop(key, None)
            CACHE_MISSES.labels(self.name).inc()
            self.stats["misses"] += 1
            self.stats["expirations"] += 1
            return None
        
        # Move to end (most recently used)
        self.store.move_to_end(key)
        
        CACHE_HITS.labels(self.name).inc()
        self.stats["hits"] += 1
        return val
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with LRU eviction."""
        self.stats["sets"] += 1
        
        # If key exists, update and move to end
        if key in self.store:
            self.store[key] = (time.time(), value)
            self.store.move_to_end(key)
            return
        
        # Evict if at capacity (remove least recently used = first item)
        if len(self.store) >= self.max_items:
            # Remove oldest (first) item
            evicted_key, _ = self.store.popitem(last=False)
            self.stats["evictions"] += 1
            logger.debug("cache_eviction", extra={"extra": {
                "cache": self.name,
                "evicted_key": evicted_key[:12] if len(evicted_key) > 12 else evicted_key
            }})
        
        # Add new item
        self.store[key] = (time.time(), value)
    
    def clear_expired(self) -> int:
        """Remove all expired entries. Returns count of cleared items."""
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
        """Get cache statistics."""
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
    
    def keys(self) -> list:
        """Get all cache keys (for debugging)."""
        return list(self.store.keys())
    
    def clear(self) -> int:
        """Clear all cache entries. Returns count of cleared items."""
        count = len(self.store)
        self.store.clear()
        return count

# Initialize cache instances
offer_cache = ImprovedTTLCache(CACHE_TTL_SECONDS, CACHE_MAX_ITEMS, "offers_by_orq")
result_index_cache = ImprovedTTLCache(CACHE_TTL_SECONDS, CACHE_MAX_ITEMS, "results_index")

# Track search frequency for analytics (using CollectionsCounter, not PrometheusCounter)
route_frequency = CollectionsCounter()

# ---------------------------
# Per-IP Rate Limiter (token bucket)
# ---------------------------
class RateLimiter:
    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self.buckets: Dict[str, Dict[str, float]] = {}  # ip -> {"tokens":float,"updated":ts}

    def allow(self, ip: str) -> bool:
        now = time.time()
        b = self.buckets.get(ip)
        if not b:
            self.buckets[ip] = {"tokens": self.limit - 1, "updated": now}
            return True
        elapsed = now - b["updated"]
        # refill
        refill = (elapsed / self.window) * self.limit
        b["tokens"] = min(self.limit, b["tokens"] + refill)
        b["updated"] = now
        if b["tokens"] >= 1.0:
            b["tokens"] -= 1.0
            return True
        return False

limiter = RateLimiter(REQUEST_RATE_LIMIT, REQUEST_RATE_WINDOW)

# ---------------------------
# Duffel HTTP client
# ---------------------------
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
        self.client = httpx.AsyncClient(base_url=DUFFEL_BASE, headers=HEADERS, timeout=CLIENT_TIMEOUT)

    async def _call(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        if not breaker.allow():
            raise HTTPException(status_code=503, detail="Duffel temporarily unavailable (circuit open)")
        start = time.time()

        async def do_req():
            resp = await self.client.request(method, url, **kwargs)
            return resp

        try:
            resp = await with_backoff(do_req)
            status_code = resp.status_code
            DUFFEL_REQ_COUNT.labels(method, url, str(status_code)).inc()
            DUFFEL_REQ_LAT.labels(url).observe(time.time() - start)

            # logging
            logger.info("duffel_call", extra={
                "extra": {
                    "method": method,
                    "path": url,
                    "status": status_code,
                    "latency_ms": round((time.time() - start) * 1000, 1),
                }
            })

            # --- CIRCUIT BREAKER FIX ---
            # Check for server-side errors (5xx) before raising
            if 500 <= status_code:
                breaker.record_failure()
                # Raise the httpx error to be caught below
                resp.raise_for_status()

            # Successful response (2xx)
            breaker.record_success()

            # Client-side errors (4xx) - do not trip breaker
            if 400 <= status_code < 500:
                try:
                    err = resp.json()
                except Exception:
                    err = {"message": resp.text}
                raise HTTPException(status_code=status_code, detail=err)
            
            return resp.json()

        except httpx.HTTPStatusError as e:
            # This catches the 5xx errors raised above
            if 500 <= e.response.status_code:
                # This failure is already recorded
                pass
            # Re-raise as HTTPException for FastAPI
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            raise HTTPException(status_code=e.response.status_code, detail=detail)

        except httpx.RequestError as e:
            # Network errors (timeout, connection refused) ARE failures
            breaker.record_failure()
            logger.error("duffel_request_error", extra={"extra": {"error": str(e), "url": str(e.request.url)}})
            raise HTTPException(status_code=503, detail=f"Duffel network error: {str(e)}")
        
        except Exception as e:
            # Any other unexpected error is also a failure
            breaker.record_failure()
            logger.error("duffel_unexpected_error", extra={"extra": {"error": str(e)}})
            raise e
        # --- END FIX ---


    # Core endpoints
    async def create_offer_request(self, data: Dict[str, Any], return_offers: bool = True):
        payload = {"data": data}
        params = {"return_offers": str(return_offers).lower()}
        return await self._call("POST", "/air/offer_requests", json=payload, params=params)

    async def get_offer_request(self, orq_id: str):
        return await self._call("GET", f"/air/offer_requests/{orq_id}")

    async def list_offers(self, *, offer_request_id: str, limit: int = 50, after: Optional[str] = None):
        params = {"offer_request_id": offer_request_id, "limit": limit}
        if after:
            params["after"] = after
        return await self._call("GET", "/air/offers", params=params)

    async def get_offer(self, offer_id: str):
        return await self._call("GET", f"/air/offers/{offer_id}")

    async def create_order(self, data: Dict[str, Any]):
        payload = {"data": data}
        return await self._call("POST", "/air/orders", json=payload)

    async def get_order(self, order_id: str):
        return await self._call("GET", f"/air/orders/{order_id}")

duffel = DuffelClient()

# ---------------------------
# Schemas
# ---------------------------
class TimeWindow(BaseModel):
    from_: Optional[str] = Field(None, alias="from", description="HH:MM 24h")
    to: Optional[str] = Field(None, description="HH:MM 24h")

class SliceIn(BaseModel):
    origin: str = Field(..., example="TAS")
    destination: str = Field(..., example="IST")
    departure_date: str = Field(..., example="2025-11-10")
    departure_time: Optional[TimeWindow] = None
    arrival_time: Optional[TimeWindow] = None

class PaxIn(BaseModel):
    type: Optional[str] = Field(None, example="adult")
    age: Optional[int] = Field(None, ge=0, le=120)

    # --- QUALITY FIX ---
    # Removed useless @validator that did nothing
    # --- END FIX ---

class SearchRequest(BaseModel):
    slices: List[SliceIn]
    passengers: List[PaxIn]
    cabin_class: Optional[str] = Field(None, description="economy|premium_economy|business|first")
    max_connections: Optional[int] = Field(1, ge=0, le=2)
    supplier_timeout_ms: Optional[int] = Field(None, description="2000..60000")
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

class OrderPassenger(BaseModel):
    id: str
    email: EmailStr
    phone_number: str
    born_on: str
    title: Optional[str] = None
    gender: Optional[str] = None
    family_name: str
    given_name: str
    infant_passenger_id: Optional[str] = None

class PaymentIn(BaseModel):
    type: str = Field(..., description="balance or arc_bsp_cash")
    currency: str
    amount: str

class CreateOrderRequest(BaseModel):
    selected_offer_id: str
    passengers: List[OrderPassenger]
    payment: PaymentIn
    type: str = Field("instant", description="instant or hold")

class OrderResponse(BaseModel):
    order_id: str
    booking_reference: Optional[str] = None
    raw: Dict[str, Any]

class OfferQuery(BaseModel):
    limit: int = Field(50, ge=1, le=250)
    max_total_amount: Optional[float] = None
    currency: Optional[str] = None
    max_connections: Optional[int] = Field(None, ge=0, le=2)
    sort: Optional[str] = Field(None, description="price_asc|price_desc|duration_asc|duration_desc")

# ---------------------------
# Utilities
# ---------------------------

# --- DEDUPLICATION FIX ---
# This is the single, correct definition of this function.
# The duplicate definition has been removed.
# --- END FIX ---
def normalize_search_key(req: SearchRequest) -> str:
    """
    Create a deterministic hash of search params.
    
    IMPORTANT: return_offers is NOT included in the key because:
    - Both true/false should cache separately (different data structures)
    - This is handled by checking the flag in cache lookup
    """
    payload = {
        "slices": [s.dict(by_alias=True) for s in req.slices],
        "passengers": [p.dict() for p in req.passengers],
        "cabin_class": req.cabin_class,
        "max_connections": req.max_connections,
        # "return_offers": req.return_offers,  # Kept commented out per logic
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()

def _offer_total_amount(o: Dict[str, Any]) -> float:
    try:
        return float(o["total_amount"])
    except Exception:
        return float("inf")

def _offer_duration_minutes(o: Dict[str, Any]) -> int:
    total = 0
    for sl in o.get("slices", []):
        dur = sl.get("duration") or sl.get("total_duration") or ""
        m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", dur)
        if m:
            h = int(m.group(1) or 0)
            mins = int(m.group(2) or 0)
            total += h * 60 + mins
    return total if total > 0 else 10**9

def filter_sort_offers(offers: List[Dict[str, Any]], q: OfferQuery) -> List[Dict[str, Any]]:
    res = offers
    if q.currency:
        res = [o for o in res if o.get("total_currency") == q.currency]
    if q.max_total_amount is not None:
        res = [o for o in res if _offer_total_amount(o) <= q.max_total_amount]
    if q.max_connections is not None:
        def conn(o):
            max_conn = 0
            for sl in o.get("slices", []):
                stops = 0
                for seg in sl.get("segments", []):
                    stops += len(seg.get("stops", []))
                max_conn = max(max_conn, stops)
            return max_conn
        res = [o for o in res if conn(o) <= q.max_connections]

    if q.sort == "price_asc":
        res.sort(key=_offer_total_amount)
    elif q.sort == "price_desc":
        res.sort(key=_offer_total_amount, reverse=True)
    elif q.sort == "duration_asc":
        res.sort(key=_offer_duration_minutes)
    elif q.sort == "duration_desc":
        res.sort(key=_offer_duration_minutes, reverse=True)
    return res

# ---------------------------
# Background Tasks
# ---------------------------
_cleanup_task: Optional[asyncio.Task] = None
_warming_task: Optional[asyncio.Task] = None


# --- DEDUPLICATION FIX ---
# This is the single, correct definition of this function.
# The duplicate definition has been removed.
# --- END FIX ---
async def periodic_cache_cleanup():
    """Background task that periodically cleans expired cache entries."""
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            
            offer_expired = offer_cache.clear_expired()
            result_expired = result_index_cache.clear_expired()
            
            if offer_expired > 0 or result_expired > 0:
                logger.info("cache_cleanup", extra={"extra": {
                    "offer_expired": offer_expired,
                    "result_expired": result_expired,
                    "offer_size": len(offer_cache.store),
                    "result_size": len(result_index_cache.store)
                }})
        
        except asyncio.CancelledError:
            logger.info("cleanup_task_cancelled")
            break
        except Exception as e:
            logger.error("cleanup_task_error", extra={"extra": {"error": str(e)}})


# --- QUALITY FIX ---
# Renamed from warm_popular_routes to be more accurate
# --- END FIX ---
async def log_popular_routes():
    """Background task to track and log popular routes for analysis."""
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
    """FastAPI lifespan context manager for startup/shutdown tasks."""
    global _cleanup_task, _warming_task
    
    # Startup
    logger.info("app_startup", extra={"extra": {
        "cache_ttl": CACHE_TTL_SECONDS,
        "cache_max": CACHE_MAX_ITEMS
    }})
    _cleanup_task = asyncio.create_task(periodic_cache_cleanup())
    
    # --- QUALITY FIX ---
    # Updated to call renamed function
    _warming_task = asyncio.create_task(log_popular_routes())
    # --- END FIX ---
    
    yield
    
    # Shutdown
    logger.info("app_shutdown")
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    if _warming_task:
        _warming_task.cancel()
        try:
            await _warming_task
        except asyncio.CancelledError:
            pass

# ---------------------------
# FastAPI app & Middleware
# ---------------------------
router = APIRouter()
app = FastAPI(

)



# Request ID + logging + rate limiting + metrics middleware
@app.middleware("http")
async def observability_mw(request: Request, call_next):
    # rate limit
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.allow(client_ip):
        REQ_COUNT.labels(request.method, request.url.path, str(status.HTTP_429_TOO_MANY_REQUESTS)).inc()
        return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

    req_id = hashlib.md5(f"{time.time()}-{client_ip}-{request.url.path}".encode()).hexdigest()[:12]
    start = time.time()
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logger.error("request_error", extra={"extra": {"request_id": req_id, "path": request.url.path, "error": str(e)}})
        response = JSONResponse({"detail": "Internal Server Error"}, status_code=500)

    latency = time.time() - start
    logger.info("request", extra={"extra": {
        "request_id": req_id, "ip": client_ip, "method": request.method,
        "path": request.url.path, "status": status_code,
        "latency_ms": round(latency * 1000, 2)
    }})
    REQ_COUNT.labels(request.method, request.url.path, str(status_code)).inc()
    REQ_LATENCY.labels(request.url.path).observe(latency)
    response.headers["X-Request-ID"] = req_id
    return response

# ---------------------------
# API Endpoints
# ---------------------------


@router.get("/healthz")
async def health():
    """Health check endpoint."""
    return {
        "ok": True,
        "circuit_open": breaker.state.open,
        "failures": breaker.state.failures,
        "cache_sizes": {
            "offers": len(offer_cache.store),
            "results": len(result_index_cache.store)
        }
    }


# --- Core Flight Search Endpoint ---
@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, format: str = "clean"):
    """
    Smart flight search with multi-layer caching and clean response formatting.
    
    Query Parameters:
        format: Response format - 'clean' (default) or 'raw'
                'clean' returns frontend-friendly JSON
                'raw' returns original Duffel response
    """
    
    # Generate deterministic cache key
    cache_key = normalize_search_key(req)
    
    # Track search frequency for analytics
    route_frequency[cache_key] += 1
    
    # ====== STEP 1: TRY CACHE FIRST ======
    cached_entry = result_index_cache.get(cache_key)
    
    if cached_entry:
        orq_id = cached_entry["offer_request_id"]
        age_seconds = time.time() - cached_entry["ts"]
        
        # Verify cache is still valid (within TTL)
        if age_seconds < CACHE_TTL_SECONDS:
            # Check if return_offers flag matches
            if cached_entry.get("return_offers") == req.return_offers:
                # Try to get offers from cache
                cached_offers = offer_cache.get(orq_id)
                
                if cached_offers:
                    logger.info("cache_hit", extra={"extra": {
                        "cache_key": cache_key[:12],
                        "orq_id": orq_id,
                        "age_seconds": round(age_seconds, 1),
                        "offers_count": len(cached_offers)
                    }})
                    
                    # Prepare response based on format
                    page = cached_offers[:req.page_size]
                    
                    meta_data = PageMeta(
                        page_size=req.page_size,
                        has_more=len(cached_offers) > len(page),
                        next_after=None,
                        source="cached",
                        cache_hit=True,
                        cache_age_seconds=round(age_seconds, 1)
                    )
                    
                    # Return in requested format
                    if format == "clean":
                        # Transform to clean format
                        response_data = {
                            "offer_request_id": orq_id,
                            "offers": page,
                            "meta": meta_data.dict()
                        }
                        clean_response = transform_search_response(response_data, limit=req.page_size)
                        return SearchResponse(
                            offer_request_id=orq_id,
                            offers=clean_response["offers"],
                            meta=meta_data
                        )
                    else:
                        # Return raw format
                        return SearchResponse(
                            offer_request_id=orq_id,
                            offers=page,
                            meta=meta_data
                        )
            else:
                logger.info("cache_miss", extra={"extra": {
                    "cache_key": cache_key[:12],
                    "reason": "return_offers_mismatch",
                    "cached_flag": cached_entry.get("return_offers"),
                    "requested_flag": req.return_offers
                }})
    
    # ====== STEP 2: CACHE MISS - FETCH FROM DUFFEL ======
    logger.info("cache_miss", extra={"extra": {
        "cache_key": cache_key[:12],
        "reason": "no_valid_cache_entry" if not cached_entry else "cache_expired"
    }})
    
    # Build Duffel Offer Request payload
    slices_payload = []
    for s in req.slices:
        sp = {"origin": s.origin, "destination": s.destination, "departure_date": s.departure_date}
        if s.departure_time:
            sp["departure_time"] = {k: v for k, v in s.departure_time.dict(by_alias=True).items() if v}
        if s.arrival_time:
            sp["arrival_time"] = {k: v for k, v in s.arrival_time.dict(by_alias=True).items() if v}
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

    # Create new Offer Request from Duffel
    resp = await duffel.create_offer_request(data=data, return_offers=req.return_offers)
    orq = resp["data"]
    orq_id = orq["id"]

    # ====== STEP 3: STORE IN CACHE ======
    # Store single entry (most recent)
    result_index_cache.set(cache_key, {
        "offer_request_id": orq_id,
        "ts": time.time(),
        "return_offers": req.return_offers,
        "page_size": req.page_size
    })

    # Handle pagination based on return_offers flag
    if not req.return_offers:
        # Fetch first page from Duffel
        first_page = await duffel.list_offers(offer_request_id=orq_id, limit=req.page_size)
        offers = first_page.get("data", []) or []
        meta = first_page.get("meta", {}) or {}
        next_after = meta.get("after")
        
        # Cache offers for subsequent requests
        if offers:
            offer_cache.set(orq_id, offers)
        
        meta_data = PageMeta(
            page_size=req.page_size,
            has_more=bool(next_after),
            next_after=next_after,
            source="fresh_paged",
            cache_hit=False
        )
        
        # Return in requested format
        if format == "clean":
            response_data = {
                "offer_request_id": orq_id,
                "offers": offers,
                "meta": meta_data.dict()
            }
            clean_response = transform_search_response(response_data, limit=req.page_size)
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

    # return_offers = true -> embedded offers in response
    embedded = orq.get("offers") or []
    if embedded:
        # Cache ALL embedded offers
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
    
    # Return in requested format
    if format == "clean":
        response_data = {
            "offer_request_id": orq_id,
            "offers": page,
            "meta": meta_data.dict()
        }
        clean_response = transform_search_response(response_data, limit=req.page_size)
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


@router.get("/offers/{offer_id}/clean")
async def get_offer_clean(offer_id: str):
    """
    Get a single offer in clean, frontend-friendly format.
    """
    raw_offer = await duffel.get_offer(offer_id)
    offer_data = raw_offer.get("data", {})
    
    return {
        "offer": transform_offer(offer_data)
    }
