# services/amadeus_service.py
import asyncio
import json
import logging
import random
import zlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import httpx
import redis.asyncio as redis
from fastapi import HTTPException

from app.core.config import settings
from schemas.flight_segment import FlightSegment, FlightLeg, Flight

logger = logging.getLogger(__name__)

# ---------------------------
# Globals / singletons
# ---------------------------
HTTP_TIMEOUT_S = 25.0
_httpx_client: Optional[httpx.AsyncClient] = None
_token_lock = asyncio.Lock()
amadeus_token_storage = {"token": None, "expires_at": datetime.min}

redis_client = redis.from_url(settings.REDIS_URI, decode_responses=False)  # store bytes for zlib
CB_KEY = "amadeus:circuit"           # circuit breaker redis key
CB_FAIL_WINDOW = 120                 # seconds to consider failures
CB_OPEN_SECONDS = 60                 # open-circuit duration after too many failures
CB_MAX_FAILS = 6                     # fails in window to open circuit

# Cache TTLs
TTL_SEARCH = getattr(settings, "CACHE_TTL_SEARCH", 900)   # 15m default
TTL_OFFERS = getattr(settings, "CACHE_TTL_OFFERS", 300)   # 5m default
TTL_RAW = max(TTL_SEARCH, 3600)                           # raw responses longer by default


# ---------------------------
# Helpers
# ---------------------------
def _now_utc() -> datetime:
    return datetime.utcnow()

def _jitter(base: float = 0.2) -> float:
    return random.uniform(0, base)

async def _get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT_S)
    return _httpx_client

async def _cb_is_open() -> bool:
    """Circuit breaker: returns True if the circuit is open (skip provider calls)."""
    try:
        val = await redis_client.get(CB_KEY)
        if not val:
            return False
        opened_until = int(val.decode())
        return opened_until > int(_now_utc().timestamp())
    except Exception:
        return False

async def _cb_record_failure():
    """Record a failure and possibly open the circuit."""
    try:
        window_key = f"{CB_KEY}:win"
        # Increment failures in the window
        cur = await redis_client.incr(window_key)
        if cur == 1:
            await redis_client.expire(window_key, CB_FAIL_WINDOW)
        if cur >= CB_MAX_FAILS:
            # open circuit for CB_OPEN_SECONDS
            opened_until = int((_now_utc() + timedelta(seconds=CB_OPEN_SECONDS)).timestamp())
            await redis_client.set(CB_KEY, str(opened_until), ex=CB_OPEN_SECONDS)
            # reset window counter
            await redis_client.delete(window_key)
            logger.warning("Amadeus circuit OPENED for %ss", CB_OPEN_SECONDS)
    except Exception:
        pass

async def _cb_record_success():
    """On success, reduce failure pressure."""
    try:
        await redis_client.delete(f"{CB_KEY}:win")
    except Exception:
        pass

async def _retry_get(url: str, headers: Dict[str, str], params: Dict[str, Any], retries: int = 3) -> httpx.Response:
    client = await _get_httpx_client()
    last_exc = None
    for attempt in range(retries):
        try:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            last_exc = e
            # 401 handled by caller for token refresh
            status = getattr(e.response, "status_code", None) if isinstance(e, httpx.HTTPStatusError) else None
            # On 429/5xx, backoff; on others, break
            if status in (429, 500, 502, 503, 504) or isinstance(e, httpx.RequestError):
                delay = (2 ** attempt) + _jitter()
                logger.warning("Amadeus GET retry %s/%s after error %s, backing off %.2fs", attempt + 1, retries, status or type(e).__name__, delay)
                await asyncio.sleep(delay)
                continue
            break
    raise last_exc  # type: ignore[misc]

def _zset(key: str, obj: Any, ttl: int):
    """Compress + set JSON to Redis (bytes)."""
    try:
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        comp = zlib.compress(data, level=6)
        return redis_client.set(key, comp, ex=ttl)
    except Exception as e:
        logger.debug("Redis zset failed for %s: %s", key, e)

async def _zget(key: str) -> Optional[Any]:
    """Get + decompress JSON from Redis."""
    try:
        raw = await redis_client.get(key)
        if not raw:
            return None
        data = zlib.decompress(raw)
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        logger.debug("Redis zget failed for %s: %s", key, e)
        return None

def _make_cache_key(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    infants: int = 0,
    cabin_class: str = "ECONOMY",
    non_stop: bool = False,
    currency: str = "USD"
) -> str:
    """
    Generate a unique cache key that includes ALL search parameters.
    This ensures different searches don't return the same cached results.
    """
    parts = [
        "flights",
        origin.upper(),
        destination.upper(),
        departure_date,
        return_date or "ow",  # "ow" for one-way
        f"a{adults}",
        f"c{children}",
        f"i{infants}",
        cabin_class.lower()[:4],  # econ, prem, busi, firs
        "ns" if non_stop else "all",
        currency.upper()
    ]
    return ":".join(parts)

def _parse_duration_to_minutes(duration_str: str) -> int:
    """
    Convert ISO 8601 duration (e.g., 'PT2H30M') to minutes.
    Returns 0 if parsing fails.
    """
    try:
        duration_str = duration_str.replace('PT', '')
        hours = 0
        minutes = 0
        
        if 'H' in duration_str:
            parts = duration_str.split('H')
            hours = int(parts[0])
            duration_str = parts[1] if len(parts) > 1 else ''
        
        if 'M' in duration_str:
            minutes = int(duration_str.replace('M', ''))
        
        return hours * 60 + minutes
    except (ValueError, AttributeError):
        return 0
    

def _format_duration(duration_str: str) -> str:
    """
    Convert ISO 8601 duration (e.g., 'PT2H30M') to human-readable format (e.g., '2h 30m').
    Returns the formatted string or original if parsing fails.
    
    Args:
        duration_str: ISO 8601 duration string (e.g., 'PT2H30M', 'PT5H', 'PT45M')
        
    Returns:
        Human-readable duration string (e.g., '2h 30m', '5h', '45m')
        
    Examples:
        'PT2H30M' -> '2h 30m'
        'PT5H' -> '5h'
        'PT45M' -> '45m'
        'PT0M' -> '0m'
    """
    try:
        if not duration_str or not duration_str.startswith('PT'):
            return duration_str  # Return as-is if not ISO format
        
        # Remove 'PT' prefix
        duration_str = duration_str.replace('PT', '')
        hours = 0
        minutes = 0
        
        # Parse hours
        if 'H' in duration_str:
            parts = duration_str.split('H')
            hours = int(parts[0])
            duration_str = parts[1] if len(parts) > 1 else ''
        
        # Parse minutes
        if 'M' in duration_str:
            minutes = int(duration_str.replace('M', ''))
        
        # Format output
        if hours > 0 and minutes > 0:
            return f"{hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return "0m"
            
    except (ValueError, AttributeError) as e:
        logger.debug("Failed to format duration %s: %s", duration_str, e)
        return duration_str  # Return original on error
    

def _calculate_layover(prev_arrival: str, next_departure: str) -> Optional[str]:
    """
    Calculate layover duration between two segments.
    Returns human-readable format (e.g., '1h 15m', '2h 30m', '45m') or None.
    
    Args:
        prev_arrival: ISO 8601 datetime string of previous flight arrival
        next_departure: ISO 8601 datetime string of next flight departure
        
    Returns:
        Human-readable duration string or None if calculation fails
        
    Examples:
        '2h 15m' for 2 hours 15 minutes layover
        '3h' for exactly 3 hours
        '45m' for 45 minutes
    """
    try:
        # Parse datetime strings, handling both 'Z' and timezone formats
        prev_time = datetime.fromisoformat(prev_arrival.replace('Z', '+00:00'))
        next_time = datetime.fromisoformat(next_departure.replace('Z', '+00:00'))
        
        # Calculate time difference
        diff = next_time - prev_time
        
        # Handle negative layover (data error)
        if diff.total_seconds() < 0:
            return None
        
        # Extract hours and minutes
        total_seconds = int(diff.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        # Format output based on duration
        if hours > 0 and minutes > 0:
            return f"{hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return "0m"  # Same airport transfer or immediate connection
            
    except (ValueError, AttributeError, TypeError) as e:
        logger.debug("Failed to calculate layover: %s", e)
        return None


# ---------------------------
# OAuth token management
# ---------------------------
class AmadeusTimeoutError(Exception):
    pass

async def get_amadeus_token() -> str:
    async with _token_lock:
        now = _now_utc()
        if amadeus_token_storage["token"] and amadeus_token_storage["expires_at"] > now:
            return amadeus_token_storage["token"]  # still valid

        logger.info("Amadeus token missing/expired. Fetching new one...")
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        body = {
            "grant_type": "client_credentials",
            "client_id": settings.AMADEUS_API_KEY,
            "client_secret": settings.AMADEUS_API_SECRET,
        }
        client = await _get_httpx_client()
        resp = await client.post(settings.AMADEUS_TOKEN_URL, data=body, headers=headers)
        resp.raise_for_status()
        token_data = resp.json()
        token = token_data["access_token"]
        expires_in = token_data["expires_in"]
        # pad 5 minutes
        amadeus_token_storage["token"] = token
        amadeus_token_storage["expires_at"] = now + timedelta(seconds=max(expires_in - 300, 60))
        logger.info("Amadeus token retrieved.")
        return token

# ---------------------------
# Public API
# ---------------------------
async def search_flights_amadeus(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    infants: int = 0,
    cabin_class: str = "ECONOMY",
    non_stop: bool = False,
    max_results: int = 20,
    currency: str = "USD"
) -> List[Flight]:
    """
    Search flights with comprehensive parameters.
    Supports both one-way and round-trip searches.
    Caches results based on ALL search parameters to avoid returning wrong data.
    """
    # Circuit breaker
    if await _cb_is_open():
        logger.warning("Circuit open: skipping Amadeus search")
        return []

    # Generate unique cache key including ALL parameters
    cache_key = _make_cache_key(
        origin, destination, departure_date, return_date,
        adults, children, infants, cabin_class, non_stop, currency
    )
    raw_key = f"raw:{cache_key}"

    # Check cache
    cached = await _zget(cache_key)
    if cached:
        logger.info("CACHE HIT %s", cache_key)
        return [Flight.model_validate(item) for item in cached]

    logger.info("CACHE MISS %s -> Amadeus API", cache_key)

    token = await get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Build Amadeus API parameters
    params = {
        "originLocationCode": origin.upper(),
        "destinationLocationCode": destination.upper(),
        "departureDate": departure_date,
        "adults": adults,
        "max": min(max_results, 250),  # Amadeus max is 250
        "currencyCode": currency.upper(),
        "travelClass": cabin_class.upper(),
        "nonStop": "true" if non_stop else "false"
    }
    
    # Add children and infants if present
    if children > 0:
        params["children"] = children
    if infants > 0:
        params["infants"] = infants
    
    # Add return date for round-trip
    if return_date:
        params["returnDate"] = return_date

    try:
        resp = await _retry_get(settings.AMADEUS_SEARCH_URL, headers, params, retries=3)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            # Token expired unexpectedly -> refresh once
            amadeus_token_storage["token"] = None
            headers["Authorization"] = f"Bearer {await get_amadeus_token()}"
            resp = await _retry_get(settings.AMADEUS_SEARCH_URL, headers, params, retries=2)
        else:
            await _cb_record_failure()
            logger.error("Amadeus HTTP error: %s %s", e.response.status_code, e.response.text[:500])
            return []
    except (httpx.RequestError, httpx.ReadTimeout) as e:
        await _cb_record_failure()
        logger.error("Amadeus request error: %s", str(e))
        raise AmadeusTimeoutError("The request to the flight provider timed out.") from e

    data = resp.json()
    await _cb_record_success()

    # Cache raw response (compressed)
    await _zset(raw_key, data, ttl=TTL_RAW)

    flights = parse_flight_offers(data)

    # Cache simplified list (compressed)
    await _zset(cache_key, [f.model_dump() for f in flights], ttl=TTL_SEARCH)
    
    logger.info(f"Found {len(flights)} flights for {origin}->{destination}")
    return flights


def _parse_itinerary(itinerary: Dict[str, Any], carriers: Dict[str, str]) -> FlightLeg:
    """
    Parse a single itinerary (outbound or return) into a FlightLeg with all segments.
    All durations are converted to human-readable format (e.g., '2h 30m').
    """
    segments_data = itinerary.get("segments", [])
    
    # Parse all segments with layover information
    parsed_segments = []
    for idx, seg in enumerate(segments_data):
        # Calculate layover if not first segment
        layover_duration = None
        if idx > 0:
            prev_seg = segments_data[idx - 1]
            layover_duration = _calculate_layover(
                prev_seg["arrival"]["at"],
                seg["departure"]["at"]
            )
        
        # Convert segment duration to human-readable format
        segment_duration = _format_duration(seg.get("duration", ""))
        
        segment = FlightSegment(
            departure_airport=seg["departure"]["iataCode"],
            departure_time=seg["departure"]["at"],
            arrival_airport=seg["arrival"]["iataCode"],
            arrival_time=seg["arrival"]["at"],
            airline=carriers.get(seg["carrierCode"], seg["carrierCode"]),
            airline_code=seg["carrierCode"],
            flight_number=f"{seg['carrierCode']}{seg['number']}",
            aircraft=seg.get("aircraft", {}).get("code"),
            duration=segment_duration,  # ✅ Human-readable format
            layover_duration=layover_duration  # ✅ Human-readable format
        )
        parsed_segments.append(segment)
    
    # Get first and last segment for summary
    first_seg = segments_data[0]
    last_seg = segments_data[-1]
    
    # Convert leg duration to human-readable format
    leg_duration = _format_duration(itinerary.get("duration", ""))
    
    return FlightLeg(
        departure_airport=first_seg["departure"]["iataCode"],
        departure_time=first_seg["departure"]["at"],
        arrival_airport=last_seg["arrival"]["iataCode"],
        arrival_time=last_seg["arrival"]["at"],
        duration=leg_duration,  # ✅ Human-readable format
        stops=max(len(segments_data) - 1, 0),
        segments=parsed_segments
    )

def parse_flight_offers(data: Dict[str, Any]) -> List[Flight]:
    """
    Parse Amadeus API response into Flight objects with nested structure.
    Handles both one-way and round-trip with full segment details.
    """
    offers = data.get("data", [])
    if not offers:
        return []

    carriers = data.get("dictionaries", {}).get("carriers", {})
    results: List[Flight] = []

    for offer in offers:
        try:
            # Parse outbound leg (always present)
            outbound = _parse_itinerary(offer["itineraries"][0], carriers)
            
            # Check for return leg
            return_leg = None
            is_round_trip = len(offer["itineraries"]) > 1
            
            if is_round_trip:
                return_leg = _parse_itinerary(offer["itineraries"][1], carriers)
            
            # Create flight object
            flight = Flight(
                id=offer["id"],
                price=float(offer["price"]["total"]),
                currency=offer["price"]["currency"],
                is_round_trip=is_round_trip,
                outbound=outbound,
                return_flight=return_leg
            )
            
            results.append(flight)
            
        except (KeyError, IndexError) as e:
            logger.debug("Offer parse skipped: %s", e)
            continue
    
    return results


async def get_full_flight_offer(
    flight_id: str, 
    search_params: dict
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the full offer + dictionaries from cached raw data.
    search_params must include: origin, destination, departure_date
    Optional: return_date, adults, children, infants, cabin_class, non_stop, currency
    """
    # Reconstruct the cache key with all parameters
    cache_key = _make_cache_key(
        origin=search_params['origin'],
        destination=search_params['destination'],
        departure_date=search_params['departure_date'],
        return_date=search_params.get('return_date'),
        adults=search_params.get('adults', 1),
        children=search_params.get('children', 0),
        infants=search_params.get('infants', 0),
        cabin_class=search_params.get('cabin_class', 'ECONOMY'),
        non_stop=search_params.get('non_stop', False),
        currency=search_params.get('currency', 'USD')
    )
    raw_key = f"raw:{cache_key}"
    
    full = await _zget(raw_key)
    if not full:
        logger.info("Raw cache miss for %s", raw_key)
        return None
    
    for offer in full.get("data", []):
        if offer.get("id") == flight_id:
            return {"offer": offer, "dictionaries": full.get("dictionaries", {})}
    
    logger.info("Offer %s not found under %s", flight_id, raw_key)
    return None


async def price_flight_offer(flight_offer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Confirm final price before booking (mandatory).
    """
    if await _cb_is_open():
        raise HTTPException(status_code=503, detail="Flight provider unavailable. Try again shortly.")

    logger.info("Pricing flight offer...")
    token = await get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"data": {"type": "flight-offers-pricing", "flightOffers": [flight_offer]}}
    client = await _get_httpx_client()

    try:
        resp = await client.post("https://test.api.amadeus.com/v2/shopping/flight-offers/", headers=headers, json=payload)
        resp.raise_for_status()
        await _cb_record_success()
        return resp.json()
    except httpx.HTTPStatusError as e:
        await _cb_record_failure()
        logger.error("Pricing failed: %s %s", e.response.status_code, e.response.text[:500])
        raise
    except (httpx.RequestError, httpx.ReadTimeout) as e:
        await _cb_record_failure()
        logger.error("Pricing timeout/error: %s", str(e))
        raise AmadeusTimeoutError("Pricing request timed out") from e


async def create_flight_order(priced_offer_data: Dict[str, Any], passengers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create final PNR (flight-order). Expects `priced_offer_data` from price API.
    """
    if await _cb_is_open():
        raise HTTPException(status_code=503, detail="Flight provider unavailable. Try again shortly.")

    logger.info("Creating flight order (PNR)...")
    token = await get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    travelers = []
    for i, p in enumerate(passengers):
        travelers.append({
            "id": str(i + 1),
            "dateOfBirth": p["date_of_birth"],
            "name": {"firstName": p["first_name"], "lastName": p["last_name"]},
            "contact": {
                "emailAddress": p["email"],
                "phones": [{"deviceType": "MOBILE", "countryCallingCode": "998", "number": p["phone_number"].replace("+", "")}],
            },
        })

    payload = {
        "data": {
            "type": "flight-order",
            "flightOffers": priced_offer_data["data"]["flightOffers"],  # must be verbatim from priced response
            "travelers": travelers,
            "remarks": {"general": [{"subType": "GENERAL_MISCELLANEOUS", "text": "FlyUz MVP Booking"}]},
            "ticketingAgreement": {"option": "DELAY_TO_CANCEL", "delay": "6H"},
            "contacts": [{
                "addresseeName": {"firstName": "FlyUz", "lastName": "Booking"},
                "companyName": "FlyUz",
                "purpose": "STANDARD",
                "phones": [{"deviceType": "MOBILE", "countryCallingCode": "998", "number": "901234567"}],
                "emailAddress": "bookings@flyuz.dev",
            }],
        }
    }

    client = await _get_httpx_client()
    try:
        resp = await client.post("https://test.api.amadeus.com/v2/booking/flight-orders", headers=headers, json=payload)
        if resp.status_code != 201:
            await _cb_record_failure()
            logger.error("Booking failed: %s %s", resp.status_code, resp.text[:800])
            resp.raise_for_status()
        await _cb_record_success()
        return resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        await _cb_record_failure()
        raise

# --- PERFORMANCE IMPROVEMENT HERE ---
async def search_flexible_flights_amadeus(
    origin: str,
    destination: str,
    duration_days: int,
    months: int = 2,
    day_step: int = 3,
    adults: int = 1,
    children: int = 0,
    cabin_class: str = "ECONOMY"
) -> List[Flight]:
    """
    Flexible search: iterate departure dates over a window (sampled every `day_step` days),
    pair each with an implied return date (dep + duration_days), and collect options by
    running all searches concurrently.
    """
    # Circuit breaker
    if await _cb_is_open():
        logger.warning("Circuit open: skipping flexible search")
        return []

    cache_key = f"flex:{origin}:{destination}:d{duration_days}:m{months}:s{day_step}:a{adults}:c{children}:{cabin_class.lower()[:4]}"
    cached = await _zget(cache_key)
    if cached:
        logger.info("CACHE HIT %s", cache_key)
        return [Flight.model_validate(item) for item in cached]

    logger.info("CACHE MISS %s -> gathering concurrent tasks", cache_key)

    start_date = _now_utc().date() + timedelta(days=30)  # start ~1 month out
    end_date = start_date + timedelta(days=30 * months)
    
    # Create a list of all search tasks to run
    tasks = []
    current_date = start_date
    while current_date < end_date:
        dep_str = current_date.strftime("%Y-%m-%d")
        ret_str = (current_date + timedelta(days=duration_days)).strftime("%Y-%m-%d")
        
        # Create a coroutine for each API call
        task = search_flights_amadeus(
            origin=origin,
            destination=destination,
            departure_date=dep_str,
            return_date=ret_str,  # Round-trip for flexible search
            adults=adults,
            children=children,
            cabin_class=cabin_class
        )
        tasks.append(task)
        current_date += timedelta(days=day_step)

    # Run all search tasks concurrently
    # return_exceptions=True ensures that one failed task doesn't stop others
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results, filtering out any exceptions
    all_results: List[Flight] = []
    for result in task_results:
        if isinstance(result, list):
            all_results.extend(result)
        elif isinstance(result, Exception):
            logger.warning("A flexible search sub-query failed: %s", result)

    # Cache the final aggregated results
    await _zset(cache_key, [f.model_dump() for f in all_results], ttl=TTL_SEARCH)
    logger.info(f"Flexible search found {len(all_results)} total options from {len(tasks)} queries")
    return all_results