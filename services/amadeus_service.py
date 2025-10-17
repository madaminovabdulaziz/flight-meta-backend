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

# Valid currencies supported by Amadeus
VALID_CURRENCIES = ["USD", "EUR", "RUB", "UZS"]

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
    try:
        val = await redis_client.get(CB_KEY)
        if not val:
            return False
        opened_until = int(val.decode())
        return opened_until > int(_now_utc().timestamp())
    except Exception:
        return False

async def _cb_record_failure():
    try:
        window_key = f"{CB_KEY}:win"
        cur = await redis_client.incr(window_key)
        if cur == 1:
            await redis_client.expire(window_key, CB_FAIL_WINDOW)
        if cur >= CB_MAX_FAILS:
            opened_until = int((_now_utc() + timedelta(seconds=CB_OPEN_SECONDS)).timestamp())
            await redis_client.set(CB_KEY, str(opened_until), ex=CB_OPEN_SECONDS)
            await redis_client.delete(window_key)
            logger.warning("Amadeus circuit OPENED for %ss", CB_OPEN_SECONDS)
    except Exception:
        pass

async def _cb_record_success():
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
            status = getattr(e.response, "status_code", None) if isinstance(e, httpx.HTTPStatusError) else None
            if status in (429, 500, 502, 503, 504) or isinstance(e, httpx.RequestError):
                delay = (2 ** attempt) + _jitter()
                logger.warning("Amadeus GET retry %s/%s after error %s, backing off %.2fs", attempt + 1, retries, status or type(e).__name__, delay)
                await asyncio.sleep(delay)
                continue
            break
    raise last_exc  # type: ignore[misc]

def _zset(key: str, obj: Any, ttl: int):
    try:
        data = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        comp = zlib.compress(data, level=6)
        return redis_client.set(key, comp, ex=ttl)
    except Exception as e:
        logger.debug("Redis zset failed for %s: %s", key, e)

async def _zget(key: str) -> Optional[Any]:
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
    currency: str = "USD",
    user_intent: str = "find_best_value",  # NEW: Include user intent
    max_price: Optional[float] = None      # NEW: Include max_price
) -> str:
    parts = [
        "flights",
        origin.upper(),
        destination.upper(),
        departure_date,
        return_date or "ow",
        f"a{adults}",
        f"c{children}",
        f"i{infants}",
        cabin_class.lower()[:4],
        "ns" if non_stop else "all",
        currency.upper(),
        user_intent.lower(),
        f"mp{max_price or 'none'}"
    ]
    return ":".join(parts)

def _parse_duration_to_minutes(duration_str: str) -> int:
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
    try:
        if not duration_str or not duration_str.startswith('PT'):
            return duration_str
        duration_str = duration_str.replace('PT', '')
        hours = 0
        minutes = 0
        if 'H' in duration_str:
            parts = duration_str.split('H')
            hours = int(parts[0])
            duration_str = parts[1] if len(parts) > 1 else ''
        if 'M' in duration_str:
            minutes = int(duration_str.replace('M', ''))
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
        return duration_str

def _calculate_layover(prev_arrival: str, next_departure: str) -> Optional[str]:
    try:
        prev_time = datetime.fromisoformat(prev_arrival.replace('Z', '+00:00'))
        next_time = datetime.fromisoformat(next_departure.replace('Z', '+00:00'))
        diff = next_time - prev_time
        if diff.total_seconds() < 0:
            return None
        total_seconds = int(diff.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        if hours > 0 and minutes > 0:
            return f"{hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return "0m"
    except (ValueError, AttributeError, TypeError) as e:
        logger.debug("Failed to calculate layover: %s", e)
        return None

async def _validate_iata_code(code: str) -> bool:
    """
    Validate IATA code by checking against Amadeus airport API.
    For simplicity, this is a placeholder. In production, query:
    https://test.api.amadeus.com/v1/reference-data/locations/airports
    """
    # Placeholder: assume 3-letter uppercase is valid
    # TODO: Implement actual API call to Amadeus airport lookup
    return len(code) == 3 and code.isupper() and code.isalpha()

# ---------------------------
# OAuth token management
# ---------------------------
class AmadeusTimeoutError(Exception):
    pass

async def get_amadeus_token() -> str:
    async with _token_lock:
        now = _now_utc()
        if amadeus_token_storage["token"] and amadeus_token_storage["expires_at"] > now:
            return amadeus_token_storage["token"]
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
    currency: str = "USD",
    max_price: Optional[float] = None,  # NEW
    user_intent: str = "find_best_value"  # NEW
) -> List[Flight]:
    """
    Search flights with comprehensive parameters.
    Supports both one-way and round-trip searches.
    Caches results based on ALL search parameters to avoid returning wrong data.
    Filters by max_price if provided.
    """
    # Validate inputs
    if not await _validate_iata_code(origin) or not await _validate_iata_code(destination):
        raise HTTPException(status_code=400, detail="Invalid origin or destination IATA code")
    if adults < 1 or children < 0 or infants < 0:
        raise HTTPException(status_code=400, detail="Invalid passenger counts")
    if currency not in VALID_CURRENCIES:
        raise HTTPException(status_code=400, detail=f"Unsupported currency: {currency}")
    try:
        dep_date = datetime.strptime(departure_date, "%Y-%m-%d")
        if dep_date.date() < _now_utc().date():
            raise HTTPException(status_code=400, detail="Departure date cannot be in the past")
        if return_date:
            ret_date = datetime.strptime(return_date, "%Y-%m-%d")
            if ret_date <= dep_date:
                raise HTTPException(status_code=400, detail="Return date must be after departure date")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    # Circuit breaker
    if await _cb_is_open():
        logger.warning("Circuit open: skipping Amadeus search")
        return []

    # Generate unique cache key
    cache_key = _make_cache_key(
        origin, destination, departure_date, return_date,
        adults, children, infants, cabin_class, non_stop, currency,
        user_intent, max_price
    )
    raw_key = f"raw:{cache_key}"

    # Check cache
    cached = await _zget(cache_key)
    if cached:
        logger.info("CACHE HIT %s", cache_key)
        flights = [Flight.model_validate(item) for item in cached]
        if max_price:
            flights = [f for f in flights if f.price <= max_price]
        return flights

    logger.info("CACHE MISS %s -> Amadeus API", cache_key)

    token = await get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    params = {
        "originLocationCode": origin.upper(),
        "destinationLocationCode": destination.upper(),
        "departureDate": departure_date,
        "adults": adults,
        "max": min(max_results, 250),
        "currencyCode": currency.upper(),
        "travelClass": cabin_class.upper(),
        "nonStop": "true" if non_stop else "false"
    }
    
    if children > 0:
        params["children"] = children
    if infants > 0:
        params["infants"] = infants
    if return_date:
        params["returnDate"] = return_date
    if max_price:
        params["maxPrice"] = int(max_price)  # Amadeus expects integer

    try:
        resp = await _retry_get(settings.AMADEUS_SEARCH_URL, headers, params, retries=3)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
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

    await _zset(raw_key, data, ttl=TTL_RAW)

    flights = parse_flight_offers(data)
    
    # Filter by max_price if not passed to Amadeus API
    if max_price and "maxPrice" not in params:
        flights = [f for f in flights if f.price <= max_price]

    # Sort based on user_intent
    if user_intent == "find_cheapest":
        flights.sort(key=lambda x: x.price)
    elif user_intent == "find_fastest":
        flights.sort(key=lambda x: _parse_duration_to_minutes(x.outbound.duration))
    
    await _zset(cache_key, [f.model_dump() for f in flights], ttl=TTL_SEARCH)
    
    logger.info(f"Found {len(flights)} flights for {origin}->{destination}")
    return flights

async def search_flexible_flights_amadeus(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    flexible_dates: bool = False,
    adults: int = 1,
    children: int = 0,
    infants: int = 0,
    cabin_class: str = "ECONOMY",
    non_stop: bool = False,
    max_results: int = 20,
    currency: str = "USD",
    max_price: Optional[float] = None,
    user_intent: str = "find_best_value"
) -> List[Flight]:
    """
    Flexible search over a date range if flexible_dates is True.
    Samples dates around departure_date (and return_date if provided).
    """
    if not flexible_dates:
        return await search_flights_amadeus(
            origin, destination, departure_date, return_date, adults, children,
            infants, cabin_class, non_stop, max_results, currency, max_price, user_intent
        )

    # Circuit breaker
    if await _cb_is_open():
        logger.warning("Circuit open: skipping flexible search")
        return []

    cache_key = f"flex:{_make_cache_key(origin, destination, departure_date, return_date, adults, children, infants, cabin_class, non_stop, currency, user_intent, max_price)}"
    cached = await _zget(cache_key)
    if cached:
        logger.info("CACHE HIT %s", cache_key)
        return [Flight.model_validate(item) for item in cached]

    logger.info("CACHE MISS %s -> gathering concurrent tasks", cache_key)

    # Define date range: ±3 days around departure (and return if provided)
    try:
        base_dep_date = datetime.strptime(departure_date, "%Y-%m-%d").date()
        date_range = [base_dep_date + timedelta(days=i) for i in range(-3, 4)]
        if return_date:
            base_ret_date = datetime.strptime(return_date, "%Y-%m-%d").date()
            ret_date_range = [base_ret_date + timedelta(days=i) for i in range(-3, 4)]
        else:
            ret_date_range = [None]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format")

    tasks = []
    for dep_date in date_range:
        for ret_date in ret_date_range:
            task = search_flights_amadeus(
                origin=origin,
                destination=destination,
                departure_date=dep_date.strftime("%Y-%m-%d"),
                return_date=ret_date.strftime("%Y-%m-%d") if ret_date else None,
                adults=adults,
                children=children,
                infants=infants,
                cabin_class=cabin_class,
                non_stop=non_stop,
                max_results=max_results,
                currency=currency,
                max_price=max_price,
                user_intent=user_intent
            )
            tasks.append(task)

    task_results = await asyncio.gather(*tasks, return_exceptions=True)
    all_results: List[Flight] = []
    for result in task_results:
        if isinstance(result, list):
            all_results.extend(result)
        elif isinstance(result, Exception):
            logger.warning("A flexible search sub-query failed: %s", result)

    # Filter by max_price if not already filtered
    if max_price:
        all_results = [f for f in all_results if f.price <= max_price]

    # Sort based on user_intent
    if user_intent == "find_cheapest":
        all_results.sort(key=lambda x: x.price)
    elif user_intent == "find_fastest":
        all_results.sort(key=lambda x: _parse_duration_to_minutes(x.outbound.duration))

    await _zset(cache_key, [f.model_dump() for f in all_results], ttl=TTL_SEARCH)
    logger.info(f"Flexible search found {len(all_results)} total options from {len(tasks)} queries")
    return all_results

# Other functions (parse_flight_offers, get_full_flight_offer, price_flight_offer, create_flight_order)
# remain unchanged unless you need specific tweaks for your use case.
# I'll include them for completeness but without modification.

def _parse_itinerary(itinerary: Dict[str, Any], carriers: Dict[str, str]) -> FlightLeg:
    segments_data = itinerary.get("segments", [])
    parsed_segments = []
    for idx, seg in enumerate(segments_data):
        layover_duration = None
        if idx > 0:
            prev_seg = segments_data[idx - 1]
            layover_duration = _calculate_layover(
                prev_seg["arrival"]["at"],
                seg["departure"]["at"]
            )
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
            duration=segment_duration,
            layover_duration=layover_duration
        )
        parsed_segments.append(segment)
    first_seg = segments_data[0]
    last_seg = segments_data[-1]
    leg_duration = _format_duration(itinerary.get("duration", ""))
    return FlightLeg(
        departure_airport=first_seg["departure"]["iataCode"],
        departure_time=first_seg["departure"]["at"],
        arrival_airport=last_seg["arrival"]["iataCode"],
        arrival_time=last_seg["arrival"]["at"],
        duration=leg_duration,
        stops=max(len(segments_data) - 1, 0),
        segments=parsed_segments
    )

def parse_flight_offers(data: Dict[str, Any]) -> List[Flight]:
    offers = data.get("data", [])
    if not offers:
        return []
    carriers = data.get("dictionaries", {}).get("carriers", {})
    results: List[Flight] = []
    for offer in offers:
        try:
            outbound = _parse_itinerary(offer["itineraries"][0], carriers)
            return_leg = None
            is_round_trip = len(offer["itineraries"]) > 1
            if is_round_trip:
                return_leg = _parse_itinerary(offer["itineraries"][1], carriers)
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
        currency=search_params.get('currency', 'USD'),
        user_intent=search_params.get('user_intent', 'find_best_value'),
        max_price=search_params.get('max_price')
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
            "flightOffers": priced_offer_data["data"]["flightOffers"],
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