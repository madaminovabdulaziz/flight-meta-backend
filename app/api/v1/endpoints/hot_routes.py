# app/api/v1/endpoints/hot_routes.py - Complete with Smart Currency
from fastapi import APIRouter, Depends, Query, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, desc
from typing import List, Optional
from datetime import datetime, timedelta
import httpx
import json
import logging
from calendar import monthrange
import calendar

from app.db.database import get_db
from app.models.models import Click, Airport, User
from app.core.config import settings
from services.amadeus_service import redis_client
from schemas.hot_routes import HotRouteResponse
from app.core.localization import get_user_preferences, LocalizationService
from app.api.v1.dependencies import get_current_user_optional

router = APIRouter()
logger = logging.getLogger(__name__)

CACHE_TTL_HOT_ROUTES = 3600
TRAVELPAYOUTS_API_URL = "https://api.travelpayouts.com/v2/prices/latest"
TRAVELPAYOUTS_CHEAP_URL = "https://api.travelpayouts.com/v2/prices/month-matrix"

POPULAR_INTERNATIONAL_DESTINATIONS = [
    "IST", "DXB", "SHJ", "DEL", "BOM", "KUL", "BKK", 
    "ICN", "NRT", "PEK", "MOW", "LED", "ALA", "FRU",
    "JFK", "LHR", "CDG", "FRA", "AMS", "BCN", "ROM"
]

DEFAULT_DESTINATION_IMAGE = "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=800&q=80"
# app/api/v1/endpoints/hot_routes.py

# --- REPLACE your existing DESTINATION_IMAGES dictionary with this one ---

DESTINATION_IMAGES = {
    # === Middle East & Turkey ===
    "IST": "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?w=400",  # Istanbul, Turkey
    "AYT": "https://images.unsplash.com/photo-1617431332082-5635f1f9e2af?w=400",  # Antalya, Turkey
    "DXB": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=400",  # Dubai, UAE
    "SHJ": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=400",  # Sharjah, UAE (using Dubai image)
    "AUH": "https://images.unsplash.com/photo-1503415397539-a9a7c64503a5?w=400",  # Abu Dhabi, UAE
    "DOH": "https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400",  # Doha, Qatar
    "JED": "https://images.unsplash.com/photo-1596422846543-75c6fc197f07?w=400",  # Jeddah, Saudi Arabia
    "RUH": "https://images.unsplash.com/photo-1587474260584-136574528ed5?w=400",  # Riyadh, Saudi Arabia
    "TLV": "https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=400",  # Tel Aviv, Israel

    # === Europe ===
    "LHR": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=400",  # London, UK
    "CDG": "https://images.unsplash.com/photo-1502602898657-3e91760c0337?w=400",  # Paris, France
    "AMS": "https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=400",  # Amsterdam, Netherlands
    "FRA": "https://images.unsplash.com/photo-1502103823424-959c9a826a38?w=400",  # Frankfurt, Germany
    "BCN": "https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=400",  # Barcelona, Spain
    "MAD": "https://images.unsplash.com/photo-1539037116277-4db20889f2d4?w=400",  # Madrid, Spain
    "FCO": "https://images.unsplash.com/photo-1529260830199-42c24129f196?w=400",  # Rome, Italy
    "ROM": "https://images.unsplash.com/photo-1529260830199-42c24129f196?w=400",  # Rome, Italy (generic)
    "DUB": "https://images.unsplash.com/photo-1553988983-ecc9f523d040?w=400",  # Dublin, Ireland
    "LIS": "https://images.unsplash.com/photo-1522228512835-26a45e54817c?w=400",  # Lisbon, Portugal
    "ATH": "https://images.unsplash.com/photo-1591129944900-84a5303a72ad?w=400",  # Athens, Greece
    "PRG": "https://images.unsplash.com/photo-1522525287612-6f25455ca8a2?w=400",  # Prague, Czech Republic
    "VIE": "https://images.unsplash.com/photo-1560840067-ddcaeb7831d2?w=400",  # Vienna, Austria
    "BUD": "https://images.unsplash.com/photo-1528114639423-4573883a8bcf?w=400",  # Budapest, Hungary

    # === CIS Region ===
    "MOW": "https://images.unsplash.com/photo-1513326738677-b964603b136d?w=400",  # Moscow, Russia
    "LED": "https://images.unsplash.com/photo-1556543961-ea5d8a2e8a02?w=400",  # St. Petersburg, Russia
    "ALA": "https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=400",  # Almaty, Kazakhstan
    "FRU": "https://images.unsplash.com/photo-1601625841499-01b34ec91c6e?w=400",  # Bishkek, Kyrgyzstan
    "TAS": "https://images.unsplash.com/photo-1596383648434-153364f378ae?w=400",  # Tashkent, Uzbekistan
    "GYD": "https://images.unsplash.com/photo-1634841923617-83688225a0b7?w=400",  # Baku, Azerbaijan
    
    # === Asia Pacific ===
    "BKK": "https://images.unsplash.com/photo-1508009603885-50cf7c579365?w=400",  # Bangkok, Thailand
    "KUL": "https://images.unsplash.com/photo-1596422846543-75c6fc197f07?w=400",  # Kuala Lumpur, Malaysia
    "SIN": "https://images.unsplash.com/photo-1525625293386-3f8f99389edd?w=400",  # Singapore
    "ICN": "https://images.unsplash.com/photo-1517154421773-0529f29ea451?w=400",  # Seoul, South Korea
    "NRT": "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?w=400",  # Tokyo, Japan
    "PEK": "https://images.unsplash.com/photo-1529989658210-a781b8548325?w=400",  # Beijing, China
    "PVG": "https://images.unsplash.com/photo-1538428331393-2d2d2a235242?w=400",  # Shanghai, China
    "HKG": "https://images.unsplash.com/photo-1513622790141-cf11e428787a?w=400",  # Hong Kong
    "DEL": "https://images.unsplash.com/photo-1587474260584-136574528ed5?w=400",  # Delhi, India
    "BOM": "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?w=400",  # Mumbai, India
    "DPS": "https://images.unsplash.com/photo-1537996194471-e657df97525p?w=400",  # Bali, Indonesia
    "SGN": "https://images.unsplash.com/photo-1559592233-10d63b2c15a7?w=400",  # Ho Chi Minh City, Vietnam

    # === Americas ===
    "JFK": "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=400",  # New York, USA
    "LAX": "https://images.unsplash.com/photo-1542065406-53817f5256e4?w=400",  # Los Angeles, USA
    "MIA": "https://images.unsplash.com/photo-1533106418989-88406e758cae?w=400",  # Miami, USA
    "YYZ": "https://images.unsplash.com/photo-1502444336025-108731b4586a?w=400",  # Toronto, Canada
    "CUN": "https://images.unsplash.com/photo-1580592354376-a12803362145?w=400",  # Cancun, Mexico
    "GRU": "https://images.unsplash.com/photo-1529260830199-42c24129f196?w=400",  # Sao Paulo, Brazil
}


async def cache_get(key: str):
    """Get from Redis cache"""
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data.decode('utf-8'))
    except Exception as e:
        logger.debug(f"Cache get failed: {e}")
        return None


async def cache_set(key: str, value: any, ttl: int):
    """Set in Redis cache"""
    try:
        await redis_client.set(
            key,
            json.dumps(value),
            ex=ttl
        )
    except Exception as e:
        logger.debug(f"Cache set failed: {e}")


async def fetch_popular_from_travelpayouts(
    origin: str = "TAS",
    limit: int = 10,
    period: str = "year"
) -> List[dict]:
    """Fetch popular routes from Travelpayouts (always in USD)"""
    try:
        aggregated_url = "https://api.travelpayouts.com/v1/prices/cheap"
        
        params = {
            "origin": origin.upper(),
            "currency": "USD",  # Always fetch in USD, convert later
            "token": settings.TRAVELPAYOUTS_API_TOKEN
        }
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            all_routes = []
            
            for destination in POPULAR_INTERNATIONAL_DESTINATIONS[:15]:
                dest_params = {**params, "destination": destination}
                
                try:
                    response = await client.get(aggregated_url, params=dest_params, timeout=5.0)
                    
                    if response.status_code == 200:
                        dest_data = response.json()
                        
                        if dest_data.get("success") and dest_data.get("data"):
                            flights = dest_data.get("data", {}).get(destination, [])
                            
                            if flights and len(flights) > 0:
                                cheapest = min(flights, key=lambda x: x.get("price", 999999))
                                
                                all_routes.append({
                                    "origin": origin.upper(),
                                    "destination": destination,
                                    "price": float(cheapest.get("price", 0)),
                                    "currency": "USD",
                                    "departure_date": cheapest.get("departure_at"),
                                    "return_date": cheapest.get("return_at"),
                                    "airline": cheapest.get("airline"),
                                    "flight_number": cheapest.get("flight_number"),
                                    "found_at": cheapest.get("created_at"),
                                    # FIX: Use destination-specific image
                                    "image_url": DESTINATION_IMAGES.get(
                                        destination, 
                                        "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400"
                                    )
                                })
                except Exception as e:
                    logger.debug(f"Skipped {destination}: {e}")
                    continue
            
            if all_routes:
                all_routes.sort(key=lambda x: x["price"])
                logger.info(f"Fetched {len(all_routes)} routes from Travelpayouts")
                return all_routes[:limit]
            
            # Fallback to latest prices endpoint
            logger.info("Falling back to latest prices endpoint")
            params = {
                "origin": origin.upper(),
                "currency": "USD",
                "limit": 50,
                "page": 1,
                "sorting": "price",
                "token": settings.TRAVELPAYOUTS_API_TOKEN
            }
            
            response = await client.get(TRAVELPAYOUTS_API_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success", False):
                logger.warning(f"Travelpayouts API returned success=false")
                return []
            
            routes = []
            seen_destinations = set()
            
            for item in data.get("data", []):
                destination = item.get("destination")
                
                if destination in seen_destinations:
                    continue
                
                if destination not in POPULAR_INTERNATIONAL_DESTINATIONS:
                    continue
                
                seen_destinations.add(destination)
                
                routes.append({
                    "origin": origin.upper(),
                    "destination": destination,
                    "price": float(item.get("value", 0)),
                    "currency": "USD",
                    "departure_date": item.get("depart_date"),
                    "return_date": item.get("return_date"),
                    "airline": item.get("airline"),
                    "flight_number": item.get("flight_number"),
                    "found_at": item.get("found_at"),
                    # FIX: Use destination-specific image here too!
                    "image_url": DESTINATION_IMAGES.get(
                        destination, 
                        "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400"
                    )
                })
                
                if len(routes) >= limit:
                    break
            
            logger.info(f"Fetched {len(routes)} routes (fallback)")
            return routes
            
    except httpx.HTTPError as e:
        logger.error(f"Travelpayouts API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching from Travelpayouts: {e}")
        return []

async def enrich_with_airport_data(routes: List[dict], db: AsyncSession) -> List[dict]:
    """Add airport names and countries"""
    if not routes:
        return []
    
    airport_codes = set()
    for route in routes:
        airport_codes.add(route["origin"])
        airport_codes.add(route["destination"])
    
    result = await db.execute(
        select(Airport)
        .where(Airport.iata_code.in_(list(airport_codes)))
    )
    airports = {a.iata_code: a for a in result.scalars().all()}
    
    enriched_routes = []
    for route in routes:
        origin_airport = airports.get(route["origin"])
        dest_airport = airports.get(route["destination"])
        
        enriched_routes.append({
            **route,
            "origin_city": origin_airport.city if origin_airport else route["origin"],
            "origin_country": origin_airport.country if origin_airport else "Unknown",
            "destination_city": dest_airport.city if dest_airport else route["destination"],
            "destination_country": dest_airport.country if dest_airport else "Unknown",
        })
    
    return enriched_routes


@router.get("/", response_model=List[HotRouteResponse])
async def get_hot_routes(
    request: Request,
    origin: str = Query("TAS", min_length=3, max_length=3),
    limit: int = Query(8, ge=1, le=20),
    period: str = Query("year", regex="^(year|month)$"),
    currency: Optional[str] = Query(None, description="Override currency (USD, UZS, EUR, etc.)"),
    language: Optional[str] = Query(None, description="Override language (en, uz, ru, tr)"),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get hot/popular INTERNATIONAL flight routes with smart currency conversion.
    
    **Smart Features:**
    - 🌍 Auto-detects currency from user location (IP-based)
    - 💰 Converts all prices to user's preferred currency
    - 👤 Uses authenticated user's saved preferences
    - 🔄 Supports explicit currency override
    
    **Currency Detection Priority:**
    1. Query parameter (?currency=UZS)
    2. Authenticated user's preference (from database)
    3. Auto-detected from IP location
    4. System default (UZS for Uzbekistan)
    
    **Supported Currencies:**
    USD, UZS, EUR, RUB, TRY, GBP, AED, KRW, INR, CNY
    """
    origin = origin.upper()
    
    # Get user preferences with smart detection
    prefs = await get_user_preferences(
        request, 
        currency, 
        language,
        current_user=current_user  # Pass authenticated user
    )
    user_currency = prefs["currency"]
    user_language = prefs["language"]
    
    logger.info(f"🌍 User preferences: {user_currency} / {user_language}")
    
    # Cache key includes currency for separate caching
    cache_key = f"hot_routes_intl:{origin}:{limit}:{period}:{user_currency}"
    cache_ttl = 10800 if period == "year" else 3600
    
    cached = await cache_get(cache_key)
    if cached:
        logger.info(f"✅ CACHE HIT - Hot routes in {user_currency}")
        return [HotRouteResponse(**item) for item in cached]
    
    logger.info(f"❌ CACHE MISS - Fetching hot routes for {origin} in {user_currency}")
    
    # Fetch routes (always in USD from API)
    routes = await fetch_popular_from_travelpayouts(origin, limit, period)
    
    if not routes:
        routes = get_fallback_routes(origin)
    
    # Enrich with airport data
    routes = await enrich_with_airport_data(routes, db)
    
    # Convert prices to user's currency
    if user_currency != "USD":
        for route in routes:
            route["price"] = await LocalizationService.convert_price(
                route["price"],
                "USD",
                user_currency
            )
            route["currency"] = user_currency
            route["original_currency"] = "USD"
    
    # Sort by converted price
    routes.sort(key=lambda x: x.get("price", 999999))
    routes = routes[:limit]
    
    # Add currency symbol to response
    for route in routes:
        route["currency_symbol"] = prefs["currency_symbol"]
    
    # Cache converted results
    await cache_set(cache_key, routes, cache_ttl)
    
    logger.info(f"📊 Returning {len(routes)} routes in {user_currency}")
    
    return [HotRouteResponse(**route) for route in routes]


# @router.get("/cheap-flights")
# async def get_cheap_flights(
#     request: Request,
#     origin: str = Query(..., description="Origin airport code"),
#     destination: str = Query(..., description="Destination airport code"),
#     currency: Optional[str] = Query(None),
#     language: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     current_user: Optional[User] = Depends(get_current_user_optional)
# ):
#     """
#     Get cheapest flights with smart currency conversion.
#     Returns price calendar for next 30-60 days in user's preferred currency.
#     """
#     origin = origin.upper()
#     destination = destination.upper()
    
#     # Get user preferences
#     prefs = await get_user_preferences(request, currency, language, current_user)
#     user_currency = prefs["currency"]
    
#     cache_key = f"cheap_flights:{origin}:{destination}:{user_currency}"
    
#     cached = await cache_get(cache_key)
#     if cached:
#         logger.info(f"✅ CACHE HIT - Cheap flights in {user_currency}")
#         return cached
    
#     logger.info(f"❌ CACHE MISS - Fetching cheap flights")
    
#     try:
#         current_month = datetime.now().strftime("%Y-%m")
#         next_month = (datetime.now().replace(day=1) + timedelta(days=32)).strftime("%Y-%m")
        
#         all_prices = []
        
#         async with httpx.AsyncClient(timeout=15.0) as client:
#             for month in [current_month, next_month]:
#                 params = {
#                     "origin": origin,
#                     "destination": destination,
#                     "month": month,
#                     "currency": "USD",  # Always fetch in USD
#                     "token": settings.TRAVELPAYOUTS_API_TOKEN
#                 }
                
#                 try:
#                     response = await client.get(TRAVELPAYOUTS_CHEAP_URL, params=params, timeout=5.0)
#                     response.raise_for_status()
                    
#                     data = response.json()
                    
#                     if isinstance(data, dict):
#                         if data.get("success") == False:
#                             continue
#                         prices_data = data.get("data", [])
#                     elif isinstance(data, list):
#                         prices_data = data
#                     else:
#                         continue
                    
#                     if isinstance(prices_data, list):
#                         for item in prices_data:
#                             if not item.get("show_to_affiliates", True):
#                                 continue
                            
#                             all_prices.append({
#                                 "date": item.get("depart_date"),
#                                 "price": float(item.get("value", 0)),
#                                 "currency": "USD",
#                                 "return_date": item.get("return_date"),
#                                 "airline": item.get("airline"),
#                                 "flight_number": item.get("flight_number"),
#                                 "number_of_changes": item.get("number_of_changes", 0)
#                             })
#                     elif isinstance(prices_data, dict):
#                         for date_str, price_data in prices_data.items():
#                             all_prices.append({
#                                 "date": date_str,
#                                 "price": float(price_data.get("value", 0)),
#                                 "currency": "USD",
#                                 "return_date": price_data.get("return_date"),
#                                 "airline": price_data.get("airline"),
#                                 "flight_number": price_data.get("flight_number"),
#                                 "number_of_changes": price_data.get("number_of_changes", 0)
#                             })
                    
#                 except Exception as e:
#                     logger.warning(f"Failed to fetch prices for {month}: {e}")
#                     continue
        
#         if not all_prices:
#             raise HTTPException(
#                 status_code=404, 
#                 detail=f"No price data found for {origin}->{destination}"
#             )
        
#         # Remove duplicates
#         seen_dates = set()
#         unique_prices = []
#         for price in all_prices:
#             if price["date"] not in seen_dates:
#                 seen_dates.add(price["date"])
#                 unique_prices.append(price)
        
#         unique_prices.sort(key=lambda x: x["date"])
        
#         # Convert prices to user currency
#         if user_currency != "USD":
#             for price in unique_prices:
#                 price["price"] = await LocalizationService.convert_price(
#                     price["price"],
#                     "USD",
#                     user_currency
#                 )
#                 price["currency"] = user_currency
        
#         # Build result with converted prices
#         cheapest = min(unique_prices, key=lambda x: x["price"]) if unique_prices else None
#         average = round(sum(p["price"] for p in unique_prices) / len(unique_prices), 2) if unique_prices else 0
        
#         result = {
#             "origin": origin,
#             "destination": destination,
#             "prices": unique_prices,
#             "cheapest": cheapest,
#             "average": average,
#             "count": len(unique_prices),
#             "currency": user_currency,
#             "currency_symbol": prefs["currency_symbol"]
#         }
        
#         # Add airport info
#         airport_result = await db.execute(
#             select(Airport)
#             .where(Airport.iata_code.in_([origin, destination]))
#         )
#         airports = {a.iata_code: a for a in airport_result.scalars().all()}
        
#         if origin in airports:
#             result["origin_city"] = airports[origin].city
#             result["origin_country"] = airports[origin].country
#         if destination in airports:
#             result["destination_city"] = airports[destination].city
#             result["destination_country"] = airports[destination].country
        
#         # Cache for 6 hours
#         await cache_set(cache_key, result, 21600)
        
#         logger.info(f"📊 Returning {len(unique_prices)} price points in {user_currency}")
        
#         return result
        
#     except HTTPException:
#         raise
#     except httpx.HTTPError as e:
#         logger.error(f"Travelpayouts API error: {e}")
#         raise HTTPException(status_code=502, detail="Failed to fetch flight prices")
#     except Exception as e:
#         logger.error(f"Error fetching cheap flights: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# @router.get("/month-calendar")
# async def get_month_price_calendar(
#     request: Request,
#     origin: str = Query(..., min_length=3, max_length=3, regex="^[A-Z]{3}$"),
#     destination: str = Query(..., min_length=3, max_length=3, regex="^[A-Z]{3}$"),
#     month: str = Query(..., regex="^20[0-9]{2}-(0[1-9]|1[0-2])$", example="2025-10"),
#     currency: Optional[str] = Query(None),
#     language: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     current_user: Optional[User] = Depends(get_current_user_optional)
# ):
#     """
#     Month price calendar with smart currency conversion.
#     Auto-detects user location and shows prices in local currency.
#     """
#     origin = origin.upper()
#     destination = destination.upper()
    
#     # Get user preferences
#     prefs = await get_user_preferences(request, currency, language, current_user)
#     user_currency = prefs["currency"]
    
#     # Validation
#     if origin == destination:
#         raise HTTPException(status_code=400, detail="Origin and destination must be different")
    
#     try:
#         target_date = datetime.strptime(month, "%Y-%m")
#     except ValueError:
#         raise HTTPException(status_code=400, detail="Invalid month format. Use YYYY-MM")
    
#     current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#     if target_date < current_month:
#         raise HTTPException(status_code=400, detail=f"Cannot fetch past months")
    
#     max_future_date = current_month + timedelta(days=365)
#     if target_date > max_future_date:
#         raise HTTPException(status_code=400, detail=f"Max 12 months ahead")
    
#     # Validate airports
#     airport_result = await db.execute(
#         select(Airport)
#         .where(Airport.iata_code.in_([origin, destination]))
#         .where(Airport.is_active == True)
#     )
#     airports = {a.iata_code: a for a in airport_result.scalars().all()}
    
#     if origin not in airports:
#         raise HTTPException(status_code=404, detail=f"Airport '{origin}' not found")
    
#     if destination not in airports:
#         raise HTTPException(status_code=404, detail=f"Airport '{destination}' not found")
    
#     # Cache key includes currency
#     cache_key = f"month_calendar:{origin}:{destination}:{month}:{user_currency}"
    
#     cached = await cache_get(cache_key)
#     if cached:
#         logger.info(f"✅ CACHE HIT - Month calendar in {user_currency}")
#         return cached
    
#     logger.info(f"❌ CACHE MISS - Fetching month calendar")
    
#     try:
#         params = {
#             "origin": origin,
#             "destination": destination,
#             "month": month,
#             "currency": "USD",  # Always fetch in USD
#             "token": settings.TRAVELPAYOUTS_API_TOKEN
#         }
        
#         async with httpx.AsyncClient(timeout=15.0) as client:
#             response = await client.get(TRAVELPAYOUTS_CHEAP_URL, params=params)
#             response.raise_for_status()
            
#             data = response.json()
            
#             prices_raw = []
#             if isinstance(data, dict):
#                 if data.get("success") == False:
#                     raise HTTPException(status_code=502, detail="No price data available")
#                 prices_raw = data.get("data", [])
#             elif isinstance(data, list):
#                 prices_raw = data
            
#             # Parse prices
#             prices_by_date = {}
            
#             if isinstance(prices_raw, list):
#                 for item in prices_raw:
#                     if not item.get("show_to_affiliates", True):
#                         continue
                    
#                     date_str = item.get("depart_date")
#                     if date_str:
#                         prices_by_date[date_str] = {
#                             "price": float(item.get("value", 0)),
#                             "currency": "USD",
#                             "return_date": item.get("return_date"),
#                             "airline": item.get("airline"),
#                             "flight_number": item.get("flight_number"),
#                             "number_of_changes": item.get("number_of_changes", 0),
#                             "found_at": item.get("found_at")
#                         }
            
#             elif isinstance(prices_raw, dict):
#                 for date_str, price_data in prices_raw.items():
#                     prices_by_date[date_str] = {
#                         "price": float(price_data.get("value", 0)),
#                         "currency": "USD",
#                         "return_date": price_data.get("return_date"),
#                         "airline": price_data.get("airline"),
#                         "flight_number": price_data.get("flight_number"),
#                         "number_of_changes": price_data.get("number_of_changes", 0),
#                         "found_at": price_data.get("found_at")
#                     }
            
#             if not prices_by_date:
#                 raise HTTPException(status_code=404, detail=f"No flights found for {origin}->{destination}")
            
#             # Convert all prices to user currency
#             if user_currency != "USD":
#                 for date_str, price_data in prices_by_date.items():
#                     price_data["price"] = await LocalizationService.convert_price(
#                         price_data["price"],
#                         "USD",
#                         user_currency
#                     )
#                     price_data["currency"] = user_currency
            
#             # Build calendar structure
#             year = target_date.year
#             month_num = target_date.month
#             num_days = monthrange(year, month_num)[1]
#             first_day = datetime(year, month_num, 1)
#             first_weekday = first_day.weekday()
            
#             calendar_weeks = []
#             current_week = []
#             week_number = 1
            
#             for _ in range(first_weekday):
#                 current_week.append(None)
            
#             for day in range(1, num_days + 1):
#                 date_obj = datetime(year, month_num, day)
#                 date_str = date_obj.strftime("%Y-%m-%d")
#                 day_name = calendar.day_name[date_obj.weekday()][:3]
                
#                 price_data = prices_by_date.get(date_str)
                
#                 day_info = {
#                     "date": date_str,
#                     "day": day,
#                     "day_name": day_name,
#                     "is_weekend": date_obj.weekday() >= 5,
#                     "is_past": date_obj < datetime.now(),
#                     "has_price": price_data is not None
#                 }
                
#                 if price_data:
#                     day_info.update(price_data)
#                 else:
#                     day_info.update({
#                         "price": None,
#                         "currency": user_currency,
#                         "airline": None
#                     })
                
#                 current_week.append(day_info)
                
#                 if len(current_week) == 7:
#                     calendar_weeks.append({
#                         "week": week_number,
#                         "days": current_week
#                     })
#                     current_week = []
#                     week_number += 1
            
#             if current_week:
#                 while len(current_week) < 7:
#                     current_week.append(None)
                
#                 calendar_weeks.append({
#                     "week": week_number,
#                     "days": current_week
#                 })
            
#             # Calculate statistics
#             all_prices = [p["price"] for p in prices_by_date.values()]
            
#             if all_prices:
#                 statistics = {
#                     "cheapest_price": min(all_prices),
#                     "average_price": round(sum(all_prices) / len(all_prices), 2),
#                     "most_expensive_price": max(all_prices),
#                     "median_price": round(sorted(all_prices)[len(all_prices) // 2], 2),
#                     "total_days_with_prices": len(all_prices),
#                     "total_days_in_month": num_days
#                 }
                
#                 cheapest_date = min(prices_by_date.items(), key=lambda x: x[1]["price"])
#                 expensive_date = max(prices_by_date.items(), key=lambda x: x[1]["price"])
                
#                 statistics["cheapest_date"] = {
#                     "date": cheapest_date[0],
#                     "price": cheapest_date[1]["price"],
#                     "day_name": datetime.strptime(cheapest_date[0], "%Y-%m-%d").strftime("%A")
#                 }
                
#                 statistics["most_expensive_date"] = {
#                     "date": expensive_date[0],
#                     "price": expensive_date[1]["price"],
#                     "day_name": datetime.strptime(expensive_date[0], "%Y-%m-%d").strftime("%A")
#                 }
#             else:
#                 statistics = {
#                     "cheapest_price": None,
#                     "average_price": None,
#                     "most_expensive_price": None,
#                     "total_days_with_prices": 0,
#                     "total_days_in_month": num_days
#                 }
            
#             result = {
#                 "origin": origin,
#                 "destination": destination,
#                 "origin_city": airports[origin].city,
#                 "origin_country": airports[origin].country,
#                 "destination_city": airports[destination].city,
#                 "destination_country": airports[destination].country,
#                 "month": month,
#                 "month_name": calendar.month_name[month_num],
#                 "year": year,
#                 "calendar": calendar_weeks,
#                 "statistics": statistics,
#                 "currency": user_currency,
#                 "currency_symbol": prefs["currency_symbol"]
#             }
            
#             await cache_set(cache_key, result, 21600)
            
#             logger.info(f"📅 Returning month calendar in {user_currency}")
            
#             return result
            
#     except HTTPException:
#         raise
#     except httpx.HTTPError as e:
#         logger.error(f"Travelpayouts API error: {e}")
#         raise HTTPException(status_code=502, detail="Failed to fetch price data")
#     except Exception as e:
#         logger.error(f"Error fetching month calendar: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def get_fallback_routes(origin: str = "TAS") -> List[dict]:
    """Fallback data for popular routes"""
    fallback = {
        "TAS": [
            {
                "origin": "TAS", 
                "destination": "IST", 
                "price": 245.0, 
                "currency": "USD", 
                "airline": "TK",
                "image_url": DESTINATION_IMAGES.get("IST")
            },
            {
                "origin": "TAS", 
                "destination": "DXB", 
                "price": 298.0, 
                "currency": "USD", 
                "airline": "FZ",
                "image_url": DESTINATION_IMAGES.get("DXB")
            },
            {
                "origin": "TAS", 
                "destination": "MOW", 
                "price": 189.0, 
                "currency": "USD", 
                "airline": "HY",
                "image_url": DESTINATION_IMAGES.get("MOW")
            },
            {
                "origin": "TAS", 
                "destination": "ICN", 
                "price": 425.0, 
                "currency": "USD", 
                "airline": "KE",
                "image_url": DESTINATION_IMAGES.get("ICN")
            },
            {
                "origin": "TAS", 
                "destination": "DEL", 
                "price": 215.0, 
                "currency": "USD", 
                "airline": "HY",
                "image_url": DESTINATION_IMAGES.get("DEL")
            },
            {
                "origin": "TAS", 
                "destination": "BKK", 
                "price": 420.0, 
                "currency": "USD", 
                "airline": "HY",
                "image_url": DESTINATION_IMAGES.get("BKK")
            },
            {
                "origin": "TAS", 
                "destination": "KUL", 
                "price": 380.0, 
                "currency": "USD", 
                "airline": "HY",
                "image_url": DESTINATION_IMAGES.get("KUL")
            },
            {
                "origin": "TAS", 
                "destination": "ALA", 
                "price": 89.0, 
                "currency": "USD", 
                "airline": "HY",
                "image_url": DESTINATION_IMAGES.get("ALA")
            },
        ]
    }
    
    return fallback.get(origin.upper(), fallback["TAS"])



# # app/api/v1/endpoints/hot_routes.py - ENHANCED VERSION
# # Add this NEW endpoint to your existing hot_routes.py
# # app/api/v1/endpoints/hot_routes.py - ENHANCED VERSION
# # Add this NEW endpoint to your existing hot_routes.py

# @router.get("/popular-directions", response_model=List[HotRouteResponse])
# async def get_popular_directions_v3(
#     request: Request,
#     destination: str = Query(..., min_length=3, max_length=3, description="Destination IATA code (e.g., 'IST', 'DXB')"),
#     limit: int = Query(10, ge=1, le=30),
#     page: int = Query(1, ge=1),
#     currency: Optional[str] = Query(None),
#     language: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     current_user: Optional[User] = Depends(get_current_user_optional)
# ):
#     """
#     🔥 **NEW** Popular Directions to a Destination (TravelPayouts v3 API)
    
#     **What Makes This Different from `/hot-routes/`?**
#     - This endpoint answers: "Where do people fly TO Istanbul FROM?"
#     - Your existing `/hot-routes/` answers: "Where can I fly FROM Tashkent TO?"
    
#     **Use Cases:**
#     - Reverse search: Find popular origins for a destination
#     - Tourism insights: See where visitors come from
#     - Marketing: Target users from high-traffic origins
#     - Alternative airports: Suggest nearby departure points
    
#     **Why This API is Better:**
#     ✅ Based on REAL user behavior (searches + bookings)
#     ✅ TravelPayouts' ML models (not your manual list)
#     ✅ Single API call (vs 15+ calls in your current method)
#     ✅ Auto-updates as trends change
    
#     **Smart Features:**
#     - 🌍 Auto-detects user currency (IP-based)
#     - 💰 Converts prices to local currency
#     - 📊 Returns origin cities ranked by popularity
#     - 🔄 Redis cached for 3 hours
    
#     **Example:**
#     ```
#     GET /hot-routes/popular-directions?destination=IST&limit=20&currency=EUR
    
#     Response: Top 20 cities people fly FROM to Istanbul
#     ```
#     """
#     destination = destination.upper()
    
#     # Get user preferences
#     prefs = await get_user_preferences(request, currency, language, current_user)
#     user_currency = prefs["currency"]
#     user_language = prefs["language"]
    
#     logger.info(f"🎯 Popular directions to {destination} in {user_currency}")
    
#     # Cache key includes destination + currency
#     cache_key = f"popular_directions_v3:{destination}:{limit}:{page}:{user_currency}"
#     cache_ttl = 10800  # 3 hours (data updates slowly)
    
#     cached = await cache_get(cache_key)
#     if cached:
#         logger.info(f"✅ CACHE HIT - Popular directions to {destination}")
#         return [HotRouteResponse(**item) for item in cached]
    
#     logger.info(f"❌ CACHE MISS - Fetching popular directions to {destination}")
    
#     # Call TravelPayouts v3 API
#     try:
#         api_url = "https://api.travelpayouts.com/aviasales/v3/get_popular_directions"
        
#         params = {
#             "destination": destination,
#             "locale": user_language,
#             "currency": "USD",  # Always fetch in USD, convert later
#             "limit": limit,
#             "page": page,
#             "token": settings.TRAVELPAYOUTS_API_TOKEN
#         }
        
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             response = await client.get(api_url, params=params)
#             response.raise_for_status()
            
#             data = response.json()
            
#             # Validate response
#             if not data.get("success"):
#                 logger.warning(f"API returned success=false for {destination}")
#                 raise HTTPException(
#                     status_code=502,
#                     detail=f"No popular directions found for {destination}"
#                 )
            
#             # Extract routes from response
#             routes = []
#             api_data = data.get("data", {})
            
#             destination_info = api_data.get("destination", {})
#             origins = api_data.get("origin", [])
            
#             if not origins:
#                 logger.warning(f"No origin cities found for {destination}")
#                 # Return empty but don't fail
#                 return []
            
#             # Parse each origin route
#             for origin_data in origins:
#                 # Parse dates properly - convert empty strings to None
#                 departure_date = origin_data.get("departure_at")
#                 return_date = origin_data.get("return_at")
                
#                 # Convert empty strings to None for Pydantic validation
#                 if not departure_date or departure_date == "":
#                     departure_date = None
#                 if not return_date or return_date == "":
#                     return_date = None
                
#                 route = {
#                     "origin": origin_data.get("city_iata"),
#                     "destination": destination,
#                     "origin_city": origin_data.get("city_name"),
#                     "destination_city": destination_info.get("city_name"),
#                     "destination_country": destination_info.get("country_name"),
#                     "price": float(origin_data.get("price", 0)),
#                     "currency": "USD",
#                     "departure_date": departure_date,
#                     "return_date": return_date,
#                     "image_url": DESTINATION_IMAGES.get(destination, 
#                         "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400")
#                 }
                
#                 routes.append(route)
            
#             logger.info(f"✅ Fetched {len(routes)} popular origins for {destination}")
            
#             # Enrich with full airport data
#             routes = await enrich_with_airport_data(routes, db)
            
#             # Convert prices to user currency
#             if user_currency != "USD":
#                 for route in routes:
#                     route["price"] = await LocalizationService.convert_price(
#                         route["price"],
#                         "USD",
#                         user_currency
#                     )
#                     route["currency"] = user_currency
            
#             # Add currency symbol
#             for route in routes:
#                 route["currency_symbol"] = prefs["currency_symbol"]
            
#             # Sort by price (already sorted by API, but ensure)
#             routes.sort(key=lambda x: x.get("price", 999999))
            
#             # Cache the results
#             await cache_set(cache_key, routes, cache_ttl)
            
#             logger.info(f"📊 Returning {len(routes)} routes in {user_currency}")
            
#             return [HotRouteResponse(**route) for route in routes]
    
#     except httpx.TimeoutException:
#         logger.error(f"Timeout fetching popular directions for {destination}")
#         raise HTTPException(
#             status_code=504,
#             detail="Request timed out"
#         )
    
#     except httpx.HTTPStatusError as e:
#         logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=f"TravelPayouts API error: {e.response.text}"
#         )
    
#     except Exception as e:
#         logger.error(f"Error fetching popular directions: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to fetch popular directions: {str(e)}"
#         )


# @router.get("/destination-insights/{iata_code}")
# async def get_destination_insights(
#     request: Request,
#     iata_code: str,
#     currency: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     current_user: Optional[User] = Depends(get_current_user_optional)
# ):
#     """
#     📊 Destination Insights Dashboard
    
#     Combines multiple data sources for a destination:
#     1. Popular origin cities (where people fly FROM)
#     2. Average prices
#     3. Best time to fly
#     4. Trending status
    
#     Perfect for:
#     - Destination landing pages
#     - Travel planning UI
#     - Marketing analytics
    
#     **Example:**
#     ```
#     GET /hot-routes/destination-insights/IST?currency=USD
    
#     Returns comprehensive data about Istanbul as a destination
#     ```
#     """
#     iata_code = iata_code.upper()
    
#     # Get user preferences
#     prefs = await get_user_preferences(request, currency, None, current_user)
#     user_currency = prefs["currency"]
    
#     cache_key = f"destination_insights:{iata_code}:{user_currency}"
    
#     cached = await cache_get(cache_key)
#     if cached:
#         return cached
    
#     try:
#         # Fetch popular directions
#         api_url = "https://api.travelpayouts.com/aviasales/v3/get_popular_directions"
        
#         params = {
#             "destination": iata_code,
#             "locale": "en",
#             "currency": "USD",
#             "limit": 30,
#             "page": 1,
#             "token": settings.TRAVELPAYOUTS_API_TOKEN
#         }
        
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             response = await client.get(api_url, params=params)
#             response.raise_for_status()
#             data = response.json()
        
#         if not data.get("success"):
#             raise HTTPException(status_code=404, detail="Destination not found")
        
#         # Get airport info
#         airport_result = await db.execute(
#             select(Airport).where(Airport.iata_code == iata_code)
#         )
#         airport = airport_result.scalar_one_or_none()
        
#         if not airport:
#             raise HTTPException(status_code=404, detail="Airport not found in database")
        
#         # Parse data
#         destination_info = data.get("data", {}).get("destination", {})
#         origins = data.get("data", {}).get("origin", [])
        
#         # Calculate insights
#         prices = [float(o.get("price", 0)) for o in origins if o.get("price")]
        
#         insights = {
#             "destination": {
#                 "iata": iata_code,
#                 "city": destination_info.get("city_name", airport.city),
#                 "country": destination_info.get("country_name", airport.country),
#                 "airport_name": airport.name,
#                 "timezone": airport.timezone
#             },
#             "statistics": {
#                 "total_origin_cities": len(origins),
#                 "average_price": round(sum(prices) / len(prices), 2) if prices else None,
#                 "cheapest_price": min(prices) if prices else None,
#                 "most_expensive_price": max(prices) if prices else None,
#                 "currency": "USD"
#             },
#             "top_origin_cities": [
#                 {
#                     "city": o.get("city_name"),
#                     "iata": o.get("city_iata"),
#                     "price": float(o.get("price", 0)),
#                     "departure_date": o.get("departure_at")
#                 }
#                 for o in origins[:10]
#             ]
#         }
        
#         # Convert prices to user currency
#         if user_currency != "USD":
#             for key in ["average_price", "cheapest_price", "most_expensive_price"]:
#                 if insights["statistics"][key]:
#                     insights["statistics"][key] = await LocalizationService.convert_price(
#                         insights["statistics"][key],
#                         "USD",
#                         user_currency
#                     )
            
#             for city in insights["top_origin_cities"]:
#                 city["price"] = await LocalizationService.convert_price(
#                     city["price"],
#                     "USD",
#                     user_currency
#                 )
            
#             insights["statistics"]["currency"] = user_currency
        
#         insights["currency_symbol"] = prefs["currency_symbol"]
        
#         # Cache for 6 hours
#         await cache_set(cache_key, insights, 21600)
        
#         return insights
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching destination insights: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
    


# # Add to your hot_routes.py or create a new flights_v3.py

# @router.get("/prices-for-dates")
# async def get_prices_for_specific_dates(
#     request: Request,
#     origin: str = Query(..., min_length=3, max_length=3, description="Origin IATA code"),
#     destination: str = Query(..., min_length=3, max_length=3, description="Destination IATA code"),
#     departure_at: str = Query(..., description="Departure date (YYYY-MM or YYYY-MM-DD)"),
#     return_at: Optional[str] = Query(None, description="Return date (YYYY-MM or YYYY-MM-DD), omit for one-way"),
#     one_way: bool = Query(True, description="One-way ticket (true) or round-trip (false)"),
#     direct: bool = Query(False, description="Direct flights only"),
#     sorting: str = Query("price", regex="^(price|route)$", description="Sort by: price or route popularity"),
#     unique: bool = Query(False, description="Return only unique routes"),
#     limit: int = Query(30, ge=1, le=1000, description="Results per page (max 1000)"),
#     page: int = Query(1, ge=1, description="Page number for pagination"),
#     currency: Optional[str] = Query(None),
#     language: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     current_user: Optional[User] = Depends(get_current_user_optional)
# ):
#     """
#     🎯 **NEW V3 API** - Flight Prices for Specific Dates
    
#     **The Modern Replacement for Old Endpoints**
    
#     Returns the cheapest tickets found by Aviasales users in the **last 48 hours**.
#     This is the most up-to-date price data available from TravelPayouts.
    
#     **What Makes This Better:**
#     - ✅ Fresher data (last 48h vs 30 days)
#     - ✅ Unified endpoint (replaces 4 old APIs)
#     - ✅ More filtering options (direct, unique, sorting)
#     - ✅ Better pagination (up to 1000 results)
#     - ✅ Route popularity sorting
    
#     **Replaces These Old Endpoints:**
#     - `/v1/prices/cheap` → Use `sorting=price`
#     - `/v1/prices/direct` → Use `direct=true`
#     - `/v1/city-directions` → Use `sorting=route&unique=true`
#     - `/v2/prices/latest` → This is the newer version
    
#     **Use Cases:**
#     1. **Specific Date Search**: "Flights on July 28"
#        - `departure_at=2025-07-28`
    
#     2. **Month Flexibility**: "Flights in July"
#        - `departure_at=2025-07`
    
#     3. **Direct Flights Only**: "Non-stop to Dubai"
#        - `direct=true`
    
#     4. **Popular Routes**: "Most searched routes from TAS"
#        - `sorting=route&unique=true`
    
#     5. **Round-trip**: "Return flights"
#        - `one_way=false&return_at=2025-08-15`
    
#     **Smart Features:**
#     - 🌍 Auto-detects user currency
#     - 💰 Converts all prices to local currency
#     - 📊 Includes flight duration and transfer info
#     - 🔗 Returns direct booking links (affiliate)
#     - ⚡ Cached for 30 minutes (prices refresh)
    
#     **Response Fields:**
#     - `price` - Ticket price in user's currency
#     - `airline` - Airline IATA code (e.g., "TK")
#     - `flight_number` - Flight number
#     - `departure_at` - Departure datetime (ISO 8601)
#     - `return_at` - Return datetime (if round-trip)
#     - `transfers` - Number of stops outbound
#     - `return_transfers` - Number of stops return
#     - `duration` - Total trip duration (minutes)
#     - `duration_to` - Outbound duration (minutes)
#     - `duration_back` - Return duration (minutes)
#     - `link` - Deep link to booking page
    
#     **Example Queries:**
#     ```
#     # One-way to Istanbul in July
#     GET /prices-for-dates?origin=TAS&destination=IST&departure_at=2025-07&one_way=true
    
#     # Round-trip direct flights
#     GET /prices-for-dates?origin=TAS&destination=DXB&departure_at=2025-07-15&return_at=2025-07-22&direct=true&one_way=false
    
#     # Popular routes from Tashkent
#     GET /prices-for-dates?origin=TAS&sorting=route&unique=true&limit=50
#     ```
#     """
#     origin = origin.upper()
#     destination = destination.upper()
    
#     # Get user preferences
#     prefs = await get_user_preferences(request, currency, language, current_user)
#     user_currency = prefs["currency"]
#     user_language = prefs["language"]
    
#     logger.info(f"🎯 V3 Price search: {origin}→{destination} on {departure_at} in {user_currency}")
    
#     # Build cache key
#     cache_key = f"prices_v3:{origin}:{destination}:{departure_at}:{return_at}:{one_way}:{direct}:{sorting}:{unique}:{limit}:{page}:{user_currency}"
#     cache_ttl = 1800  # 30 minutes (prices refresh every 30-60 min)
    
#     cached = await cache_get(cache_key)
#     if cached:
#         logger.info(f"✅ CACHE HIT - V3 prices")
#         return cached
    
#     logger.info(f"❌ CACHE MISS - Fetching V3 prices")
    
#     try:
#         # Call TravelPayouts V3 API
#         api_url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
        
#         params = {
#             "origin": origin,
#             "destination": destination,
#             "departure_at": departure_at,
#             "one_way": str(one_way).lower(),
#             "direct": str(direct).lower(),
#             "sorting": sorting,
#             "unique": str(unique).lower(),
#             "currency": "USD",  # Always fetch in USD, convert later
#             "limit": limit,
#             "page": page,
#             "token": settings.TRAVELPAYOUTS_API_TOKEN
#         }
        
#         # Add return date if round-trip
#         if return_at and not one_way:
#             params["return_at"] = return_at
        
#         async with httpx.AsyncClient(timeout=15.0) as client:
#             response = await client.get(api_url, params=params)
#             response.raise_for_status()
            
#             data = response.json()
            
#             # Validate response
#             if not data.get("success"):
#                 logger.warning(f"V3 API returned success=false")
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"No flights found for {origin}→{destination} on {departure_at}"
#                 )
            
#             flights = data.get("data", [])
            
#             if not flights:
#                 logger.info(f"No results for {origin}→{destination}")
#                 return []
            
#             logger.info(f"✅ Found {len(flights)} flights from V3 API")
            
#             # Enrich and transform data
#             enriched_flights = []
            
#             for flight in flights:
#                 # Parse dates properly
#                 departure_datetime = flight.get("departure_at")
#                 return_datetime = flight.get("return_at")
                
#                 # Build deep link for booking
#                 link_fragment = flight.get("link", "")
#                 booking_url = f"https://www.aviasales.com{link_fragment}" if link_fragment else None
                
#                 enriched_flight = {
#                     # Route info
#                     "origin": flight.get("origin"),
#                     "destination": flight.get("destination"),
#                     "origin_airport": flight.get("origin_airport"),
#                     "destination_airport": flight.get("destination_airport"),
                    
#                     # Pricing
#                     "price": float(flight.get("price", 0)),
#                     "currency": "USD",
                    
#                     # Flight details
#                     "airline": flight.get("airline"),
#                     "flight_number": flight.get("flight_number"),
#                     "departure_at": departure_datetime,
#                     "return_at": return_datetime,
                    
#                     # Transfer info
#                     "transfers": flight.get("transfers", 0),
#                     "return_transfers": flight.get("return_transfers", 0),
#                     "is_direct": flight.get("transfers", 0) == 0,
                    
#                     # Duration (in minutes)
#                     "duration_total": flight.get("duration"),
#                     "duration_outbound": flight.get("duration_to"),
#                     "duration_return": flight.get("duration_back"),
                    
#                     # Booking
#                     "booking_url": booking_url,
#                     "link_fragment": link_fragment,
                    
#                     # Trip type
#                     "trip_type": "one-way" if one_way else "round-trip"
#                 }
                
#                 enriched_flights.append(enriched_flight)
            
#             # Convert prices to user currency
#             if user_currency != "USD":
#                 for flight in enriched_flights:
#                     flight["price"] = await LocalizationService.convert_price(
#                         flight["price"],
#                         "USD",
#                         user_currency
#                     )
#                     flight["currency"] = user_currency
            
#             # Add currency symbol
#             for flight in enriched_flights:
#                 flight["currency_symbol"] = prefs["currency_symbol"]
            
#             # Enrich with airport names
#             enriched_flights = await enrich_with_airport_data(enriched_flights, db)
            
#             # Sort by price (API sorts, but ensure after conversion)
#             if sorting == "price":
#                 enriched_flights.sort(key=lambda x: x.get("price", 999999))
            
#             # Add statistics
#             prices = [f["price"] for f in enriched_flights]
#             result = {
#                 "success": True,
#                 "flights": enriched_flights,
#                 "metadata": {
#                     "origin": origin,
#                     "destination": destination,
#                     "departure_at": departure_at,
#                     "return_at": return_at,
#                     "one_way": one_way,
#                     "direct": direct,
#                     "count": len(enriched_flights),
#                     "currency": user_currency,
#                     "currency_symbol": prefs["currency_symbol"],
#                     "statistics": {
#                         "cheapest": min(prices) if prices else None,
#                         "average": round(sum(prices) / len(prices), 2) if prices else None,
#                         "most_expensive": max(prices) if prices else None,
#                         "direct_flights_count": sum(1 for f in enriched_flights if f.get("is_direct"))
#                     },
#                     "filters_applied": {
#                         "direct_only": direct,
#                         "unique_routes": unique,
#                         "sorting": sorting
#                     },
#                     "pagination": {
#                         "page": page,
#                         "limit": limit,
#                         "has_more": len(enriched_flights) == limit
#                     }
#                 }
#             }
            
#             # Cache the results
#             await cache_set(cache_key, result, cache_ttl)
            
#             logger.info(f"📊 Returning {len(enriched_flights)} flights in {user_currency}")
            
#             return result
    
#     except httpx.TimeoutException:
#         logger.error(f"Timeout fetching V3 prices")
#         raise HTTPException(status_code=504, detail="Request timed out")
    
#     except httpx.HTTPStatusError as e:
#         logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=f"TravelPayouts API error: {e.response.text}"
#         )
    
#     except Exception as e:
#         logger.error(f"Error fetching V3 prices: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to fetch prices: {str(e)}"
#         )


# @router.get("/search-alternatives")
# async def search_flight_alternatives(
#     request: Request,
#     origin: str = Query(..., description="Origin IATA"),
#     destination: str = Query(..., description="Destination IATA"),
#     departure_month: str = Query(..., description="Month (YYYY-MM)"),
#     currency: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     current_user: Optional[User] = Depends(get_current_user_optional)
# ):
#     """
#     🔍 Smart Alternative Flight Finder
    
#     Uses the V3 API to find:
#     1. Cheapest dates in the month
#     2. Direct vs connecting options
#     3. Different date combinations
    
#     Perfect for "flexible dates" users.
#     """
#     origin = origin.upper()
#     destination = destination.upper()
    
#     prefs = await get_user_preferences(request, currency, None, current_user)
#     user_currency = prefs["currency"]
    
#     cache_key = f"alternatives_v3:{origin}:{destination}:{departure_month}:{user_currency}"
    
#     cached = await cache_get(cache_key)
#     if cached:
#         return cached
    
#     try:
#         api_url = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
        
#         # Fetch both direct and connecting flights
#         results = {
#             "direct_flights": [],
#             "connecting_flights": [],
#             "cheapest_overall": None,
#             "recommendations": []
#         }
        
#         async with httpx.AsyncClient(timeout=15.0) as client:
#             # Get direct flights
#             direct_params = {
#                 "origin": origin,
#                 "destination": destination,
#                 "departure_at": departure_month,
#                 "direct": "true",
#                 "sorting": "price",
#                 "currency": "USD",
#                 "limit": 10,
#                 "token": settings.TRAVELPAYOUTS_API_TOKEN
#             }
            
#             direct_response = await client.get(api_url, params=direct_params)
#             if direct_response.status_code == 200:
#                 direct_data = direct_response.json()
#                 if direct_data.get("success"):
#                     results["direct_flights"] = direct_data.get("data", [])[:5]
            
#             # Get all flights (including connections)
#             all_params = {
#                 **direct_params,
#                 "direct": "false",
#                 "limit": 20
#             }
            
#             all_response = await client.get(api_url, params=all_params)
#             if all_response.status_code == 200:
#                 all_data = all_response.json()
#                 if all_data.get("success"):
#                     all_flights = all_data.get("data", [])
                    
#                     # Separate connecting flights
#                     results["connecting_flights"] = [
#                         f for f in all_flights 
#                         if f.get("transfers", 0) > 0
#                     ][:5]
                    
#                     # Find cheapest overall
#                     if all_flights:
#                         results["cheapest_overall"] = min(
#                             all_flights,
#                             key=lambda x: x.get("price", 999999)
#                         )
        
#         # Convert prices
#         if user_currency != "USD":
#             for category in ["direct_flights", "connecting_flights"]:
#                 for flight in results[category]:
#                     flight["price"] = await LocalizationService.convert_price(
#                         flight["price"],
#                         "USD",
#                         user_currency
#                     )
#                     flight["currency"] = user_currency
            
#             if results["cheapest_overall"]:
#                 results["cheapest_overall"]["price"] = await LocalizationService.convert_price(
#                     results["cheapest_overall"]["price"],
#                     "USD",
#                     user_currency
#                 )
#                 results["cheapest_overall"]["currency"] = user_currency
        
#         # Add recommendations
#         if results["direct_flights"]:
#             results["recommendations"].append(
#                 f"✈️ {len(results['direct_flights'])} direct flights available"
#             )
        
#         if results["connecting_flights"]:
#             avg_savings = 0
#             if results["direct_flights"] and results["connecting_flights"]:
#                 avg_direct = sum(f["price"] for f in results["direct_flights"]) / len(results["direct_flights"])
#                 avg_connecting = sum(f["price"] for f in results["connecting_flights"]) / len(results["connecting_flights"])
#                 avg_savings = avg_direct - avg_connecting
            
#             if avg_savings > 0:
#                 results["recommendations"].append(
#                     f"💰 Save ~{round(avg_savings)} {user_currency} with connecting flights"
#                 )
        
#         results["currency"] = user_currency
#         results["currency_symbol"] = prefs["currency_symbol"]
        
#         await cache_set(cache_key, results, 3600)
        
#         return results
    
#     except Exception as e:
#         logger.error(f"Error in alternatives search: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))




    

### ----- ENABLE THIS ROUTER IN PRODUCTION ---------
# @router.get("/trending", response_model=List[HotRouteResponse])
# async def get_trending_routes(
#     request: Request,
#     limit: int = Query(5, ge=1, le=10),
#     days: int = Query(7, ge=1, le=30),
#     currency: Optional[str] = Query(None),
#     language: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     current_user: Optional[User] = Depends(get_current_user_optional)
# ):
#     """
#     Get trending routes with smart currency conversion.
#     Shows routes seeing increased interest.
#     """
#     # Get user preferences
#     prefs = await get_user_preferences(request, currency, language, current_user)
#     user_currency = prefs["currency"]
    
#     cache_key = f"trending_routes:{limit}:{days}:{user_currency}"
    
#     cached = await cache_get(cache_key)
#     if cached:
#         logger.info(f"✅ CACHE HIT - Trending routes in {user_currency}")
#         return [HotRouteResponse(**item) for item in cached]
    
#     logger.info(f"❌ CACHE MISS - Calculating trending routes")
    
#     period_end = datetime.utcnow()
#     period_start = period_end - timedelta(days=days)
#     comparison_start = period_start - timedelta(days=days)
    
#     # Recent period clicks
#     recent_result = await db.execute(
#         select(
#             Click.origin,
#             Click.destination,
#             func.count(Click.id).label("recent_count"),
#             func.avg(Click.price).label("avg_price"),
#             Click.currency
#         )
#         .where(Click.clicked_at >= period_start)
#         .group_by(Click.origin, Click.destination, Click.currency)
#     )
#     recent_clicks = {
#         (row.origin, row.destination): {
#             "count": row.recent_count,
#             "price": row.avg_price,
#             "currency": row.currency
#         }
#         for row in recent_result
#     }
    
#     # Previous period clicks
#     previous_result = await db.execute(
#         select(
#             Click.origin,
#             Click.destination,
#             func.count(Click.id).label("previous_count")
#         )
#         .where(
#             and_(
#                 Click.clicked_at >= comparison_start,
#                 Click.clicked_at < period_start
#             )
#         )
#         .group_by(Click.origin, Click.destination)
#     )
#     previous_clicks = {
#         (row.origin, row.destination): row.previous_count
#         for row in previous_result
#     }
    
#     # Calculate growth
#     trending = []
#     for (origin, dest), recent_data in recent_clicks.items():
#         previous_count = previous_clicks.get((origin, dest), 0)
#         recent_count = recent_data["count"]
        
#         if previous_count > 0:
#             growth = ((recent_count - previous_count) / previous_count) * 100
#         else:
#             growth = 100 if recent_count > 0 else 0
        
#         # Only international destinations with positive growth
#         if growth > 0 and recent_count >= 3 and dest in POPULAR_INTERNATIONAL_DESTINATIONS:
#             trending.append({
#                 "origin": origin,
#                 "destination": dest,
#                 "price": round(recent_data["price"], 2),
#                 "currency": recent_data["currency"],
#                 "click_count": recent_count,
#                 "growth_percent": round(growth, 1),
#                 "image_url": DESTINATION_IMAGES.get(dest, "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400")
#             })
    
#     trending.sort(key=lambda x: x["growth_percent"], reverse=True)
#     trending = trending[:limit]
    
#     # Enrich with airport data
#     trending = await enrich_with_airport_data(trending, db)
    
#     # Convert prices to user currency
#     if user_currency != "USD":
#         for route in trending:
#             # Convert from stored currency to user currency
#             route["price"] = await LocalizationService.convert_price(
#                 route["price"],
#                 route["currency"],
#                 user_currency
#             )
#             route["currency"] = user_currency
    
#     # Add currency symbol
#     for route in trending:
#         route["currency_symbol"] = prefs["currency_symbol"]
    
#     await cache_set(cache_key, trending, CACHE_TTL_HOT_ROUTES)
    
#     logger.info(f"📈 Returning {len(trending)} trending routes in {user_currency}")
    
#     return [HotRouteResponse(**route) for route in trending]
































