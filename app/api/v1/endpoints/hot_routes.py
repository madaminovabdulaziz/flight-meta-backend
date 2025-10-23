# app/api/v1/endpoints/hot_routes.py - FIXED VERSION
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

# Popular destinations by region
POPULAR_INTERNATIONAL_DESTINATIONS = [
    # Middle East & Turkey
    "IST", "DXB", "AUH", "DOH", "SHJ", "AYT", "JED", "RUH", "AMM", "BEY",
    
    # Europe
    "LHR", "CDG", "AMS", "FRA", "BCN", "MAD", "FCO", "MXP", "ATH", "PRG",
    "VIE", "BUD", "DUB", "LIS", "ZRH", "CPH", "OSL", "ARN", "HEL", "WAW",
    
    # CIS & Central Asia
    "MOW", "LED", "ALA", "FRU", "TAS", "GYD", "KIV", "TBS", "EVN", "ASB",
    
    # Asia Pacific
    "BKK", "KUL", "SIN", "ICN", "NRT", "HND", "PEK", "PVG", "HKG", "DEL",
    "BOM", "DPS", "SGN", "HAN", "MNL", "CGK", "SYD", "MEL", "BNE", "AKL",
    
    # Americas
    "JFK", "LAX", "MIA", "ORD", "SFO", "YYZ", "YVR", "MEX", "CUN", "GRU",
    "EZE", "SCL", "LIM", "BOG"
]

# High-quality destination images (real photos from Unsplash)
DESTINATION_IMAGES = {
    # === Middle East & Turkey ===
    "IST": "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200",  # Blue Mosque Istanbul
    "AYT": "https://images.unsplash.com/photo-1605116955535-2b1fb95c6fdb",  # Antalya beach
    "DXB": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c",  # Burj Khalifa
    "SHJ": "https://images.unsplash.com/photo-1580674684081-7617fbf3d745",  # Sharjah mosque
    "AUH": "https://images.unsplash.com/photo-1503415397539-a9a7c64503a5",  # Abu Dhabi Sheikh Zayed
    "DOH": "https://images.unsplash.com/photo-1559827260-dc66d52bef19",  # Doha skyline
    "JED": "https://images.unsplash.com/photo-1596422846543-75c6fc197f07",  # Jeddah corniche
    "RUH": "https://images.unsplash.com/photo-1587474260584-136574528ed5",  # Riyadh Kingdom Tower
    "AMM": "https://images.unsplash.com/photo-1580850750852-981c8e62dc25",  # Amman Citadel
    "BEY": "https://images.unsplash.com/photo-1580492516014-4a28466d55df",  # Beirut

    # === Europe ===
    "LHR": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad",  # London Bridge
    "CDG": "https://images.unsplash.com/photo-1502602898657-3e91760c0337",  # Paris Eiffel
    "AMS": "https://images.unsplash.com/photo-1534351590666-13e3e96b5017",  # Amsterdam canals
    "FRA": "https://images.unsplash.com/photo-1559827260-dc66d52bef19",  # Frankfurt skyline
    "BCN": "https://images.unsplash.com/photo-1583422409516-2895a77efded",  # Barcelona Sagrada
    "MAD": "https://images.unsplash.com/photo-1539037116277-4db20889f2d4",  # Madrid Plaza
    "FCO": "https://images.unsplash.com/photo-1552832230-c0197dd311b5",  # Rome Colosseum
    "ROM": "https://images.unsplash.com/photo-1552832230-c0197dd311b5",  # Rome
    "MXP": "https://images.unsplash.com/photo-1513581166391-887a96ddeafd",  # Milan Cathedral
    "ATH": "https://images.unsplash.com/photo-1555993539-1732b0258235",  # Athens Acropolis
    "PRG": "https://images.unsplash.com/photo-1541849546-216549ae216d",  # Prague Bridge
    "VIE": "https://images.unsplash.com/photo-1516550893923-42d28e5677af",  # Vienna Palace
    "BUD": "https://images.unsplash.com/photo-1541849546-216549ae216d",  # Budapest Parliament
    "DUB": "https://images.unsplash.com/photo-1549918864-48ac978761a4",  # Dublin Ha'penny
    "LIS": "https://images.unsplash.com/photo-1555881400-74d7acaacd8b",  # Lisbon tram
    "ZRH": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",  # Zurich lake
    "CPH": "https://images.unsplash.com/photo-1513622790141-00f80d57bfc8",  # Copenhagen Nyhavn
    "OSL": "https://images.unsplash.com/photo-1592214541514-0cd33f1feee0",  # Oslo Opera House
    "ARN": "https://images.unsplash.com/photo-1509356843151-3e7d96241e11",  # Stockholm
    "HEL": "https://images.unsplash.com/photo-1526778548025-fa2f459cd5c1",  # Helsinki Cathedral
    "WAW": "https://images.unsplash.com/photo-1605704817967-d50b8f9a26dc",  # Warsaw

    # === CIS Region ===
    "MOW": "https://images.unsplash.com/photo-1513326738677-b964603b136d",  # Moscow Red Square
    "LED": "https://images.unsplash.com/photo-1558505012-817176f84c6d",  # St Petersburg Church
    "ALA": "https://images.unsplash.com/photo-1602069877324-7c8bcf0e6c85",  # Almaty mountains
    "FRU": "https://images.unsplash.com/photo-1601625841499-01b34ec91c6e",  # Bishkek Ala-Too
    "TAS": "https://images.unsplash.com/photo-1596383648434-153364f378ae",  # Tashkent
    "GYD": "https://images.unsplash.com/photo-1634841923617-83688225a0b7",  # Baku Flame Towers
    "TBS": "https://images.unsplash.com/photo-1591123120675-6f7f1aae0e5b",  # Tbilisi
    "EVN": "https://images.unsplash.com/photo-1599642194462-a40e7f03a64a",  # Yerevan
    
    # === Asia Pacific ===
    "BKK": "https://images.unsplash.com/photo-1508009603885-50cf7c579365",  # Bangkok temple
    "KUL": "https://images.unsplash.com/photo-1596422846543-75c6fc197f07",  # Kuala Lumpur Petronas
    "SIN": "https://images.unsplash.com/photo-1525625293386-3f8f99389edd",  # Singapore Marina Bay
    "ICN": "https://images.unsplash.com/photo-1517154421773-0529f29ea451",  # Seoul skyline
    "NRT": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf",  # Tokyo
    "HND": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf",  # Tokyo
    "PEK": "https://images.unsplash.com/photo-1508804185872-d7badad00f7d",  # Beijing Forbidden City
    "PVG": "https://images.unsplash.com/photo-1474181487882-5abf3f0ba6c2",  # Shanghai Bund
    "HKG": "https://images.unsplash.com/photo-1536599018102-9f803c140fc1",  # Hong Kong skyline
    "DEL": "https://images.unsplash.com/photo-1587474260584-136574528ed5",  # Delhi India Gate
    "BOM": "https://images.unsplash.com/photo-1529253355930-ddbe423a2ac7",  # Mumbai Gateway
    "DPS": "https://images.unsplash.com/photo-1555400038-63f5ba517a47",  # Bali temple
    "SGN": "https://images.unsplash.com/photo-1583417319070-4a69db38a482",  # Ho Chi Minh
    "HAN": "https://images.unsplash.com/photo-1509233725247-49e657c54213",  # Hanoi
    "MNL": "https://images.unsplash.com/photo-1583417319070-4a69db38a482",  # Manila
    "CGK": "https://images.unsplash.com/photo-1555425751-7b8e49d6f1f0",  # Jakarta
    "SYD": "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9",  # Sydney Opera House
    "MEL": "https://images.unsplash.com/photo-1514395462725-fb4566210144",  # Melbourne
    "AKL": "https://images.unsplash.com/photo-1507699622108-4be3abd695ad",  # Auckland

    # === Americas ===
    "JFK": "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9",  # New York
    "LAX": "https://images.unsplash.com/photo-1518416177092-ec985e4d6c14",  # Los Angeles
    "MIA": "https://images.unsplash.com/photo-1505118380757-91f5f5632de0",  # Miami beach
    "ORD": "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df",  # Chicago
    "SFO": "https://images.unsplash.com/photo-1501594907352-04cda38ebc29",  # San Francisco
    "YYZ": "https://images.unsplash.com/photo-1517935706615-2717063c2225",  # Toronto CN Tower
    "YVR": "https://images.unsplash.com/photo-1527719327859-c6ce80353573",  # Vancouver
    "MEX": "https://images.unsplash.com/photo-1518638150340-f706e86654de",  # Mexico City
    "CUN": "https://images.unsplash.com/photo-1561035490-8dd6edf36c58",  # Cancun
    "GRU": "https://images.unsplash.com/photo-1483729558449-99ef09a8c325",  # Sao Paulo
    "EZE": "https://images.unsplash.com/photo-1589909202802-8f4aadce1849",  # Buenos Aires
    "SCL": "https://images.unsplash.com/photo-1469854523086-cc02fe5d8800",  # Santiago
    "LIM": "https://images.unsplash.com/photo-1531968455001-5c5272a41129",  # Lima
    "BOG": "https://images.unsplash.com/photo-1568632234157-ce7aecd03d0d",  # Bogota
}

# Default fallback image
DEFAULT_DESTINATION_IMAGE = "https://images.unsplash.com/photo-1436491865332-7a61a109cc05"


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
    origin: str,  # REMOVED DEFAULT - origin is now required!
    limit: int = 10,
    period: str = "year"
) -> List[dict]:
    """
    Fetch popular routes FROM the specified origin TO popular destinations.
    Always returns prices in USD for consistent conversion.
    
    FIX: Properly uses the origin parameter instead of ignoring it!
    """
    try:
        # Use aggregated prices endpoint for better data
        aggregated_url = "https://api.travelpayouts.com/v1/prices/cheap"
        
        origin = origin.upper()  # Normalize to uppercase
        
        params = {
            "origin": origin,  # NOW ACTUALLY USES THE ORIGIN PARAMETER!
            "currency": "USD",  # Always fetch in USD, convert later
            "token": settings.TRAVELPAYOUTS_API_TOKEN
        }
        
        logger.info(f"🔍 Searching routes FROM {origin} to popular destinations...")
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            all_routes = []
            
            # Search FROM origin TO each popular destination
            for destination in POPULAR_INTERNATIONAL_DESTINATIONS[:20]:
                # Skip if destination is same as origin
                if destination == origin:
                    continue
                
                dest_params = {**params, "destination": destination}
                
                try:
                    response = await client.get(aggregated_url, params=dest_params, timeout=5.0)
                    
                    if response.status_code == 200:
                        dest_data = response.json()
                        
                        if dest_data.get("success") and dest_data.get("data"):
                            flights = dest_data.get("data", {}).get(destination, [])
                            
                            if flights and len(flights) > 0:
                                # Get the cheapest flight to this destination
                                cheapest = min(flights, key=lambda x: x.get("price", 999999))
                                
                                route = {
                                    "origin": origin,  # Use the actual origin parameter
                                    "destination": destination,
                                    "price": float(cheapest.get("price", 0)),
                                    "currency": "USD",
                                    "departure_date": cheapest.get("departure_at"),
                                    "return_date": cheapest.get("return_at"),
                                    "airline": cheapest.get("airline"),
                                    "flight_number": cheapest.get("flight_number"),
                                    "found_at": cheapest.get("created_at"),
                                    # Use destination-specific image, fallback to default
                                    "image_url": DESTINATION_IMAGES.get(destination, DEFAULT_DESTINATION_IMAGE)
                                }
                                
                                all_routes.append(route)
                                logger.debug(f"✓ Found {origin} → {destination}: ${route['price']}")
                                
                except httpx.TimeoutException:
                    logger.debug(f"⏱ Timeout for {destination}")
                    continue
                except Exception as e:
                    logger.debug(f"⚠ Skipped {destination}: {e}")
                    continue
            
            if all_routes:
                # Sort by price (cheapest first)
                all_routes.sort(key=lambda x: x["price"])
                logger.info(f"✅ Fetched {len(all_routes)} routes from {origin}")
                return all_routes[:limit]
            
            # FALLBACK: Try the latest prices endpoint if cheap endpoint fails
            logger.info(f"📋 Falling back to latest prices endpoint for {origin}")
            
            fallback_params = {
                "origin": origin,
                "currency": "USD",
                "limit": 50,
                "page": 1,
                "sorting": "price",
                "token": settings.TRAVELPAYOUTS_API_TOKEN
            }
            
            response = await client.get(TRAVELPAYOUTS_API_URL, params=fallback_params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success") and data.get("data"):
                    routes = []
                    
                    for flight_data in data["data"][:limit]:
                        destination = flight_data.get("destination")
                        
                        route = {
                            "origin": origin,
                            "destination": destination,
                            "price": float(flight_data.get("price", 0)),
                            "currency": "USD",
                            "departure_date": flight_data.get("depart_date"),
                            "return_date": flight_data.get("return_date"),
                            "airline": flight_data.get("airline"),
                            "flight_number": flight_data.get("flight_number"),
                            "found_at": flight_data.get("found_at"),
                            "image_url": DESTINATION_IMAGES.get(destination, DEFAULT_DESTINATION_IMAGE)
                        }
                        
                        routes.append(route)
                    
                    logger.info(f"✅ Fallback successful: {len(routes)} routes from {origin}")
                    return routes
            
            logger.warning(f"❌ No routes found from {origin}")
            return []
            
    except Exception as e:
        logger.error(f"❌ Error fetching from Travelpayouts: {e}", exc_info=True)
        return []

async def enrich_with_airport_data(routes: List[dict], db: AsyncSession) -> List[dict]:
    """Add airport names and countries to routes"""
    try:
        all_codes = set()
        for route in routes:
            all_codes.add(route["origin"])
            all_codes.add(route["destination"])
        
        result = await db.execute(
            select(Airport).where(Airport.iata_code.in_(all_codes))
        )
        airports = {airport.iata_code: airport for airport in result.scalars()}
        
        for route in routes:
            origin_airport = airports.get(route["origin"])
            dest_airport = airports.get(route["destination"])
            
            if origin_airport:
                route["origin_city"] = origin_airport.city
                route["origin_country"] = origin_airport.country
            else:
                route["origin_city"] = route["origin"]
                route["origin_country"] = "Unknown"
            
            if dest_airport:
                route["destination_city"] = dest_airport.city
                route["destination_country"] = dest_airport.country
            else:
                route["destination_city"] = route["destination"]
                route["destination_country"] = "Unknown"
        
        return routes
        
    except Exception as e:
        logger.error(f"Error enriching airport data: {e}")
        return routes


@router.get("/", response_model=List[HotRouteResponse])
async def get_hot_routes(
    request: Request,
    origin: str = Query("TAS", description="Origin airport code (IATA)"),
    limit: int = Query(10, ge=1, le=20),
    currency: Optional[str] = Query(None),
    language: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get hot deals FROM the specified origin airport.
    
    FIXED: Now properly searches FROM the origin you specify!
    
    Examples:
    - origin=JFK → Shows cheap flights FROM New York
    - origin=TAS → Shows cheap flights FROM Tashkent
    - origin=LHR → Shows cheap flights FROM London
    """
    
    # Normalize origin to uppercase
    origin = origin.upper()
    
    # Get user preferences for currency/language
    prefs = await get_user_preferences(request, currency, language, current_user)
    user_currency = prefs["currency"]
    
    # Check cache
    cache_key = f"hot_routes:{origin}:{limit}:{user_currency}"
    cached = await cache_get(cache_key)
    
    if cached:
        logger.info(f"✅ CACHE HIT - Hot routes from {origin} in {user_currency}")
        return [HotRouteResponse(**item) for item in cached]
    
    logger.info(f"❌ CACHE MISS - Fetching hot routes from {origin}")
    
    # Fetch from Travelpayouts (returns USD prices)
    routes = await fetch_popular_from_travelpayouts(
        origin=origin,  # NOW PROPERLY USES THE ORIGIN!
        limit=limit
    )
    
    if not routes:
        logger.warning(f"No routes found from {origin}")
        raise HTTPException(
            status_code=404, 
            detail=f"No routes found from {origin}. Try another origin airport."
        )
    
    # Enrich with airport data (city names, countries)
    routes = await enrich_with_airport_data(routes, db)
    
    # Convert prices from USD to user's preferred currency
    if user_currency != "USD":
        logger.info(f"💱 Converting prices from USD to {user_currency}")
        
        for route in routes:
            route["price"] = await LocalizationService.convert_price(
                route["price"],
                "USD",  # Source currency (Travelpayouts returns USD)
                user_currency  # Target currency
            )
            route["currency"] = user_currency
    
    # Add currency symbol to each route
    for route in routes:
        route["currency_symbol"] = prefs["currency_symbol"]
    
    # Cache the results
    await cache_set(cache_key, routes, CACHE_TTL_HOT_ROUTES)
    
    logger.info(f"📤 Returning {len(routes)} hot routes from {origin} in {user_currency}")
    
    return [HotRouteResponse(**route) for route in routes]