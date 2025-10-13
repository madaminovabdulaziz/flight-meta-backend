# app/api/v1/endpoints/destinations.py
"""
Popular destinations endpoint for SkySearch AI.
Fetches the most popular flight directions from a specified city.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends
from fastapi.responses import JSONResponse
import httpx
from datetime import datetime, timedelta
import json
import logging
from app.core.config import settings
from services.amadeus_service import redis_client  # Reuse Redis connection
from schemas.popular_destinations import PopularDestinationsResponse, DestinationItem
from app.db.database import get_db
from app.models.models import Airport

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL_SECONDS = 900  # 15 minutes - popular destinations change slowly
CACHE_KEY_PREFIX = "popular_destinations"


async def get_cached_destinations(
    cache_key: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached popular destinations from Redis.
    
    Args:
        cache_key: Redis cache key
        
    Returns:
        Cached data if available, None otherwise
    """
    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            logger.info(f"Cache HIT for key: {cache_key}")
            return json.loads(cached_data.decode('utf-8'))
        logger.info(f"Cache MISS for key: {cache_key}")
        return None
    except Exception as e:
        logger.error(f"Redis cache retrieval error: {str(e)}")
        return None


async def set_cached_destinations(
    cache_key: str,
    data: Dict[str, Any],
    ttl: int = CACHE_TTL_SECONDS
) -> None:
    """
    Store popular destinations in Redis cache.
    
    Args:
        cache_key: Redis cache key
        data: Data to cache
        ttl: Time-to-live in seconds
    """
    try:
        await redis_client.set(
            cache_key,
            json.dumps(data),
            ex=ttl
        )
        logger.info(f"Cached data with key: {cache_key}, TTL: {ttl}s")
    except Exception as e:
        logger.error(f"Redis cache storage error: {str(e)}")


async def fetch_from_travelpayouts(
    origin: str,
    currency: str,
    token: str
) -> Dict[str, Any]:
    """
    Fetch popular destinations from TravelPayouts API.
    
    Args:
        origin: Origin city IATA code
        currency: Currency code (USD, RUB, etc.)
        token: TravelPayouts API token
        
    Returns:
        API response data
        
    Raises:
        HTTPException: If API request fails
    """
    url = "https://api.travelpayouts.com/v1/city-directions"
    params = {
        "origin": origin.upper(),
        "currency": currency.upper(),
        "token": token
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info(f"Fetching popular destinations from TravelPayouts: {origin} -> currency: {currency}")
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success"):
                error_msg = data.get("error", "Unknown error from TravelPayouts API")
                logger.error(f"TravelPayouts API error: {error_msg}")
                raise HTTPException(status_code=502, detail=f"External API error: {error_msg}")
            
            return data
            
    except httpx.HTTPStatusError as e:
        logger.error(f"TravelPayouts API HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=502,
            detail=f"External API returned status {e.response.status_code}"
        )
    except httpx.TimeoutException:
        logger.error("TravelPayouts API request timeout")
        raise HTTPException(status_code=504, detail="External API timeout")
    except Exception as e:
        logger.error(f"Unexpected error calling TravelPayouts API: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def transform_response(
    api_data: Dict[str, Any],
    airport_details: Dict[str, Dict[str, str]]
) -> PopularDestinationsResponse:
    """
    Transform TravelPayouts API response into our schema format.
    Filter for international routes only and enrich with location details.
    
    Args:
        api_data: Raw API response
        airport_details: Dictionary mapping IATA codes to airport details
        
    Returns:
        Structured response matching our schema
    """
    destinations = []
    data_dict = api_data.get("data", {})
    
    # Get origin country for filtering
    origin_country = None
    for dest_code, dest_data in data_dict.items():
        origin_iata = dest_data.get("origin")
        if origin_iata and origin_iata in airport_details:
            origin_country = airport_details[origin_iata].get("country_code")
            break
    
    # Filter and sort destinations
    for dest_code, dest_data in data_dict.items():
        dest_iata = dest_data.get("destination")
        
        # Skip if destination details not found
        if not dest_iata or dest_iata not in airport_details:
            continue
        
        dest_country = airport_details[dest_iata].get("country_code")
        
        # ONLY include international routes (different countries)
        if origin_country and dest_country and origin_country == dest_country:
            logger.debug(f"Skipping domestic route: {dest_code} (same country: {origin_country})")
            continue
        
        # Enrich with location details
        origin_details = airport_details.get(dest_data.get("origin", ""), {})
        dest_details = airport_details.get(dest_iata, {})
        
        destinations.append(
            DestinationItem(
                destination_code=dest_code,
                origin=dest_data.get("origin"),
                origin_city=origin_details.get("city", ""),
                origin_country=origin_details.get("country", ""),
                origin_country_code=origin_details.get("country_code", ""),
                destination=dest_iata,
                destination_city=dest_details.get("city", ""),
                destination_country=dest_details.get("country", ""),
                destination_country_code=dest_details.get("country_code", ""),
                price=dest_data.get("price"),
                transfers=dest_data.get("transfers", 0),
                airline=dest_data.get("airline"),
                flight_number=dest_data.get("flight_number"),
                departure_at=dest_data.get("departure_at"),
                return_at=dest_data.get("return_at"),
                expires_at=dest_data.get("expires_at")
            )
        )
    
    # Sort by price (cheapest first)
    destinations.sort(key=lambda x: x.price if x.price else float('inf'))
    
    return PopularDestinationsResponse(
        success=True,
        currency=api_data.get("currency", "rub"),
        destinations=destinations,
        total_count=len(destinations)
    )


async def get_airport_details(
    iata_codes: list[str],
    db: AsyncSession
) -> Dict[str, Dict[str, str]]:
    """
    Fetch airport details (city, country) from database.
    
    Args:
        iata_codes: List of IATA codes to lookup
        db: Database session
        
    Returns:
        Dictionary mapping IATA codes to airport details
    """
    if not iata_codes:
        return {}
    
    try:
        result = await db.execute(
            select(Airport).where(
                Airport.iata_code.in_(iata_codes),
                Airport.is_active == True
            )
        )
        airports = result.scalars().all()
        
        return {
            airport.iata_code: {
                "city": airport.city,
                "country": airport.country,
                "country_code": airport.country_code
            }
            for airport in airports
        }
    except Exception as e:
        logger.error(f"Failed to fetch airport details: {str(e)}")
        return {}


@router.get(
    "/popular-destinations",
    response_model=PopularDestinationsResponse,
    summary="Get popular destinations from a city",
    description="Fetches the most popular flight directions from a specified origin city with smart caching"
)
async def get_popular_destinations(
    origin: str = Query(
        ...,
        description="Origin city IATA code (e.g., LON, NYC, IST)",
        min_length=2,
        max_length=3,
        example="LON"
    ),
    currency: Optional[str] = Query(
        "USD",
        description="Currency code for prices (USD, RUB, EUR, etc.)",
        min_length=3,
        max_length=3,
        example="USD"
    ),
    db: AsyncSession = Depends(get_db)
) -> PopularDestinationsResponse:
    """
    Get popular INTERNATIONAL destinations from a specified city.
    
    This endpoint provides:
    - Most popular INTERNATIONAL flight routes (excludes domestic)
    - Enriched with city and country names for origin and destination
    - Pricing in your preferred currency
    - Smart caching for performance (15-minute TTL)
    - Sorted by price (cheapest first)
    
    **Example usage:**
    ```
    GET /api/v1/popular-destinations?origin=LON&currency=USD
    ```
    
    **Cache behavior:**
    - Results cached for 15 minutes per origin-currency combination
    - Cache key format: `popular_destinations:{origin}:{currency}`
    - Automatic cache invalidation after TTL
    
    Args:
        origin: Origin city IATA code (2-3 characters)
        currency: Currency code (default: USD)
        db: Database session for airport lookup
        
    Returns:
        PopularDestinationsResponse with international destinations only
        
    Raises:
        HTTPException: 400 for invalid input, 502/504 for API errors
    """
    # Input validation
    origin = origin.strip().upper()
    currency = currency.strip().upper()
    
    if not origin or len(origin) < 2:
        raise HTTPException(
            status_code=400,
            detail="Origin must be a valid 2-3 character IATA code"
        )
    
    # Generate cache key
    cache_key = f"{CACHE_KEY_PREFIX}:{origin}:{currency}"
    
    # Try cache first
    cached_result = await get_cached_destinations(cache_key)
    if cached_result:
        logger.info(f"Returning cached popular destinations for {origin}")
        return PopularDestinationsResponse(**cached_result)
    
    # Fetch from TravelPayouts API
    logger.info(f"Fetching fresh data from TravelPayouts for {origin}")
    api_response = await fetch_from_travelpayouts(
        origin=origin,
        currency=currency,
        token=settings.TRAVELPAYOUTS_API_TOKEN
    )
    
    # Extract all IATA codes from response
    iata_codes = set()
    data_dict = api_response.get("data", {})
    for dest_data in data_dict.values():
        if origin_iata := dest_data.get("origin"):
            iata_codes.add(origin_iata)
        if dest_iata := dest_data.get("destination"):
            iata_codes.add(dest_iata)
    
    # Fetch airport details from database
    airport_details = await get_airport_details(list(iata_codes), db)
    
    # Transform response with enriched data and international filtering
    structured_response = transform_response(api_response, airport_details)
    
    # Cache the result
    await set_cached_destinations(
        cache_key,
        structured_response.model_dump()
    )
    
    logger.info(
        f"Successfully fetched {structured_response.total_count} "
        f"international destinations from {origin}"
    )
    
    return structured_response
