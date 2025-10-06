# app/api/v1/endpoints/airports.py
from fastapi import APIRouter, Depends, Query, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, func
from typing import List
import json
import time
from app.db.database import get_db
from app.models.models import Airport
from schemas.airports import AirportResponse, AirportSearchResponse
from services.amadeus_service import redis_client  # Reuse Redis connection
import logging
router = APIRouter()

# Cache TTLs (airports rarely change)
CACHE_TTL_SEARCH = 86400  # 24 hours
CACHE_TTL_POPULAR = 604800  # 7 days

logger = logging.getLogger(__name__)

async def cache_get(key: str):
    """Get from Redis cache"""
    try:
        data = await redis_client.get(key)
        if data:
            return json.loads(data.decode('utf-8'))
    except Exception:
        pass
    return None


async def cache_set(key: str, value: any, ttl: int):
    """Set in Redis cache"""
    try:
        await redis_client.set(
            key,
            json.dumps(value),
            ex=ttl
        )
    except Exception:
        pass


@router.get("/search", response_model=List[AirportSearchResponse])
async def search_airports(
    q: str = Query(..., min_length=2, max_length=50, description="Search query (IATA, city, or name)"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Autocomplete search for airports.
    
    Searches by:
    - IATA code (e.g., "IST", "JFK")
    - City name (e.g., "Istanbul", "New York")
    - Airport name (e.g., "Istanbul Airport")
    
    Used for search form autocomplete.
    Cached for 24 hours since airports rarely change.
    """
    start_time = time.time()
    
    # Check cache first
    cache_key = f"airport_search:{q.lower()}:{limit}"
    cached = await cache_get(cache_key)
    
    if cached:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"✅ CACHE HIT - Airport search '{q}' | {len(cached)} results | {elapsed_ms:.2f}ms")
        return [AirportSearchResponse(**item) for item in cached]
    
    logger.info(f"❌ CACHE MISS - Airport search '{q}' | Querying database...")
    
    search_term = f"%{q.upper()}%"
    
    # Prioritize exact IATA matches, then city, then airport name
    result = await db.execute(
        select(Airport)
        .where(Airport.is_active == True)
        .where(
            or_(
                Airport.iata_code.ilike(search_term),
                Airport.city.ilike(search_term),
                Airport.name.ilike(search_term)
            )
        )
        .order_by(
            # Prioritize exact IATA matches
            Airport.iata_code.ilike(q.upper()).desc(),
            # Then city matches
            Airport.city.ilike(f"{q}%").desc(),
            # Then sort by city name
            Airport.city
        )
        .limit(limit)
    )
    airports = result.scalars().all()
    
    response_data = [
        AirportSearchResponse(
            iata_code=airport.iata_code,
            name=airport.name,
            city=airport.city,
            country=airport.country,
            country_code=airport.country_code
        )
        for airport in airports
    ]
    
    # Cache the result
    await cache_set(
        cache_key,
        [item.model_dump() for item in response_data],
        CACHE_TTL_SEARCH
    )
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"📊 DB QUERY - Airport search '{q}' | {len(response_data)} results | {elapsed_ms:.2f}ms | Cached for 24h")
    
    return response_data


@router.get("/popular", response_model=List[AirportResponse])
async def get_popular_airports(
    limit: int = Query(20, ge=1, le=100, description="Number of airports to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get popular destination airports.
    
    Returns major hubs and popular tourist destinations.
    Cached for 7 days as this rarely changes.
    
    TODO: Make this dynamic based on click analytics from clicks table.
    """
    start_time = time.time()
    
    # Check cache first
    cache_key = f"popular_airports:{limit}"
    cached = await cache_get(cache_key)
    
    if cached:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"✅ CACHE HIT - Popular airports | {len(cached)} results | {elapsed_ms:.2f}ms")
        return [AirportResponse(**item) for item in cached]
    
    logger.info(f"❌ CACHE MISS - Popular airports | Querying database...")
    
    # Major international hubs and popular destinations
    popular_codes = [
        'IST', 'DXB', 'LHR', 'JFK', 'CDG', 'FRA', 'AMS', 'MAD', 
        'BCN', 'MUC', 'FCO', 'ATH', 'VIE', 'PRG', 'WAW', 'BUD',
        'SIN', 'HKG', 'NRT', 'ICN', 'BKK', 'DEL', 'BOM', 'SYD'
    ]
    
    result = await db.execute(
        select(Airport)
        .where(Airport.iata_code.in_(popular_codes))
        .where(Airport.is_active == True)
        .order_by(Airport.city)
    )
    airports = result.scalars().all()[:limit]
    
    # Cache the result
    await cache_set(
        cache_key,
        [AirportResponse.model_validate(a).model_dump() for a in airports],
        CACHE_TTL_POPULAR
    )
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"📊 DB QUERY - Popular airports | {len(airports)} results | {elapsed_ms:.2f}ms | Cached for 7 days")
    
    return airports


@router.get("/nearby", response_model=List[AirportResponse])
async def get_nearby_airports(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lng: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius_km: int = Query(100, ge=1, le=500, description="Search radius in kilometers"),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """
    Find airports near given coordinates.
    
    Uses Haversine formula to calculate distance.
    Useful for "find flights from my location" feature.
    """
    start_time = time.time()
    logger.info(f"🌍 Searching airports near ({lat}, {lng}) within {radius_km}km...")
    
    # Haversine formula in SQL
    # This is a simplified version - for production, use PostGIS
    result = await db.execute(
        select(
            Airport,
            # Approximate distance calculation (works for small distances)
            (
                6371 * func.acos(
                    func.cos(func.radians(lat)) * 
                    func.cos(func.radians(Airport.latitude)) * 
                    func.cos(func.radians(Airport.longitude) - func.radians(lng)) +
                    func.sin(func.radians(lat)) * 
                    func.sin(func.radians(Airport.latitude))
                )
            ).label('distance')
        )
        .where(Airport.is_active == True)
        .where(Airport.latitude.isnot(None))
        .where(Airport.longitude.isnot(None))
        .order_by('distance')
        .limit(limit)
    )
    
    airports = []
    for row in result:
        airport = row[0]
        distance = row[1]
        if distance <= radius_km:
            airports.append(airport)
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"📊 DB QUERY - Nearby airports | {len(airports)} results | {elapsed_ms:.2f}ms")
    
    return airports


@router.get("/country/{country_code}", response_model=List[AirportResponse])
async def get_airports_by_country(
    country_code: str = Path(..., min_length=2, max_length=2, description="ISO country code"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all airports in a specific country.
    
    Useful for showing domestic destinations.
    """
    start_time = time.time()
    logger.info(f"🌐 Fetching airports for country: {country_code.upper()}")
    
    result = await db.execute(
        select(Airport)
        .where(Airport.country_code == country_code.upper())
        .where(Airport.is_active == True)
        .order_by(Airport.city)
        .limit(limit)
    )
    airports = result.scalars().all()
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"📊 DB QUERY - Country airports | {len(airports)} results | {elapsed_ms:.2f}ms")
    
    return airports


@router.get("/{iata_code}", response_model=AirportResponse)
async def get_airport_details(
    iata_code: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific airport.
    
    Returns full airport details including coordinates, timezone, etc.
    """
    start_time = time.time()
    
    # Validate IATA code format
    if len(iata_code) != 3:
        raise HTTPException(
            status_code=400,
            detail="IATA code must be exactly 3 characters"
        )
    
    logger.info(f"🔍 Fetching details for airport: {iata_code.upper()}")
    
    result = await db.execute(
        select(Airport)
        .where(Airport.iata_code == iata_code.upper())
        .where(Airport.is_active == True)
    )
    airport = result.scalar_one_or_none()
    
    if not airport:
        logger.warning(f"❌ Airport not found: {iata_code.upper()}")
        raise HTTPException(
            status_code=404, 
            detail=f"Airport with code '{iata_code}' not found"
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"✅ Airport found: {airport.name} | {elapsed_ms:.2f}ms")
    
    return airport