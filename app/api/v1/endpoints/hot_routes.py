# app/api/v1/endpoints/hot_routes.py
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, desc
from typing import List, Optional
from datetime import datetime, timedelta
import httpx
import json
import logging

from app.db.database import get_db
from app.models.models import Click, Airport
from app.core.config import settings
from services.amadeus_service import redis_client  # Reuse Redis
from schemas.hot_routes import HotRouteResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Cache TTL
CACHE_TTL_HOT_ROUTES = 3600  # 1 hour (hot routes don't change often)

# Travelpayouts Data API endpoint (FREE - no approval needed!)
TRAVELPAYOUTS_API_URL = "https://api.travelpayouts.com/v2/prices/latest"


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
    limit: int = 10
) -> List[dict]:
    """
    Fetch popular routes from Travelpayouts Data API.
    This is the FREE API that doesn't require approval!
    
    Returns cached flight data from last 48 hours of user searches.
    """
    try:
        params = {
            "origin": origin.upper(),
            "currency": "USD",
            "limit": limit,
            "page": 1,
            "sorting": "price",  # Cheapest flights
            "token": settings.TRAVELPAYOUTS_API_TOKEN  # Add this to your .env
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(TRAVELPAYOUTS_API_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success", False):
                logger.warning(f"Travelpayouts API returned success=false")
                return []
            
            # Parse Travelpayouts response
            routes = []
            seen_destinations = set()
            
            for item in data.get("data", []):
                destination = item.get("destination")
                
                # Skip duplicates (we want unique destinations)
                if destination in seen_destinations:
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
                    "found_at": item.get("found_at")
                })
            
            logger.info(f"Fetched {len(routes)} routes from Travelpayouts for {origin}")
            return routes
            
    except httpx.HTTPError as e:
        logger.error(f"Travelpayouts API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching from Travelpayouts: {e}")
        return []


async def fetch_popular_from_analytics(
    origin: str,
    db: AsyncSession,
    days: int = 30,
    limit: int = 10
) -> List[dict]:
    """
    Fetch popular routes from your own analytics (Click table).
    This shows what YOUR users are actually searching for.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await db.execute(
            select(
                Click.destination,
                func.count(Click.id).label("click_count"),
                func.avg(Click.price).label("avg_price"),
                func.min(Click.price).label("min_price"),
                Click.currency
            )
            .where(
                and_(
                    Click.origin == origin.upper(),
                    Click.clicked_at >= cutoff_date
                )
            )
            .group_by(Click.destination, Click.currency)
            .order_by(desc("click_count"))
            .limit(limit)
        )
        
        routes = []
        for row in result:
            routes.append({
                "origin": origin.upper(),
                "destination": row.destination,
                "price": round(row.min_price, 2),
                "currency": row.currency,
                "click_count": row.click_count,
                "avg_price": round(row.avg_price, 2)
            })
        
        logger.info(f"Fetched {len(routes)} routes from analytics for {origin}")
        return routes
        
    except Exception as e:
        logger.error(f"Error fetching from analytics: {e}")
        return []


async def enrich_with_airport_data(
    routes: List[dict],
    db: AsyncSession
) -> List[dict]:
    """
    Add airport names and countries to the routes.
    Makes the data more user-friendly.
    """
    if not routes:
        return []
    
    # Get all unique airport codes
    airport_codes = set()
    for route in routes:
        airport_codes.add(route["origin"])
        airport_codes.add(route["destination"])
    
    # Fetch airport data
    result = await db.execute(
        select(Airport)
        .where(Airport.iata_code.in_(list(airport_codes)))
    )
    airports = {a.iata_code: a for a in result.scalars().all()}
    
    # Enrich routes
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
    origin: str = Query(
        "TAS",
        description="Origin airport IATA code",
        min_length=3,
        max_length=3
    ),
    limit: int = Query(
        10,
        ge=1,
        le=20,
        description="Maximum number of routes to return"
    ),
    source: str = Query(
        "travelpayouts",
        description="Data source: 'travelpayouts', 'analytics', or 'hybrid'",
        regex="^(travelpayouts|analytics|hybrid)$"
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    Get hot/popular flight routes from a given origin.
    
    Data sources:
    - **travelpayouts**: Real price data from Travelpayouts API (last 48h searches)
    - **analytics**: Your own user click analytics (shows what YOUR users search)
    - **hybrid**: Combines both sources for best results
    
    Perfect for:
    - Landing page "Popular Routes" section
    - Destination inspiration
    - Price trends
    
    Cached for 1 hour since routes don't change frequently.
    """
    origin = origin.upper()
    cache_key = f"hot_routes:{origin}:{limit}:{source}"
    
    # Check cache
    cached = await cache_get(cache_key)
    if cached:
        logger.info(f"✅ CACHE HIT - Hot routes for {origin}")
        return [HotRouteResponse(**item) for item in cached]
    
    logger.info(f"❌ CACHE MISS - Fetching hot routes for {origin}")
    
    routes = []
    
    if source == "travelpayouts":
        routes = await fetch_popular_from_travelpayouts(origin, limit)
    
    elif source == "analytics":
        routes = await fetch_popular_from_analytics(origin, db, limit=limit)
    
    elif source == "hybrid":
        # Combine both sources
        tp_routes = await fetch_popular_from_travelpayouts(origin, limit)
        analytics_routes = await fetch_popular_from_analytics(origin, db, limit=limit)
        
        # Merge and deduplicate by destination
        seen_destinations = set()
        for route in tp_routes + analytics_routes:
            dest = route["destination"]
            if dest not in seen_destinations:
                routes.append(route)
                seen_destinations.add(dest)
                
                if len(routes) >= limit:
                    break
    
    # Enrich with airport data (city names, countries)
    routes = await enrich_with_airport_data(routes, db)
    
    # Sort by price (cheapest first)
    routes.sort(key=lambda x: x.get("price", 999999))
    
    # Limit results
    routes = routes[:limit]
    
    if not routes:
        logger.warning(f"No hot routes found for {origin}")
        # Return fallback data for demo purposes
        routes = get_fallback_routes(origin)
        routes = await enrich_with_airport_data(routes, db)
    
    # Cache the result
    await cache_set(cache_key, routes, CACHE_TTL_HOT_ROUTES)
    
    logger.info(f"📊 Returning {len(routes)} hot routes for {origin}")
    
    return [HotRouteResponse(**route) for route in routes]


def get_fallback_routes(origin: str = "TAS") -> List[dict]:
    """
    Fallback data when APIs are unavailable.
    These are real popular routes from Tashkent.
    """
    fallback = {
        "TAS": [
            {"origin": "TAS", "destination": "DXB", "price": 245.0, "currency": "USD", "airline": "FZ"},
            {"origin": "TAS", "destination": "IST", "price": 198.0, "currency": "USD", "airline": "TK"},
            {"origin": "TAS", "destination": "MOW", "price": 189.0, "currency": "USD", "airline": "HY"},
            {"origin": "TAS", "destination": "ICN", "price": 425.0, "currency": "USD", "airline": "KE"},
            {"origin": "TAS", "destination": "DEL", "price": 215.0, "currency": "USD", "airline": "HY"},
            {"origin": "TAS", "destination": "ALA", "price": 89.0, "currency": "USD", "airline": "HY"},
            {"origin": "TAS", "destination": "KUL", "price": 380.0, "currency": "USD", "airline": "HY"},
            {"origin": "TAS", "destination": "BKK", "price": 420.0, "currency": "USD", "airline": "HY"},
        ]
    }
    
    return fallback.get(origin.upper(), fallback["TAS"])


@router.get("/trending", response_model=List[HotRouteResponse])
async def get_trending_routes(
    limit: int = Query(5, ge=1, le=10),
    days: int = Query(7, ge=1, le=30, description="Look at trends from last N days"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get trending routes based on recent search activity growth.
    
    Shows routes that are seeing increased interest (clicks growing).
    Great for "Trending Now" sections.
    """
    cache_key = f"trending_routes:{limit}:{days}"
    
    cached = await cache_get(cache_key)
    if cached:
        logger.info(f"✅ CACHE HIT - Trending routes")
        return [HotRouteResponse(**item) for item in cached]
    
    logger.info(f"❌ CACHE MISS - Calculating trending routes")
    
    # Get clicks from two periods to calculate trend
    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=days)
    comparison_start = period_start - timedelta(days=days)
    
    # Recent period clicks
    recent_result = await db.execute(
        select(
            Click.origin,
            Click.destination,
            func.count(Click.id).label("recent_count"),
            func.avg(Click.price).label("avg_price"),
            Click.currency
        )
        .where(Click.clicked_at >= period_start)
        .group_by(Click.origin, Click.destination, Click.currency)
    )
    recent_clicks = {
        (row.origin, row.destination): {
            "count": row.recent_count,
            "price": row.avg_price,
            "currency": row.currency
        }
        for row in recent_result
    }
    
    # Previous period clicks
    previous_result = await db.execute(
        select(
            Click.origin,
            Click.destination,
            func.count(Click.id).label("previous_count")
        )
        .where(
            and_(
                Click.clicked_at >= comparison_start,
                Click.clicked_at < period_start
            )
        )
        .group_by(Click.origin, Click.destination)
    )
    previous_clicks = {
        (row.origin, row.destination): row.previous_count
        for row in previous_result
    }
    
    # Calculate growth rate
    trending = []
    for (origin, dest), recent_data in recent_clicks.items():
        previous_count = previous_clicks.get((origin, dest), 0)
        recent_count = recent_data["count"]
        
        # Calculate percentage growth
        if previous_count > 0:
            growth = ((recent_count - previous_count) / previous_count) * 100
        else:
            growth = 100 if recent_count > 0 else 0
        
        # Only include routes with positive growth and minimum activity
        if growth > 0 and recent_count >= 5:
            trending.append({
                "origin": origin,
                "destination": dest,
                "price": round(recent_data["price"], 2),
                "currency": recent_data["currency"],
                "click_count": recent_count,
                "growth_percent": round(growth, 1)
            })
    
    # Sort by growth rate
    trending.sort(key=lambda x: x["growth_percent"], reverse=True)
    trending = trending[:limit]
    
    # Enrich with airport data
    trending = await enrich_with_airport_data(trending, db)
    
    # Cache
    await cache_set(cache_key, trending, CACHE_TTL_HOT_ROUTES)
    
    logger.info(f"📈 Returning {len(trending)} trending routes")
    
    return [HotRouteResponse(**route) for route in trending]