# app/api/v1/endpoints/airports.py

from fastapi import APIRouter, Query, Path, HTTPException, Request
from typing import List, Optional
from services.airport_service import AirportService
from schemas.airports import AirportResponse

# --- IMPORTING YOUR REAL GEOLOCATION SERVICE ---
from services.ip_geolocation import IPGeolocationService 
# -----------------------------------------------
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
# Initialize your real service
geo_service = IPGeolocationService()


async def _get_client_country_code(request: Request) -> Optional[str]:
    """
    Extracts client IP and determines country code using IPGeolocationService.
    The response from the service is filtered for the 'country' field.
    """
    # Logic to extract client IP, prioritizing Cloudflare/Proxy headers
    client_ip = request.client.host
    forwarded = request.headers.get("X-Forwarded-For")
    cf_ip = request.headers.get("CF-Connecting-IP")
    
    if cf_ip:
        client_ip = cf_ip
    elif forwarded:
        client_ip = forwarded.split(",")[0].strip()

    if client_ip in ["127.0.0.1", "::1", "testclient"]:
        logger.debug(f"Skipping location lookup for local IP: {client_ip}")
        return None

    try:
        # We call the full service method and extract the country code
        result = await geo_service.get_nearest_airport_from_ip(client_ip)
        
        # Extract the country code from the detected_location part of your service's response
        country_code = result.get("detected_location", {}).get("country")
        
        return country_code.upper() if country_code else None
        
    except Exception as e:
        logger.warning(f"Failed to get country code for IP {client_ip}: {e}")
        return None


@router.get("/search", response_model=List[AirportResponse])
async def search_airports(
    request: Request,
    query: str = Query(
        ..., 
        min_length=1, # Allows single character search
        max_length=100,
        description="Search term (city, airport name, or IATA code)"
    ),
    limit: int = Query(
        10, 
        ge=1, 
        le=50,
        description="Maximum number of results"
    )
):
    """
    🔍 Smart airport autocomplete search.
    
    Results are prioritized by:
    1. Match Quality (Exact IATA, City Match)
    2. Multi-Airport City membership (e.g., LHR for LON query)
    3. User's Country (via IP Geolocation)
    4. Airport Popularity
    
    **Non-commercial airfields are automatically filtered out.**
    """
    # 1. Get the user's country code for location-based prioritization
    user_country_code = await _get_client_country_code(request)
    
    try:
        results = await AirportService.search_airports(
            query=query, 
            limit=limit,
            user_country_code=user_country_code # Pass the country code to the service
        )
        return results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/{iata_code}", response_model=AirportResponse)
async def get_airport(
    iata_code: str = Path(
        ..., 
        min_length=3, 
        max_length=3, 
        description="3-letter IATA code",
        example="JFK"
    )
):
    """
    Get specific airport details by IATA code.
    """
    airport = await AirportService.get_airport_by_iata(iata_code)
    
    if not airport:
        raise HTTPException(
            status_code=404, 
            detail=f"Airport with IATA code '{iata_code}' not found"
        )
    
    return airport


@router.post("/refresh-cache", status_code=204)
async def refresh_airport_cache():
    """
    🔄 Manually refresh the airport data cache.
    """
    await AirportService.refresh_cache()
    return None
