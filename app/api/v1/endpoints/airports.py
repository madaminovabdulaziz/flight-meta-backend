# app/api/v1/endpoints/airports.py

from fastapi import APIRouter, Query, Path, HTTPException
from typing import List
from services.airport_service import AirportService
from schemas.airports import AirportResponse

router = APIRouter()

@router.get("/search", response_model=List[AirportResponse])
async def search_airports(
    query: str = Query(
        ..., 
        min_length=2, 
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
    
    **Search by:**
    - IATA codes (e.g., "JFK", "LHR")
    - City names (e.g., "New York", "London")
    - Airport names (e.g., "Heathrow", "Kennedy")
    - Country names (e.g., "United States")
    
    **Ranking:**
    1. Exact IATA code match (highest)
    2. IATA code prefix match
    3. City name match
    4. Airport name match
    5. Country name match
    
    **Data Source:** OpenFlights (cached for 24 hours)
    """
    try:
        results = await AirportService.search_airports(query=query, limit=limit)
        return results
    except Exception as e:
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
    
    Example: `/airports/JFK`
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
    
    Use this endpoint if you need to force-update the airport data
    without waiting for the 24-hour TTL.
    """
    await AirportService.refresh_cache()
    return None  # 204 No Content response