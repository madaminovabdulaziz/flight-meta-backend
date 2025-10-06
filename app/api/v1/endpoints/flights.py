from fastapi import APIRouter, Query, HTTPException, Path, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import logging
from services import amadeus_service
from schemas.flight_segment import Flight
from app.models.models import User
from app.api.v1.dependencies import get_current_user

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Enums for validation
class CabinClass(str, Enum):
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"

class TripType(str, Enum):
    ONE_WAY = "one-way"
    ROUND_TRIP = "round-trip"

def validate_date(date_str: str, field_name: str) -> datetime:
    """Validate and parse date string"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj.date() < datetime.now().date():
            raise HTTPException(
                status_code=400, 
                detail=f"{field_name} cannot be in the past"
            )
        return date_obj
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid {field_name} format. Use YYYY-MM-DD"
        )

def validate_iata_code(code: str, field_name: str) -> str:
    """Validate IATA airport code"""
    if not code or len(code) != 3 or not code.isalpha():
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be a valid 3-letter IATA code"
        )
    return code.upper()

@router.get("/search", response_model=List[Flight])
async def search_flights(
    # Required parameters
    origin: str = Query(
        ..., 
        description="Departure airport IATA code",
        example="TAS",
        min_length=3,
        max_length=3
    ),
    destination: str = Query(
        ..., 
        description="Arrival airport IATA code",
        example="IST",
        min_length=3,
        max_length=3
    ),
    departure_date: str = Query(
        ..., 
        description="Departure date in YYYY-MM-DD format",
        example="2025-11-15"
    ),
    
    # Trip type and return date
    trip_type: TripType = Query(
        TripType.ONE_WAY,
        description="Type of trip: one-way or round-trip"
    ),
    return_date: Optional[str] = Query(
        None,
        description="Return date for round-trip (YYYY-MM-DD format)",
        example="2025-11-22"
    ),
    
    # Passenger details
    adults: int = Query(
        1,
        ge=1,
        le=8,
        description="Number of adult passengers (12+ years)"
    ),
    children: int = Query(
        0,
        ge=0,
        le=8,
        description="Number of child passengers (2-11 years)"
    ),
    infants: int = Query(
        0,
        ge=0,
        le=8,
        description="Number of infant passengers (under 2 years)"
    ),
    
    # Flight preferences
    cabin_class: CabinClass = Query(
        CabinClass.ECONOMY,
        description="Preferred cabin class"
    ),
    non_stop: bool = Query(
        False,
        description="Search only non-stop flights"
    ),
    
    # Results control
    max_results: int = Query(
        20,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    ),
    currency: str = Query(
        "USD",
        description="Preferred currency code (3-letter ISO)",
        min_length=3,
        max_length=3
    )
):
    """
    Search for flights with comprehensive filtering options.
    Supports both one-way and round-trip searches.
    """
    try:
        # Validate IATA codes
        origin = validate_iata_code(origin, "Origin")
        destination = validate_iata_code(destination, "Destination")
        
        # Validate origin != destination
        if origin == destination:
            raise HTTPException(
                status_code=400,
                detail="Origin and destination must be different"
            )
        
        # Validate departure date
        dep_date = validate_date(departure_date, "Departure date")
        
        # Validate passenger count
        total_passengers = adults + children + infants
        if total_passengers > 9:
            raise HTTPException(
                status_code=400,
                detail="Maximum 9 passengers allowed per booking"
            )
        
        if infants > adults:
            raise HTTPException(
                status_code=400,
                detail="Number of infants cannot exceed number of adults"
            )
        
        # Handle round-trip validation
        if trip_type == TripType.ROUND_TRIP:
            if not return_date:
                raise HTTPException(
                    status_code=400,
                    detail="Return date is required for round-trip searches"
                )
            
            ret_date = validate_date(return_date, "Return date")
            
            if ret_date <= dep_date:
                raise HTTPException(
                    status_code=400,
                    detail="Return date must be after departure date"
                )
            
            # Check if trip duration is reasonable (not more than 1 year)
            if (ret_date - dep_date).days > 365:
                raise HTTPException(
                    status_code=400,
                    detail="Trip duration cannot exceed 365 days"
                )
        
        # Log search request
        logger.info(
            f"Flight search: {origin}->{destination}, "
            f"Dep: {departure_date}, Type: {trip_type}, "
            f"PAX: {adults}A/{children}C/{infants}I, Class: {cabin_class}"
        )
        
        # Call Amadeus service
        # Note: You'll need to extend amadeus_service to handle these new parameters
        flights = await amadeus_service.search_flights_amadeus(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date if trip_type == TripType.ROUND_TRIP else None,
            adults=adults,
            children=children,
            infants=infants,
            cabin_class=cabin_class.value,
            non_stop=non_stop,
            max_results=max_results,
            currency=currency.upper()
        )
        
        logger.info(f"Found {len(flights)} flights for {origin}->{destination}")
        return flights
        
    except amadeus_service.AmadeusTimeoutError as e:
        logger.error(f"Amadeus timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="Flight search timed out. Please try again."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching for flights"
        )


@router.get("/{flight_id}/details", response_model=Dict[str, Any])
async def get_flight_details(
    flight_id: str = Path(..., description="The ID of the flight offer"),
    origin: str = Query(...),
    destination: str = Query(...),
    departure_date: str = Query(...)
):
    """
    Get detailed information about a specific flight offer.
    Requires the original search parameters to retrieve from cache.
    """
    try:
        search_params = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date
        }
        
        flight_details = await amadeus_service.get_full_flight_offer(
            flight_id, 
            search_params
        )
        
        if not flight_details:
            raise HTTPException(
                status_code=404,
                detail="Flight not found. The offer may have expired or search parameters are incorrect."
            )
        
        return flight_details
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching flight details: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve flight details"
        )


@router.get("/search/flexible", response_model=List[Flight])
async def search_flights_flexible(
    origin: str = Query(
        ..., 
        description="Departure airport IATA code",
        example="TAS"
    ),
    destination: str = Query(
        ..., 
        description="Arrival airport IATA code",
        example="IST"
    ),
    duration_days: int = Query(
        ..., 
        ge=1,
        le=30,
        description="Trip duration in days (1-30)",
        example=7
    ),
    adults: int = Query(1, ge=1, le=8),
    children: int = Query(0, ge=0, le=8),
    cabin_class: CabinClass = Query(CabinClass.ECONOMY)
):
    """
    Flexible date search: finds the cheapest flights for a given trip duration
    across multiple departure dates.
    """
    try:
        # Validate IATA codes
        origin = validate_iata_code(origin, "Origin")
        destination = validate_iata_code(destination, "Destination")
        
        if origin == destination:
            raise HTTPException(
                status_code=400,
                detail="Origin and destination must be different"
            )
        
        logger.info(
            f"Flexible search: {origin}->{destination}, "
            f"Duration: {duration_days}d, PAX: {adults}A/{children}C"
        )
        
        flights = await amadeus_service.search_flexible_flights_amadeus(
            origin=origin,
            destination=destination,
            duration_days=duration_days
        )
        
        # Sort by price ascending for flexible search
        flights.sort(key=lambda f: f.price)
        
        logger.info(f"Flexible search returned {len(flights)} options")
        return flights
        
    except amadeus_service.AmadeusTimeoutError as e:
        logger.error(f"Flexible search timeout: {e}")
        raise HTTPException(
            status_code=504,
            detail="Flexible search timed out. Please try again."
        )
    except Exception as e:
        logger.error(f"Error in flexible search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during flexible search"
        )


@router.get("/search/multi-city", response_model=Dict[str, Any])
async def search_multi_city_flights(
    # This is a placeholder for future multi-city functionality
):
    """
    Multi-city flight search (coming soon).
    For complex itineraries with multiple stops.
    """
    raise HTTPException(
        status_code=501,
        detail="Multi-city search is not yet implemented"
    )










# from fastapi import APIRouter, Query, HTTPException, Path, Depends
# from typing import List, Dict, Any
# import logging
# from services import amadeus_service
# from schemas.flight import Flight
# # We need this for the new saved searches integration
# from app.models.models import User
# from app.api.v1.dependencies import get_current_user

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# @router.get("/search", response_model=List[Flight])
# async def search_flights(
#     origin: str = Query(..., description="Departure city IATA code", example="TAS"),
#     destination: str = Query(..., description="Arrival city IATA code", example="IST"),
#     departure_date: str = Query(..., description="Departure date in YYYY-MM-DD format", example="2025-10-25")
# ):
#     try:
#         flights = await amadeus_service.search_flights_amadeus(
#             origin=origin, destination=destination, departure_date=departure_date
#         )
#         return flights
#     except Exception as e:
#         logger.error(f"An unexpected error occurred in search endpoint: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An internal server error occurred.")

# # --- THE FIX IS HERE ---
# # The path should NOT include '/flights' again, because the prefix is already
# # handled in main.py.
# @router.get("/{flight_id}/details", response_model=Dict[str, Any])
# async def get_flight_details(
#     flight_id: str = Path(..., description="The ID of the flight offer"),
#     origin: str = Query(...),
#     destination: str = Query(...),
#     departure_date: str = Query(...)
# ):
#     try:
#         search_params = {"origin": origin, "destination": destination, "departure_date": departure_date}
#         flight_details = await amadeus_service.get_full_flight_offer(flight_id, search_params)
        
#         if not flight_details:
#             raise HTTPException(status_code=404, detail="Flight details not found or cache expired.")
#         return flight_details
#     except Exception as e:
#         logger.error(f"Error fetching flight details in endpoint: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Could not retrieve flight details.")



# @router.get("/search/flexible", response_model=List[Flight])
# async def search_flights_flexible(
#     origin: str = Query(..., description="Departure city IATA code", example="TAS"),
#     destination: str = Query(..., description="Arrival city IATA code", example="IST"),
#     duration_days: int = Query(..., description="The duration of the trip in days", example=7)
# ):
#     try:
#         flights = await amadeus_service.search_flexible_flights_amadeus(
#             origin=origin, destination=destination, duration_days=duration_days
#         )
#         # We sort by price ascending for the flexible search results
#         flights.sort(key=lambda f: f.price)
#         return flights
#     except amadeus_service.AmadeusTimeoutError as e:
#         raise HTTPException(status_code=504, detail=str(e))
#     except Exception as e:
#         logger.error(f"An unexpected error in flexible search endpoint: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="An internal server error occurred.")

