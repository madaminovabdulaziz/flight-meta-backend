# # app/api/v1/endpoints/flights.py
# """
# Flight Search Endpoints - Production Ready
# Supports both AI-powered conversational search and traditional parameter-based search.
# """

# from fastapi import APIRouter, Query, HTTPException, Path, Depends, Body
# from typing import List, Dict, Any, Optional
# from datetime import datetime, timedelta
# from enum import Enum
# import logging
# from services import amadeus_service
# from schemas.flight_segment import Flight
# # NEW IMPORT: Import the new booking schemas
# from schemas.booking import PriceFlightRequest, CreateOrderRequest 
# from app.models.models import User
# from app.api.v1.dependencies import get_current_user
# from services.ai_service import get_ai_service, AIFlightSearchService

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# # Enums for validation
# class CabinClass(str, Enum):
#     ECONOMY = "ECONOMY"
#     PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
#     BUSINESS = "BUSINESS"
#     FIRST = "FIRST"

# class TripType(str, Enum):
#     ONE_WAY = "one-way"
#     ROUND_TRIP = "round-trip"

# class UserIntent(str, Enum):
#     CHEAPEST = "find_cheapest"
#     FASTEST = "find_fastest"
#     BEST_VALUE = "find_best_value"
#     COMPARE = "compare_options"

# def validate_date(date_str: str, field_name: str) -> datetime:
#     """Validate and parse date string"""
#     try:
#         date_obj = datetime.strptime(date_str, "%Y-%m-%d")
#         if date_obj.date() < datetime.now().date():
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"{field_name} cannot be in the past"
#             )
#         return date_obj
#     except ValueError:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"Invalid {field_name} format. Use YYYY-MM-DD"
#         )

# def validate_iata_code(code: str, field_name: str) -> str:
#     """Validate IATA airport code"""
#     if not code or len(code) != 3 or not code.isalpha():
#         raise HTTPException(
#             status_code=400,
#             detail=f"{field_name} must be a valid 3-letter IATA code"
#         )
#     return code.upper()


# # ==================== AI-POWERED SEARCH ====================

# @router.post("/search/ai", response_model=Dict[str, Any])
# async def search_flights_ai(
#     query: str = Body(
#         ...,
#         description="Natural language query for flight search",
#         example="I want a cheap, non-stop flight to Dubai next Friday under $400"
#     ),
#     user_id: Optional[str] = Body(
#         None, 
#         description="Optional user ID for conversation context"
#     ),
#     ai_service: AIFlightSearchService = Depends(get_ai_service)
# ):
#     """
#     AI-Powered Conversational Flight Search
    
#     Translates natural language into structured search and returns results.
#     Handles conversation state for follow-up questions.
    
#     Examples:
#     - "Cheap tickets to Dubai next month"
#     - "I need to be in Istanbul on November 25th"
#     - "Weekend trip to Moscow for 2 people"
#     - "Show me business class flights to Seoul"
#     """
#     logger.info(f"AI Search Query from user {user_id}: {query}")
    
#     # Parse query using AI service
#     ai_result = await ai_service.parse_query(query, user_id=user_id)
    
#     # Handle clarification requests
#     if not ai_result.get("success"):
#         if ai_result.get("needs_clarification"):
#             return {
#                 "status": "clarification_needed",
#                 "question": ai_result["question"],
#                 "missing_field": ai_result["missing_field"],
#                 "ai_interpretation": ai_result.get("ai_response", "Please provide more details.")
#             }
#         else:
#             raise HTTPException(
#                 status_code=422,
#                 detail=ai_result.get("error", "Could not understand the search request. Please try rephrasing.")
#             )
    
#     # Extract and validate structured parameters
#     params = ai_result["search_params"]
    
#     # Backend validation (IATA codes and dates)
#     try:
#         origin_iata = validate_iata_code(params["origin"], "Origin")
#         destination_iata = validate_iata_code(params["destination"], "Destination")
        
#         validate_date(params["departure_date"], "Departure date")
#         if params["return_date"]:
#             validate_date(params["return_date"], "Return date")
             
#     except HTTPException as e:
#         return {
#             "status": "validation_failed",
#             "detail": e.detail,
#             "ai_interpretation": f"I found the parameters, but there was an issue: {e.detail}"
#         }

#     logger.info(f"Structured search parameters: {params}")

#     # Call Amadeus service
#     try:
#         # Choose search function based on flexibility
#         if params.get("flexible_dates"):
#             search_func = amadeus_service.search_flexible_flights_amadeus
#         else:
#             search_func = amadeus_service.search_flights_amadeus
        
#         flights = await search_func(
#             origin=origin_iata,
#             destination=destination_iata,
#             departure_date=params["departure_date"],
#             return_date=params["return_date"],
#             adults=params["adults"],
#             children=params["children"],
#             infants=params["infants"],
#             cabin_class=params["cabin_class"],
#             non_stop=params["non_stop"],
#             currency=params.get("currency", "USD"),
#             max_price=params.get("max_price"),
#             user_intent=params["user_intent"]
#         )

#         # Generate AI summary
#         flights_dump = [f.model_dump() for f in flights]
#         summary = await ai_service.explain_results(query, flights_dump, params["user_intent"])

#         return {
#             "status": "success",
#             "ai_summary": summary,
#             "search_params_used": params,
#             "total_results": len(flights),
#             "results": flights
#         }

#     except amadeus_service.AmadeusTimeoutError as e:
#         logger.error(f"Amadeus timeout: {e}")
#         raise HTTPException(
#             status_code=504,
#             detail="Flight search timed out. Please try again."
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during Amadeus call: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail="An unexpected error occurred while processing the search."
#         )


# # ==================== TRADITIONAL SEARCH ====================

# @router.get("/search", response_model=List[Flight])
# async def search_flights(
#     # Required parameters
#     origin: str = Query(
#         ..., 
#         description="Departure airport IATA code",
#         example="TAS",
#         min_length=3,
#         max_length=3
#     ),
#     destination: str = Query(
#         ..., 
#         description="Arrival airport IATA code",
#         example="IST",
#         min_length=3,
#         max_length=3
#     ),
#     departure_date: str = Query(
#         ..., 
#         description="Departure date in YYYY-MM-DD format",
#         example="2025-11-15"
#     ),
    
#     # Trip type and return date
#     trip_type: TripType = Query(
#         TripType.ONE_WAY,
#         description="Type of trip: one-way or round-trip"
#     ),
#     return_date: Optional[str] = Query(
#         None,
#         description="Return date for round-trip (YYYY-MM-DD format)",
#         example="2025-11-22"
#     ),
    
#     # Passenger details
#     adults: int = Query(
#         1,
#         ge=1,
#         le=8,
#         description="Number of adult passengers (12+ years)"
#     ),
#     children: int = Query(
#         0,
#         ge=0,
#         le=8,
#         description="Number of child passengers (2-11 years)"
#     ),
#     infants: int = Query(
#         0,
#         ge=0,
#         le=8,
#         description="Number of infant passengers (under 2 years)"
#     ),
    
#     # Flight preferences
#     cabin_class: CabinClass = Query(
#         CabinClass.ECONOMY,
#         description="Preferred cabin class"
#     ),
#     non_stop: bool = Query(
#         False,
#         description="Search only non-stop flights"
#     ),
    
#     # NEW: User intent and price filtering
#     user_intent: UserIntent = Query(
#         UserIntent.BEST_VALUE,
#         description="Search priority: cheapest, fastest, best value, or compare options"
#     ),
#     max_price: Optional[float] = Query(
#         None,
#         ge=0,
#         description="Maximum price in USD (optional price filter)"
#     ),
    
#     # Results control
#     max_results: int = Query(
#         20,
#         ge=1,
#         le=50,
#         description="Maximum number of results to return"
#     ),
#     currency: str = Query(
#         "USD",
#         description="Preferred currency code (3-letter ISO)",
#         min_length=3,
#         max_length=3
#     )
# ):
#     """
#     Traditional Flight Search with Comprehensive Filtering
    
#     Supports both one-way and round-trip searches with full parameter control.
#     Now includes user_intent and max_price for better result filtering.
#     """
#     try:
#         # Validate IATA codes
#         origin = validate_iata_code(origin, "Origin")
#         destination = validate_iata_code(destination, "Destination")
        
#         # Validate origin != destination
#         if origin == destination:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Origin and destination must be different"
#             )
        
#         # Validate departure date
#         dep_date = validate_date(departure_date, "Departure date")
        
#         # Validate passenger count
#         total_passengers = adults + children + infants
#         if total_passengers > 9:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Maximum 9 passengers allowed per booking"
#             )
        
#         if infants > adults:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Number of infants cannot exceed number of adults"
#             )
        
#         # Handle round-trip validation
#         if trip_type == TripType.ROUND_TRIP:
#             if not return_date:
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Return date is required for round-trip searches"
#                 )
            
#             ret_date = validate_date(return_date, "Return date")
            
#             if ret_date <= dep_date:
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Return date must be after departure date"
#                 )
            
#             # Check if trip duration is reasonable (not more than 1 year)
#             if (ret_date - dep_date).days > 365:
#                 raise HTTPException(
#                     status_code=400,
#                     detail="Trip duration cannot exceed 365 days"
#                 )
        
#         # Log search request
#         logger.info(
#             f"Flight search: {origin}->{destination}, "
#             f"Dep: {departure_date}, Type: {trip_type}, "
#             f"PAX: {adults}A/{children}C/{infants}I, Class: {cabin_class}, "
#             f"Intent: {user_intent}, MaxPrice: {max_price}"
#         )
        
#         # Call Amadeus service with ALL parameters including user_intent and max_price
#         flights = await amadeus_service.search_flights_amadeus(
#             origin=origin,
#             destination=destination,
#             departure_date=departure_date,
#             return_date=return_date if trip_type == TripType.ROUND_TRIP else None,
#             adults=adults,
#             children=children,
#             infants=infants,
#             cabin_class=cabin_class.value,
#             non_stop=non_stop,
#             max_results=max_results,
#             currency=currency.upper(),
#             max_price=max_price,  # NEW
#             user_intent=user_intent.value  # NEW
#         )
        
#         logger.info(f"Found {len(flights)} flights for {origin}->{destination}")
#         return flights
        
#     except amadeus_service.AmadeusTimeoutError as e:
#         logger.error(f"Amadeus timeout: {e}")
#         raise HTTPException(
#             status_code=504,
#             detail="Flight search timed out. Please try again."
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error in search endpoint: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail="An error occurred while searching for flights"
#         )


# # ==================== FLIGHT DETAILS / RE-PRICING ====================

# @router.post("/price", response_model=Dict[str, Any])
# async def price_selected_flight(
#     request: PriceFlightRequest = Body(...)
# ):
#     """
#     Step 2: Re-validates and re-prices a flight offer to ensure availability and current fare.
#     The response contains the final, guaranteed price before booking.
#     """
#     logger.info(f"Pricing flight ID: {request.flight_id}")
    
#     # 1. Retrieve the full Amadeus offer object from the cache
#     search_params = request.model_dump(exclude_none=True)
#     full_offer_data = await amadeus_service.get_full_flight_offer(
#         request.flight_id, 
#         search_params
#     )
    
#     if not full_offer_data:
#         raise HTTPException(
#             status_code=404,
#             detail="Flight offer not found in cache. Please perform a new search."
#         )
    
#     # 2. Call the Amadeus Pricing API
#     try:
#         priced_offer = await amadeus_service.price_flight_offer(full_offer_data["offer"])
#         logger.info(f"Flight ID {request.flight_id} successfully priced.")
#         return priced_offer
        
#     except amadeus_service.AmadeusTimeoutError:
#         raise HTTPException(
#             status_code=504,
#             detail="Pricing request timed out. The Amadeus service may be busy."
#         )
#     except HTTPException as e:
#         if e.status_code == 400:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Flight price is no longer valid or offer is expired."
#             )
#         raise
#     except Exception as e:
#         logger.error(f"Error during pricing: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail="An unexpected error occurred while re-pricing the flight."
#         )


# # ==================== FLIGHT ORDER / BOOKING ====================

# @router.post("/order", response_model=Dict[str, Any], status_code=201)
# async def create_flight_order(
#     request: CreateOrderRequest = Body(...)
# ):
#     """
#     Step 3: Creates the final flight order (PNR) using the priced offer data.
#     Note: In the Amadeus Self-Service environment, this creates the booking but requires
#     a separate consolidator agreement for final ticket issuance.
#     """
#     if not request.passengers:
#         raise HTTPException(status_code=400, detail="Passenger details are required for booking.")
    
#     logger.info(f"Attempting to create flight order for {len(request.passengers)} passengers.")
    
#     try:
#         # The service function expects a list of dictionaries for passengers
#         passenger_list = [p.model_dump() for p in request.passengers]
        
#         order_response = await amadeus_service.create_flight_order(
#             request.priced_offer_data,
#             passenger_list
#         )
        
#         # Log PNR confirmation
#         pnr = order_response.get("data", {}).get("id")
#         logger.info(f"Successfully created PNR/Order ID: {pnr}")
        
#         return {
#             "status": "success",
#             "message": "Flight order created successfully. PNR ready for ticketing.",
#             "order_id": pnr,
#             "amadeus_response": order_response
#         }
        
#     except amadeus_service.AmadeusTimeoutError:
#         raise HTTPException(
#             status_code=504,
#             detail="Booking request timed out. The Amadeus service may be busy."
#         )
#     except HTTPException as e:
#         # Handle 400s which could mean invalid data or expired fare
#         if e.status_code in (400, 404):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Booking failed due to invalid data or expired fare: {e.detail}"
#             )
#         raise
#     except Exception as e:
#         logger.error(f"Error during order creation: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail="An unexpected error occurred while creating the flight order."
#         )


# # ==================== FLIGHT DETAILS / CACHE RETRIEVAL ====================

# @router.get("/{flight_id}/details", response_model=Dict[str, Any])
# async def get_flight_details(
#     flight_id: str = Path(..., description="The ID of the flight offer"),
#     origin: str = Query(...),
#     destination: str = Query(...),
#     departure_date: str = Query(...)
# ):
#     """
#     Get detailed information about a specific flight offer.
#     Requires the original search parameters to retrieve the full Amadeus offer from cache.
#     """
#     try:
#         search_params = {
#             "origin": origin,
#             "destination": destination,
#             "departure_date": departure_date
#         }
        
#         flight_details = await amadeus_service.get_full_flight_offer(
#             flight_id, 
#             search_params
#         )
        
#         if not flight_details:
#             raise HTTPException(
#                 status_code=404,
#                 detail="Flight not found. The offer may have expired or search parameters are incorrect."
#             )
        
#         return flight_details
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching flight details: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail="Could not retrieve flight details"
#         )


# # ==================== FLEXIBLE DATE SEARCH ====================

# @router.get("/search/flexible", response_model=List[Flight])
# async def search_flights_flexible(
#     origin: str = Query(
#         ..., 
#         description="Departure airport IATA code",
#         example="TAS"
#     ),
#     destination: str = Query(
#         ..., 
#         description="Arrival airport IATA code",
#         example="IST"
#     ),
#     departure_date: str = Query(
#         ...,
#         description="Target departure date (will search ±3 days)",
#         example="2025-11-15"
#     ),
#     return_date: Optional[str] = Query(
#         None,
#         description="Target return date (will search ±3 days if provided)",
#         example="2025-11-22"
#     ),
#     adults: int = Query(1, ge=1, le=8),
#     children: int = Query(0, ge=0, le=8),
#     infants: int = Query(0, ge=0, le=8),
#     cabin_class: CabinClass = Query(CabinClass.ECONOMY),
#     non_stop: bool = Query(False),
#     max_price: Optional[float] = Query(None, ge=0),
#     user_intent: UserIntent = Query(UserIntent.BEST_VALUE),
#     currency: str = Query("USD", min_length=3, max_length=3)
# ):
#     """
#     Flexible date search: finds flights across multiple dates (±3 days).
#     Perfect for travelers with flexible schedules looking for the best deals.
#     """
#     try:
#         # Validate IATA codes
#         origin = validate_iata_code(origin, "Origin")
#         destination = validate_iata_code(destination, "Destination")
        
#         if origin == destination:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Origin and destination must be different"
#             )
        
#         # Validate dates
#         validate_date(departure_date, "Departure date")
#         if return_date:
#             validate_date(return_date, "Return date")
        
#         logger.info(
#             f"Flexible search: {origin}->{destination}, "
#             f"Dep: {departure_date}±3d, PAX: {adults}A/{children}C/{infants}I, "
#             f"Intent: {user_intent}"
#         )
        
#         flights = await amadeus_service.search_flexible_flights_amadeus(
#             origin=origin,
#             destination=destination,
#             departure_date=departure_date,
#             return_date=return_date,
#             flexible_dates=True,
#             adults=adults,
#             children=children,
#             infants=infants,
#             cabin_class=cabin_class.value,
#             non_stop=non_stop,
#             currency=currency.upper(),
#             max_price=max_price,
#             user_intent=user_intent.value
#         )
        
#         logger.info(f"Flexible search returned {len(flights)} options")
#         return flights
        
#     except amadeus_service.AmadeusTimeoutError as e:
#         logger.error(f"Flexible search timeout: {e}")
#         raise HTTPException(
#             status_code=504,
#             detail="Flexible search timed out. Please try again."
#         )
#     except Exception as e:
#         logger.error(f"Error in flexible search: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail="An error occurred during flexible search"
#         )


# # ==================== MULTI-CITY (PLACEHOLDER) ====================

# @router.get("/search/multi-city", response_model=Dict[str, Any])
# async def search_multi_city_flights():
#     """
#     Multi-city flight search (coming soon).
#     For complex itineraries with multiple stops.
#     """
#     raise HTTPException(
#         status_code=501,
#         detail="Multi-city search is not yet implemented. Stay tuned!"
#     )
