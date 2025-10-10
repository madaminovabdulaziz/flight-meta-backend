# app/api/v1/endpoints/ai_search.py
"""
AI-Powered Conversational Flight Search - SIMPLIFIED

TravelPayouts only - no Amadeus complexity.
Clean, fast, revenue-generating from day 1.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

from services.ai_service import get_ai_service
from services.flight_search import get_flight_search
from schemas.flight_segment import Flight
from app.api.v1.dependencies import get_current_user_optional
from app.models.models import User

logger = logging.getLogger(__name__)
router = APIRouter()


# ==================== SCHEMAS ====================

class AISearchRequest(BaseModel):
    """Request for AI-powered flight search"""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language flight search query"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "I want a cheap weekend trip to Dubai in November"
            }
        }


class AISearchResponse(BaseModel):
    """Response from AI-powered search"""
    
    # AI Understanding
    understood: bool = Field(..., description="Whether AI understood the query")
    interpretation: Optional[str] = Field(None, description="AI's interpretation")
    
    # Extracted Parameters
    search_params: Optional[Dict[str, Any]] = Field(None)
    
    # Results
    flights: List[Flight] = Field(default_factory=list)
    total_results: int = 0
    
    # AI Insights
    ai_summary: Optional[str] = None
    recommendations: Optional[List[str]] = None
    
    # Clarification
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


# ==================== ENDPOINTS ====================

@router.post("/search", response_model=AISearchResponse)
async def ai_search_flights(
    request: AISearchRequest,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    🤖 AI-Powered Flight Search (TravelPayouts)
    
    Natural language → Flight results
    
    **Examples:**
    - "Cheap flight to Dubai next Friday"
    - "Weekend trip to Istanbul under $300"
    - "2 tickets to Moscow next month"
    - "Покажи рейсы в Москву" (Russian)
    """
    try:
        # Step 1: Parse query with AI
        logger.info(f"AI Search: {request.query[:100]}...")
        ai_service = get_ai_service()
        
        parse_result = await ai_service.parse_query(
            query=request.query,
            user_id=str(current_user.id) if current_user else None
        )
        
        # Step 2: Handle clarification
        if parse_result.get("needs_clarification"):
            return AISearchResponse(
                understood=False,
                needs_clarification=True,
                clarification_question=parse_result["question"],
                interpretation=f"Need clarification: {parse_result['missing_field']}"
            )
        
        # Step 3: Check parsing success
        if not parse_result.get("success"):
            return AISearchResponse(
                understood=False,
                interpretation=parse_result.get("error", "Could not understand query"),
                recommendations=[
                    "Try: 'Cheap flight to Dubai next Friday'",
                    "Or: 'Weekend trip to Istanbul under $300'",
                    "Or use the traditional search form"
                ]
            )
        
        # Step 4: Execute search with TravelPayouts
        search_params = parse_result["search_params"]
        logger.info(f"Search params: {search_params}")
        
        search_service = get_flight_search()
        search_result = await search_service.search_flights(
            origin=search_params["origin"],
            destination=search_params["destination"],
            departure_date=search_params["departure_date"],
            return_date=search_params.get("return_date"),
            adults=search_params.get("adults", 1),
            children=search_params.get("children", 0),
            cabin_class=search_params.get("cabin_class", "ECONOMY"),
            non_stop=search_params.get("non_stop", False),
            max_price=search_params.get("max_price"),
            currency="USD"
        )
        
        flights = search_result["flights"]
        metadata = search_result["metadata"]
        
        # Step 5: Generate AI summary
        if flights:
            flight_dicts = [f.model_dump() for f in flights[:10]]
            ai_summary = await ai_service.explain_results(
                query=request.query,
                results=flight_dicts,
                user_intent=search_params.get("user_intent", "find_best_value")
            )
        else:
            ai_summary = "Sorry, I couldn't find any flights for those dates. Try different dates or nearby airports."
        
        # Step 6: Combine recommendations
        all_recommendations = metadata.get("recommendations", [])
        
        return AISearchResponse(
            understood=True,
            interpretation=parse_result.get("ai_interpretation", "Search understood!"),
            search_params=search_params,
            flights=flights[:20],  # Top 20
            total_results=search_result["total_results"],
            ai_summary=ai_summary,
            recommendations=all_recommendations,
            needs_clarification=False
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during search. Please try again."
        )


@router.get("/cheapest")
async def get_cheapest_flights(
    origin: str = "TAS",
    limit: int = 10
):
    """
    💰 Cheapest Flights from Origin
    
    Find the absolute cheapest destinations from an airport.
    Perfect for "Where can I fly cheaply?" queries.
    """
    search = get_flight_search()
    
    try:
        results = await search.get_cheapest_from_origin(
            origin=origin.upper(),
            limit=limit
        )
        
        return {
            "origin": origin.upper(),
            "cheapest_destinations": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Cheapest flights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calendar")
async def get_price_calendar(
    origin: str,
    destination: str,
    month: Optional[str] = None
):
    """
    📅 Price Calendar - Flexible Dates
    
    See prices for every day of the month.
    Perfect for "I'm flexible with dates" users.
    
    Month format: YYYY-MM (e.g., "2025-11")
    """
    search = get_flight_search()
    
    try:
        calendar = await search.get_price_calendar(
            origin=origin.upper(),
            destination=destination.upper(),
            month=month
        )
        
        return calendar
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot-routes")
async def get_hot_routes_endpoint(
    origin: str = "TAS",
    limit: int = 10
):
    """
    🔥 Hot Routes - Popular Destinations
    
    Trending routes from an origin.
    Based on search volume and price trends.
    """
    search = get_flight_search()
    
    try:
        routes = await search.get_hot_routes(
            origin=origin.upper(),
            limit=limit
        )
        
        return {
            "origin": origin.upper(),
            "hot_routes": routes,
            "count": len(routes)
        }
    except Exception as e:
        logger.error(f"Hot routes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/booking-link")
async def generate_booking_link(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    adults: int = 1
):
    """
    🔗 Generate Affiliate Booking Link
    
    THIS IS HOW YOU MAKE MONEY! 💰
    
    Returns TravelPayouts deeplink with your affiliate ID.
    User clicks → books → you earn 3-5% commission.
    """
    search = get_flight_search()
    
    try:
        link = search.build_booking_link(
            origin=origin.upper(),
            destination=destination.upper(),
            departure_date=departure_date,
            return_date=return_date,
            adults=adults
        )
        
        return {
            "booking_url": link,
            "note": "User will be redirected to TravelPayouts booking page"
        }
    except Exception as e:
        logger.error(f"Booking link error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/demo-queries")
async def get_demo_queries():
    """
    📝 Example Queries for AI Search
    
    Show users what they can ask.
    """
    return {
        "examples": [
            {
                "query": "Cheap flight to Dubai next Friday",
                "language": "English",
                "expected": "One-way to DXB, next Friday, sorted by price"
            },
            {
                "query": "Weekend trip to Istanbul under $300",
                "language": "English",
                "expected": "Round-trip, Fri-Sun, max price $300"
            },
            {
                "query": "2 tickets to Moscow next month",
                "language": "English",
                "expected": "2 passengers, next month dates"
            },
            {
                "query": "Покажи рейсы в Москву на следующей неделе",
                "language": "Russian",
                "expected": "Flights to Moscow next week"
            },
            {
                "query": "Business class to London, December 15",
                "language": "English",
                "expected": "Business cabin, specific date"
            }
        ],
        "tips": [
            "Be specific about dates and destination",
            "Mention budget if price matters: 'cheap', 'under $300'",
            "Specify passengers: '2 people', 'family of 4'",
            "Say 'direct' or 'non-stop' if you don't want layovers"
        ]
    }