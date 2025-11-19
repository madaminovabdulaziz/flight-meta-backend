"""
Production Flight Search Node
Integrates with the enhanced FlightService to retrieve complete flight data.
"""

import logging
from datetime import datetime, date
from typing import Any, Dict, List

from app.langgraph_flow.state import ConversationState, update_state

logger = logging.getLogger(__name__)


async def flight_search_node(state: ConversationState) -> ConversationState:
    """
    Invoke the flight search backend using the collected slots.
    Populates raw_flights in the state with complete, ranking-ready data.
    """
    
    if not getattr(state, "ready_for_search", False):
        logger.warning("[FlightSearchNode] Called when ready_for_search=False")
        return state
    
    # Extract search parameters (dataclass-safe)
    origin = getattr(state, "origin", None)
    destination = getattr(state, "destination", None)
    departure_date = getattr(state, "departure_date", None)
    return_date = getattr(state, "return_date", None)
    passengers = getattr(state, "passengers", 1)
    travel_class = getattr(state, "travel_class", None)
    budget = getattr(state, "budget", None)
    flexibility = getattr(state, "flexibility", 0)
    
    logger.info(
        f"[FlightSearchNode] Searching flights {origin} → {destination} on {departure_date} "
        f"(return: {return_date}, pax={passengers}, class={travel_class}, budget={budget}, flex={flexibility})"
    )
    
    # Import FlightService
    try:
        from services.flight_service import FlightService
    except ImportError:
        logger.warning("[FlightSearchNode] Production flight service not found, using mock service")
        from services.flight_service import FlightService
    
    service = FlightService()
    
    try:
        # Call flight search service
        flights: List[Dict[str, Any]] = await service.search_flights(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            passengers=passengers,
            travel_class=travel_class,
            budget=budget,
            flexibility=flexibility,
        )
        
        logger.info(f"[FlightSearchNode] Retrieved {len(flights)} flights with complete metadata")
        
        # Log a sample flight
        if flights:
            sample = flights[0]
            logger.debug(
                f"[FlightSearchNode] Sample flight: {sample.get('airline_name')} "
                f"{sample.get('price')} {sample.get('currency')}, "
                f"{sample.get('stops')} stops, "
                f"rating={sample.get('airline_rating')}, "
                f"on_time={sample.get('on_time_score')}"
            )
        
        return update_state(state, {
            "raw_flights": flights,
            "search_executed": True,
            "search_timestamp": datetime.utcnow(),
            "errors": getattr(state, "errors", []) or [],
        })
    
    except Exception as e:
        logger.error(f"[FlightSearchNode] Flight search failed: {e}", exc_info=True)
        
        # Safely accumulate errors
        errors = getattr(state, "errors", []) or []
        errors.append(f"Flight search failed: {str(e)}")
        
        return update_state(state, {
            "raw_flights": [],
            "search_executed": False,
            "errors": errors,
            "assistant_message": (
                "I encountered an issue searching for flights. "
                "Would you like to try adjusting your search parameters?"
            ),
            "suggestions": [
                "Change dates",
                "Try different airports",
                "Adjust budget",
            ],
        })
