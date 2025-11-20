"""
Production Flight Search Node
Integrates REAL Duffel API with robust validation
"""

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from app.langgraph_flow.state import ConversationState, update_state
from app.api.v1.endpoints.duffel_new import duffel

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def _validate_search_params(state: ConversationState) -> Optional[str]:
    """
    Validate search parameters before calling Duffel.
    Returns: Error message if validation fails, None if OK.
    """
    # Check origin
    origin = _resolve_airport(state.origin_airports, state.origin)
    if not origin:
        return "Missing or invalid origin airport"
    
    # Check destination
    destination = _resolve_airport(state.destination_airports, state.destination)
    if not destination:
        return "Missing or invalid destination airport"
    
    # Check if origin == destination
    if origin.upper() == destination.upper():
        return f"Origin and destination cannot be the same ({origin}). Please specify a different destination."
    
    # Check departure date
    if not state.departure_date:
        return "Missing departure date"
    
    # Check passengers
    if not state.passengers or state.passengers < 1:
        return "Invalid passenger count"
    
    return None


def _resolve_airport(airport_list: List[str], fallback: Optional[str]) -> Optional[str]:
    """
    Resolve airport code from list or fallback.
    """
    if airport_list and len(airport_list) > 0:
        return airport_list[0].upper()
    
    if fallback and isinstance(fallback, str) and len(fallback) >= 3:
        return fallback.upper()
    
    return None


def _iso_duration_to_minutes(iso_duration: str) -> int:
    """
    Parse ISO 8601 duration (PT1H30M) to minutes.
    """
    if not iso_duration or not iso_duration.startswith("PT"):
        return 0
    
    try:
        duration = iso_duration[2:]
        hours = 0
        minutes = 0
        
        if "H" in duration:
            hours_part, duration = duration.split("H")
            hours = int(hours_part)
        
        if "M" in duration:
            minutes_part = duration.split("M")[0]
            minutes = int(minutes_part)
        
        return (hours * 60) + minutes
    
    except Exception as e:
        logger.warning(f"Failed to parse ISO duration '{iso_duration}': {e}")
        return 0


# ============================================================================
# FLIGHT NORMALIZATION (FIXED)
# ============================================================================

def _normalize_duffel_offer(offer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform Duffel offer into internal flight schema.
    Handles NULL values defensively to prevent crashes.
    """
    try:
        # Defensive: Handle explicit None/Null in high-level keys
        owner = offer.get("owner") or {}
        slices = offer.get("slices") or []
        
        if not slices:
            return None
        
        # Safe extract amount
        total_amount_str = offer.get("total_amount")
        total_amount = float(total_amount_str) if total_amount_str else 0.0
        currency = offer.get("total_currency", "USD")
        
        # Extract first slice (outbound)
        first_slice = slices[0]
        segments = first_slice.get("segments") or []
        
        if not segments:
            return None
        
        # Calculate stops
        stops = max(0, len(segments) - 1)
        
        # Get duration
        duration_minutes = _iso_duration_to_minutes(first_slice.get("duration", "PT0M"))
        
        # Get departure/arrival times
        first_seg = segments[0]
        last_seg = segments[-1]
        
        departure_time = first_seg.get("departing_at", "")
        arrival_time = last_seg.get("arriving_at", "")
        
        # Get airports (Safe access)
        origin_dict = first_seg.get("origin") or {}
        dest_dict = last_seg.get("destination") or {}
        
        origin = origin_dict.get("iata_code", "")
        destination = dest_dict.get("iata_code", "")
        
        # 🛡️ FIXED: Defensive extraction for nested conditions
        # "conditions" can be None in JSON, so .get("conditions", {}) fails if value is explicitly null
        conditions = offer.get("conditions") or {}
        refund_policy = conditions.get("refund_before_departure") or {}
        is_refundable = refund_policy.get("allowed", False)
        
        # Safe baggage check
        baggage_included = _has_baggage(offer)

        return {
            "id": offer.get("id"),
            "airline_name": owner.get("name", "Unknown Airline"),
            "airline": owner.get("iata_code", "XX"),
            "price": total_amount,
            "currency": currency,
            "origin": origin,
            "destination": destination,
            "departure_time": departure_time,
            "arrival_time": arrival_time,
            "duration_minutes": duration_minutes,
            "stops": stops,
            "layovers": _extract_layovers(segments),
            "travel_class": offer.get("cabin_class", "economy").title(),
            "refundable": is_refundable,
            "baggage_included": baggage_included,
            "airline_rating": 4.0,
            "on_time_score": 0.8,
            "departure_airport_distance_to_city_km": 30,
            "arrival_airport_distance_to_city_km": 30,
            "booking_link": f"https://duffel.com/book/{offer.get('id')}",
            "raw_data": offer,
            "is_mock": False,
        }
    
    except Exception as e:
        # Log strict warning but don't crash the whole search
        # Only log the ID to keep logs clean
        logger.warning(f"Skipping malformed Duffel offer {offer.get('id', 'unknown')}: {e}")
        return None


def _extract_layovers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract layover information from segments."""
    layovers = []
    
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]
        
        current_arrival = current.get("arriving_at", "")
        next_departure = next_seg.get("departing_at", "")
        
        if current_arrival and next_departure:
            try:
                arr_time = datetime.fromisoformat(current_arrival.replace("Z", "+00:00"))
                dep_time = datetime.fromisoformat(next_departure.replace("Z", "+00:00"))
                
                layover_duration = (dep_time - arr_time).total_seconds() / 60  # minutes
                
                dest_dict = current.get("destination") or {}
                
                layovers.append({
                    "airport": dest_dict.get("iata_code", ""),
                    "airport_name": dest_dict.get("name", ""),
                    "duration_minutes": int(layover_duration),
                    "overnight": arr_time.date() != dep_time.date(),
                    "min_connection_minutes": int(layover_duration),
                })
            except:
                pass
    
    return layovers


def _has_baggage(offer: Dict[str, Any]) -> bool:
    """Check if checked baggage is included."""
    passengers = offer.get("passengers") or []
    
    for pax in passengers:
        baggages = pax.get("baggages") or []
        for bag in baggages:
            if bag.get("type") == "checked" and bag.get("quantity", 0) > 0:
                return True
    
    return False


# ============================================================================
# MAIN FLIGHT SEARCH NODE
# ============================================================================

async def flight_search_node(state: ConversationState) -> ConversationState:
    """
    Execute REAL flight search using Duffel API.
    """
    
    if not getattr(state, "ready_for_search", False):
        logger.warning("[FlightSearch] Called when ready_for_search=False")
        return state
    
    logger.info("[FlightSearch] 🚀 Starting REAL flight search via Duffel")
    
    # 1. Validate
    validation_error = _validate_search_params(state)
    if validation_error:
        return update_state(state, {
            "raw_flights": [],
            "search_executed": False,
            "errors": [validation_error],
            "assistant_message": f"I can't search: {validation_error}",
            "suggestions": ["Change destination", "Start over"],
        })
    
    # 2. Resolve Airports
    origin = _resolve_airport(state.origin_airports, state.origin)
    destination = _resolve_airport(state.destination_airports, state.destination)
    
    # 3. Build Payload
    try:
        if isinstance(state.departure_date, date):
            dep_date_str = state.departure_date.strftime("%Y-%m-%d")
        else:
            dep_date_str = str(state.departure_date)
        
        slices = [{
            "origin": origin,
            "destination": destination,
            "departure_date": dep_date_str
        }]
        
        if state.return_date:
            if isinstance(state.return_date, date):
                ret_date_str = state.return_date.strftime("%Y-%m-%d")
            else:
                ret_date_str = str(state.return_date)
            slices.append({
                "origin": destination,
                "destination": origin,
                "departure_date": ret_date_str
            })
        
        passengers = [{"type": "adult"} for _ in range(state.passengers or 1)]
        
        cabin_map = {"Economy": "economy", "Business": "business", "First": "first"}
        cabin_class = cabin_map.get(state.travel_class or "Economy", "economy")
        
        duffel_payload = {
            "slices": slices,
            "passengers": passengers,
            "cabin_class": cabin_class,
        }
        
    except Exception as e:
        logger.error(f"[FlightSearch] Payload build failed: {e}")
        return state
    
    # 4. Call Duffel
    try:
        response = await duffel.create_offer_request(duffel_payload, return_offers=True)
        raw_offers = response.get("data", {}).get("offers", [])
        logger.info(f"[FlightSearch] ✅ Received {len(raw_offers)} raw offers")
    except Exception as e:
        logger.error(f"[FlightSearch] API Failed: {e}")
        return update_state(state, {
            "assistant_message": "Flight search service temporarily unavailable.",
            "suggestions": ["Try again later"]
        })
    
    # 5. Normalize (Using Fixed Function)
    normalized_flights = []
    for offer in raw_offers:
        normalized = _normalize_duffel_offer(offer)
        if normalized:
            normalized_flights.append(normalized)
    
    logger.info(f"[FlightSearch] ✅ Normalized {len(normalized_flights)} flights")
    
    return update_state(state, {
        "raw_flights": normalized_flights,
        "search_executed": True,
        "search_timestamp": datetime.utcnow(),
        "errors": []
    })