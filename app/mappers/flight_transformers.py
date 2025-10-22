# flight_transformers.py - Production-Ready Frontend Response Transformers
"""
COMPLETE PRODUCTION SOLUTION with:
✓ Proper duration calculations (hours + minutes)
✓ Correct stops/connections counting
✓ Expiration filtering
✓ Operating vs Marketing carrier distinction
✓ Timezone information
✓ Layover calculations
✓ Better baggage details
✓ Identity document requirements
✓ Loyalty program support
✓ Available services handling
✓ Fare grouping by flight
✓ Smart caching signatures
✓ Price change detection
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from collections import defaultdict
import hashlib
import json
import re


# ============================================================================
# CORE UTILITY FUNCTIONS
# ============================================================================

def parse_duration(duration_str: str) -> Dict[str, int]:
    """Parse ISO 8601 duration (PT6H20M) into hours and minutes."""
    if not duration_str:
        return {"hours": 0, "minutes": 0, "total_minutes": 0}
    
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration_str)
    if not match:
        return {"hours": 0, "minutes": 0, "total_minutes": 0}
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    total_minutes = hours * 60 + minutes
    
    return {
        "hours": hours,
        "minutes": minutes,
        "total_minutes": total_minutes
    }


def calculate_duration_from_minutes(total_minutes: int) -> Dict[str, int]:
    """Convert total minutes into hours and minutes."""
    return {
        "hours": total_minutes // 60,
        "minutes": total_minutes % 60,
        "total_minutes": total_minutes
    }


def format_datetime(dt_str: str) -> Dict[str, str]:
    """Format ISO datetime into frontend-friendly format."""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return {
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M"),
            "datetime": dt.strftime("%Y-%m-%d %H:%M"),
            "iso": dt_str
        }
    except Exception:
        return {
            "date": "",
            "time": "",
            "datetime": "",
            "iso": dt_str
        }


def is_offer_expired(expires_at: str) -> bool:
    """Check if an offer has expired."""
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return expiry <= now
    except Exception:
        return True


def get_expiration_info(expires_at: str) -> Dict[str, Any]:
    """Get expiration information including countdown."""
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = expiry - now
        minutes_left = int(delta.total_seconds() / 60)
        
        return {
            "expires_at": expires_at,
            "expires_in_minutes": minutes_left,
            "is_expired": minutes_left <= 0,
            "expires_soon": 0 < minutes_left <= 5
        }
    except Exception:
        return {
            "expires_at": expires_at,
            "expires_in_minutes": -1,
            "is_expired": True,
            "expires_soon": False
        }


# ============================================================================
# FLIGHT SIGNATURE GENERATION (for grouping & caching)
# ============================================================================

def generate_flight_signature(offer: Dict[str, Any]) -> str:
    """
    Generate unique signature for a flight (not offer).
    Same flight with different fares will have same signature.
    
    Signature includes:
    - Route (all segments origin/destination)
    - Departure times
    - Operating carriers
    - Aircraft types (to differentiate equipment changes)
    
    Does NOT include:
    - Offer ID
    - Price
    - Fare brand
    - Policies
    """
    signature_parts = []
    
    for slice_data in offer.get("slices", []):
        for seg in slice_data.get("segments", []):
            operating_carrier = seg.get("operating_carrier", {})
            aircraft = seg.get("aircraft") or {}
            
            part = (
                f"{seg.get('origin', {}).get('iata_code', '')}-"
                f"{seg.get('destination', {}).get('iata_code', '')}-"
                f"{seg.get('departing_at', '')}-"
                f"{operating_carrier.get('iata_code', '')}-"
                f"{aircraft.get('iata_code', '')}"
            )
            signature_parts.append(part)
    
    # Create hash for compact signature
    full_signature = "|".join(signature_parts)
    return hashlib.md5(full_signature.encode()).hexdigest()


def generate_offer_cache_key(offer_id: str, include_services: bool = False) -> str:
    """
    Generate cache key for a specific offer fetch.
    """
    key = f"offer:{offer_id}"
    if include_services:
        key += ":with_services"
    return key


# ============================================================================
# BAGGAGE TRANSFORMATION
# ============================================================================

def transform_baggage(passengers_baggage: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Transform baggage information with more details."""
    carry_on_bags = []
    checked_bags = []
    
    for bag in passengers_baggage:
        bag_type = bag.get("type", "")
        quantity = bag.get("quantity", 0)
        
        bag_details = {
            "quantity": quantity,
            "weight_kg": None,
            "dimensions": None
        }
        
        if bag.get("weight"):
            bag_details["weight_kg"] = bag["weight"].get("value")
        
        if bag.get("dimensions"):
            bag_details["dimensions"] = bag["dimensions"]
        
        if bag_type == "carry_on":
            carry_on_bags.append(bag_details)
        elif bag_type == "checked":
            checked_bags.append(bag_details)
    
    return {
        "carry_on": {
            "allowed": len(carry_on_bags) > 0,
            "quantity": sum(b["quantity"] for b in carry_on_bags),
            "details": carry_on_bags[0] if carry_on_bags else None
        },
        "checked": {
            "allowed": len(checked_bags) > 0,
            "quantity": sum(b["quantity"] for b in checked_bags),
            "details": checked_bags[0] if checked_bags else None
        }
    }


# ============================================================================
# SEGMENT TRANSFORMATION
# ============================================================================

def transform_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a single flight segment into clean format."""
    origin = segment.get("origin", {})
    destination = segment.get("destination", {})
    operating_carrier = segment.get("operating_carrier", {})
    marketing_carrier = segment.get("marketing_carrier", {})
    
    # Get passenger baggage info (from first passenger)
    passengers = segment.get("passengers", [])
    baggage_info = {"carry_on": {"allowed": False, "quantity": 0}, "checked": {"allowed": False, "quantity": 0}}
    if passengers:
        baggages = passengers[0].get("baggages", [])
        baggage_info = transform_baggage(baggages)
    
    # Aircraft info
    aircraft = segment.get("aircraft") or {}
    
    # Calculate stops correctly
    stops_count = len(segment.get("stops", []))
    
    return {
        "segment_id": segment.get("id"),
        "origin": {
            "airport_code": origin.get("iata_code"),
            "airport_name": origin.get("name"),
            "city": origin.get("city_name"),
            "terminal": segment.get("origin_terminal"),
            "timezone": origin.get("time_zone")
        },
        "destination": {
            "airport_code": destination.get("iata_code"),
            "airport_name": destination.get("name"),
            "city": destination.get("city_name"),
            "terminal": segment.get("destination_terminal"),
            "timezone": destination.get("time_zone")
        },
        "departure": format_datetime(segment.get("departing_at", "")),
        "arrival": format_datetime(segment.get("arriving_at", "")),
        "duration": parse_duration(segment.get("duration", "")),
        "operating_carrier": {
            "name": operating_carrier.get("name"),
            "code": operating_carrier.get("iata_code"),
            "logo": operating_carrier.get("logo_symbol_url")
        },
        "marketing_carrier": {
            "name": marketing_carrier.get("name"),
            "code": marketing_carrier.get("iata_code"),
            "logo": marketing_carrier.get("logo_symbol_url")
        },
        "airline": {
            "name": operating_carrier.get("name"),
            "code": operating_carrier.get("iata_code"),
            "logo": operating_carrier.get("logo_symbol_url")
        },
        "flight_number": f"{marketing_carrier.get('iata_code', '')}{segment.get('marketing_carrier_flight_number', '')}",
        "aircraft": {
            "model": aircraft.get("name"),
            "code": aircraft.get("iata_code")
        },
        "baggage": baggage_info,
        "stops": stops_count,
        "cabin_class": passengers[0].get("cabin_class") if passengers else None
    }


# ============================================================================
# LAYOVER CALCULATION
# ============================================================================

def calculate_layovers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Calculate layover information between segments."""
    if len(segments) <= 1:
        return []
    
    layovers = []
    for i in range(len(segments) - 1):
        current_seg = segments[i]
        next_seg = segments[i + 1]
        
        try:
            arrival = datetime.fromisoformat(current_seg.get("arriving_at", "").replace("Z", "+00:00"))
            departure = datetime.fromisoformat(next_seg.get("departing_at", "").replace("Z", "+00:00"))
            layover_minutes = int((departure - arrival).total_seconds() / 60)
            
            layover_airport = current_seg.get("destination", {})
            
            layovers.append({
                "airport_code": layover_airport.get("iata_code"),
                "airport_name": layover_airport.get("name"),
                "city": layover_airport.get("city_name"),
                "duration_minutes": layover_minutes,
                "duration_formatted": f"{layover_minutes // 60}h {layover_minutes % 60}m",
                "is_short": layover_minutes < 90,
                "is_long": layover_minutes > 360
            })
        except Exception:
            continue
    
    return layovers


# ============================================================================
# SLICE TRANSFORMATION
# ============================================================================

def transform_slice(slice_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a slice (journey leg) into clean format."""
    origin = slice_data.get("origin", {})
    destination = slice_data.get("destination", {})
    segments = slice_data.get("segments", [])
    
    # Stops = segments - 1
    total_stops = max(0, len(segments) - 1)
    
    # Calculate layovers
    layovers = calculate_layovers(segments)
    
    # Transform segments
    transformed_segments = [transform_segment(seg) for seg in segments]
    
    return {
        "slice_id": slice_data.get("id"),
        "origin": {
            "airport_code": origin.get("iata_code"),
            "city": origin.get("city_name"),
            "timezone": origin.get("time_zone")
        },
        "destination": {
            "airport_code": destination.get("iata_code"),
            "city": destination.get("city_name"),
            "timezone": destination.get("time_zone")
        },
        "departure": format_datetime(segments[0].get("departing_at", "")) if segments else None,
        "arrival": format_datetime(segments[-1].get("arriving_at", "")) if segments else None,
        "duration": parse_duration(slice_data.get("duration", "")),
        "stops": total_stops,
        "segments": transformed_segments,
        "layovers": layovers,
        "fare_brand_name": slice_data.get("fare_brand_name")
    }


# ============================================================================
# OFFER TRANSFORMATION
# ============================================================================

def transform_offer(offer: Dict[str, Any], skip_expiration_check: bool = False) -> Optional[Dict[str, Any]]:
    """
    Transform a full Duffel offer into clean, frontend-friendly format.
    
    Args:
        offer: Raw Duffel offer
        skip_expiration_check: Set to True when getting single offer (user selected it)
        
    Returns:
        Transformed offer dict, or None if expired (and not skipping check)
    """
    # Filter expired offers (unless explicitly skipping)
    expires_at = offer.get("expires_at")
    if not skip_expiration_check and expires_at and is_offer_expired(expires_at):
        return None
    
    owner = offer.get("owner", {})
    slices = offer.get("slices", [])
    conditions = offer.get("conditions", {})
    payment_reqs = offer.get("payment_requirements", {})
    
    # Price breakdown
    price = {
        "total": float(offer.get("total_amount", 0)),
        "currency": offer.get("total_currency", "USD"),
        "base": float(offer.get("base_amount", 0)),
        "tax": float(offer.get("tax_amount", 0))
    }
    
    # Refund and change policies
    refund_policy = conditions.get("refund_before_departure", {}) or {}
    change_policy = conditions.get("change_before_departure", {}) or {}
    
    policies = {
        "refundable": refund_policy.get("allowed", False),
        "refund_penalty": {
            "amount": float(refund_policy.get("penalty_amount", 0)) if refund_policy.get("penalty_amount") else None,
            "currency": refund_policy.get("penalty_currency")
        } if refund_policy.get("allowed") and refund_policy.get("penalty_amount") else None,
        "changeable": change_policy.get("allowed", False),
        "change_penalty": {
            "amount": float(change_policy.get("penalty_amount", 0)) if change_policy.get("penalty_amount") else None,
            "currency": change_policy.get("penalty_currency")
        } if change_policy.get("allowed") and change_policy.get("penalty_amount") else None
    }
    
    # Transform slices
    transformed_slices = [transform_slice(s) for s in slices]
    
    # Calculate total duration
    total_duration_mins = sum(
        parse_duration(s.get("duration", ""))["total_minutes"] 
        for s in slices
    )
    
    # Calculate max stops
    max_stops = max(
        (max(0, len(s.get("segments", [])) - 1) for s in slices),
        default=0
    )
    
    # Determine if direct flight
    is_direct = max_stops == 0 and all(len(s.get("segments", [])) == 1 for s in slices)
    
    # Get cabin class
    cabin_class = "economy"
    if slices and slices[0].get("segments"):
        first_seg = slices[0]["segments"][0]
        if first_seg.get("passengers"):
            cabin_class = first_seg["passengers"][0].get("cabin_class", "economy")
    
    # Emissions categorization
    emissions_kg = offer.get("total_emissions_kg")
    emissions_label = None
    if emissions_kg:
        try:
            emissions_value = float(emissions_kg)
            if emissions_value < 200:
                emissions_label = "Low"
            elif emissions_value < 400:
                emissions_label = "Medium"
            else:
                emissions_label = "High"
        except Exception:
            pass
    
    # Available services
    available_services = offer.get("available_services") or []
    has_services = len(available_services) > 0
    
    # Generate flight signature for grouping
    flight_signature = generate_flight_signature(offer)
    
    return {
        "offer_id": offer.get("id"),
        "flight_signature": flight_signature,  # NEW: For grouping
        "airline": {
            "name": owner.get("name"),
            "code": owner.get("iata_code"),
            "logo": owner.get("logo_symbol_url")
        },
        "price": price,
        "slices": transformed_slices,
        "summary": {
            "is_direct": is_direct,
            "total_duration": calculate_duration_from_minutes(total_duration_mins),
            "max_stops": max_stops,
            "cabin_class": cabin_class
        },
        "policies": policies,
        "booking_info": {
            **get_expiration_info(expires_at),
            "payment_required_by": payment_reqs.get("payment_required_by"),
            "instant_payment_required": payment_reqs.get("requires_instant_payment", False),
            "passenger_identity_documents_required": offer.get("passenger_identity_documents_required", False),
            "supported_identity_document_types": offer.get("supported_passenger_identity_document_types", [])
        },
        "metadata": {
            "created_at": offer.get("created_at"),
            "updated_at": offer.get("updated_at"),
            "live_mode": offer.get("live_mode", False),
            "fare_brand": slices[0].get("fare_brand_name") if slices else None,
            "emissions_kg": emissions_kg,
            "emissions_label": emissions_label,
            "supported_loyalty_programmes": offer.get("supported_loyalty_programmes", []),
            "has_available_services": has_services,
            "partial_offer": offer.get("partial", False)
        },
        "available_services": available_services if has_services else []
    }


# ============================================================================
# SEARCH RESPONSE TRANSFORMATION
# ============================================================================

def transform_search_response(
    duffel_response: Dict[str, Any],
    limit: Optional[int] = None,
    filter_expired: bool = True
) -> Dict[str, Any]:
    """Transform full Duffel search response into clean frontend format."""
    offers = duffel_response.get("offers", [])
    
    # Transform and filter expired offers
    transformed_offers = []
    expired_count = 0
    
    for offer in offers:
        transformed = transform_offer(offer)
        if transformed is None:
            expired_count += 1
        else:
            transformed_offers.append(transformed)
    
    # Sort by price (cheapest first)
    transformed_offers.sort(key=lambda x: x["price"]["total"])
    
    # Apply limit if specified
    if limit:
        transformed_offers = transformed_offers[:limit]
    
    # Extract metadata
    meta = duffel_response.get("meta", {})
    
    return {
        "offer_request_id": duffel_response.get("offer_request_id"),
        "offers": transformed_offers,
        "count": len(transformed_offers),
        "meta": {
            "page_size": meta.get("page_size"),
            "has_more": meta.get("has_more", False),
            "next_after": meta.get("next_after"),
            "source": meta.get("source", "fresh"),
            "cache_hit": meta.get("cache_hit", False),
            "cache_age_seconds": meta.get("cache_age_seconds"),
            "expired_offers_filtered": expired_count if filter_expired else 0
        }
    }


# ============================================================================
# FARE GROUPING (NEW FEATURE)
# ============================================================================

def group_offers_by_flight(offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group offers by flight (same route/time/airline) showing different fares.
    
    Returns list of flight groups, each containing:
    - flight: Base flight information
    - fares: List of fare options for that flight
    
    Example:
    [
        {
            "flight_signature": "abc123...",
            "flight": {...},  # Base flight info
            "fares": [
                {"offer_id": "off_1", "fare_brand": "Basic", "price": 300, ...},
                {"offer_id": "off_2", "fare_brand": "Flexible", "price": 450, ...}
            ]
        }
    ]
    """
    grouped = defaultdict(lambda: {"flight": None, "fares": []})
    
    for offer in offers:
        flight_sig = offer.get("flight_signature")
        
        if not flight_sig:
            # Shouldn't happen, but handle gracefully
            continue
        
        # Store base flight info (use first occurrence)
        if not grouped[flight_sig]["flight"]:
            grouped[flight_sig]["flight"] = {
                "flight_signature": flight_sig,
                "slices": offer["slices"],
                "airline": offer["airline"],
                "summary": offer["summary"]
            }
        
        # Extract fare-specific info
        fare = {
            "offer_id": offer["offer_id"],
            "fare_brand": offer["metadata"].get("fare_brand") or "Standard",
            "price": offer["price"],
            "policies": offer["policies"],
            "baggage": {
                "carry_on": offer["slices"][0]["segments"][0]["baggage"]["carry_on"],
                "checked": offer["slices"][0]["segments"][0]["baggage"]["checked"]
            } if offer.get("slices") and offer["slices"][0].get("segments") else None,
            "booking_info": offer["booking_info"],
            "metadata": {
                "emissions_kg": offer["metadata"].get("emissions_kg"),
                "emissions_label": offer["metadata"].get("emissions_label")
            }
        }
        
        grouped[flight_sig]["fares"].append(fare)
    
    # Convert to list and sort fares by price within each group
    result = []
    for flight_sig, flight_data in grouped.items():
        # Sort fares by price (cheapest first)
        flight_data["fares"].sort(key=lambda f: f["price"]["total"])
        
        # Add flight signature at root level
        flight_data["flight_signature"] = flight_sig
        
        result.append(flight_data)
    
    # Sort flights by cheapest fare
    result.sort(key=lambda g: g["fares"][0]["price"]["total"] if g["fares"] else float('inf'))
    
    return result


# ============================================================================
# HELPER FILTERS
# ============================================================================

def get_cheapest_offer(offers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the cheapest offer from a list."""
    if not offers:
        return None
    return min(offers, key=lambda x: x["price"]["total"])


def get_fastest_offer(offers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the fastest offer from a list."""
    if not offers:
        return None
    return min(offers, key=lambda x: x["summary"]["total_duration"]["total_minutes"])


def get_direct_flights(offers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter only direct flights."""
    return [o for o in offers if o["summary"]["is_direct"]]


def filter_by_max_price(
    offers: List[Dict[str, Any]], 
    max_price: float
) -> List[Dict[str, Any]]:
    """Filter offers by maximum price."""
    return [o for o in offers if o["price"]["total"] <= max_price]


def filter_by_airline(
    offers: List[Dict[str, Any]], 
    airline_codes: List[str]
) -> List[Dict[str, Any]]:
    """Filter offers by airline codes."""
    return [
        o for o in offers 
        if o["airline"]["code"] in airline_codes
    ]


def filter_by_max_duration(
    offers: List[Dict[str, Any]],
    max_hours: float
) -> List[Dict[str, Any]]:
    """Filter offers by maximum duration in hours."""
    max_minutes = max_hours * 60
    return [
        o for o in offers
        if o["summary"]["total_duration"]["total_minutes"] <= max_minutes
    ]


def filter_by_stops(
    offers: List[Dict[str, Any]],
    max_stops: int
) -> List[Dict[str, Any]]:
    """Filter offers by maximum number of stops."""
    return [
        o for o in offers
        if o["summary"]["max_stops"] <= max_stops
    ]


# ============================================================================
# PRICE CHANGE DETECTION
# ============================================================================

def detect_price_change(
    cached_offer: Dict[str, Any],
    fresh_offer: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Detect if price has changed between cached and fresh offer.
    
    Returns:
        None if no change, or dict with change details if changed
    """
    cached_price = cached_offer.get("price", {}).get("total")
    fresh_price = fresh_offer.get("price", {}).get("total")
    
    if cached_price is None or fresh_price is None:
        return None
    
    if cached_price == fresh_price:
        return None
    
    change = fresh_price - cached_price
    percent_change = (change / cached_price) * 100
    
    return {
        "changed": True,
        "old_price": cached_price,
        "new_price": fresh_price,
        "difference": change,
        "percent_change": round(percent_change, 2),
        "increased": change > 0
    }