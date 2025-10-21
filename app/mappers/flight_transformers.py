# flight_transformers.py - Clean Frontend Response Transformers
"""
Data transformation layer to convert verbose Duffel API responses 
into clean, frontend-friendly JSON structures.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import re


def parse_duration(duration_str: str) -> Dict[str, int]:
    """
    Parse ISO 8601 duration (PT6H20M) into hours and minutes.
    
    Args:
        duration_str: ISO 8601 duration like "PT6H20M"
        
    Returns:
        Dict with 'hours', 'minutes', 'total_minutes'
    """
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


def format_datetime(dt_str: str) -> Dict[str, str]:
    """
    Format ISO datetime into frontend-friendly format.
    
    Args:
        dt_str: ISO datetime like "2025-10-28T20:54:00"
        
    Returns:
        Dict with 'date', 'time', 'datetime', 'iso'
    """
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


def transform_segment(segment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single flight segment into clean format.
    """
    origin = segment.get("origin", {})
    destination = segment.get("destination", {})
    operating_carrier = segment.get("operating_carrier", {})
    marketing_carrier = segment.get("marketing_carrier", {})
    
    # Get passenger baggage info (from first passenger)
    passengers = segment.get("passengers", [])
    baggage_info = {}
    if passengers:
        baggages = passengers[0].get("baggages", [])
        baggage_info = {
            "carry_on": sum(1 for b in baggages if b.get("type") == "carry_on"),
            "checked": sum(1 for b in baggages if b.get("type") == "checked")
        }
    
    # Aircraft info
    aircraft = segment.get("aircraft") or {}
    
    return {
        "segment_id": segment.get("id"),
        "origin": {
            "airport_code": origin.get("iata_code"),
            "airport_name": origin.get("name"),
            "city": origin.get("city_name"),
            "terminal": segment.get("origin_terminal")
        },
        "destination": {
            "airport_code": destination.get("iata_code"),
            "airport_name": destination.get("name"),
            "city": destination.get("city_name"),
            "terminal": segment.get("destination_terminal")
        },
        "departure": format_datetime(segment.get("departing_at", "")),
        "arrival": format_datetime(segment.get("arriving_at", "")),
        "duration": parse_duration(segment.get("duration", "")),
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
        "stops": len(segment.get("stops", []))
    }


def transform_slice(slice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a slice (journey leg) into clean format.
    """
    origin = slice_data.get("origin", {})
    destination = slice_data.get("destination", {})
    segments = slice_data.get("segments", [])
    
    # Calculate total stops (connections)
    total_stops = sum(len(seg.get("stops", [])) for seg in segments)
    
    return {
        "slice_id": slice_data.get("id"),
        "origin": {
            "airport_code": origin.get("iata_code"),
            "city": origin.get("city_name")
        },
        "destination": {
            "airport_code": destination.get("iata_code"),
            "city": destination.get("city_name")
        },
        "departure": format_datetime(segments[0].get("departing_at", "")) if segments else None,
        "arrival": format_datetime(segments[-1].get("arriving_at", "")) if segments else None,
        "duration": parse_duration(slice_data.get("duration", "")),
        "stops": total_stops,
        "segments": [transform_segment(seg) for seg in segments]
    }


def transform_offer(offer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a full Duffel offer into clean, frontend-friendly format.
    
    This is the main transformation function that should be called for each offer.
    """
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
        } if refund_policy.get("allowed") else None,
        "changeable": change_policy.get("allowed", False),
        "change_penalty": {
            "amount": float(change_policy.get("penalty_amount", 0)) if change_policy.get("penalty_amount") else None,
            "currency": change_policy.get("penalty_currency")
        } if change_policy.get("allowed") else None
    }
    
    # Calculate total duration and stops
    total_duration_mins = sum(
        parse_duration(s.get("duration", ""))["total_minutes"] 
        for s in slices
    )
    max_stops = max(
        (sum(len(seg.get("stops", [])) for seg in s.get("segments", [])) for s in slices),
        default=0
    )
    
    # Determine if direct flight
    is_direct = max_stops == 0 and all(len(s.get("segments", [])) == 1 for s in slices)
    
    return {
        "offer_id": offer.get("id"),
        "airline": {
            "name": owner.get("name"),
            "code": owner.get("iata_code"),
            "logo": owner.get("logo_symbol_url")
        },
        "price": price,
        "slices": [transform_slice(s) for s in slices],
        "summary": {
            "is_direct": is_direct,
            "total_duration": parse_duration(f"PT{total_duration_mins}M"),
            "max_stops": max_stops,
            "cabin_class": slices[0]["segments"][0]["passengers"][0]["cabin_class"] if slices and slices[0].get("segments") and slices[0]["segments"][0].get("passengers") else "economy"
        },
        "policies": policies,
        "booking_info": {
            "expires_at": offer.get("expires_at"),
            "payment_required_by": payment_reqs.get("payment_required_by"),
            "instant_payment_required": payment_reqs.get("requires_instant_payment", False)
        },
        "metadata": {
            "created_at": offer.get("created_at"),
            "live_mode": offer.get("live_mode", False),
            "fare_brand": slices[0].get("fare_brand_name") if slices else None,
            "emissions_kg": offer.get("total_emissions_kg")
        }
    }


def transform_search_response(
    duffel_response: Dict[str, Any],
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Transform full Duffel search response into clean frontend format.
    
    Args:
        duffel_response: Raw response from Duffel API
        limit: Optional limit for number of offers to return
        
    Returns:
        Clean, frontend-friendly response
    """
    offers = duffel_response.get("offers", [])
    
    # Apply limit if specified
    if limit:
        offers = offers[:limit]
    
    transformed_offers = [transform_offer(offer) for offer in offers]
    
    # Sort by price (cheapest first) by default
    transformed_offers.sort(key=lambda x: x["price"]["total"])
    
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
            "cache_age_seconds": meta.get("cache_age_seconds")
        }
    }


# Additional helper functions for specific use cases

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
    """Filter offers by airline codes (e.g., ['BA', 'AA'])."""
    return [
        o for o in offers 
        if o["airline"]["code"] in airline_codes
    ]