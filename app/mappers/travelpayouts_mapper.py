# app/mappers/travelpayouts_mapper.py
"""
TravelPayouts API Response Mapper

Converts TravelPayouts API responses into our domain models (Flight, FlightLeg, etc.)
Handles data validation, normalization, and transformation.

Why separate mapper?
- Keeps services lean (SRP)
- Easier to test data transformation logic
- Isolates API format changes
- Reusable across multiple endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from schemas.flight_segment import Flight, FlightLeg, FlightSegment

logger = logging.getLogger(__name__)


class TravelPayoutsMapper:
    """
    Maps TravelPayouts API responses to domain models.
    
    TravelPayouts format quirks:
    - Times are often just dates (no time component)
    - Missing segment details (they aggregate data)
    - Currency in various fields
    - Flight numbers not always present
    """
    
    # Default values for missing data
    DEFAULT_DURATION = "PT5H0M"  # 5 hours default
    DEFAULT_DEPARTURE_TIME = "09:00:00"
    DEFAULT_ARRIVAL_TIME = "14:00:00"
    
    @staticmethod
    def to_flight(item: Dict[str, Any]) -> Optional[Flight]:
        """
        Convert TravelPayouts 'latest prices' item to Flight object.
        
        Args:
            item: Raw item from TravelPayouts /v2/prices/latest response
        
        Returns:
            Flight object or None if parsing fails
        
        Example TravelPayouts item:
        {
            "value": 245.0,
            "trip_class": 0,
            "show_to_affiliates": true,
            "origin": "TAS",
            "destination": "DXB",
            "gate": "AMADEUS",
            "depart_date": "2025-11-15",
            "return_date": "2025-11-22",
            "number_of_changes": 1,
            "found_at": "2025-10-07T12:00:00Z",
            "distance": 2345,
            "actual": true,
            "airline": "FZ",
            "flight_number": "123"
        }
        """
        try:
            # Validate required fields
            if not TravelPayoutsMapper._validate_required_fields(item):
                return None
            
            # Extract basic info
            origin = item["origin"].upper()
            destination = item["destination"].upper()
            price = float(item["value"])
            currency = item.get("currency", "USD").upper()
            
            # Dates
            departure_date = item["depart_date"]
            return_date = item.get("return_date")
            
            # Airline info
            airline_code = item.get("airline", "").upper()
            flight_number = item.get("flight_number")
            if flight_number:
                flight_number = f"{airline_code}{flight_number}"
            else:
                flight_number = f"{airline_code}XXX"  # Placeholder
            
            # Stops
            stops = item.get("number_of_changes", 0)
            
            # Build flight ID
            flight_id = TravelPayoutsMapper._build_flight_id(item)
            
            # Create outbound leg
            outbound = TravelPayoutsMapper._build_flight_leg(
                origin=origin,
                destination=destination,
                date=departure_date,
                airline_code=airline_code,
                flight_number=flight_number,
                stops=stops
            )
            
            # Create return leg if round-trip
            return_leg = None
            is_round_trip = return_date is not None
            
            if is_round_trip:
                return_leg = TravelPayoutsMapper._build_flight_leg(
                    origin=destination,  # Reversed
                    destination=origin,
                    date=return_date,
                    airline_code=airline_code,
                    flight_number=flight_number,
                    stops=0  # Usually direct on return
                )
            
            # Create Flight object
            flight = Flight(
                id=flight_id,
                price=price,
                currency=currency,
                is_round_trip=is_round_trip,
                outbound=outbound,
                return_flight=return_leg
            )
            
            return flight
            
        except KeyError as e:
            logger.debug(f"Missing required field in TravelPayouts item: {e}")
            return None
        except ValueError as e:
            logger.debug(f"Invalid value in TravelPayouts item: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing TravelPayouts item: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _validate_required_fields(item: Dict[str, Any]) -> bool:
        """Validate that required fields are present and non-empty"""
        required = ["origin", "destination", "value", "depart_date"]
        
        for field in required:
            if field not in item or not item[field]:
                logger.debug(f"Missing or empty required field: {field}")
                return False
        
        # Validate price is positive
        try:
            price = float(item["value"])
            if price <= 0:
                logger.debug(f"Invalid price: {price}")
                return False
        except (ValueError, TypeError):
            logger.debug(f"Price is not a valid number: {item.get('value')}")
            return False
        
        return True
    
    @staticmethod
    def _build_flight_id(item: Dict[str, Any]) -> str:
        """
        Generate unique flight ID from TravelPayouts data.
        
        Format: tp_{gate}_{origin}_{dest}_{date}
        """
        gate = item.get("gate", "TP")
        origin = item["origin"]
        dest = item["destination"]
        date = item["depart_date"].replace("-", "")
        
        return f"tp_{gate}_{origin}_{dest}_{date}".lower()
    
    @staticmethod
    def _build_flight_leg(
        origin: str,
        destination: str,
        date: str,
        airline_code: str,
        flight_number: str,
        stops: int = 0
    ) -> FlightLeg:
        """
        Build a FlightLeg with reasonable defaults for missing data.
        
        TravelPayouts doesn't provide:
        - Exact departure/arrival times (only dates)
        - Segment breakdown for multi-stop flights
        - Aircraft types
        
        So we estimate sensible defaults.
        """
        # Parse date and add estimated times
        departure_datetime = f"{date}T{TravelPayoutsMapper.DEFAULT_DEPARTURE_TIME}"
        
        # Estimate arrival time (5 hours later by default)
        try:
            dep_dt = datetime.fromisoformat(date)
            arr_dt = dep_dt + timedelta(hours=5)
            arrival_datetime = f"{arr_dt.date()}T{TravelPayoutsMapper.DEFAULT_ARRIVAL_TIME}"
        except:
            arrival_datetime = f"{date}T{TravelPayoutsMapper.DEFAULT_ARRIVAL_TIME}"
        
        # Create single segment (TravelPayouts doesn't give segment details)
        segment = FlightSegment(
            departure_airport=origin,
            departure_time=departure_datetime,
            arrival_airport=destination,
            arrival_time=arrival_datetime,
            airline=airline_code,  # TODO: Map code to full name
            airline_code=airline_code,
            flight_number=flight_number,
            aircraft=None,  # Not available
            duration=TravelPayoutsMapper.DEFAULT_DURATION,
            layover_duration=None
        )
        
        # Create FlightLeg
        leg = FlightLeg(
            departure_airport=origin,
            departure_time=departure_datetime,
            arrival_airport=destination,
            arrival_time=arrival_datetime,
            duration=TravelPayoutsMapper.DEFAULT_DURATION,
            stops=stops,
            segments=[segment]
        )
        
        return leg
    
    @staticmethod
    def to_flight_from_cheap(item: Dict[str, Any]) -> Optional[Flight]:
        """
        Convert item from 'cheapest tickets' endpoint to Flight.
        
        Format is similar to 'latest prices' but nested differently.
        """
        # The cheap endpoint returns already-formatted dicts
        # So we can reuse to_flight() if we normalize the structure
        
        try:
            normalized = {
                "origin": item.get("origin"),
                "destination": item.get("destination"),
                "value": item.get("price"),
                "currency": item.get("currency", "USD"),
                "depart_date": item.get("departure_at", "")[:10] if item.get("departure_at") else None,
                "return_date": item.get("return_at", "")[:10] if item.get("return_at") else None,
                "airline": item.get("airline"),
                "flight_number": item.get("flight_number"),
                "number_of_changes": 0,  # Cheap endpoint doesn't specify
                "gate": "TP_CHEAP"
            }
            
            return TravelPayoutsMapper.to_flight(normalized)
            
        except Exception as e:
            logger.debug(f"Failed to parse cheap ticket item: {e}")
            return None


# Airline code to name mapping (partial - extend as needed)
AIRLINE_NAMES = {
    "FZ": "FlyDubai",
    "TK": "Turkish Airlines",
    "HY": "Uzbekistan Airways",
    "SU": "Aeroflot",
    "EK": "Emirates",
    "QR": "Qatar Airways",
    "KE": "Korean Air",
    "AI": "Air India",
    "BA": "British Airways",
    "LH": "Lufthansa",
    "AF": "Air France",
    # Add more as needed
}

def get_airline_name(code: str) -> str:
    """Get airline full name from IATA code"""
    return AIRLINE_NAMES.get(code.upper(), code.upper())