from pydantic import BaseModel, Field
from typing import Optional, List


class FlightSegment(BaseModel):
    """
    Represents a single flight segment (one leg of a journey).
    For example: TAS -> IST via AYT would have 2 segments:
    1. TAS -> AYT
    2. AYT -> IST
    """
    departure_airport: str = Field(..., description="IATA code of departure airport")
    departure_time: str = Field(..., description="Departure time in ISO 8601 format")
    arrival_airport: str = Field(..., description="IATA code of arrival airport")
    arrival_time: str = Field(..., description="Arrival time in ISO 8601 format")
    
    airline: str = Field(..., description="Airline name")
    airline_code: str = Field(..., description="IATA airline code")
    flight_number: str = Field(..., description="Flight number (e.g., 'TK1234')")
    
    aircraft: Optional[str] = Field(None, description="Aircraft type code")
    duration: str = Field(..., description="Segment duration in ISO 8601 format (e.g., 'PT2H30M')")
    
    # Layover information (only for segments after the first one)
    layover_duration: Optional[str] = Field(
        None, 
        description="Waiting time before this segment in ISO 8601 format (e.g., 'PT1H15M')"
    )


class FlightLeg(BaseModel):
    """
    Represents one leg of a trip (outbound or return).
    Contains summary info + detailed segments with layover information.
    """
    # Summary information (first departure to final arrival)
    departure_airport: str = Field(..., description="Origin airport IATA code")
    departure_time: str = Field(..., description="Departure time in ISO 8601 format")
    arrival_airport: str = Field(..., description="Destination airport IATA code")
    arrival_time: str = Field(..., description="Arrival time in ISO 8601 format")
    
    duration: str = Field(..., description="Total leg duration in ISO 8601 format")
    stops: int = Field(..., description="Number of stops (0 for non-stop)")
    
    # Detailed segment-by-segment breakdown with layovers
    segments: List[FlightSegment] = Field(..., description="All flight segments in order")


class Flight(BaseModel):
    """
    Represents a complete flight offer (one-way or round-trip).
    Uses nested structure for clean separation of outbound/return legs.
    """
    id: str = Field(..., description="Unique flight offer ID from Amadeus")
    
    # Price information (total for entire trip)
    price: float = Field(..., description="Total price for all passengers and legs")
    currency: str = Field(..., description="Price currency code (e.g., 'USD')")
    
    # Trip type
    is_round_trip: bool = Field(..., description="True if round-trip, False if one-way")
    
    # Outbound flight (always present)
    outbound: FlightLeg = Field(..., description="Outbound flight leg with all segments")
    
    # Return flight (only for round-trips)
    return_flight: Optional[FlightLeg] = Field(
        None, 
        description="Return flight leg with all segments (only for round-trip)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "1",
                "price": 450.00,
                "currency": "USD",
                "is_round_trip": True,
                "outbound": {
                    "departure_airport": "TAS",
                    "departure_time": "2025-11-15T10:30:00",
                    "arrival_airport": "IST",
                    "arrival_time": "2025-11-15T16:45:00",
                    "duration": "PT6H15M",
                    "stops": 1,
                    "segments": [
                        {
                            "departure_airport": "TAS",
                            "departure_time": "2025-11-15T10:30:00",
                            "arrival_airport": "AYT",
                            "arrival_time": "2025-11-15T13:00:00",
                            "airline": "Turkish Airlines",
                            "airline_code": "TK",
                            "flight_number": "TK1234",
                            "aircraft": "320",
                            "duration": "PT2H30M",
                            "layover_duration": None
                        },
                        {
                            "departure_airport": "AYT",
                            "departure_time": "2025-11-15T15:30:00",
                            "arrival_airport": "IST",
                            "arrival_time": "2025-11-15T16:45:00",
                            "airline": "Turkish Airlines",
                            "airline_code": "TK",
                            "flight_number": "TK5678",
                            "aircraft": "738",
                            "duration": "PT1H15M",
                            "layover_duration": "PT2H30M"
                        }
                    ]
                },
                "return_flight": {
                    "departure_airport": "IST",
                    "departure_time": "2025-11-22T08:00:00",
                    "arrival_airport": "TAS",
                    "arrival_time": "2025-11-22T13:15:00",
                    "duration": "PT5H15M",
                    "stops": 0,
                    "segments": [
                        {
                            "departure_airport": "IST",
                            "departure_time": "2025-11-22T08:00:00",
                            "arrival_airport": "TAS",
                            "arrival_time": "2025-11-22T13:15:00",
                            "airline": "Uzbekistan Airways",
                            "airline_code": "HY",
                            "flight_number": "HY255",
                            "aircraft": "763",
                            "duration": "PT5H15M",
                            "layover_duration": None
                        }
                    ]
                }
            }
        }