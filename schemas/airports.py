# schemas/airports.py
from pydantic import BaseModel, Field
from typing import Optional


class AirportSearchResponse(BaseModel):
    """
    Lightweight airport info for autocomplete/search results.
    Only includes essential fields for search dropdown.
    """
    iata_code: str = Field(..., description="3-letter IATA code")
    name: str = Field(..., description="Airport name")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")
    country_code: str = Field(..., description="ISO country code")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "iata_code": "IST",
                "name": "Istanbul Airport",
                "city": "Istanbul",
                "country": "Turkey",
                "country_code": "TR"
            }
        }


# app/schemas/airport.py


class AirportResponse(BaseModel):
    iata_code: str = Field(..., example="JFK")
    icao_code: Optional[str] = Field(None, example="KJFK")
    name: str = Field(..., example="John F. Kennedy International Airport")
    city: str = Field(..., example="New York")
    country: str = Field(..., example="United States")
    country_code: str = Field(..., example="US")
    timezone: str = Field(..., example="America/New_York")
    latitude: Optional[float] = Field(None, example=40.6413)
    longitude: Optional[float] = Field(None, example=-73.7781)
    
    @property
    def display_name(self) -> str:
        """Format for dropdown: 'JFK - New York, United States'"""
        return f"{self.iata_code} - {self.city}, {self.country}"
    
    class Config:
        json_schema_extra = {
            "example": {
                "iata_code": "JFK",
                "icao_code": "KJFK",
                "name": "John F. Kennedy International Airport",
                "city": "New York",
                "country": "United States",
                "country_code": "US",
                "timezone": "America/New_York",
                "latitude": 40.6413,
                "longitude": -73.7781
            }
        }