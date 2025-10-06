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


class AirportResponse(BaseModel):
    """
    Complete airport information including coordinates and timezone.
    Used for detailed airport pages.
    """
    id: int
    iata_code: str = Field(..., description="3-letter IATA code")
    icao_code: Optional[str] = Field(None, description="4-letter ICAO code")
    name: str
    city: str
    country: str
    country_code: str
    timezone: str = Field(..., description="Timezone (e.g., 'Europe/Istanbul')")
    latitude: Optional[float] = Field(None, description="Latitude coordinate")
    longitude: Optional[float] = Field(None, description="Longitude coordinate")
    is_active: bool = Field(True, description="Whether airport is operational")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "iata_code": "IST",
                "icao_code": "LTFM",
                "name": "Istanbul Airport",
                "city": "Istanbul",
                "country": "Turkey",
                "country_code": "TR",
                "timezone": "Europe/Istanbul",
                "latitude": 41.2619,
                "longitude": 28.7414,
                "is_active": True
            }
        }