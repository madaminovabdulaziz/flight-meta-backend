# schemas/hot_routes.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class HotRouteResponse(BaseModel):
    """Response model for hot/popular routes"""
    
    # Route information
    origin: str = Field(..., description="Origin airport IATA code")
    destination: str = Field(..., description="Destination airport IATA code")
    
    # Price information
    price: float = Field(..., description="Lowest price found")
    currency: str = Field(default="USD", description="Currency code")
    
    # Enriched airport data (optional - added by endpoint)
    origin_city: Optional[str] = Field(None, description="Origin city name")
    origin_country: Optional[str] = Field(None, description="Origin country")
    destination_city: Optional[str] = Field(None, description="Destination city name")
    destination_country: Optional[str] = Field(None, description="Destination country")
    
    # Flight details (optional - from Travelpayouts)
    airline: Optional[str] = Field(None, description="Airline IATA code")
    flight_number: Optional[str] = Field(None, description="Flight number")
    departure_date: Optional[date] = Field(None, description="Sample departure date")
    return_date: Optional[date] = Field(None, description="Sample return date")
    
    # Analytics data (optional - from your own data)
    click_count: Optional[int] = Field(None, description="Number of clicks (from analytics)")
    avg_price: Optional[float] = Field(None, description="Average price (from analytics)")
    
    # Trending data (optional - from trending endpoint)
    growth_percent: Optional[float] = Field(None, description="Growth percentage (for trending)")
    
    # Metadata
    found_at: Optional[str] = Field(None, description="When this price was found")
    
    class Config:
        json_schema_extra = {
            "example": {
                "origin": "TAS",
                "destination": "DXB",
                "price": 245.0,
                "currency": "USD",
                "origin_city": "Tashkent",
                "origin_country": "Uzbekistan",
                "destination_city": "Dubai",
                "destination_country": "United Arab Emirates",
                "airline": "FZ",
                "departure_date": "2025-11-15",
                "click_count": 156
            }
        }