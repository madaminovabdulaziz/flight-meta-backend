from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DestinationItem(BaseModel):
    """Individual destination item with enriched location data."""
    
    destination_code: str = Field(
        ...,
        description="Destination IATA code",
        example="IST"
    )
    origin: str = Field(
        ...,
        description="Origin city IATA code",
        example="LON"
    )
    origin_city: str = Field(
        "",
        description="Origin city name",
        example="London"
    )
    origin_country: str = Field(
        "",
        description="Origin country name",
        example="United Kingdom"
    )
    origin_country_code: str = Field(
        "",
        description="Origin country ISO code",
        example="GB"
    )
    destination: str = Field(
        ...,
        description="Destination city IATA code",
        example="IST"
    )
    destination_city: str = Field(
        "",
        description="Destination city name",
        example="Istanbul"
    )
    destination_country: str = Field(
        "",
        description="Destination country name",
        example="Turkey"
    )
    destination_country_code: str = Field(
        "",
        description="Destination country ISO code",
        example="TR"
    )
    price: float = Field(
        ...,
        description="Flight price in specified currency",
        example=3673.0
    )
    transfers: int = Field(
        0,
        description="Number of transfers/stops",
        example=0
    )
    airline: Optional[str] = Field(
        None,
        description="IATA airline code",
        example="WZ"
    )
    flight_number: Optional[int] = Field(
        None,
        description="Flight number",
        example=125
    )
    departure_at: Optional[str] = Field(
        None,
        description="Departure datetime (ISO 8601)",
        example="2021-03-08T16:35:00Z"
    )
    return_at: Optional[str] = Field(
        None,
        description="Return datetime (ISO 8601)",
        example="2021-03-17T16:05:00Z"
    )
    expires_at: Optional[str] = Field(
        None,
        description="Price expiration datetime (ISO 8601)",
        example="2021-02-22T09:32:44Z"
    )


class PopularDestinationsResponse(BaseModel):
    """Response schema for popular international destinations endpoint."""
    
    success: bool = Field(
        True,
        description="Request success status"
    )
    currency: str = Field(
        ...,
        description="Currency code used for pricing",
        example="usd"
    )
    destinations: List[DestinationItem] = Field(
        ...,
        description="List of popular international destinations sorted by price"
    )
    total_count: int = Field(
        ...,
        description="Total number of international destinations returned",
        example=15
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "currency": "usd",
                "total_count": 2,
                "destinations": [
                    {
                        "destination_code": "IST",
                        "origin": "LON",
                        "origin_city": "London",
                        "origin_country": "United Kingdom",
                        "origin_country_code": "GB",
                        "destination": "IST",
                        "destination_city": "Istanbul",
                        "destination_country": "Turkey",
                        "destination_country_code": "TR",
                        "price": 245.50,
                        "transfers": 0,
                        "airline": "TK",
                        "flight_number": 1985,
                        "departure_at": "2025-10-20T14:30:00Z",
                        "return_at": "2025-10-27T18:15:00Z",
                        "expires_at": "2025-10-13T12:00:00Z"
                    },
                    {
                        "destination_code": "DXB",
                        "origin": "LON",
                        "origin_city": "London",
                        "origin_country": "United Kingdom",
                        "origin_country_code": "GB",
                        "destination": "DXB",
                        "destination_city": "Dubai",
                        "destination_country": "United Arab Emirates",
                        "destination_country_code": "AE",
                        "price": 389.00,
                        "transfers": 0,
                        "airline": "EK",
                        "flight_number": 7,
                        "departure_at": "2025-10-22T09:45:00Z",
                        "return_at": "2025-10-29T21:30:00Z",
                        "expires_at": "2025-10-13T15:30:00Z"
                    }
                ]
            }
        }
