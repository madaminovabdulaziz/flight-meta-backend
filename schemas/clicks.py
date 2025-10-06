# schemas/clicks.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date, datetime


class ClickCreate(BaseModel):
    """Schema for creating a new click record"""
    
    # Flight route
    origin: str = Field(..., min_length=3, max_length=3, description="Origin airport IATA code")
    destination: str = Field(..., min_length=3, max_length=3, description="Destination airport IATA code")
    departure_date: date = Field(..., description="Departure date")
    return_date: Optional[date] = Field(None, description="Return date for round-trip")
    
    # Pricing
    price: float = Field(..., gt=0, description="Flight price")
    currency: str = Field(..., min_length=3, max_length=3, description="Price currency code")
    
    # Partner info
    partner_name: str = Field(..., description="Partner name (e.g., 'Amadeus')")
    deeplink: str = Field(..., description="Full URL to redirect user to partner site")
    
    # Flight data snapshot
    flight_offer_snapshot: Optional[dict] = Field(None, description="Complete flight offer data as JSON")
    
    @field_validator('origin', 'destination', 'currency')
    @classmethod
    def uppercase_codes(cls, v: str) -> str:
        """Ensure IATA codes and currency are uppercase"""
        return v.upper() if v else v
    
    @field_validator('return_date')
    @classmethod
    def validate_return_date(cls, v: Optional[date], info) -> Optional[date]:
        """Ensure return date is after departure date"""
        if v and 'departure_date' in info.data:
            if v <= info.data['departure_date']:
                raise ValueError("Return date must be after departure date")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "origin": "TAS",
                "destination": "IST",
                "departure_date": "2025-11-15",
                "return_date": "2025-11-22",
                "price": 450.00,
                "currency": "USD",
                "partner_name": "Amadeus",
                "deeplink": "https://partner.com/book?ref=abc123&origin=TAS&dest=IST",
                "flight_offer_snapshot": {
                    "id": "1",
                    "outbound": {"departure_time": "2025-11-15T10:30:00"}
                }
            }
        }


class ClickResponse(BaseModel):
    """Response after creating click record"""
    
    click_id: int = Field(..., description="Unique click ID for tracking")
    redirect_url: str = Field(..., description="URL to redirect user to")
    session_id: str = Field(..., description="Session ID for tracking")
    partner_name: str = Field(..., description="Partner name")
    
    class Config:
        json_schema_extra = {
            "example": {
                "click_id": 12345,
                "redirect_url": "https://partner.com/book?ref=abc123",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "partner_name": "Amadeus"
            }
        }


class ClickHistoryResponse(BaseModel):
    """Schema for click history items"""
    
    id: int
    origin: str
    destination: str
    departure_date: date
    return_date: Optional[date]
    price: float
    currency: str
    partner_name: str
    converted: bool = Field(..., description="Whether this click resulted in a booking")
    clicked_at: datetime
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 12345,
                "origin": "TAS",
                "destination": "IST",
                "departure_date": "2025-11-15",
                "return_date": "2025-11-22",
                "price": 450.00,
                "currency": "USD",
                "partner_name": "Amadeus",
                "converted": True,
                "clicked_at": "2025-10-01T14:30:00Z"
            }
        }


class ClickStats(BaseModel):
    """User's click statistics"""
    
    total_clicks: int = Field(..., description="Total number of clicks")
    total_amount_clicked: float = Field(..., description="Total amount of flights clicked")
    converted_bookings: int = Field(..., description="Number of confirmed bookings")
    recent_clicks_30d: int = Field(..., description="Clicks in last 30 days")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_clicks": 15,
                "total_amount_clicked": 6750.50,
                "converted_bookings": 3,
                "recent_clicks_30d": 5
            }
        }


class ClickConversionUpdate(BaseModel):
    """Schema for marking click as converted (webhook use)"""
    
    click_id: int = Field(..., description="Click ID to mark as converted")
    commission_amount: Optional[float] = Field(None, description="Actual commission earned")
    partner_booking_ref: Optional[str] = Field(None, description="Partner's booking reference")
    
    class Config:
        json_schema_extra = {
            "example": {
                "click_id": 12345,
                "commission_amount": 15.50,
                "partner_booking_ref": "ABC123XYZ"
            }
        }


class TopRouteResponse(BaseModel):
    """Schema for top clicked routes analytics"""
    
    route: str = Field(..., description="Route in format 'ORIGIN-DESTINATION'")
    click_count: int = Field(..., description="Number of clicks")
    avg_price: float = Field(..., description="Average price clicked")
    
    class Config:
        json_schema_extra = {
            "example": {
                "route": "TAS-IST",
                "click_count": 1250,
                "avg_price": 425.75
            }
        }


class ConversionRateResponse(BaseModel):
    """Schema for conversion rate analytics"""
    
    period_days: int = Field(..., description="Period analyzed in days")
    total_clicks: int = Field(..., description="Total clicks in period")
    converted_clicks: int = Field(..., description="Clicks that resulted in bookings")
    conversion_rate_percent: float = Field(..., description="Conversion rate as percentage")
    
    class Config:
        json_schema_extra = {
            "example": {
                "period_days": 30,
                "total_clicks": 1500,
                "converted_clicks": 225,
                "conversion_rate_percent": 15.0
            }
        }