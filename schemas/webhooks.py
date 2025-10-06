# schemas/webhooks.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PartnerConversionWebhook(BaseModel):
    """
    Schema for partner webhook when a booking is confirmed.
    
    This is what partners send to your webhook endpoint
    to notify you that a click resulted in a booking.
    """
    
    partner_name: str = Field(..., description="Partner identifier")
    click_id: int = Field(..., description="Your click tracking ID")
    partner_booking_ref: str = Field(..., description="Partner's booking reference")
    
    # Commission details
    commission_amount: float = Field(..., gt=0, description="Actual commission earned")
    commission_currency: str = Field("USD", description="Commission currency")
    
    # Booking details (optional, for reconciliation)
    booking_amount: Optional[float] = Field(None, description="Total booking amount")
    booking_currency: Optional[str] = Field(None, description="Booking currency")
    
    # Timestamp from partner
    booking_timestamp: Optional[datetime] = Field(None, description="When booking was made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "partner_name": "Amadeus",
                "click_id": 12345,
                "partner_booking_ref": "ABC123XYZ",
                "commission_amount": 15.50,
                "commission_currency": "USD",
                "booking_amount": 450.00,
                "booking_currency": "USD",
                "booking_timestamp": "2025-10-01T14:30:00Z"
            }
        }


class WebhookSignatureTest(BaseModel):
    """Schema for testing webhook signatures"""
    
    partner_name: str = Field(..., description="Partner to test")
    test_payload: dict = Field({}, description="Any test data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "partner_name": "Amadeus",
                "test_payload": {"test": "data"}
            }
        }