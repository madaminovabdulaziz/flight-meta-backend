from typing import List, Optional
from pydantic import BaseModel, Field

# --- Traveler Sub-Schemas ---

class TravelerContact(BaseModel):
    """Schema for traveler contact details."""
    email: str = Field(..., description="Traveler's primary email address.")
    phone_number: str = Field(..., description="Traveler's mobile phone number, including country code (e.g., +998901234567).")

class TravelerName(BaseModel):
    """Schema for traveler name details."""
    first_name: str = Field(..., description="Traveler's first name (as per passport).", example="ALI")
    last_name: str = Field(..., description="Traveler's last name/surname (as per passport).", example="SAIDOV")

class Traveler(BaseModel):
    """Detailed schema for a single passenger required for Amadeus booking."""
    first_name: str = Field(..., description="Traveler's first name.", example="ALI")
    last_name: str = Field(..., description="Traveler's last name.", example="SAIDOV")
    date_of_birth: str = Field(..., description="Traveler's date of birth (YYYY-MM-DD).", example="1990-01-01")
    email: str = Field(..., description="Traveler's primary email address.")
    phone_number: str = Field(..., description="Traveler's mobile phone number (+CountryCodeNumber).")
    # You might add more fields here later (e.g., gender, document ID)

# --- Request Body Schemas ---

class PriceFlightRequest(BaseModel):
    """Request body for re-pricing a flight offer."""
    # This ID is the 'id' field returned by the Flight Offers Search API
    flight_id: str = Field(..., description="The unique ID of the flight offer to be priced.")
    
    # Required original search parameters to retrieve the full offer from cache
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "flight_id": "1",
                "origin": "TAS",
                "destination": "IST",
                "departure_date": "2025-12-01",
                "return_date": "2025-12-08"
            }
        }

class CreateOrderRequest(BaseModel):
    """
    Request body for creating a flight order (booking/PNR).
    This uses the full priced flight offer data structure from the previous step.
    """
    # This should be the WHOLE JSON response from the 'price_flight_offer' call
    priced_offer_data: dict = Field(..., description="The complete JSON response from the /price endpoint.")
    
    passengers: List[Traveler] = Field(..., description="List of traveler details for the booking.")

    class Config:
        json_schema_extra = {
            "example": {
                "priced_offer_data": {
                    # ... The full Amadeus JSON structure goes here ...
                },
                "passengers": [
                    {
                        "first_name": "ALI",
                        "last_name": "SAIDOV",
                        "date_of_birth": "1990-01-01",
                        "email": "ali.saidov@skysearchai.dev",
                        "phone_number": "+998901234567"
                    }
                ]
            }
        }
