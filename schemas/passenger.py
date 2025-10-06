# # schemas/passenger.py
# from pydantic import BaseModel, Field
# from typing import Optional
# from datetime import date
# from enum import Enum

# class PassengerTypeEnum(str, Enum):
#     ADULT = "adult"
#     CHILD = "child"
#     INFANT = "infant"

# class PassengerBase(BaseModel):
#     passenger_type: PassengerTypeEnum = PassengerTypeEnum.ADULT
#     title: str = Field(..., max_length=10)  # Mr, Mrs, Ms, Miss
#     first_name: str = Field(..., min_length=1, max_length=100)
#     last_name: str = Field(..., min_length=1, max_length=100)
#     gender: str = Field(..., max_length=10)  # male, female
#     date_of_birth: date

# class PassengerCreate(PassengerBase):
#     passport_number: Optional[str] = Field(None, max_length=50)
#     passport_expiry_date: Optional[date] = None
#     nationality: Optional[str] = Field(None, max_length=10)
#     email: Optional[str] = None
#     phone_number: Optional[str] = None

# class PassengerResponse(PassengerCreate):
#     id: int
#     booking_id: int
    
#     class Config:
#         from_attributes = True

# class PassengerUpdate(BaseModel):
#     passport_number: Optional[str] = None
#     passport_expiry_date: Optional[date] = None
#     email: Optional[str] = None
#     phone_number: Optional[str] = None