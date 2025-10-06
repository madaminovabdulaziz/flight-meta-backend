from pydantic import BaseModel
from typing import Optional

class SavedSearchCreate(BaseModel):
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0
    infants: int = 0


class SavedSearchUpdate(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    adults: Optional[int] = None
    children: Optional[int] = None
    infants: Optional[int] = None


class SavedSearch(BaseModel):
    id: int
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str] = None
    adults: int
    children: int
    infants: int

    class Config:
        from_attributes = True
