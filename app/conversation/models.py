# app/conversation/models.py
"""
Centralized data models for conversational flight search.
All Pydantic v2 models and enums live here to prevent circular imports.
Enhanced with validators to handle Redis serialization/deserialization.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ============================================================
# ENUMS
# ============================================================

class ConversationFlowState(str, Enum):
    """
    Optimized 5-6 step conversational flow for flight search.
    """
    INIT = "init"
    CONFIRM_DESTINATION = "confirm_destination"
    DATES = "dates"
    PASSENGERS = "passengers"
    PREFERENCES = "preferences"
    CONFIRMATION = "confirmation"
    SEARCH_READY = "search_ready"
    ERROR = "error"


class SuggestionType(str, Enum):
    """Types of suggestion chips"""
    QUICK_ACTION = "quick_action"
    DESTINATION = "destination"
    DATE_PRESET = "date_preset"
    PASSENGER_COUNT = "passenger_count"
    PREFERENCE = "preference"
    CONFIRMATION = "confirmation"
    CORRECTION = "correction"


class TripType(str, Enum):
    """Trip type classifications"""
    SOLO = "solo"
    COUPLE = "couple"
    FRIENDS = "friends"
    FAMILY = "family"
    BUSINESS = "business"


class CabinClass(str, Enum):
    """Cabin class options"""
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


# ============================================================
# CORE DATA MODELS
# ============================================================

class TripSpec(BaseModel):
    """
    Structured trip specification extracted from conversation.
    """
    origin: Optional[str] = None
    destination: Optional[str] = None
    depart_date: Optional[str] = None  # ISO format YYYY-MM-DD
    return_date: Optional[str] = None
    passengers: int = 1
    cabin_class: CabinClass = CabinClass.ECONOMY
    trip_type: Optional[TripType] = None
    flexible_dates: bool = False
    flexibility_days: Optional[int] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # ✅ FIX: Handle enum string conversion from Redis
    @field_validator("cabin_class", mode="before")
    @classmethod
    def parse_cabin_class(cls, value):
        """Convert string to CabinClass enum if needed"""
        if value is None:
            return CabinClass.ECONOMY
        if isinstance(value, str):
            try:
                return CabinClass(value)
            except ValueError:
                return CabinClass.ECONOMY
        return value
    
    @field_validator("trip_type", mode="before")
    @classmethod
    def parse_trip_type(cls, value):
        """Convert string to TripType enum if needed"""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return TripType(value)
            except ValueError:
                return None
        return value
    
    class Config:
        use_enum_values = True


class Suggestion(BaseModel):
    """
    Suggestion chip for user interaction.
    """
    type: SuggestionType
    label: str
    value: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # ✅ FIX: Handle enum string conversion
    @field_validator("type", mode="before")
    @classmethod
    def parse_suggestion_type(cls, value):
        """Convert string to SuggestionType enum if needed"""
        if isinstance(value, str):
            try:
                return SuggestionType(value)
            except ValueError:
                return SuggestionType.QUICK_ACTION
        return value
    
    class Config:
        use_enum_values = True


class Message(BaseModel):
    """
    Single message in conversation history.
    """
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationContext(BaseModel):
    """
    Complete conversation session state.
    Persisted in Redis with TTL.
    """
    session_id: str
    user_id: Optional[str] = None
    state: ConversationFlowState = ConversationFlowState.INIT
    trip_spec: TripSpec = Field(default_factory=TripSpec)
    messages: List[Message] = Field(default_factory=list)
    detected_origin: Optional[str] = None  # From IP geolocation
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # ✅ FIX: Handle state enum string conversion from Redis
    @field_validator("state", mode="before")
    @classmethod
    def parse_state(cls, value):
        """Convert string to ConversationFlowState enum if needed"""
        if isinstance(value, str):
            try:
                return ConversationFlowState(value)
            except ValueError:
                # Default to INIT if invalid state
                return ConversationFlowState.INIT
        return value
    
    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def parse_datetime(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================
# RESPONSE MODELS
# ============================================================

class ConversationResponse(BaseModel):
    """
    Standard response format for conversation endpoints.
    """
    session_id: str
    state: ConversationFlowState
    message: str
    suggestions: List[Suggestion] = Field(default_factory=list)
    placeholder: Optional[str] = None
    search_ready: bool = False
    trip_spec: Optional[TripSpec] = None
    error: Optional[str] = None
    validation_errors: Optional[List[str]] = None
    
    # ✅ FIX: Handle state enum string conversion
    @field_validator("state", mode="before")
    @classmethod
    def parse_state(cls, value):
        """Convert string to ConversationFlowState enum if needed"""
        if isinstance(value, str):
            try:
                return ConversationFlowState(value)
            except ValueError:
                return ConversationFlowState.INIT
        return value
    
    class Config:
        use_enum_values = True


class SearchParams(BaseModel):
    """
    Final search parameters ready for flight API.
    """
    origin: str
    destination: str
    depart_date: str
    return_date: Optional[str] = None
    passengers: int = 1
    cabin_class: str = "economy"
    currency: str = "USD"
    locale: str = "en"
    flexible_dates: bool = False
    flexibility_days: Optional[int] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# VALIDATION RESULT
# ============================================================

class ValidationResult(BaseModel):
    """
    Result of validation operations.
    """
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)