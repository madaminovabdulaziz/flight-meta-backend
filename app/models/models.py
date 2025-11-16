# app/models/models.py - Production-Ready Schema (Metasearch → OTA Ready)

from sqlalchemy import (
    Column, String, Integer, Float, ForeignKey,
    DateTime, Enum as SQLAlchemyEnum, CHAR, JSON, Date, Text, Boolean, Index
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()

# ==========================
# ENUMS
# ==========================

class BookingSource(enum.Enum):
    """Where the booking originated"""
    METASEARCH_REDIRECT = "METASEARCH_REDIRECT"  # User clicked to partner (current)
    DIRECT_BOOKING = "DIRECT_BOOKING"             # User booked with us (future OTA)


class BookingStatus(enum.Enum):
    """Unified status for both metasearch tracking AND direct bookings"""
    # Metasearch statuses
    CLICK_INITIATED = "CLICK_INITIATED"      # User clicked through to partner
    PARTNER_CONFIRMED = "PARTNER_CONFIRMED"   # Partner reported booking
    
    # OTA statuses (for future use)
    PENDING_PAYMENT = "PENDING_PAYMENT"
    PAYMENT_PROCESSING = "PAYMENT_PROCESSING"
    CONFIRMED = "CONFIRMED"
    TICKETED = "TICKETED"
    
    # Common statuses
    CANCELLED = "CANCELLED"
    REFUNDED = "REFUNDED"
    FAILED = "FAILED"


class PassengerType(enum.Enum):
    ADULT = "adult"
    CHILD = "child"
    INFANT = "infant"


class PaymentStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class PaymentProvider(enum.Enum):
    """Payment gateway providers"""
    STRIPE = "stripe"
    CLICK = "click"
    PAYME = "payme"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"


# ==========================
# AI TRIP PLANNER ENUMS
# ==========================

class MemoryType(enum.Enum):
    """Types of memories for AI Trip Planner personalization"""
    PREFERENCE = "preference"      # Explicit user preferences
    TRIP_HISTORY = "trip_history"  # Past trip patterns
    BEHAVIOR = "behavior"          # Observed patterns from choices
    FEEDBACK = "feedback"          # Corrections or dislikes


class SessionStatus(enum.Enum):
    """AI Trip Planner conversation session status"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class TripPlanStatus(enum.Enum):
    """AI Trip Planner trip planning status (distinct from booking status)"""
    PLANNING = "planning"                # User is planning
    PLANNED = "planned"                  # User completed planning
    BOOKED_ELSEWHERE = "booked_elsewhere"  # User booked on partner site
    CANCELLED = "cancelled"              # User cancelled plan


# ==========================
# CORE: USERS
# ==========================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    
    # Preferences
    preferred_currency = Column(String(3), default="USD")
    preferred_language = Column(String(5), default="en")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    bookings = relationship("Booking", back_populates="user", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="user")
    saved_searches = relationship("SavedSearch", back_populates="user", cascade="all, delete-orphan")
    
    # AI Trip Planner relationships
    ai_sessions = relationship("AISession", back_populates="user", cascade="all, delete-orphan")
    trip_plans = relationship("TripPlan", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    memory_events = relationship("MemoryEvent", back_populates="user", cascade="all, delete-orphan")


# ==========================
# BOOKINGS (Unified for Metasearch + OTA)
# ==========================

class Booking(Base):
    """
    Unified booking table that handles BOTH:
    1. Metasearch click-throughs (booking_source = METASEARCH_REDIRECT)
    2. Direct OTA bookings (booking_source = DIRECT_BOOKING)
    """
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # FIXED: Use UUID instead of sequential number to avoid collisions
    internal_booking_id = Column(String(36), unique=True, index=True, nullable=False)  # UUID
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    booking_source = Column(SQLAlchemyEnum(BookingSource), nullable=False, index=True)
    
    # Flight details (always stored)
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(DateTime(timezone=True), nullable=False)
    return_date = Column(DateTime(timezone=True), nullable=True)
    
    # Pricing
    total_price = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    
    # Status
    status = Column(SQLAlchemyEnum(BookingStatus), nullable=False, index=True)
    
    # Partner info (for metasearch redirects)
    partner_id = Column(Integer, ForeignKey("partners.id", ondelete="SET NULL"), nullable=True)
    partner_booking_ref = Column(String(255), nullable=True, index=True)  # Partner's own reference
    partner_deeplink = Column(Text, nullable=True)
    
    # Commission tracking (metasearch only)
    commission_expected = Column(Float, nullable=True)
    commission_earned = Column(Float, nullable=True)
    commission_currency = Column(String(3), nullable=True)
    
    # OTA-specific fields (NULL for metasearch bookings)
    pnr = Column(String(20), nullable=True, index=True)  # Amadeus PNR for direct bookings
    ticket_numbers = Column(JSON, nullable=True)  # Array of ticket numbers
    
    # Flight data storage strategy:
    # - Metasearch: Full snapshot in raw_flight_data (segments not normalized)
    # - OTA: Normalize into BookingSegment table, keep raw for audit only
    raw_flight_data = Column(JSON, nullable=True)  # Raw API response (always keep for audit)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    booked_at = Column(DateTime(timezone=True), nullable=True)  # When confirmed
    
    # Relationships
    user = relationship("User", back_populates="bookings")
    partner = relationship("Partner", back_populates="bookings")
    passengers = relationship("Passenger", back_populates="booking", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="booking", cascade="all, delete-orphan")
    segments = relationship("BookingSegment", back_populates="booking", cascade="all, delete-orphan")
    status_history = relationship("BookingStatusHistory", back_populates="booking", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_booking_user_date', 'user_id', 'departure_date'),
        Index('idx_booking_route', 'origin', 'destination', 'departure_date'),
        Index('idx_booking_source_status', 'booking_source', 'status'),
    )


class BookingStatusHistory(Base):
    """
    FIXED: Audit trail for all booking status changes.
    Immutable log - records are never updated or deleted.
    Critical for compliance, dispute resolution, and debugging.
    """
    __tablename__ = "booking_status_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Status transition
    from_status = Column(SQLAlchemyEnum(BookingStatus), nullable=True)  # NULL on first record
    to_status = Column(SQLAlchemyEnum(BookingStatus), nullable=False)
    
    # Who/what made the change
    changed_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    changed_by = Column(String(100), nullable=False)  # "system", "admin:user@example.com", "payment_webhook"
    reason = Column(Text, nullable=True)
    
    # Metadata
    changed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    
    # Optional: snapshot of entire booking at this moment (for forensics)
    booking_snapshot = Column(JSON, nullable=True)
    
    # Relationships
    booking = relationship("Booking", back_populates="status_history")
    
    __table_args__ = (
        Index('idx_status_history_booking_time', 'booking_id', 'changed_at'),
    )


class Passenger(Base):
    """
    Passenger details - used for:
    1. Metasearch: Optional pre-fill data for faster partner booking
    2. OTA: Required passenger info for actual booking
    """
    __tablename__ = "passengers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    passenger_type = Column(SQLAlchemyEnum(PassengerType), default=PassengerType.ADULT)
    
    # Personal details
    title = Column(String(10), nullable=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=True)
    date_of_birth = Column(Date, nullable=False)
    
    # Travel documents (required for OTA, optional for metasearch)
    passport_number = Column(String(50), nullable=True)
    passport_expiry_date = Column(Date, nullable=True)
    nationality = Column(String(2), nullable=True)  # ISO 3166-1 alpha-2
    
    # Contact (lead passenger only)
    email = Column(String(255), nullable=True)
    phone_number = Column(String(50), nullable=True)
    
    # Relationships
    booking = relationship("Booking", back_populates="passengers")


class BookingSegment(Base):
    """
    Flight segments for OTA bookings ONLY.
    For metasearch bookings, this table is EMPTY (data stays in JSON).
    Only populated when booking_source = DIRECT_BOOKING.
    """
    __tablename__ = "booking_segments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Segment ordering
    segment_order = Column(Integer, nullable=False)  # 1, 2, 3...
    is_return = Column(Boolean, default=False)  # Outbound or return leg
    
    # Flight details
    airline_id = Column(Integer, ForeignKey("airlines.id", ondelete="SET NULL"), nullable=True)
    flight_number = Column(String(10), nullable=False)
    aircraft_id = Column(Integer, ForeignKey("aircraft.id", ondelete="SET NULL"), nullable=True)
    
    departure_airport = Column(String(3), ForeignKey("airports.iata_code"), nullable=False)
    arrival_airport = Column(String(3), ForeignKey("airports.iata_code"), nullable=False)
    departure_time = Column(DateTime(timezone=True), nullable=False)
    arrival_time = Column(DateTime(timezone=True), nullable=False)
    
    duration_minutes = Column(Integer, nullable=False)
    
    # Fare details (for OTA)
    cabin_class = Column(String(50), nullable=True)
    fare_class = Column(String(10), nullable=True)
    booking_class = Column(String(2), nullable=True)
    
    # Baggage (simplified - can be JSON for complex cases)
    baggage_allowance = Column(JSON, nullable=True)  # {carry_on: "1x8kg", checked: "1x23kg"}
    
    # Relationships
    booking = relationship("Booking", back_populates="segments")
    airline = relationship("Airline")
    aircraft = relationship("Aircraft")
    departure = relationship("Airport", foreign_keys=[departure_airport])
    arrival = relationship("Airport", foreign_keys=[arrival_airport])
    
    __table_args__ = (
        Index('idx_segment_booking_order', 'booking_id', 'segment_order'),
    )


class Payment(Base):
    """
    Payment records - ONLY used for OTA direct bookings.
    Metasearch bookings have no payment records (they pay the partner).
    """
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(String(100), unique=True, index=True, nullable=False)  # Your internal ID
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Payment details
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    method = Column(String(50), nullable=False)  # "card", "bank_transfer", etc.
    
    # FIXED: Use enum for payment provider
    provider = Column(SQLAlchemyEnum(PaymentProvider), nullable=False)
    status = Column(SQLAlchemyEnum(PaymentStatus), default=PaymentStatus.PENDING, index=True)
    
    # Provider integration
    provider_transaction_id = Column(String(255), nullable=True, index=True)  # Stripe charge ID, etc.
    provider_response = Column(JSON, nullable=True)  # Full response from payment gateway
    
    # Refunds
    refunded_amount = Column(Float, default=0.0)
    refund_reason = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    paid_at = Column(DateTime(timezone=True), nullable=True)
    refunded_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="payments")
    booking = relationship("Booking", back_populates="payments")
    
    __table_args__ = (
        Index('idx_payment_provider_status', 'provider', 'status'),
    )


# ==========================
# SEARCH & ALERTS (MERGED)
# ==========================

class SavedSearch(Base):
    """
    FIXED: Merged SavedSearch and PriceAlert into one table.
    Users can save searches AND optionally enable price alerts on them.
    """
    __tablename__ = "saved_searches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Search parameters
    origin = Column(String(3), nullable=False)
    destination = Column(String(3), nullable=False)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    
    passengers_adults = Column(Integer, default=1)
    passengers_children = Column(Integer, default=0)
    passengers_infants = Column(Integer, default=0)
    cabin_class = Column(String(20), default="ECONOMY")
    
    # User customization
    search_name = Column(String(100), nullable=True)  # User's friendly name
    
    # MERGED: Price alert functionality
    is_price_alert = Column(Boolean, default=False, index=True)
    target_price = Column(Float, nullable=True)
    alert_currency = Column(String(3), nullable=True)
    alert_triggered_at = Column(DateTime(timezone=True), nullable=True)
    alert_triggered_price = Column(Float, nullable=True)
    
    # Lifecycle
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)  # For alerts
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="saved_searches")
    
    __table_args__ = (
        Index('idx_search_user_active', 'user_id', 'is_active'),
        Index('idx_alert_active_expires', 'is_price_alert', 'is_active', 'expires_at'),
    )


# ==========================
# AI TRIP PLANNER MODELS
# ==========================

class AISession(Base):
    """
    AI Trip Planner conversation sessions.
    Tracks the state machine flow for collecting flight parameters.
    """
    __tablename__ = "ai_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User tracking (NULL for anonymous users)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Session identification
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    
    # Complete state object (from Part 2 of spec)
    # Contains: destination, origin, dates, passengers, class, budget, flexibility
    # Also contains: missing_parameter, last_question, conversation_history, etc.
    state_json = Column(JSON, nullable=False)
    
    # Status
    status = Column(SQLAlchemyEnum(SessionStatus), default=SessionStatus.ACTIVE, nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="ai_sessions")
    trip_plans = relationship("TripPlan", back_populates="session")
    memory_events = relationship("MemoryEvent", back_populates="session")
    
    __table_args__ = (
        Index('idx_ai_session_user_status', 'user_id', 'status'),
        Index('idx_ai_session_created', 'created_at'),
    )


class TripPlan(Base):
    """
    AI Trip Planner trip plans (distinct from Booking).
    Represents a planned trip that may or may not result in a booking.
    Links to Booking when user books through partners or directly.
    """
    __tablename__ = "trip_plans"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User and session
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("ai_sessions.id", ondelete="SET NULL"), nullable=True)
    
    # Flight details (collected from AI conversation)
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    
    passengers = Column(Integer, default=1, nullable=False)
    travel_class = Column(String(50), nullable=True)
    budget = Column(Float, nullable=True)
    flexibility = Column(Integer, nullable=True)  # Days of flexibility
    
    # Status
    status = Column(SQLAlchemyEnum(TripPlanStatus), default=TripPlanStatus.PLANNING, nullable=False, index=True)
    
    # Link to actual booking if user booked
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="SET NULL"), nullable=True)
    
    # Recommended flights snapshot (from ranking engine)
    recommended_flights_json = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="trip_plans")
    session = relationship("AISession", back_populates="trip_plans")
    booking = relationship("Booking")
    
    __table_args__ = (
        Index('idx_trip_plan_user_status', 'user_id', 'status'),
        Index('idx_trip_plan_route_date', 'origin', 'destination', 'departure_date'),
    )


class UserPreference(Base):
    """
    AI Trip Planner structured preferences (Layer 2 of memory system).
    Stores machine-readable preferences as key-value pairs.
    
    Examples:
    - key="preferred_airports", value_json=["LGW", "LHR"]
    - key="preferred_airlines", value_json=["TK", "QR"]
    - key="budget_range", value_json=[200, 500]
    - key="prefers_direct", value_json=true
    """
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User reference
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Preference key (e.g., "preferred_airports", "budget_range")
    key = Column(String(100), nullable=False)
    
    # Preference value as JSON (supports lists, dicts, primitives)
    value_json = Column(JSON, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    __table_args__ = (
        Index('idx_user_preference_key', 'user_id', 'key', unique=True),
    )


class MemoryEvent(Base):
    """
    AI Trip Planner memory events (Layer 3 bridge to Qdrant vector DB).
    Logs memory-worthy events that get embedded and stored in Qdrant.
    This MySQL table serves as an audit trail and bridge.
    
    Memory Types:
    - PREFERENCE: Explicit user preferences ("I prefer Turkish Airlines")
    - TRIP_HISTORY: Past trips and experiences
    - BEHAVIOR: Observed patterns ("User always chooses cheapest option")
    - FEEDBACK: Corrections or dislikes ("I hate overnight layovers")
    """
    __tablename__ = "memory_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User and session references
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("ai_sessions.id", ondelete="SET NULL"), nullable=True)
    
    # Memory details
    type = Column(SQLAlchemyEnum(MemoryType), nullable=False, index=True)
    
    # Natural language summary of the memory
    content = Column(Text, nullable=False)
    
    # Additional metadata as JSON
    # Examples:
    # - {"category": "airline", "airline": "TK", "importance": "high"}
    # - {"category": "stops", "importance": "high", "confidence": 0.9}
    # - {"destination": "IST", "trip_date": "2025-03-15", "satisfaction": "high"}
    metadata_json = Column(JSON, nullable=True)
    
    # Reference to corresponding vector in Qdrant
    vector_id = Column(String(255), nullable=True, index=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User", back_populates="memory_events")
    session = relationship("AISession", back_populates="memory_events")
    
    __table_args__ = (
        Index('idx_memory_user_type', 'user_id', 'type'),
        Index('idx_memory_created', 'created_at'),
    )


# ==========================
# REFERENCE DATA
# ==========================

class Airport(Base):
    __tablename__ = "airports"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    iata_code = Column(CHAR(3), unique=True, index=True, nullable=False)
    icao_code = Column(String(4), unique=True, nullable=True)
    
    name = Column(String(255), nullable=False)
    city = Column(String(100), nullable=False, index=True)
    country = Column(String(100), nullable=False, index=True)
    country_code = Column(String(2), nullable=False, index=True)
    
    timezone = Column(String(50), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    
    is_active = Column(Boolean, default=True, index=True)
    
    __table_args__ = (
        Index('idx_airport_city_country', 'city', 'country'),
        Index('idx_airport_city_active', 'city', 'is_active'),
        Index('idx_airport_coordinates', 'latitude', 'longitude'),
    )


class Airline(Base):
    __tablename__ = "airlines"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    iata_code = Column(CHAR(2), unique=True, nullable=False, index=True)
    icao_code = Column(String(3), unique=True, nullable=True)
    
    name = Column(String(255), nullable=False)
    country = Column(String(100), nullable=True)
    
    alliance = Column(String(50), nullable=True, index=True)
    logo_url = Column(String(512), nullable=True)
    is_low_cost = Column(Boolean, default=False, index=True)
    is_active = Column(Boolean, default=True, index=True)


class Aircraft(Base):
    __tablename__ = "aircraft"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    iata_code = Column(String(4), unique=True, nullable=False, index=True)
    
    name = Column(String(100), nullable=False)
    manufacturer = Column(String(100), nullable=True)
    
    is_widebody = Column(Boolean, default=False)
    typical_seats = Column(Integer, nullable=True)


# ==========================
# PARTNERS
# ==========================

class Partner(Base):
    """
    Booking partners for metasearch (Amadeus, Kiwi, airlines, OTAs).
    When you become OTA, create a partner record with is_self=True.
    """
    __tablename__ = "partners"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    
    partner_type = Column(String(50), nullable=False)  # "GDS", "Airline", "OTA", "Aggregator"
    
    # Special flag for when you become an OTA
    is_self = Column(Boolean, default=False, index=True)  # True for your own direct bookings
    
    # Metasearch integration
    api_enabled = Column(Boolean, default=True)
    deeplink_template = Column(Text, nullable=True)  # URL template with {params}
    
    # Business terms
    commission_model = Column(String(50), nullable=True)  # "CPA", "CPC", "Revenue Share"
    commission_rate = Column(Float, nullable=True)
    commission_currency = Column(String(3), nullable=True)
    
    # Technical
    api_credentials = Column(JSON, nullable=True)  # Store encrypted
    webhook_secret = Column(String(255), nullable=True)  # For validating partner callbacks
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    notes = Column(Text, nullable=True)
    
    # Relationships
    bookings = relationship("Booking", back_populates="partner")


# ==========================
# ANALYTICS & CLICK TRACKING
# ==========================

class Click(Base):
    """
    CRITICAL TABLE: Track when users click through to booking partners.
    This is your primary revenue tracking mechanism.
    Every click = potential commission.
    """
    __tablename__ = "clicks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # User tracking (optional - may be anonymous)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    session_id = Column(String(64), index=True, nullable=False)  # Track anonymous users
    
    # Flight details (denormalized for analytics)
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    
    # Pricing
    price = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    
    # Partner info
    partner_id = Column(Integer, ForeignKey("partners.id", ondelete="SET NULL"), nullable=True)
    partner_name = Column(String(100), nullable=False)  # Denormalized for speed
    partner_deeplink = Column(Text, nullable=False)
    
    # Flight offer snapshot (for later analysis)
    flight_offer_snapshot = Column(JSON, nullable=True)
    
    # Conversion tracking
    converted = Column(Boolean, default=False, index=True)  # Did they book?
    commission_expected = Column(Float, nullable=True)  # Expected commission at click time
    commission_earned = Column(Float, nullable=True)  # Actual commission from partner webhook
    conversion_tracked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Technical metadata
    ip_hash = Column(String(64), nullable=True)  # Hashed for privacy
    user_agent = Column(String(512), nullable=True)
    referrer = Column(String(512), nullable=True)
    
    clicked_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Relationships
    user = relationship("User")
    partner = relationship("Partner")
    
    __table_args__ = (
        Index('idx_click_session_date', 'session_id', 'clicked_at'),
        Index('idx_click_partner_converted', 'partner_name', 'converted'),
        Index('idx_click_route', 'origin', 'destination', 'departure_date'),
    )


class PriceSnapshot(Base):
    """
    Historical price data for route analysis, price graphs, and "best time to book" features.
    Populated by scheduled jobs that periodically check prices.
    """
    __tablename__ = "price_snapshots"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Route
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    
    # Price data
    lowest_price = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    
    # Source
    partner_name = Column(String(100), nullable=False)
    cabin_class = Column(String(20), default="ECONOMY")
    passengers_adults = Column(Integer, default=1)
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_price_route_date', 'origin', 'destination', 'departure_date', 'recorded_at'),
    )