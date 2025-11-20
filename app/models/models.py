# app/models/models.py

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

class BookingSource(str, enum.Enum):
    METASEARCH_REDIRECT = "METASEARCH_REDIRECT"
    DIRECT_BOOKING = "DIRECT_BOOKING"

class BookingStatus(str, enum.Enum):
    CLICK_INITIATED = "CLICK_INITIATED"
    PARTNER_CONFIRMED = "PARTNER_CONFIRMED"
    PENDING_PAYMENT = "PENDING_PAYMENT"
    PAYMENT_PROCESSING = "PAYMENT_PROCESSING"
    CONFIRMED = "CONFIRMED"
    TICKETED = "TICKETED"
    CANCELLED = "CANCELLED"
    REFUNDED = "REFUNDED"
    FAILED = "FAILED"

class PassengerType(str, enum.Enum):
    ADULT = "adult"
    CHILD = "child"
    INFANT = "infant"

class PaymentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class PaymentProvider(str, enum.Enum):
    STRIPE = "stripe"
    CLICK = "click"
    PAYME = "payme"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"

# ==========================
# AI TRIP PLANNER ENUMS
# ==========================

class MemoryType(str, enum.Enum):
    PREFERENCE = "preference"
    TRIP_HISTORY = "trip_history"
    BEHAVIOR = "behavior"
    FEEDBACK = "feedback"  

class SessionStatus(str, enum.Enum):
    """
    AI Trip Planner conversation session status.
    Values are lowercase for DB compatibility.
    """
    ACTIVE = "active"
    SEARCHED = "searched"
    BOOKING = "booking"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

class TripPlanStatus(str, enum.Enum):
    PLANNING = "planning"
    PLANNED = "planned"
    BOOKED_ELSEWHERE = "booked_elsewhere"
    CANCELLED = "cancelled"


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
    
    preferred_currency = Column(String(3), default="USD")
    preferred_language = Column(String(5), default="en")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    bookings = relationship("Booking", back_populates="user", cascade="all, delete-orphan")
    payments = relationship("Payment", back_populates="user")
    saved_searches = relationship("SavedSearch", back_populates="user", cascade="all, delete-orphan")
    
    ai_sessions = relationship("AISession", back_populates="user", cascade="all, delete-orphan")
    trip_plans = relationship("TripPlan", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    memory_events = relationship("MemoryEvent", back_populates="user", cascade="all, delete-orphan")


# ==========================
# BOOKINGS
# ==========================

class Booking(Base):
    __tablename__ = "bookings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    internal_booking_id = Column(String(36), unique=True, index=True, nullable=False)
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    booking_source = Column(SQLAlchemyEnum(BookingSource), nullable=False, index=True)
    
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(DateTime(timezone=True), nullable=False)
    return_date = Column(DateTime(timezone=True), nullable=True)
    
    total_price = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    
    status = Column(SQLAlchemyEnum(BookingStatus), nullable=False, index=True)
    
    partner_id = Column(Integer, ForeignKey("partners.id", ondelete="SET NULL"), nullable=True)
    partner_booking_ref = Column(String(255), nullable=True, index=True)
    partner_deeplink = Column(Text, nullable=True)
    
    commission_expected = Column(Float, nullable=True)
    commission_earned = Column(Float, nullable=True)
    commission_currency = Column(String(3), nullable=True)
    
    pnr = Column(String(20), nullable=True, index=True)
    ticket_numbers = Column(JSON, nullable=True)
    raw_flight_data = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    booked_at = Column(DateTime(timezone=True), nullable=True)
    
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
    __tablename__ = "booking_status_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    from_status = Column(SQLAlchemyEnum(BookingStatus), nullable=True)
    to_status = Column(SQLAlchemyEnum(BookingStatus), nullable=False)
    
    changed_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    changed_by = Column(String(100), nullable=False)
    reason = Column(Text, nullable=True)
    
    changed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ip_address = Column(String(45), nullable=True)
    booking_snapshot = Column(JSON, nullable=True)
    
    booking = relationship("Booking", back_populates="status_history")
    
    __table_args__ = (
        Index('idx_status_history_booking_time', 'booking_id', 'changed_at'),
    )

class Passenger(Base):
    __tablename__ = "passengers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    passenger_type = Column(SQLAlchemyEnum(PassengerType), default=PassengerType.ADULT)
    
    title = Column(String(10), nullable=True)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    gender = Column(String(10), nullable=True)
    date_of_birth = Column(Date, nullable=False)
    
    passport_number = Column(String(50), nullable=True)
    passport_expiry_date = Column(Date, nullable=True)
    nationality = Column(String(2), nullable=True)
    
    email = Column(String(255), nullable=True)
    phone_number = Column(String(50), nullable=True)
    
    booking = relationship("Booking", back_populates="passengers")

class BookingSegment(Base):
    __tablename__ = "booking_segments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    segment_order = Column(Integer, nullable=False)
    is_return = Column(Boolean, default=False)
    
    airline_id = Column(Integer, ForeignKey("airlines.id", ondelete="SET NULL"), nullable=True)
    flight_number = Column(String(10), nullable=False)
    aircraft_id = Column(Integer, ForeignKey("aircraft.id", ondelete="SET NULL"), nullable=True)
    
    departure_airport = Column(String(3), ForeignKey("airports.iata_code"), nullable=False)
    arrival_airport = Column(String(3), ForeignKey("airports.iata_code"), nullable=False)
    departure_time = Column(DateTime(timezone=True), nullable=False)
    arrival_time = Column(DateTime(timezone=True), nullable=False)
    
    duration_minutes = Column(Integer, nullable=False)
    
    cabin_class = Column(String(50), nullable=True)
    fare_class = Column(String(10), nullable=True)
    booking_class = Column(String(2), nullable=True)
    baggage_allowance = Column(JSON, nullable=True)
    
    booking = relationship("Booking", back_populates="segments")
    airline = relationship("Airline")
    aircraft = relationship("Aircraft")
    departure = relationship("Airport", foreign_keys=[departure_airport])
    arrival = relationship("Airport", foreign_keys=[arrival_airport])
    
    __table_args__ = (
        Index('idx_segment_booking_order', 'booking_id', 'segment_order'),
    )

class Payment(Base):
    __tablename__ = "payments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    payment_id = Column(String(100), unique=True, index=True, nullable=False)
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="CASCADE"), nullable=False, index=True)
    
    amount = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    method = Column(String(50), nullable=False)
    
    provider = Column(SQLAlchemyEnum(PaymentProvider), nullable=False)
    status = Column(SQLAlchemyEnum(PaymentStatus), default=PaymentStatus.PENDING, index=True)
    
    provider_transaction_id = Column(String(255), nullable=True, index=True)
    provider_response = Column(JSON, nullable=True)
    
    refunded_amount = Column(Float, default=0.0)
    refund_reason = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    paid_at = Column(DateTime(timezone=True), nullable=True)
    refunded_at = Column(DateTime(timezone=True), nullable=True)
    
    user = relationship("User", back_populates="payments")
    booking = relationship("Booking", back_populates="payments")
    
    __table_args__ = (
        Index('idx_payment_provider_status', 'provider', 'status'),
    )

# ==========================
# SAVED SEARCH
# ==========================

class SavedSearch(Base):
    __tablename__ = "saved_searches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    origin = Column(String(3), nullable=False)
    destination = Column(String(3), nullable=False)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    
    passengers_adults = Column(Integer, default=1)
    passengers_children = Column(Integer, default=0)
    passengers_infants = Column(Integer, default=0)
    cabin_class = Column(String(20), default="ECONOMY")
    
    search_name = Column(String(100), nullable=True)
    
    is_price_alert = Column(Boolean, default=False, index=True)
    target_price = Column(Float, nullable=True)
    alert_currency = Column(String(3), nullable=True)
    alert_triggered_at = Column(DateTime(timezone=True), nullable=True)
    alert_triggered_price = Column(Float, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="saved_searches")
    
    __table_args__ = (
        Index('idx_search_user_active', 'user_id', 'is_active'),
        Index('idx_alert_active_expires', 'is_price_alert', 'is_active', 'expires_at'),
    )

# ==========================
# AI TRIP PLANNER MODELS
# ==========================

class AISession(Base):
    __tablename__ = "ai_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    state_json = Column(JSON, nullable=False)
    
    # FIXED: Use values_callable so SQLAlchemy persists lowercase 'active'
    status = Column(
        SQLAlchemyEnum(
            SessionStatus,
            values_callable=lambda obj: [e.value for e in obj]
        ),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True
    )
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    user = relationship("User", back_populates="ai_sessions")
    trip_plans = relationship("TripPlan", back_populates="session")
    memory_events = relationship("MemoryEvent", back_populates="session")
    
    __table_args__ = (
        Index('idx_ai_session_user_status', 'user_id', 'status'),
        Index('idx_ai_session_created', 'created_at'),
    )

class TripPlan(Base):
    __tablename__ = "trip_plans"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("ai_sessions.id", ondelete="SET NULL"), nullable=True)
    
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    
    passengers = Column(Integer, default=1, nullable=False)
    travel_class = Column(String(50), nullable=True)
    budget = Column(Float, nullable=True)
    flexibility = Column(Integer, nullable=True)
    
    status = Column(SQLAlchemyEnum(TripPlanStatus), default=TripPlanStatus.PLANNING, nullable=False, index=True)
    
    booking_id = Column(Integer, ForeignKey("bookings.id", ondelete="SET NULL"), nullable=True)
    recommended_flights_json = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    user = relationship("User", back_populates="trip_plans")
    session = relationship("AISession", back_populates="trip_plans")
    booking = relationship("Booking")
    
    __table_args__ = (
        Index('idx_trip_plan_user_status', 'user_id', 'status'),
        Index('idx_trip_plan_route_date', 'origin', 'destination', 'departure_date'),
    )

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    key = Column(String(100), nullable=False)
    value_json = Column(JSON, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    user = relationship("User", back_populates="preferences")
    
    __table_args__ = (
        Index('idx_user_preference_key', 'user_id', 'key', unique=True),
    )

class MemoryEvent(Base):
    __tablename__ = "memory_events"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("ai_sessions.id", ondelete="SET NULL"), nullable=True)
    
    type = Column(SQLAlchemyEnum(MemoryType), nullable=False, index=True)
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    vector_id = Column(String(255), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    user = relationship("User", back_populates="memory_events")
    session = relationship("AISession", back_populates="memory_events")
    
    __table_args__ = (
        Index('idx_memory_user_type', 'user_id', 'type'),
        Index('idx_memory_created', 'created_at'),
    )

# ==========================
# REFERENCE & ANALYTICS
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

class Partner(Base):
    __tablename__ = "partners"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(100), nullable=False)
    partner_type = Column(String(50), nullable=False)
    is_self = Column(Boolean, default=False, index=True)
    api_enabled = Column(Boolean, default=True)
    deeplink_template = Column(Text, nullable=True)
    commission_model = Column(String(50), nullable=True)
    commission_rate = Column(Float, nullable=True)
    commission_currency = Column(String(3), nullable=True)
    api_credentials = Column(JSON, nullable=True)
    webhook_secret = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text, nullable=True)
    bookings = relationship("Booking", back_populates="partner")

class Click(Base):
    __tablename__ = "clicks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    session_id = Column(String(64), index=True, nullable=False)
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    price = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    partner_id = Column(Integer, ForeignKey("partners.id", ondelete="SET NULL"), nullable=True)
    partner_name = Column(String(100), nullable=False)
    partner_deeplink = Column(Text, nullable=False)
    flight_offer_snapshot = Column(JSON, nullable=True)
    converted = Column(Boolean, default=False, index=True)
    commission_expected = Column(Float, nullable=True)
    commission_earned = Column(Float, nullable=True)
    conversion_tracked_at = Column(DateTime(timezone=True), nullable=True)
    ip_hash = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)
    referrer = Column(String(512), nullable=True)
    clicked_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    user = relationship("User")
    partner = relationship("Partner")
    __table_args__ = (
        Index('idx_click_session_date', 'session_id', 'clicked_at'),
        Index('idx_click_partner_converted', 'partner_name', 'converted'),
        Index('idx_click_route', 'origin', 'destination', 'departure_date'),
    )

class PriceSnapshot(Base):
    __tablename__ = "price_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    origin = Column(String(3), nullable=False, index=True)
    destination = Column(String(3), nullable=False, index=True)
    departure_date = Column(Date, nullable=False)
    return_date = Column(Date, nullable=True)
    lowest_price = Column(Float, nullable=False)
    currency = Column(String(3), nullable=False)
    partner_name = Column(String(100), nullable=False)
    cabin_class = Column(String(20), default="ECONOMY")
    passengers_adults = Column(Integer, default=1)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    __table_args__ = (
        Index('idx_price_route_date', 'origin', 'destination', 'departure_date', 'recorded_at'),
    )