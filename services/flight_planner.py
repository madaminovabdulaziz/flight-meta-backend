"""State-machine orchestrator for structured Duffel flight searches.

This module implements the production-grade state flow described in
``flight-planner-v1.3``.  It keeps the user's conversational context in Redis,
handles automatic origin detection, enforces validation rules, and produces the
exact Duffel search payload when all parameters are collected.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from services.amadeus_service import redis_client
from services.ip_geolocation import IPGeolocationService


class FlightPlannerState(str, Enum):
    """All supported conversation states."""

    DETECT_ORIGIN = "DETECT_ORIGIN"
    ASK_ORIGIN_OVERRIDE = "ASK_ORIGIN_OVERRIDE"
    ASK_DESTINATION = "ASK_DESTINATION"
    ASK_DEPARTURE_DATE = "ASK_DEPARTURE_DATE"
    ASK_RETURN_DATE_OR_ONEWAY = "ASK_RETURN_DATE_OR_ONEWAY"
    ASK_RETURN_DATE = "ASK_RETURN_DATE"
    ASK_CABIN = "ASK_CABIN"
    ASK_PASSENGERS = "ASK_PASSENGERS"
    ASK_MAX_CONNECTIONS = "ASK_MAX_CONNECTIONS"
    ASK_TIME_WINDOW = "ASK_TIME_WINDOW"
    CONFIRMATION = "CONFIRMATION"
    FINALIZE_JSON = "FINALIZE_JSON"


class Passenger(BaseModel):
    """Passenger descriptor with strict validation rules."""

    type: str = Field(..., description="adult | child | infant")
    age: Optional[int] = Field(default=None, description="Age in years for children/infants")

    @field_validator("type", mode="before")
    @classmethod
    def _normalise_type(cls, value: str) -> str:
        if value is None:
            raise ValueError("Passenger type is required")
        value = value.strip().lower()
        allowed = {"adult", "child", "infant"}
        if value not in allowed:
            raise ValueError(f"Unsupported passenger type '{value}'")
        return value

    @model_validator(mode="after")
    def _validate_age(self) -> "Passenger":
        if self.type in {"child", "infant"}:
            if self.age is None:
                raise ValueError("Children and infants must include age")
            if self.age < 0:
                raise ValueError("Passenger age must be non-negative")
        return self

    def to_duffel(self) -> Dict[str, Any]:
        """Convert to Duffel-compatible passenger block."""
        return {"type": self.type}


class TimeWindow(BaseModel):
    """Time window for departures/arrivals (HH:MM)."""

    start: str = Field(..., alias="from")
    end: str = Field(..., alias="to")

    @field_validator("start", "end")
    @classmethod
    def _validate_time(cls, value: str) -> str:
        datetime.strptime(value, "%H:%M")
        return value

    def to_duffel(self) -> Dict[str, str]:
        return {"from": self.start, "to": self.end}


class FlightPlannerContext(BaseModel):
    """Container for all flight-search parameters."""

    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    flexible: Optional[bool] = None
    passengers: List[Passenger] = Field(default_factory=list)
    cabin_class: Optional[str] = None
    max_connections: Optional[int] = None
    time_window: Optional[TimeWindow] = None

    @field_validator("origin", "destination")
    @classmethod
    def _validate_iata(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Airports must be valid IATA codes")
        return value

    @field_validator("departure_date", "return_date")
    @classmethod
    def _validate_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        datetime.strptime(value, "%Y-%m-%d")
        return value

    @field_validator("cabin_class")
    @classmethod
    def _validate_cabin(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip().lower()
        allowed = {"economy", "premium_economy", "business", "first"}
        if value not in allowed:
            raise ValueError("Unsupported cabin class")
        return value

    @field_validator("max_connections")
    @classmethod
    def _validate_connections(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value < 0 or value > 3:
            raise ValueError("Connections must be between 0 and 3")
        return value

    def ensure_default_passenger(self) -> None:
        if not self.passengers:
            self.passengers.append(Passenger(type="adult"))

    def to_redis(self) -> bytes:
        payload = self.model_dump(mode="json")
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_redis(cls, raw: Optional[bytes]) -> "FlightPlannerContext":
        if not raw:
            return cls()
        data = json.loads(raw.decode("utf-8"))
        return cls(**data)


@dataclass
class ConversationTurn:
    """Response payload for each conversational step."""

    state: FlightPlannerState
    prompt: Optional[str]
    suggestions: List[str]
    context: FlightPlannerContext
    duffel_request: Optional[Dict[str, Any]] = None


class FlightPlannerStateMachine:
    """State-machine orchestrator backed by Redis."""

    STATE_TTL_HOURS = 48

    def __init__(
        self,
        redis_conn=None,
        ip_service: Optional[IPGeolocationService] = None,
        ttl_hours: int = STATE_TTL_HOURS,
    ) -> None:
        self.redis = redis_conn or redis_client
        self.ip_service = ip_service or IPGeolocationService()
        self.ttl_seconds = int(ttl_hours * 3600)
        self._memory_store: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _state_key(user_id: str) -> str:
        return f"state:{user_id}"

    @staticmethod
    def _context_key(user_id: str) -> str:
        return f"context:{user_id}"

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    async def _redis_set(self, key: str, value: bytes) -> None:
        try:
            await self.redis.set(key, value, ex=self.ttl_seconds)
        except Exception:
            expire_at = self._now() + timedelta(seconds=self.ttl_seconds)
            self._memory_store[key] = {"value": value, "expires_at": expire_at}

    async def _redis_get(self, key: str) -> Optional[bytes]:
        try:
            return await self.redis.get(key)
        except Exception:
            entry = self._memory_store.get(key)
            if not entry:
                return None
            if entry["expires_at"] < self._now():
                self._memory_store.pop(key, None)
                return None
            return entry["value"]

    async def get_state(self, user_id: str) -> FlightPlannerState:
        raw = await self._redis_get(self._state_key(user_id))
        if not raw:
            return await self._write_state(user_id, FlightPlannerState.DETECT_ORIGIN)
        try:
            payload = json.loads(raw.decode("utf-8"))
            updated_at = datetime.fromisoformat(payload.get("updated_at"))
        except Exception:
            return await self._write_state(user_id, FlightPlannerState.DETECT_ORIGIN)

        if self._now() - updated_at > timedelta(seconds=self.ttl_seconds):
            return await self._write_state(user_id, FlightPlannerState.DETECT_ORIGIN)

        state_value = payload.get("current_state", FlightPlannerState.DETECT_ORIGIN.value)
        try:
            return FlightPlannerState(state_value)
        except ValueError:
            return await self._write_state(user_id, FlightPlannerState.DETECT_ORIGIN)

    async def _write_state(self, user_id: str, state: FlightPlannerState) -> FlightPlannerState:
        payload = {
            "current_state": state.value,
            "updated_at": self._now().isoformat(),
        }
        await self._redis_set(self._state_key(user_id), json.dumps(payload).encode("utf-8"))
        return state

    async def set_state(self, user_id: str, state: FlightPlannerState) -> FlightPlannerState:
        return await self._write_state(user_id, state)

    async def reset(self, user_id: str) -> None:
        await self._write_state(user_id, FlightPlannerState.DETECT_ORIGIN)
        await self._redis_set(self._context_key(user_id), FlightPlannerContext().to_redis())

    async def get_context(self, user_id: str) -> FlightPlannerContext:
        raw = await self._redis_get(self._context_key(user_id))
        context = FlightPlannerContext.from_redis(raw)
        await self._redis_set(self._context_key(user_id), context.to_redis())
        return context

    async def update_context(self, user_id: str, **fields: Any) -> FlightPlannerContext:
        context = await self.get_context(user_id)
        try:
            updated = context.model_copy(update=fields)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        await self._redis_set(self._context_key(user_id), updated.to_redis())
        return updated

    async def autodetect_origin(self, user_id: str, ip_address: str) -> str:
        airport = await self.ip_service.get_nearest_airport_from_ip(ip_address)
        iata = airport["airport"]["iata"]
        await self.update_context(user_id, origin=iata)
        await self.set_state(user_id, FlightPlannerState.ASK_DESTINATION)
        return iata

    async def build_final_request(self, user_id: str) -> Dict[str, Any]:
        context = await self.get_context(user_id)
        return self.build_duffel_request(context)

    def build_duffel_request(self, context: FlightPlannerContext) -> Dict[str, Any]:
        if not context.origin:
            raise ValueError("Origin is missing")
        if not context.destination:
            raise ValueError("Destination is missing")
        if not context.departure_date:
            raise ValueError("Departure date is missing")

        context.ensure_default_passenger()

        slices: List[Dict[str, Any]] = [
            {
                "origin": context.origin,
                "destination": context.destination,
                "departure_date": context.departure_date,
            }
        ]

        if context.return_date:
            slices.append(
                {
                    "origin": context.destination,
                    "destination": context.origin,
                    "departure_date": context.return_date,
                }
            )

        if context.time_window:
            window = context.time_window.to_duffel()
            slices[0]["departure_time"] = window

        duffel_request = {
            "slices": slices,
            "passengers": [p.to_duffel() for p in context.passengers],
            "cabin_class": context.cabin_class or "economy",
            "max_connections": context.max_connections if context.max_connections is not None else 1,
            "supplier_timeout_ms": 0,
            "return_offers": True,
            "page_size": 50,
            "currency": "USD",
        }
        return duffel_request

    async def handle_turn(
        self,
        user_id: str,
        *,
        ip_address: Optional[str] = None,
        user_input: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Process user input (if any) and return the next question/response."""

        state = await self.get_state(user_id)
        if user_input:
            await self._apply_state_input(user_id, state, user_input)
            state = await self.get_state(user_id)

        prompt, suggestions, duffel_request = await self._build_prompt(
            user_id, state, ip_address
        )
        context = await self.get_context(user_id)
        return ConversationTurn(
            state=state,
            prompt=prompt,
            suggestions=suggestions,
            context=context,
            duffel_request=duffel_request,
        )

    async def _apply_state_input(
        self, user_id: str, state: FlightPlannerState, user_input: Dict[str, Any]
    ) -> None:
        """Mutate context/state based on the latest input."""

        if state == FlightPlannerState.DETECT_ORIGIN:
            await self._handle_detect_origin_input(user_id, user_input)
        elif state == FlightPlannerState.ASK_ORIGIN_OVERRIDE:
            await self._handle_origin_override(user_id, user_input)
        elif state == FlightPlannerState.ASK_DESTINATION:
            await self._handle_destination(user_id, user_input)
        elif state == FlightPlannerState.ASK_DEPARTURE_DATE:
            await self._handle_departure_date(user_id, user_input)
        elif state == FlightPlannerState.ASK_RETURN_DATE_OR_ONEWAY:
            await self._handle_trip_type(user_id, user_input)
        elif state == FlightPlannerState.ASK_RETURN_DATE:
            await self._handle_return_date(user_id, user_input)
        elif state == FlightPlannerState.ASK_CABIN:
            await self._handle_cabin(user_id, user_input)
        elif state == FlightPlannerState.ASK_PASSENGERS:
            await self._handle_passengers(user_id, user_input)
        elif state == FlightPlannerState.ASK_MAX_CONNECTIONS:
            await self._handle_connections(user_id, user_input)
        elif state == FlightPlannerState.ASK_TIME_WINDOW:
            await self._handle_time_window(user_id, user_input)
        elif state == FlightPlannerState.CONFIRMATION:
            await self._handle_confirmation(user_id, user_input)

    async def _handle_detect_origin_input(self, user_id: str, user_input: Dict[str, Any]) -> None:
        if user_input.get("confirm_origin"):
            await self.set_state(user_id, FlightPlannerState.ASK_DESTINATION)
            return
        if user_input.get("change_origin"):
            await self.set_state(user_id, FlightPlannerState.ASK_ORIGIN_OVERRIDE)
            return
        if "origin" in user_input:
            await self.update_context(user_id, origin=user_input["origin"])
            await self.set_state(user_id, FlightPlannerState.ASK_DESTINATION)
            return
        raise ValueError("A confirmation or a new origin is required")

    async def _handle_origin_override(self, user_id: str, user_input: Dict[str, Any]) -> None:
        origin = user_input.get("origin")
        if not origin:
            raise ValueError("Origin is required")
        await self.update_context(user_id, origin=origin)
        await self.set_state(user_id, FlightPlannerState.ASK_DESTINATION)

    async def _handle_destination(self, user_id: str, user_input: Dict[str, Any]) -> None:
        destination = user_input.get("destination")
        if not destination:
            raise ValueError("Destination is required")
        await self.update_context(user_id, destination=destination)
        await self.set_state(user_id, FlightPlannerState.ASK_DEPARTURE_DATE)

    async def _handle_departure_date(self, user_id: str, user_input: Dict[str, Any]) -> None:
        departure_date = user_input.get("departure_date")
        if not departure_date:
            raise ValueError("Departure date is required")
        await self.update_context(user_id, departure_date=departure_date)
        await self.set_state(user_id, FlightPlannerState.ASK_RETURN_DATE_OR_ONEWAY)

    async def _handle_trip_type(self, user_id: str, user_input: Dict[str, Any]) -> None:
        trip_type = user_input.get("trip_type")
        if trip_type is None and "round_trip" in user_input:
            trip_type = "round_trip" if user_input["round_trip"] else "one_way"
        if trip_type not in {"one_way", "round_trip"}:
            raise ValueError("Trip type must be 'one_way' or 'round_trip'")
        if trip_type == "one_way":
            await self.update_context(user_id, return_date=None)
            await self.set_state(user_id, FlightPlannerState.ASK_CABIN)
        else:
            await self.set_state(user_id, FlightPlannerState.ASK_RETURN_DATE)

    async def _handle_return_date(self, user_id: str, user_input: Dict[str, Any]) -> None:
        return_date = user_input.get("return_date")
        if not return_date:
            raise ValueError("Return date is required for round trips")
        await self.update_context(user_id, return_date=return_date)
        await self.set_state(user_id, FlightPlannerState.ASK_CABIN)

    async def _handle_cabin(self, user_id: str, user_input: Dict[str, Any]) -> None:
        cabin = user_input.get("cabin_class")
        if not cabin:
            raise ValueError("Cabin class is required")
        await self.update_context(user_id, cabin_class=cabin)
        await self.set_state(user_id, FlightPlannerState.ASK_PASSENGERS)

    async def _handle_passengers(self, user_id: str, user_input: Dict[str, Any]) -> None:
        passengers = user_input.get("passengers")
        if not passengers:
            raise ValueError("Passenger details are required")
        passenger_models = [p if isinstance(p, Passenger) else Passenger(**p) for p in passengers]
        await self.update_context(user_id, passengers=passenger_models)
        await self.set_state(user_id, FlightPlannerState.ASK_MAX_CONNECTIONS)

    async def _handle_connections(self, user_id: str, user_input: Dict[str, Any]) -> None:
        max_conn = user_input.get("max_connections")
        if max_conn is None:
            raise ValueError("Maximum connections is required")
        await self.update_context(user_id, max_connections=int(max_conn))
        await self.set_state(user_id, FlightPlannerState.ASK_TIME_WINDOW)

    async def _handle_time_window(self, user_id: str, user_input: Dict[str, Any]) -> None:
        if user_input.get("no_preference"):
            await self.update_context(user_id, time_window=None)
        else:
            window = user_input.get("time_window")
            if window is None:
                raise ValueError("Provide a time window or mark no preference")
            if isinstance(window, TimeWindow):
                time_window = window
            else:
                time_window = TimeWindow(**window)
            await self.update_context(user_id, time_window=time_window)
        await self.set_state(user_id, FlightPlannerState.CONFIRMATION)

    async def _handle_confirmation(self, user_id: str, user_input: Dict[str, Any]) -> None:
        if user_input.get("confirm"):
            await self.set_state(user_id, FlightPlannerState.FINALIZE_JSON)
            return
        edit_field = user_input.get("edit_field")
        if not edit_field:
            raise ValueError("Specify confirm=true or the field to edit")
        await self._rollback_to_field(user_id, edit_field)

    async def _rollback_to_field(self, user_id: str, field: str) -> None:
        mapping = {
            "origin": FlightPlannerState.ASK_ORIGIN_OVERRIDE,
            "destination": FlightPlannerState.ASK_DESTINATION,
            "departure_date": FlightPlannerState.ASK_DEPARTURE_DATE,
            "trip_type": FlightPlannerState.ASK_RETURN_DATE_OR_ONEWAY,
            "return_date": FlightPlannerState.ASK_RETURN_DATE,
            "cabin_class": FlightPlannerState.ASK_CABIN,
            "passengers": FlightPlannerState.ASK_PASSENGERS,
            "max_connections": FlightPlannerState.ASK_MAX_CONNECTIONS,
            "time_window": FlightPlannerState.ASK_TIME_WINDOW,
        }
        target_state = mapping.get(field)
        if not target_state:
            raise ValueError("Unsupported field edit request")
        await self.set_state(user_id, target_state)

    async def _build_prompt(
        self,
        user_id: str,
        state: FlightPlannerState,
        ip_address: Optional[str],
    ) -> Tuple[Optional[str], List[str], Optional[Dict[str, Any]]]:
        if state == FlightPlannerState.FINALIZE_JSON:
            context = await self.get_context(user_id)
            payload = self.build_duffel_request(context)
            return None, [], payload

        context = await self.get_context(user_id)
        builders = {
            FlightPlannerState.DETECT_ORIGIN: self._prompt_detect_origin,
            FlightPlannerState.ASK_ORIGIN_OVERRIDE: self._prompt_origin_override,
            FlightPlannerState.ASK_DESTINATION: self._prompt_destination,
            FlightPlannerState.ASK_DEPARTURE_DATE: self._prompt_departure_date,
            FlightPlannerState.ASK_RETURN_DATE_OR_ONEWAY: self._prompt_trip_type,
            FlightPlannerState.ASK_RETURN_DATE: self._prompt_return_date,
            FlightPlannerState.ASK_CABIN: self._prompt_cabin,
            FlightPlannerState.ASK_PASSENGERS: self._prompt_passengers,
            FlightPlannerState.ASK_MAX_CONNECTIONS: self._prompt_connections,
            FlightPlannerState.ASK_TIME_WINDOW: self._prompt_time_window,
            FlightPlannerState.CONFIRMATION: self._prompt_confirmation,
        }
        builder = builders.get(state)
        if not builder:
            raise ValueError(f"No prompt builder for {state}")
        return await builder(user_id, context, ip_address)

    async def _prompt_detect_origin(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        if not context.origin:
            if not ip_address:
                raise ValueError("IP address is required to detect origin")
            airport = await self.ip_service.get_nearest_airport_from_ip(ip_address)
            iata = airport["airport"]["iata"]
            await self.update_context(user_id, origin=iata)
            context = await self.get_context(user_id)
        prompt = (
            f"I detected {context.origin} as your departure airport. Would you like to use this or change it?"
        )
        suggestions = [f"Use {context.origin}", "Change origin"]
        return prompt, suggestions, None

    async def _prompt_origin_override(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "Which airport should we use for departure? Please provide a 3-letter IATA code."
        suggestions = ["LAX", "JFK", "LHR"]
        return prompt, suggestions, None

    async def _prompt_destination(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "Great! Where would you like to fly to?"
        suggestions = ["New York (JFK)", "London (LHR)", "Tokyo (HND)"]
        return prompt, suggestions, None

    async def _prompt_departure_date(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "When do you want to depart? Use YYYY-MM-DD."
        suggestions = self._date_suggestions([3, 10, 17])
        return prompt, suggestions, None

    async def _prompt_trip_type(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "Is this a round trip or one-way?"
        suggestions = ["Round trip", "One-way"]
        return prompt, suggestions, None

    async def _prompt_return_date(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "What is your return date?"
        suggestions = self._date_suggestions([10, 17, 24])
        return prompt, suggestions, None

    async def _prompt_cabin(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "Which cabin class do you prefer?"
        suggestions = ["Economy", "Premium Economy", "Business", "First"]
        return prompt, suggestions, None

    async def _prompt_passengers(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "How many passengers are traveling? Include ages for children/infants."
        suggestions = ["1 adult", "2 adults", "2 adults + 1 child (8)"]
        return prompt, suggestions, None

    async def _prompt_connections(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "What's the maximum number of connections you're comfortable with?"
        suggestions = ["0 (nonstop)", "1 connection", "2 connections"]
        return prompt, suggestions, None

    async def _prompt_time_window(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        prompt = "Do you have a preferred departure time window?"
        suggestions = [
            "Morning (06:00-12:00)",
            "Afternoon (12:00-17:00)",
            "Evening (17:00-22:00)",
            "No preference",
        ]
        return prompt, suggestions, None

    async def _prompt_confirmation(
        self,
        user_id: str,
        context: FlightPlannerContext,
        ip_address: Optional[str],
    ) -> Tuple[str, List[str], Optional[Dict[str, Any]]]:
        summary = f"Review & confirm: {self._build_summary(context)}"
        suggestions = ["Looks good", "Change origin", "Change destination", "Change dates"]
        return summary, suggestions, None

    def _build_summary(self, context: FlightPlannerContext) -> str:
        origin = context.origin or "TBD"
        destination = context.destination or "TBD"
        parts = [f"{origin} → {destination}"]
        if context.departure_date:
            if context.return_date:
                date_part = f"Depart {context.departure_date}, return {context.return_date}"
            else:
                date_part = f"Depart {context.departure_date} (one-way)"
        else:
            date_part = "Departure date pending"
        parts.append(date_part)
        cabin = context.cabin_class or "economy"
        parts.append(f"Cabin: {cabin}")
        if context.passengers:
            pax_summary = ", ".join(p.type for p in context.passengers)
        else:
            pax_summary = "adult"
        parts.append(f"Passengers: {pax_summary}")
        max_conn = context.max_connections if context.max_connections is not None else 1
        parts.append(f"Up to {max_conn} connections")
        if context.time_window:
            parts.append(
                f"Depart between {context.time_window.start}-{context.time_window.end}"
            )
        return " | ".join(parts)

    def _date_suggestions(self, offsets: List[int]) -> List[str]:
        base = self._now().date()
        return [str(base + timedelta(days=offset)) for offset in offsets]


__all__ = [
    "FlightPlannerState",
    "FlightPlannerContext",
    "FlightPlannerStateMachine",
    "Passenger",
    "ConversationTurn",
]
