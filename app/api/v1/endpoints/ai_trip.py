"""FastAPI endpoint exposing the Flight Planner state machine."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from services.flight_planner import (
    ConversationTurn,
    FlightPlannerContext,
    FlightPlannerState,
    FlightPlannerStateMachine,
)

router = APIRouter(prefix="/ai-trip", tags=["ai-trip"])
state_machine = FlightPlannerStateMachine()


class ConversationTurnRequest(BaseModel):
    user_id: str = Field(..., description="Stable user/session identifier")
    message: Optional[str] = Field(
        default=None,
        description="JSON string payload for the current state input",
    )


class ConversationTurnResponse(BaseModel):
    state: FlightPlannerState
    prompt: Optional[str]
    suggestions: List[str]
    context: FlightPlannerContext
    duffel_request: Optional[Dict[str, Any]] = None

    @classmethod
    def from_turn(cls, turn: ConversationTurn) -> "ConversationTurnResponse":
        return cls(
            state=turn.state,
            prompt=turn.prompt,
            suggestions=turn.suggestions,
            context=turn.context,
            duffel_request=turn.duffel_request,
        )


def _parse_message(message: Optional[str]) -> Optional[Dict[str, Any]]:
    if not message:
        return None
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="message must be valid JSON") from exc
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="message JSON must be an object")
    return parsed


@router.post("/turn", response_model=ConversationTurnResponse)
async def ai_trip_turn(payload: ConversationTurnRequest, request: Request) -> ConversationTurnResponse:
    """Run one conversational turn through the planner state machine."""

    user_input = _parse_message(payload.message)
    client_ip = request.client.host if request.client else None
    turn = await state_machine.handle_turn(
        payload.user_id,
        ip_address=client_ip,
        user_input=user_input,
    )
    return ConversationTurnResponse.from_turn(turn)
