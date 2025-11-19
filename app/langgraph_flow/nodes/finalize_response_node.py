# langgraph_flow/nodes/finalize_response_node.py

import logging
from datetime import datetime
from typing import List, Dict, Any

from app.langgraph_flow.state import ConversationState, update_state
from services.session_store import save_session_state_to_redis

logger = logging.getLogger(__name__)


def _format_flight_brief(f: Dict[str, Any]) -> str:
    raw = f.get("raw_data", {})
    airline = raw.get("airline_name") or raw.get("airline") or "Unknown airline"
    origin = raw.get("origin")
    dest = raw.get("destination")
    price = raw.get("price")
    currency = raw.get("currency", "")
    stops = raw.get("stops", 0)
    dep = raw.get("departure_time")
    arr = raw.get("arrival_time")

    stop_text = "direct" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"

    return f"{airline} {origin} → {dest}, {stop_text}, {price} {currency}, dep: {dep}, arr: {arr}"


async def finalize_response_node(state: ConversationState) -> ConversationState:
    """
    Prepare final assistant_message, suggestions, and save state to Redis.
    If ranked_flights is non-empty, we present top options.
    """

    ranked = getattr(state, "ranked_flights", []) or []
    assistant_message = getattr(state, "assistant_message", None)
    suggestions = getattr(state, "suggestions", []) or []

    # =====================================================
    # CASE 1 — We have flight results
    # =====================================================
    if ranked:
        lines = ["Here are the top flight options I found for you:\n"]

        for i, f in enumerate(ranked[:3], start=1):
            lines.append(f"{i}. {_format_flight_brief(f)}")

        lines.append(
            "\nThese were selected based on price, duration, layovers, airline quality, "
            "airport convenience, and your preferences."
        )
        lines.append("You can ask to adjust dates, origin, or see more options.")

        assistant_message = "\n".join(lines)
        suggestions = [
            "Show more options",
            "Change dates",
            "Change origin",
            "Filter by airline",
        ]
        is_complete = True

    # =====================================================
    # CASE 2 — No flights found or user isn't ready for search
    # =====================================================
    else:
        if not assistant_message:
            assistant_message = (
                "I couldn't find suitable flights with the current parameters. "
                "Try adjusting dates, origin, or budget."
            )
            suggestions = ["Change dates", "Change budget", "Change origin"]

        is_complete = False

    # =====================================================
    # SAVE TO REDIS
    # =====================================================
    try:
        await save_session_state_to_redis(
            state.session_id,
            state.to_dict()   # <— correct serialization for dataclass
        )
    except Exception as e:
        logger.error(f"[FinalizeResponseNode] Failed to save session state: {e}", exc_info=True)

    # =====================================================
    # UPDATE STATE
    # =====================================================
    return update_state(state, {
        "assistant_message": assistant_message,
        "suggestions": suggestions,
        "is_complete": is_complete,
        "updated_at": datetime.utcnow(),
    })
