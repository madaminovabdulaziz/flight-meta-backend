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
    Otherwise, we rely on previous question / error.
    """

    ranked = state.get("ranked_flights") or []
    assistant_message = state.get("assistant_message")
    suggestions = state.get("suggestions") or []

    if ranked:
        # Build a simple, expert-style summary
        lines = ["Here are the top flight options I found for you:\n"]

        for i, f in enumerate(ranked[:3], start=1):
            brief = _format_flight_brief(f)
            lines.append(f"{i}. {brief}")

        lines.append(
            "\nThese are selected based on a balance of price, duration, layovers, airline quality, "
            "airport convenience, and your preferences."
        )
        lines.append("You can ask to adjust dates, origin, budget, or see more options.")

        assistant_message = "\n".join(lines)
        suggestions = [
            "Show more options",
            "Change dates",
            "Change origin",
            "Filter by airline",
        ]

        is_complete = True
    else:
        # No results or error scenario
        if not assistant_message:
            assistant_message = (
                "I couldn't find suitable flights with the current parameters. "
                "You can try adjusting your dates, origin, or budget."
            )
            suggestions = [
                "Change dates",
                "Change budget",
                "Change origin",
            ]
        is_complete = False

    # Persist state (short-term) to Redis
    try:
        await save_session_state_to_redis(state["session_id"], dict(state))
    except Exception as e:
        logger.error(f"[FinalizeResponseNode] Failed to save session state: {e}", exc_info=True)

    return update_state(state, {
        "assistant_message": assistant_message,
        "suggestions": suggestions,
        "is_complete": is_complete,
        "updated_at": datetime.utcnow(),
    })
