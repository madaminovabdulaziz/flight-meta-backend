# langgraph_flow/nodes/extract_parameters_node.py

import logging
from datetime import date
from typing import Any, Dict, Optional, List

from app.langgraph_flow.state import ConversationState, update_state
from services.llm_service import generate_json_response

logger = logging.getLogger(__name__)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        # Expecting YYYY-MM-DD
        return date.fromisoformat(value)
    except Exception:
        return None


def _maybe_airport_list(code: Optional[str]) -> List[str]:
    if code and isinstance(code, str) and len(code) == 3 and code.isalpha():
        return [code.upper()]
    return []


async def extract_parameters_node(state: ConversationState) -> ConversationState:
    """
    Extract structured flight parameters from the user's latest message using Gemini.
    Handles new info and change_of_plan updates.
    """

    latest = state.get("latest_user_message", "")
    history = state.get("conversation_history", [])
    long_term_prefs = state.get("long_term_preferences", {})
    intent = state.get("intent", "travel_query")

    system_prompt = """
You are a travel parameter extractor for a flight-only trip planner.

Your job:
- Read the latest user message (with a bit of context).
- Extract any of the following fields, IF they are present or can be reasonably inferred:
  - destination: city or airport code (e.g., "IST" or "Istanbul")
  - origin: city or airport code (e.g., "LGW" or "London")
  - departure_date: departure date in YYYY-MM-DD, or null
  - return_date: return date in YYYY-MM-DD, or null (one-way if null)
  - passengers: integer >= 1
  - travel_class: Economy, Premium Economy, Business, First
  - budget: numeric maximum budget in user currency (just number, no symbol)
  - flexibility: integer days of flexibility (e.g., 2 means ±2 days)

If the user is changing a previous choice (change_of_plan intent), the new value should override the old one.

If something is NOT clearly present, set it to null, do NOT guess aggressively.

Return STRICT JSON with all keys:
{
  "destination": string | null,
  "origin": string | null,
  "departure_date": "YYYY-MM-DD" | null,
  "return_date": "YYYY-MM-DD" | null,
  "passengers": int | null,
  "travel_class": string | null,
  "budget": float | null,
  "flexibility": int | null
}
"""

    tail = history[-4:]
    conv_snippet = "\n".join([f"{m['role']}: {m['content']}" for m in tail])

    user_prompt = f"""
Conversation so far (last few turns):
{conv_snippet}

Latest user message:
{latest}

Long-term preferences (if any, may help infer defaults but do NOT override explicit user input):
{long_term_prefs}

Current intent: {intent}
"""

    resp = await generate_json_response(system_prompt, user_prompt)

    def get_field(name: str) -> Any:
        return resp.get(name)

    # Parsed values
    new_destination = get_field("destination")
    new_origin = get_field("origin")
    new_dep_date = _parse_date(get_field("departure_date"))
    new_ret_date = _parse_date(get_field("return_date"))
    new_passengers = get_field("passengers")
    new_travel_class = get_field("travel_class")
    new_budget = get_field("budget")
    new_flex = get_field("flexibility")

    updated: Dict[str, Any] = {}
    change_of_plan = (intent == "change_of_plan")

    def maybe_update(key: str, new_val: Any):
        if new_val is None or new_val == "":
            return
        if change_of_plan:
            # Overwrite old value
            updated[key] = new_val
        else:
            # Only set if not already present in state
            if state.get(key) in (None, "", 0):
                updated[key] = new_val

    maybe_update("destination", new_destination)
    maybe_update("origin", new_origin)
    maybe_update("departure_date", new_dep_date)
    maybe_update("return_date", new_ret_date)
    maybe_update("passengers", new_passengers)
    maybe_update("travel_class", new_travel_class)
    maybe_update("budget", new_budget)
    maybe_update("flexibility", new_flex)

    # Derived fields: airports + round-trip flag
    dest = updated.get("destination", state.get("destination"))
    orig = updated.get("origin", state.get("origin"))
    ret = updated.get("return_date", state.get("return_date"))

    dest_airports = state.get("destination_airports", [])
    orig_airports = state.get("origin_airports", [])

    if dest and not dest_airports:
        dest_airports = _maybe_airport_list(dest)

    if orig and not orig_airports:
        orig_airports = _maybe_airport_list(orig)

    updated["destination_airports"] = dest_airports
    updated["origin_airports"] = orig_airports
    updated["is_round_trip"] = bool(ret)

    logger.info(f"[ExtractParametersNode] Updated slots: {list(updated.keys())}")

    return update_state(state, updated)
