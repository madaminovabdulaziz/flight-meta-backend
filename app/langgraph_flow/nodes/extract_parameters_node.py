"""
Extract Parameters Node - Production-Ready LLM Fallback
=======================================================
LLM-based parameter extraction for low-confidence queries.
Enhanced with full conversation context support for multi-turn dialogs.
"""

import logging
from datetime import date
from typing import Any, Dict, Optional, List

from app.langgraph_flow.state import ConversationState, update_state

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURATION
# ==========================================

PARAMETER_FIELDS = [
    "destination",
    "origin",
    "departure_date",
    "return_date",
    "passengers",
    "travel_class",
    "budget",
    "flexibility",
]

SYSTEM_PROMPT = """Extract flight parameters from the conversation.

You are analyzing a conversation history where the user is booking a flight.
Extract any flight parameters mentioned in the conversation.

Return JSON with these keys (use null if not present):
{
  "destination": string|null,
  "origin": string|null,
  "departure_date": "YYYY-MM-DD"|null,
  "return_date": "YYYY-MM-DD"|null,
  "passengers": int|null,
  "travel_class": "Economy"|"Premium Economy"|"Business"|"First"|null,
  "budget": float|null,
  "flexibility": int|null
}

IMPORTANT:
- Consider the FULL conversation context, not just the latest message
- Pay attention to which parameter the assistant is asking for
- Extract relative dates like "next week", "tomorrow", "next Monday" as specific dates
- If the user says "To Paris", that's the destination
- If the user says "From London", that's the origin
- If the user says just a date/time, that's likely the departure_date
- Do not guess. Use null when unsure."""


# ==========================================
# UTILITIES
# ==========================================

def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        logger.warning(f"[ExtractParams] Invalid date format: {value}")
        return None


def _normalize_airport_code(code: Optional[str]) -> List[str]:
    if not code or not isinstance(code, str):
        return []
    code = code.strip().upper()
    if len(code) == 3 and code.isalpha():
        return [code]
    return []


def _build_conversation_context(state: ConversationState) -> str:
    """
    Build rich conversation context from history.

    This is critical for multi-turn conversations where parameters
    are provided incrementally across multiple messages.
    """
    history = getattr(state, "conversation_history", None)

    if not history or len(history) == 0:
        # Fallback to latest message
        return getattr(state, "latest_user_message", "")

    # Take last 6 messages (3 turns = 3 user + 3 assistant)
    recent_history = history[-6:] if len(history) > 6 else history

    # Build context string with role labels
    context_parts = []
    for msg in recent_history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            context_parts.append(f"{role}: {content}")

    context = "\n".join(context_parts)

    logger.debug(f"[ExtractParams] Built context from {len(recent_history)} messages")

    return context


def _merge_extracted_params(
    state: ConversationState,
    extracted: Dict[str, Any],
    is_change_of_plan: bool = False
) -> Dict[str, Any]:
    """
    Merge extracted parameters with existing state.

    Args:
        state: Current conversation state
        extracted: Newly extracted parameters
        is_change_of_plan: If True, overwrite existing values

    Returns:
        Dictionary of updates to apply
    """
    updates = {}

    for key, value in extracted.items():
        if value is None or value == "":
            continue

        if is_change_of_plan:
            updates[key] = value
            logger.info(f"[ExtractParams] Overwriting {key}: {value} (change_of_plan)")

        else:
            existing = getattr(state, key, None)
            if existing in (None, "", 0):
                updates[key] = value
                logger.debug(f"[ExtractParams] Setting {key}: {value}")

    return updates


def _add_derived_fields(state: ConversationState, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Add derived fields like airport lists and is_round_trip."""
    destination = updates.get("destination", getattr(state, "destination", None))
    origin = updates.get("origin", getattr(state, "origin", None))
    return_date = updates.get("return_date", getattr(state, "return_date", None))

    if destination:
        dest_airports = _normalize_airport_code(destination)
        if dest_airports:
            updates["destination_airports"] = dest_airports

    if origin:
        orig_airports = _normalize_airport_code(origin)
        if orig_airports:
            updates["origin_airports"] = orig_airports

    updates["is_round_trip"] = bool(return_date)

    return updates


# ==========================================
# MAIN NODE
# ==========================================

async def extract_parameters_node(state: ConversationState) -> ConversationState:
    """
    Extract parameters using LLM with full conversation context.

    This node is only called when the rule-based path has low confidence.
    It uses the LLM to extract parameters from the conversation history,
    providing much better multi-turn support than single-message extraction.
    """

    # Guard rail: skip if rule-based path
    if getattr(state, "use_rule_based_path", False):
        logger.info("[ExtractParams] Rule-based path active → skipping LLM extraction")
        return state

    # Build conversation context (with history!)
    context = _build_conversation_context(state)

    if not context or not context.strip():
        logger.warning("[ExtractParams] Empty context, skipping extraction")
        return state

    logger.info(f"[ExtractParams] Extracting from conversation context ({len(context)} chars)")

    # LLM extraction
    try:
        from services.llm_service import generate_json_response

        response = await generate_json_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=context,  # Use full context, not just latest message
        )

        logger.debug(f"[ExtractParams] Raw LLM response: {response}")

    except Exception as e:
        logger.error(f"[ExtractParams] LLM extraction failed: {e}", exc_info=True)
        return state

    # Parse fields
    extracted_raw = {}

    for field in PARAMETER_FIELDS:
        value = response.get(field)

        if value is not None:

            if field in ("departure_date", "return_date"):
                value = _parse_date(value)

            elif field == "passengers":
                try:
                    value = int(value) if value else None
                except:
                    logger.warning(f"[ExtractParams] Invalid passengers: {value}")
                    value = None

            elif field == "budget":
                try:
                    value = float(value) if value else None
                except:
                    logger.warning(f"[ExtractParams] Invalid budget: {value}")
                    value = None

            elif field == "flexibility":
                try:
                    value = int(value) if value else None
                except:
                    logger.warning(f"[ExtractParams] Invalid flexibility: {value}")
                    value = None

            extracted_raw[field] = value

    logger.info(f"[ExtractParams] Extracted fields: {[k for k,v in extracted_raw.items() if v is not None]}")

    # Merge with state
    intent = getattr(state, "intent", "")
    is_change_of_plan = (intent == "change_of_plan")

    updates = _merge_extracted_params(
        state=state,
        extracted=extracted_raw,
        is_change_of_plan=is_change_of_plan
    )

    updates = _add_derived_fields(state, updates)

    logger.info(f"[ExtractParams] Applying updates: {list(updates.keys())}")

    return update_state(state, updates)


# ==========================================
# TESTING UTILITIES
# ==========================================

async def test_extraction(message: str) -> Dict[str, Any]:
    """Test extraction on a single message."""
    import time
    from services.llm_service import generate_json_response

    start = time.time()

    try:
        response = await generate_json_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=message,
        )

        extracted = {field: response.get(field) for field in PARAMETER_FIELDS}

        return {
            "message": message,
            "extracted": extracted,
            "elapsed_ms": int((time.time() - start) * 1000),
            "raw_response": response,
        }

    except Exception as e:
        return {
            "message": message,
            "extracted": {},
            "elapsed_ms": int((time.time() - start) * 1000),
            "error": str(e),
        }


def get_extraction_coverage(state: ConversationState) -> Dict[str, Any]:
    """Get extraction coverage statistics."""
    filled = []
    missing = []

    for field in PARAMETER_FIELDS:
        value = getattr(state, field, None)
        if value not in (None, "", 0):
            filled.append(field)
        else:
            missing.append(field)

    return {
        "filled_params": filled,
        "missing_params": missing,
        "coverage_percent": len(filled) / len(PARAMETER_FIELDS) * 100,
    }


def should_skip_llm_extraction(state: ConversationState) -> bool:
    """Determine if LLM extraction should be skipped."""
    if getattr(state, "use_rule_based_path", False):
        return True

    context = _build_conversation_context(state)
    if not context.strip():
        return True

    critical = ["destination", "origin", "departure_date"]
    all_filled = all(getattr(state, c, None) for c in critical)

    if all_filled and getattr(state, "intent", "") != "change_of_plan":
        return True

    return False
