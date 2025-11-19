import logging
from typing import Dict, Any, List, Optional

from app.langgraph_flow.state import ConversationState, update_state

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Generate a short, friendly question to get missing travel information.

The question should be:
- Clear and specific
- Natural and conversational
- Professional but warm

Return JSON:
{
  "assistant_message": "the question to ask",
  "next_placeholder": "short input placeholder",
  "suggestions": ["option1", "option2", "option3"]
}

Keep everything concise."""


FALLBACK_QUESTIONS = {
    "destination": {
        "assistant_message": "Where would you like to fly to?",
        "next_placeholder": "Enter destination",
        "suggestions": ["Dubai", "Istanbul", "London", "Paris"],
    },
    "origin": {
        "assistant_message": "Where are you flying from?",
        "next_placeholder": "Enter departure city",
        "suggestions": ["Tashkent", "Samarkand", "Bukhara"],
    },
    "departure_date": {
        "assistant_message": "When would you like to depart?",
        "next_placeholder": "Select or enter date",
        "suggestions": ["Tomorrow", "This Weekend", "Next Week"],
    },
    "return_date": {
        "assistant_message": "When would you like to return?",
        "next_placeholder": "Select return date",
        "suggestions": ["Same Day", "Next Day", "One Week Later"],
    },
}

# ============================================================
# CONTEXT BUILDER — DATACLASS SAFE
# ============================================================

def _build_minimal_context(missing_parameter: str, state: ConversationState) -> str:
    context_parts = [f"Ask about: {missing_parameter}"]

    if missing_parameter == "origin" and state.destination:
        context_parts.append(f"Destination: {state.destination}")

    elif missing_parameter == "departure_date":
        if state.destination:
            context_parts.append(f"Going to: {state.destination}")
        if state.origin:
            context_parts.append(f"From: {state.origin}")

    elif missing_parameter == "return_date" and state.departure_date:
        context_parts.append(f"Departing: {state.departure_date}")

    return "\n".join(context_parts)


def _get_fallback_question(missing_parameter: str) -> Dict[str, Any]:
    return FALLBACK_QUESTIONS.get(
        missing_parameter,
        {
            "assistant_message": f"Could you provide your {missing_parameter}?",
            "next_placeholder": f"Enter {missing_parameter}",
            "suggestions": [],
        }
    )

# ============================================================
# MAIN NODE — DATACLASS SAFE
# ============================================================

async def ask_next_question_node(state: ConversationState) -> ConversationState:

    # Access safely (field does not exist in dataclass)
    use_rule_based_path = getattr(state, "use_rule_based_path", False)

    if use_rule_based_path:
        logger.info("[AskQuestion] Rule-based path → skipping LLM question generation")
        return state

    missing = state.missing_parameter

    if not missing:
        logger.warning("[AskQuestion] missing_parameter not set, defaulting to 'destination'")
        missing = "destination"

    logger.info(f"[AskQuestion] Generating question for missing: '{missing}'")

    # TRY LLM
    try:
        from services.llm_service import generate_json_response

        context = _build_minimal_context(missing, state)
        logger.debug(f"[AskQuestion] Context: {context}")

        response = await generate_json_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=context,
        )

        assistant_message = response.get("assistant_message")
        next_placeholder = response.get("next_placeholder")
        suggestions = response.get("suggestions", [])

        if not assistant_message or not isinstance(assistant_message, str):
            raise ValueError("Invalid assistant_message from LLM")

        logger.info(f"[AskQuestion] ✅ Generated: '{assistant_message[:50]}...'")

    except Exception as e:
        logger.warning(f"[AskQuestion] LLM failed: {e} → using fallback template")
        fallback = _get_fallback_question(missing)
        assistant_message = fallback["assistant_message"]
        next_placeholder = fallback["next_placeholder"]
        suggestions = fallback["suggestions"]

    # SANITIZE OUTPUT
    if not next_placeholder or len(next_placeholder) > 100:
        next_placeholder = f"Enter {missing}"

    if not isinstance(suggestions, list):
        suggestions = []

    suggestions = suggestions[:5]

    logger.info(
        f"[AskQuestion] Question ready: message={len(assistant_message)} chars, suggestions={len(suggestions)}"
    )

    return update_state(state, {
        "assistant_message": assistant_message,
        "next_placeholder": next_placeholder,
        "suggestions": suggestions,
        "last_question": assistant_message,
    })

# ============================================================
# UTILITIES — DATACLASS SAFE
# ============================================================

def should_skip_llm_question(state: ConversationState) -> bool:
    return getattr(state, "use_rule_based_path", False) or not state.missing_parameter


def generate_question_from_template(missing_parameter: str, state: Optional[ConversationState] = None) -> Dict[str, Any]:
    base = _get_fallback_question(missing_parameter)

    if state:
        if missing_parameter == "destination" and state.origin:
            base["assistant_message"] = f"Where would you like to fly from {state.origin}?"

        if missing_parameter == "departure_date" and state.origin and state.destination:
            base["assistant_message"] = (
                f"When would you like to fly from {state.origin} to {state.destination}?"
            )

    return base


def get_question_metrics(state: ConversationState) -> Dict[str, Any]:
    return {
        "missing_parameter": state.missing_parameter,
        "has_question": bool(state.assistant_message),
        "has_suggestions": bool(state.suggestions),
        "suggestion_count": len(state.suggestions),
        "used_llm": not getattr(state, "use_rule_based_path", False),
    }
