# langgraph_flow/nodes/refusal_node.py

import logging
from app.langgraph_flow.state import ConversationState, update_state, validate_required_slots

logger = logging.getLogger(__name__)


async def refusal_node(state: ConversationState) -> ConversationState:
    """
    Politely refuse non-travel or chitchat intents and steer user back to trip planning.
    """

    intent = state.get("intent", "irrelevant")
    latest = state.get("latest_user_message", "")

    logger.info(f"[RefusalNode] Handling non-travel intent '{intent}'")

    # Decide what to ask next based on required slots
    missing = validate_required_slots(state) or "destination"

    if missing == "destination":
        assistant_message = (
            "I'm here to help you plan flights. Let's start with your destination — "
            "where would you like to fly to?"
        )
        suggestions = ["Istanbul", "Dubai", "London", "Paris"]
        placeholder = "Where are you flying to?"
    elif missing == "departure_date":
        assistant_message = (
            "I'm focused on trip planning. When would you like to depart?"
        )
        suggestions = ["This month", "Next month", "Give exact dates"]
        placeholder = "When do you want to fly?"
    else:  # origin
        assistant_message = (
            "Let’s get back to your trip. Which city or airport are you flying from?"
        )
        suggestions = ["London", "New York", "Dubai"]
        placeholder = "Departure city or airport"

    return update_state(state, {
        "assistant_message": assistant_message,
        "next_placeholder": placeholder,
        "suggestions": suggestions,
        "missing_parameter": missing,
    })
