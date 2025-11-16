# langgraph_flow/nodes/ask_next_question_node.py

import logging
from app.langgraph_flow.state import ConversationState, update_state
from services.llm_service import generate_json_response

logger = logging.getLogger(__name__)


async def ask_next_question_node(state: ConversationState) -> ConversationState:
    """
    Generate the next question to ask the user, based on the missing_parameter and context.
    Uses Gemini to produce:
      - next_placeholder
      - suggestions (buttons)
      - assistant_message
    """

    missing = state.get("missing_parameter")
    if not missing:
        logger.warning("[AskNextQuestionNode] No missing_parameter set; falling back to generic question")
        missing = "destination"

    system_prompt = """
You are a premium AI travel assistant.

Your job:
- Ask ONE clear, friendly follow-up question to get the missing information.
- Adapt the question to the missing slot:
  - destination        → where they are flying to
  - origin             → where they are flying from
  - departure_date     → when they want to depart
  - return_date        → when they want to return
  - passengers         → how many people are flying
  - travel_class       → Economy / Premium Economy / Business / First
  - budget             → max budget for the trip
  - flexibility        → how flexible they are in days (±N)

You must return STRICT JSON with:
{
  "next_placeholder": "short placeholder text for the input box",
  "suggestions": ["short button 1", "short button 2", "..."],
  "assistant_message": "natural language message displayed in chat"
}

Rules:
- assistant_message should be short, friendly, and expert-level.
- suggestions should be 2–5 items, tailored to the missing parameter when possible.
- Do NOT ask about anything other than the missing parameter.
"""

    user_context = f"""
Missing parameter: {missing}

Current known data:
- origin: {state.get("origin")}
- destination: {state.get("destination")}
- departure_date: {state.get("departure_date")}
- return_date: {state.get("return_date")}
- passengers: {state.get("passengers")}
- travel_class: {state.get("travel_class")}
- budget: {state.get("budget")}
- flexibility: {state.get("flexibility")}

Long-term preferences (may help tone/suggestions):
{state.get("long_term_preferences", {})}
"""

    resp = await generate_json_response(system_prompt, user_context)

    next_placeholder = resp.get("next_placeholder") or "Type your answer..."
    suggestions = resp.get("suggestions") or []
    assistant_message = resp.get("assistant_message") or "Could you share this detail with me?"

    logger.info(f"[AskNextQuestionNode] Asking about '{missing}'")

    return update_state(state, {
        "next_placeholder": next_placeholder,
        "suggestions": suggestions,
        "assistant_message": assistant_message,
        "last_question": assistant_message,
    })
