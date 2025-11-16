# langgraph_flow/nodes/classify_intent_node.py

import logging
from typing import Literal

from app.langgraph_flow.state import ConversationState, update_state
from services.llm_service import generate_json_response

logger = logging.getLogger(__name__)


INTENT_LABELS = [
    "travel_query",
    "destination_provided",
    "origin_provided",
    "date_provided",
    "budget_provided",
    "preference_provided",
    "change_of_plan",
    "irrelevant",
    "chitchat",
]


async def classify_intent_node(state: ConversationState) -> ConversationState:
    """
    Use Gemini to classify the user's latest message into one of the allowed intents.
    """

    latest = state.get("latest_user_message", "")
    history = state.get("conversation_history", [])

    system_prompt = """
You are an intent classifier for an AI Trip Planner. 
Classify the user's LAST message into ONE of these labels:

- travel_query          → General travel question or plan
- destination_provided  → User gives destination or mentions where they want to go
- origin_provided       → User specifies origin airport/city
- date_provided         → User gives specific or fuzzy dates
- budget_provided       → User mentions budget or price constraints
- preference_provided   → User states some preference (airline, airport, class, time)
- change_of_plan        → User changes previously given info (origin, dates, etc.)
- irrelevant            → Not about travel at all
- chitchat              → Casual chat, greetings, small talk (but travel-related or neutral)

Rules:
- Only output one label.
- If unsure but it's still about travel, use "travel_query".
- If the message clearly changes something previously stated (dates, origin, destination), use "change_of_plan".
- If non-travel, use "irrelevant".

Return STRICT JSON:
{"intent": "<one_of_labels>"}
"""

    conv_snippet = ""
    if history:
        # Include last 3 messages for extra context
        tail = history[-3:]
        parts = [f"{m['role']}: {m['content']}" for m in tail]
        conv_snippet = "\n".join(parts)

    user_prompt = f"""
Conversation (last few turns):
{conv_snippet}

Latest user message:
{latest}
"""

    resp = await generate_json_response(system_prompt, user_prompt)
    intent = resp.get("intent")

    if intent not in INTENT_LABELS:
        logger.warning(f"[ClassifyIntentNode] Invalid or missing intent '{intent}', defaulting to 'travel_query'")
        intent = "travel_query"

    logger.info(f"[ClassifyIntentNode] Classified intent: {intent}")

    return update_state(state, {
        "intent": intent,
    })
