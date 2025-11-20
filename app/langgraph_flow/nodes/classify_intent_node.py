"""
Classify Intent Node - Production-Ready LLM Fallback
====================================================
LLM-based intent classification for low-confidence queries.
"""

import logging
from typing import Dict, Any

from app.langgraph_flow.state import ConversationState
from services.llm_service import generate_json_response

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

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

DEFAULT_INTENT = "travel_query"

SYSTEM_PROMPT = """
You are an intent classifier for a travel chatbot.
Allowed Intents:
- travel_query
- destination_provided
- origin_provided
- date_provided
- budget_provided
- preference_provided
- change_of_plan
- irrelevant
- chitchat

INSTRUCTIONS:
1. Analyze the user message.
2. Select EXACTLY ONE intent from the list above.
3. If multiple apply, pick the most dominant one (e.g. "To Paris tomorrow" -> destination_provided).
4. Return JSON: {"intent": "your_label"}
"""

# ==========================================
# MAIN NODE – DATACLASS SAFE VERSION
# ==========================================

async def classify_intent_node(state: ConversationState) -> ConversationState:
    """
    LLM-based fallback classifier.
    This function NEVER overwrites:
    - use_rule_based_path
    - routing_confidence
    - confidence_breakdown
    """

    # ----------------------------------------
    # 1. Skip if rule-based path is active
    # ----------------------------------------
    if getattr(state, "use_rule_based_path", False):
        logger.info("[ClassifyIntent] Rule-based path → SKIP LLM classifier")
        return state

    latest_message = state.latest_user_message or ""

    if not latest_message.strip():
        state.intent = DEFAULT_INTENT
        state.llm_intent = DEFAULT_INTENT
        logger.warning("[ClassifyIntent] Empty message → fallback intent")
        return state

    logger.info(f"[ClassifyIntent] Classifying: '{latest_message[:80]}...'")

    # ----------------------------------------
    # 2. LLM classification attempt
    # ----------------------------------------
    try:
        response = await generate_json_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=latest_message,
        )

        intent = response.get("intent")

        if not intent or intent not in INTENT_LABELS:
            logger.warning(f"[ClassifyIntent] Invalid intent '{intent}', using fallback")
            intent = DEFAULT_INTENT

    except Exception as e:
        logger.error(f"[ClassifyIntent] LLM classification failed: {e}")
        intent = DEFAULT_INTENT

    # ----------------------------------------
    # 3. Apply ONLY the intent
    # ----------------------------------------
    state.intent = intent
    state.llm_intent = intent  # For debugging

    logger.info(f"[ClassifyIntent] Final intent → {intent}")

    # NOTE: DO NOT CALL update_state() — it would overwrite routing fields
    return state


# ==========================================
# UTILITIES — DATACLASS SAFE
# ==========================================

def get_intent_description(intent: str) -> str:
    return {
        "travel_query": "General travel question or plan",
        "destination_provided": "User specified destination",
        "origin_provided": "User specified origin",
        "date_provided": "User provided travel dates",
        "budget_provided": "User mentioned budget constraints",
        "preference_provided": "User stated preferences",
        "change_of_plan": "User changed previous information",
        "irrelevant": "Non-travel content",
        "chitchat": "Casual conversation",
    }.get(intent, "Unknown intent")


def validate_intent(intent: str) -> bool:
    return intent in INTENT_LABELS


def should_skip_llm_classification(state: ConversationState) -> bool:
    # Skip if rule-based path
    if getattr(state, "use_rule_based_path", False):
        return True

    # Skip if intent already classified w/ high confidence
    if state.intent and getattr(state, "routing_confidence", 0) >= 0.90:
        return True

    # Skip empty message
    if not (state.latest_user_message or "").strip():
        return True

    return False


def get_classification_metrics(state: ConversationState) -> Dict[str, Any]:
    return {
        "intent": getattr(state, "intent", None),
        "llm_intent": getattr(state, "llm_intent", None),
        "used_llm_classification": getattr(state, "llm_intent", None) is not None,
        "routing_confidence": getattr(state, "routing_confidence", None),
        "path": "llm" if not getattr(state, "use_rule_based_path", False) else "rule_based",
    }

# ==========================================
# BATCH CLASSIFIER — DATACLASS SAFE
# ==========================================

async def classify_intents_batch(messages: list[str]) -> list[str]:
    if not messages:
        return []

    batch_prompt = "Classify each message:\n"
    for i, msg in enumerate(messages):
        batch_prompt += f"{i+1}. {msg}\n"

    batch_prompt += "\nReturn JSON: {\"intents\": [\"<label1>\", ...]}"

    try:
        response = await generate_json_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=batch_prompt,
        )

        raw = response.get("intents", [])
        validated = []

        for intent in raw:
            validated.append(intent if intent in INTENT_LABELS else DEFAULT_INTENT)

        while len(validated) < len(messages):
            validated.append(DEFAULT_INTENT)

        return validated[:len(messages)]

    except Exception as e:
        logger.error(f"[ClassifyIntentBatch] Failed: {e}")
        return [DEFAULT_INTENT] * len(messages)


# ==========================================
# TESTING
# ==========================================

async def test_classification(message: str) -> Dict[str, Any]:
    import time

    start = time.time()

    try:
        response = await generate_json_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=message,
        )

        intent = response.get("intent", DEFAULT_INTENT)

        return {
            "intent": intent,
            "valid": intent in INTENT_LABELS,
            "elapsed_ms": int((time.time() - start) * 1000),
            "raw_response": response,
        }

    except Exception as e:
        return {
            "intent": DEFAULT_INTENT,
            "valid": False,
            "elapsed_ms": int((time.time() - start) * 1000),
            "error": str(e),
        }
