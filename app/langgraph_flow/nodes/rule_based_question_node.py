"""
Rule-Based Question Node - Final Production Version
===================================================
Template-based question generation with zero LLM calls.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.langgraph_flow.state import ConversationState, update_state
from services.template_engine import template_engine

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURATION
# ==========================================

DEFAULT_QUESTION_DATA = {
    "destination": {
        "message": "Where would you like to fly to?",
        "placeholder": "Enter destination city or airport",
        "suggestions": ["Dubai", "Istanbul", "London", "Paris"],
    },
    "origin": {
        "message": "Where are you flying from?",
        "placeholder": "Enter departure city",
        "suggestions": ["Tashkent", "Samarkand", "Bukhara"],
    },
    "departure_date": {
        "message": "When would you like to depart?",
        "placeholder": "Select or enter date",
        "suggestions": ["Tomorrow", "This Weekend", "Next Week"],
    },
}


# ==========================================
# VALIDATION
# ==========================================

def _normalize_suggestion(suggestion: str) -> str:
    normalized = suggestion.strip()
    return normalized


def _validate_question_data(question_data: Dict[str, Any]) -> Dict[str, Any]:
    validated = {}
    
    # message
    message = question_data.get("message", "")
    if not message or not isinstance(message, str) or len(message) > 500:
        logger.warning(f"Invalid message: {message}")
        message = "Could you tell me more?"
    validated["message"] = message.strip()
    
    # placeholder
    placeholder = question_data.get("placeholder", "")
    if not placeholder or not isinstance(placeholder, str) or len(placeholder) > 100:
        placeholder = "Type here..."
    validated["placeholder"] = placeholder.strip()
    
    # suggestions
    suggestions = question_data.get("suggestions", [])
    if not isinstance(suggestions, list):
        suggestions = []
    
    valid_suggestions = []
    for suggestion in suggestions[:5]:
        if isinstance(suggestion, str) and 0 < len(suggestion) <= 50:
            normalized = _normalize_suggestion(suggestion)
            if normalized:
                valid_suggestions.append(normalized)
    
    validated["suggestions"] = valid_suggestions
    
    return validated


def _get_fallback_question(missing_parameter: str) -> Dict[str, Any]:
    return DEFAULT_QUESTION_DATA.get(
        missing_parameter,
        {
            "message": f"Could you provide your {missing_parameter}?",
            "placeholder": f"Enter {missing_parameter}",
            "suggestions": [],
        }
    )


# ==========================================
# GUARD CONDITION CHECKER
# ==========================================

def should_skip_rule_question(state: ConversationState) -> bool:
    """
    Skip if:
    1. Not on rule-based path
    2. No missing_parameter
    3. Already ready_for_search
    """
    # Guard 1
    if not getattr(state, "use_rule_based_path", False):
        return True
    
    # Guard 2
    if not getattr(state, "missing_parameter", None):
        return True
    
    # Guard 3
    if getattr(state, "ready_for_search", False):
        return True
    
    return False


# ==========================================
# MAIN NODE FUNCTION
# ==========================================

async def rule_based_question_node(state: ConversationState) -> ConversationState:
    """
    Generate question using templates (NO LLM).
    """
    
    # Guard 1: Only rule-based path
    if not getattr(state, "use_rule_based_path", False):
        logger.info(
            "[RuleQuestion] LLM path active → skipping template generation "
            "(using ask_next_question_node instead)"
        )
        return state
    
    # Guard 2: missing parameter
    missing_param = getattr(state, "missing_parameter", None)
    if not missing_param:
        logger.warning("[RuleQuestion] No missing_parameter set, cannot generate question")
        return state
    
    # Guard 3: already ready_for_search
    if getattr(state, "ready_for_search", False):
        logger.info("[RuleQuestion] Already ready for search, skipping question generation")
        return state
    
    logger.info(f"[RuleQuestion] Generating template question for: '{missing_param}'")
    
    # Generate question
    try:
        question_data = template_engine.generate_question(missing_param, state)
        logger.debug(f"[RuleQuestion] Raw template output: {question_data}")
    except Exception as e:
        logger.error(f"[RuleQuestion] Template engine failed: {e}, using fallback")
        question_data = _get_fallback_question(missing_param)
    
    # Validate
    question_data = _validate_question_data(question_data)
    
    message = question_data["message"]
    placeholder = question_data["placeholder"]
    suggestions = question_data["suggestions"]
    
    logger.info(
        f"[RuleQuestion] ✅ Generated: '{message[:60]}...' "
        f"(suggestions: {len(suggestions)})"
    )
    
    updates = {
        "assistant_message": message,
        "next_placeholder": placeholder,
        "suggestions": suggestions,
        "last_question_slot": missing_param,
        "last_question_text": message,
        "flow_stage": "collecting",
        "updated_at": datetime.now(timezone.utc),
    }
    
    return update_state(state, updates)


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def generate_template_question(
    missing_parameter: str,
    state: Optional[ConversationState] = None
) -> Dict[str, Any]:
    try:
        if state is not None:
            question_data = template_engine.generate_question(missing_parameter, state)
        else:
            question_data = template_engine.generate_question(missing_parameter, {})
    except Exception as e:
        logger.error(f"Template generation failed: {e}")
        question_data = _get_fallback_question(missing_parameter)
    
    return _validate_question_data(question_data)


def test_template_generation(missing_parameter: str) -> Dict[str, Any]:
    import time
    
    start_time = time.time()
    try:
        question_data = generate_template_question(missing_parameter, None)
        elapsed = time.time() - start_time
        return {
            "missing_parameter": missing_parameter,
            "question_data": question_data,
            "elapsed_ms": int(elapsed * 1000),
            "success": True,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "missing_parameter": missing_parameter,
            "error": str(e),
            "elapsed_ms": int(elapsed * 1000),
            "success": False,
        }


def get_question_quality_score(question_data: Dict[str, Any]) -> float:
    score = 0.0
    
    message = question_data.get("message", "")
    if message and 10 <= len(message) <= 200:
        score += 0.4
    
    placeholder = question_data.get("placeholder", "")
    if placeholder and 5 <= len(placeholder) <= 50:
        score += 0.2
    
    suggestions = question_data.get("suggestions", [])
    if suggestions:
        score += 0.2
    
    count = len(suggestions)
    if count >= 3:
        score += 0.2
    elif count >= 1:
        score += 0.1
    
    return score


def get_template_metrics(state: ConversationState) -> Dict[str, Any]:
    """
    Extract template generation metrics.
    """
    suggestions = getattr(state, "suggestions", []) or []
    return {
        "missing_parameter": getattr(state, "missing_parameter", None),
        "last_question_slot": getattr(state, "last_question_slot", None),
        "has_question": bool(getattr(state, "assistant_message", None)),
        "has_suggestions": bool(suggestions),
        "suggestion_count": len(suggestions),
        "flow_stage": getattr(state, "flow_stage", None),
        "used_templates": getattr(state, "use_rule_based_path", False),
        "updated_at": getattr(state, "updated_at", None),
    }
