"""
Rule-Based Question Node - Production Version with Premium UX
============================================================
Context-aware question generation with zero LLM calls.

Key Features:
1. Context-aware questions ("Where to IST from?" instead of "Where from?")
2. Smart suggestions based on known information
3. Zero LLM calls (100% deterministic)
4. <50ms latency

Author: AI Trip Planner Team
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.langgraph_flow.state import ConversationState, update_state

logger = logging.getLogger(__name__)


# ============================================================================
# CONTEXTUAL QUESTION BUILDER (PREMIUM UX)
# ============================================================================

def _build_contextual_question(
    missing_param: str, 
    state: ConversationState
) -> Dict[str, Any]:
    """
    Build questions that reference what we already know.
    
    Examples:
    - If destination="IST", ask: "Where are you flying to IST from?"
    - If origin="TAS", ask: "Where would you like to fly from TAS?"
    
    This creates a natural, context-aware conversation flow.
    """
    
    # ========================================
    # ORIGIN (Where flying FROM)
    # ========================================
    if missing_param == "origin":
        if state.destination:
            # We know destination → reference it
            dest_name = _get_city_name(state.destination)
            return {
                "message": f"Great! Where are you flying to {dest_name} from?",
                "placeholder": "Enter departure city or airport",
                "suggestions": [
                    "Tashkent",
                    "Samarkand", 
                    "Bukhara",
                    "📍 Use my location"
                ],
            }
        else:
            # Generic question
            return {
                "message": "Where are you flying from?",
                "placeholder": "Enter departure city or airport",
                "suggestions": [
                    "Tashkent",
                    "📍 Use my location"
                ],
            }
    
    # ========================================
    # DESTINATION (Where flying TO)
    # ========================================
    elif missing_param == "destination":
        if state.origin:
            # We know origin → reference it
            origin_name = _get_city_name(state.origin)
            return {
                "message": f"Perfect! Where would you like to fly from {origin_name}?",
                "placeholder": "Enter destination city",
                "suggestions": [
                    "Istanbul",
                    "Dubai", 
                    "London",
                    "Paris",
                    "✨ Surprise me"
                ],
            }
        else:
            # Generic question
            return {
                "message": "Where would you like to fly to?",
                "placeholder": "Enter destination city or airport",
                "suggestions": [
                    "Istanbul",
                    "Dubai",
                    "London", 
                    "Paris"
                ],
            }
    
    # ========================================
    # DEPARTURE DATE
    # ========================================
    elif missing_param == "departure_date":
        if state.origin and state.destination:
            # We know both → reference them
            origin_name = _get_city_name(state.origin)
            dest_name = _get_city_name(state.destination)
            
            return {
                "message": f"When would you like to fly from {origin_name} to {dest_name}?",
                "placeholder": "Select date or say 'next week'",
                "suggestions": [
                    "Tomorrow",
                    "This Weekend",
                    "Next Week",
                    "Next Month",
                    "📅 Flexible"
                ],
            }
        else:
            # Partial context
            return {
                "message": "When would you like to depart?",
                "placeholder": "Select departure date",
                "suggestions": [
                    "Tomorrow",
                    "This Weekend",
                    "Next Week"
                ],
            }
    
    # ========================================
    # RETURN DATE
    # ========================================
    elif missing_param == "return_date":
        if state.departure_date:
            return {
                "message": "When would you like to return?",
                "placeholder": "Select return date",
                "suggestions": [
                    "Same day",
                    "Next day",
                    "One week later",
                    "✈️ One-way trip"
                ],
            }
        else:
            return {
                "message": "When would you like to return?",
                "placeholder": "Select return date",
                "suggestions": [
                    "One week later",
                    "Two weeks later",
                    "✈️ One-way trip"
                ],
            }
    
    # ========================================
    # PASSENGERS
    # ========================================
    elif missing_param == "passengers":
        return {
            "message": "How many passengers?",
            "placeholder": "Number of travelers",
            "suggestions": [
                "👤 Just me",
                "👥 2 people",
                "👨‍👩‍👧 Family (3-4)",
                "👥 Group (5+)"
            ],
        }
    
    # ========================================
    # TRAVEL CLASS
    # ========================================
    elif missing_param == "travel_class":
        if state.budget:
            return {
                "message": f"Which class would you prefer? (Budget: ${state.budget})",
                "placeholder": "Select travel class",
                "suggestions": [
                    "💺 Economy",
                    "✨ Premium Economy",
                    "💼 Business",
                    "👑 First Class"
                ],
            }
        else:
            return {
                "message": "Which class would you prefer?",
                "placeholder": "Select travel class",
                "suggestions": [
                    "💺 Economy",
                    "✨ Premium Economy",
                    "💼 Business"
                ],
            }
    
    # ========================================
    # BUDGET
    # ========================================
    elif missing_param == "budget":
        return {
            "message": "What's your maximum budget for this trip?",
            "placeholder": "Enter maximum budget (USD)",
            "suggestions": [
                "$200",
                "$500",
                "$1000",
                "💰 No limit"
            ],
        }
    
    # ========================================
    # FLEXIBILITY
    # ========================================
    elif missing_param == "flexibility":
        return {
            "message": "How flexible are you with travel dates?",
            "placeholder": "Date flexibility",
            "suggestions": [
                "📌 Exact dates only",
                "±1 day",
                "±2 days",
                "±3 days"
            ],
        }
    
    # ========================================
    # FALLBACK (Unknown parameter)
    # ========================================
    else:
        logger.warning(f"[RuleQuestion] Unknown parameter: {missing_param}")
        return {
            "message": f"Could you provide your {missing_param}?",
            "placeholder": f"Enter {missing_param}",
            "suggestions": [],
        }


# ============================================================================
# UTILITIES
# ============================================================================

def _get_city_name(airport_code: str) -> str:
    """
    Convert airport code to friendly city name.
    
    Examples:
    - "IST" → "Istanbul"
    - "LHR" → "London"
    - "TAS" → "Tashkent"
    """
    
    # Known mappings
    city_map = {
        "IST": "Istanbul",
        "SAW": "Istanbul",
        "LHR": "London",
        "LGW": "London",
        "STN": "London",
        "LTN": "London",
        "CDG": "Paris",
        "ORY": "Paris",
        "DXB": "Dubai",
        "TAS": "Tashkent",
        "JFK": "New York",
        "LAX": "Los Angeles",
        "SIN": "Singapore",
        "HKG": "Hong Kong",
        "NRT": "Tokyo",
        "HND": "Tokyo",
        "BCN": "Barcelona",
        "MAD": "Madrid",
        "FCO": "Rome",
        "MUC": "Munich",
        "FRA": "Frankfurt",
        "AMS": "Amsterdam",
        "BER": "Berlin",
    }
    
    # Return city name if known, otherwise return code
    return city_map.get(airport_code.upper(), airport_code.upper())


def _normalize_suggestion(suggestion: str) -> str:
    """Clean and normalize suggestion text."""
    return suggestion.strip()


def _validate_question_data(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize question data.
    
    Ensures:
    - Message is valid string (10-500 chars)
    - Placeholder is valid (5-100 chars)
    - Suggestions is list of strings (<= 5 items)
    """
    
    validated = {}
    
    # ========================================
    # Validate message
    # ========================================
    message = question_data.get("message", "")
    if not message or not isinstance(message, str):
        logger.warning(f"[RuleQuestion] Invalid message: {message}")
        message = "Could you tell me more?"
    elif len(message) > 500:
        logger.warning(f"[RuleQuestion] Message too long: {len(message)} chars")
        message = message[:500]
    elif len(message) < 10:
        logger.warning(f"[RuleQuestion] Message too short: {len(message)} chars")
        message = "Could you tell me more?"
    
    validated["message"] = message.strip()
    
    # ========================================
    # Validate placeholder
    # ========================================
    placeholder = question_data.get("placeholder", "")
    if not placeholder or not isinstance(placeholder, str):
        placeholder = "Type here..."
    elif len(placeholder) > 100:
        placeholder = placeholder[:100]
    elif len(placeholder) < 5:
        placeholder = "Type here..."
    
    validated["placeholder"] = placeholder.strip()
    
    # ========================================
    # Validate suggestions
    # ========================================
    suggestions = question_data.get("suggestions", [])
    
    if not isinstance(suggestions, list):
        suggestions = []
    
    valid_suggestions = []
    for suggestion in suggestions[:5]:  # Max 5 suggestions
        if isinstance(suggestion, str) and 0 < len(suggestion) <= 50:
            normalized = _normalize_suggestion(suggestion)
            if normalized:
                valid_suggestions.append(normalized)
    
    validated["suggestions"] = valid_suggestions
    
    return validated


# ============================================================================
# GUARD CONDITIONS
# ============================================================================

def should_skip_rule_question(state: ConversationState) -> bool:
    """
    Determine if we should skip question generation.
    
    Skip if:
    1. Not on rule-based path
    2. No missing parameter
    3. Already ready for search
    """
    
    # Guard 1: Not rule-based path
    if not getattr(state, "use_rule_based_path", False):
        return True
    
    # Guard 2: No missing parameter
    if not getattr(state, "missing_parameter", None):
        return True
    
    # Guard 3: Already ready for search
    if getattr(state, "ready_for_search", False):
        return True
    
    return False


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

async def rule_based_question_node(state: ConversationState) -> ConversationState:
    """
    Generate contextual question using templates (NO LLM).
    
    Performance: <50ms, 0 LLM calls, $0.00
    
    Flow:
    1. Check guards (skip if not applicable)
    2. Generate contextual question based on known info
    3. Validate output
    4. Update state with question, placeholder, suggestions
    """
    
    # ========================================
    # Guard 1: Only rule-based path
    # ========================================
    if not getattr(state, "use_rule_based_path", False):
        logger.info(
            "[RuleQuestion] LLM path active → skipping template generation "
            "(using ask_next_question_node instead)"
        )
        return state
    
    # ========================================
    # Guard 2: Must have missing parameter
    # ========================================
    missing_param = getattr(state, "missing_parameter", None)
    if not missing_param:
        logger.warning(
            "[RuleQuestion] No missing_parameter set, cannot generate question"
        )
        return state
    
    # ========================================
    # Guard 3: Not ready for search yet
    # ========================================
    if getattr(state, "ready_for_search", False):
        logger.info(
            "[RuleQuestion] Already ready for search, skipping question generation"
        )
        return state
    
    logger.info(f"[RuleQuestion] Generating contextual question for: '{missing_param}'")
    
    # ========================================
    # Generate contextual question
    # ========================================
    try:
        question_data = _build_contextual_question(missing_param, state)
        logger.debug(f"[RuleQuestion] Raw question data: {question_data}")
        
    except Exception as e:
        logger.error(
            f"[RuleQuestion] Question generation failed: {e}, using fallback",
            exc_info=True
        )
        
        # Fallback to generic question
        question_data = {
            "message": f"Could you provide your {missing_param}?",
            "placeholder": f"Enter {missing_param}",
            "suggestions": [],
        }
    
    # ========================================
    # Validate output
    # ========================================
    question_data = _validate_question_data(question_data)
    
    message = question_data["message"]
    placeholder = question_data["placeholder"]
    suggestions = question_data["suggestions"]
    
    logger.info(
        f"[RuleQuestion] ✅ Generated: '{message[:60]}...' "
        f"(suggestions: {len(suggestions)})"
    )
    
    # ========================================
    # Update state
    # ========================================
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


# ============================================================================
# UTILITY FUNCTIONS FOR EXTERNAL USE
# ============================================================================

def generate_template_question(
    missing_parameter: str,
    state: Optional[ConversationState] = None
) -> Dict[str, Any]:
    """
    Public API for generating questions programmatically.
    
    Usage:
        question = generate_template_question("origin", state)
        print(question["message"])
    """
    
    try:
        if state is not None:
            question_data = _build_contextual_question(missing_parameter, state)
        else:
            # Create minimal state for context-less generation
            from app.langgraph_flow.state import create_initial_state
            minimal_state = create_initial_state(
                session_id="temp",
                latest_message=""
            )
            question_data = _build_contextual_question(missing_parameter, minimal_state)
        
        return _validate_question_data(question_data)
        
    except Exception as e:
        logger.error(f"[RuleQuestion] Template generation failed: {e}")
        
        return {
            "message": f"Could you provide your {missing_parameter}?",
            "placeholder": f"Enter {missing_parameter}",
            "suggestions": [],
        }


def get_question_quality_score(question_data: Dict[str, Any]) -> float:
    """
    Calculate quality score for a generated question (0.0-1.0).
    
    Scoring criteria:
    - Message quality: 0.4 points (length, clarity)
    - Placeholder quality: 0.2 points (brevity, usefulness)
    - Has suggestions: 0.2 points (existence)
    - Suggestion count: 0.2 points (3+ suggestions is ideal)
    """
    
    score = 0.0
    
    # Message quality
    message = question_data.get("message", "")
    if message and 10 <= len(message) <= 200:
        score += 0.4
    
    # Placeholder quality
    placeholder = question_data.get("placeholder", "")
    if placeholder and 5 <= len(placeholder) <= 50:
        score += 0.2
    
    # Has suggestions
    suggestions = question_data.get("suggestions", [])
    if suggestions:
        score += 0.2
    
    # Suggestion count (3+ is ideal)
    count = len(suggestions)
    if count >= 3:
        score += 0.2
    elif count >= 1:
        score += 0.1
    
    return round(score, 2)


def get_template_metrics(state: ConversationState) -> Dict[str, Any]:
    """
    Extract template generation metrics from state.
    
    Useful for monitoring and debugging.
    """
    
    suggestions = getattr(state, "suggestions", []) or []
    
    return {
        "missing_parameter": getattr(state, "missing_parameter", None),
        "last_question_slot": getattr(state, "last_question_slot", None),
        "last_question_text": getattr(state, "last_question", None),
        "has_question": bool(getattr(state, "assistant_message", None)),
        "has_suggestions": bool(suggestions),
        "suggestion_count": len(suggestions),
        "flow_stage": getattr(state, "flow_stage", None),
        "used_templates": getattr(state, "use_rule_based_path", False),
        "updated_at": getattr(state, "updated_at", None),
    }


# ============================================================================
# TESTING & DEBUGGING
# ============================================================================

def test_template_generation(missing_parameter: str) -> Dict[str, Any]:
    """
    Test question generation for a specific parameter.
    
    Returns performance metrics and quality score.
    
    Usage:
        result = test_template_generation("origin")
        print(result["success"])
        print(result["elapsed_ms"])
    """
    
    import time
    
    start_time = time.time()
    
    try:
        question_data = generate_template_question(missing_parameter, None)
        elapsed = time.time() - start_time
        
        return {
            "missing_parameter": missing_parameter,
            "question_data": question_data,
            "quality_score": get_question_quality_score(question_data),
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


# ============================================================================
# DOCUMENTATION
# ============================================================================

"""
USAGE EXAMPLES:

1. Basic usage in LangGraph:
   
   state = await rule_based_question_node(state)
   print(state.assistant_message)

2. Generate question programmatically:
   
   question = generate_template_question("origin", state)
   print(question["message"])
   print(question["suggestions"])

3. Test question quality:
   
   result = test_template_generation("destination")
   print(f"Quality: {result['quality_score']}/1.0")
   print(f"Speed: {result['elapsed_ms']}ms")

4. Get metrics:
   
   metrics = get_template_metrics(state)
   print(f"Suggestions: {metrics['suggestion_count']}")
"""