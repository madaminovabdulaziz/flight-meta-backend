"""
Should Use LLM Node – Dataclass-Safe Version
============================================
"""

import logging
from typing import Literal, Dict, Any, Tuple

from app.langgraph_flow.state import ConversationState, update_state
from services.local_intent_classifier import intent_classifier
from services.rule_based_extractor import rule_extractor

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

CONFIDENCE_THRESHOLD = 0.85
INTENT_WEIGHT = 0.6
EXTRACTION_WEIGHT = 0.4
MAX_COMPLETENESS_BONUS = 0.15
MAX_DEPTH_BONUS = 0.10
COMPLETENESS_BONUS_PER_SLOT = 0.04
DEPTH_BONUS_PER_TURN = 0.02


# ============================================================================
# CONFIDENCE COMPONENTS
# ============================================================================

def _calculate_overall_confidence(
    intent_confidence: float,
    extraction_confidence: float,
    filled_slots: int,
    turn_count: int
) -> Tuple[float, Dict[str, Any]]:

    base = (
        intent_confidence * INTENT_WEIGHT +
        extraction_confidence * EXTRACTION_WEIGHT
    )

    completeness_bonus = min(
        MAX_COMPLETENESS_BONUS,
        filled_slots * COMPLETENESS_BONUS_PER_SLOT
    )

    depth_bonus = min(
        MAX_DEPTH_BONUS,
        turn_count * DEPTH_BONUS_PER_TURN
    )

    overall = min(1.0, base + completeness_bonus + depth_bonus)

    breakdown = {
        "intent_conf": round(intent_confidence, 3),
        "extract_conf": round(extraction_confidence, 3),
        "base": round(base, 3),
        "completeness_bonus": round(completeness_bonus, 3),
        "depth_bonus": round(depth_bonus, 3),
        "overall": round(overall, 3),
        "filled_slots": filled_slots,
        "turns": turn_count,
    }

    return overall, breakdown


def _count_filled_slots(state: ConversationState) -> int:
    return sum([
        1 if getattr(state, "destination", None) else 0,
        1 if getattr(state, "origin", None) else 0,
        1 if getattr(state, "departure_date", None) else 0,
        1 if getattr(state, "passengers", None) else 0,
    ])


def _normalize_extracted_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = {}

    for k, v in params.items():
        if v is not None:
            out[k] = v

    dest = params.get("destination")
    if isinstance(dest, str) and len(dest) == 3:
        out["destination_airports"] = [dest.upper()]

    origin = params.get("origin")
    if isinstance(origin, str) and len(origin) == 3:
        out["origin_airports"] = [origin.upper()]

    if "return_date" in params:
        out["is_round_trip"] = params["return_date"] is not None

    return out


# ============================================================================
# MAIN NODE – SINGLE SOURCE OF TRUTH
# ============================================================================

async def should_use_llm_node(state: ConversationState) -> ConversationState:

    message = getattr(state, "latest_user_message", "")

    if not message:
        logger.warning("[ShouldUseLLM] Empty message → defaulting to LLM path")
        return update_state(state, {
            "use_rule_based_path": False,
            "routing_confidence": 0.0,
            "confidence_breakdown": {"error": "empty_message"},
        })

    logger.info(f"[ShouldUseLLM] Processing: '{message[:80]}...'")

    # ----------------------------------------
    # 1. Intent Classification
    # ----------------------------------------
    try:
        intent, intent_conf = intent_classifier.classify(message, state)
    except Exception as e:
        logger.error(f"Intent classifier failed: {e}")
        intent, intent_conf = "unknown", 0.0

    # ----------------------------------------
    # 2. Rule-Based Extraction
    # ----------------------------------------
    try:
        extracted, extract_conf = rule_extractor.extract(message, state)
    except Exception as e:
        logger.error(f"Extractor failed: {e}")
        extracted, extract_conf = {}, 0.0

    # ----------------------------------------
    # 3. Confidence Calculation
    # ----------------------------------------
    filled_slots = _count_filled_slots(state)
    turn_count = getattr(state, "turn_count", 0)

    overall, breakdown = _calculate_overall_confidence(
        intent_conf,
        extract_conf,
        filled_slots,
        turn_count,
    )

    logger.info(
        f"[ShouldUseLLM] Overall Confidence {overall:.3f} "
        f"(slots={filled_slots}, turns={turn_count})"
    )

    # ----------------------------------------
    # 4. Routing Decision
    # ----------------------------------------
    use_rule = overall >= CONFIDENCE_THRESHOLD

    if use_rule:
        logger.info(f"[ShouldUseLLM] ✅ HIGH confidence → RULE-BASED path")
    else:
        logger.info(f"[ShouldUseLLM] ⚠️ LOW confidence → LLM path")

    # ----------------------------------------
    # 5. Prepare State Updates
    # ----------------------------------------
    updates = {
        "use_rule_based_path": use_rule,
        "routing_confidence": overall,
        "confidence_breakdown": breakdown,
        "intent": intent,
        "extracted_params": extracted,
    }

    # Apply extracted params immediately if using rule-based path
    # if use_rule:
    #     normalized = _normalize_extracted_params(extracted)
    #     updates.update(normalized)
    #     logger.info(f"[ShouldUseLLM] Applied {len(normalized)} extracted params to state")

    # 🔧 FIX: Always return updated state (was missing for LLM path)
    updated_state = update_state(state, updates)
    
    # Debug: Verify state was updated correctly
    logger.debug(
        f"[ShouldUseLLM] State after update: use_rule={getattr(updated_state, 'use_rule_based_path', 'NOT SET')}, "
        f"conf={getattr(updated_state, 'routing_confidence', 'NOT SET')}"
    )
    
    return updated_state


# ============================================================================
# ROUTER – READS THE DECISION
# ============================================================================

# ... inside route_based_on_confidence ...

def route_based_on_confidence(state: ConversationState) -> Literal["rule_based_path", "llm_path"]:
    """
    Router function that reads the decision made by should_use_llm_node.
    """
    # Safe attribute access
    use_rule = getattr(state, "use_rule_based_path", False)
    conf = getattr(state, "routing_confidence", 0.0)
    
    routing_path = "rule_based_path" if use_rule else "llm_path"

    # Info log is sufficient
    logger.info(f"[Router] Decided: {routing_path} (Conf: {conf:.2f})")

    return routing_path

# ============================================================================
# DEBUGGING / METRICS
# ============================================================================

def get_routing_metrics(state: ConversationState) -> Dict[str, Any]:
    return {
        "routing_confidence": getattr(state, "routing_confidence", None),
        "use_rule_based_path": getattr(state, "use_rule_based_path", None),
        "confidence_breakdown": getattr(state, "confidence_breakdown", None),
        "intent": getattr(state, "intent", None),
        "extracted_params": getattr(state, "extracted_params", None),
        "threshold": CONFIDENCE_THRESHOLD,
    }


def validate_routing_state(state: ConversationState) -> bool:
    try:
        assert isinstance(getattr(state, "use_rule_based_path"), bool)
        conf = getattr(state, "routing_confidence")
        assert isinstance(conf, (int, float)) and 0 <= conf <= 1
        assert getattr(state, "intent") is not None
        return True
    except Exception as e:
        logger.error(f"Routing validation failed: {e}")
        return False