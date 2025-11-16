"""
Should Use LLM Node
The routing decision node that enables hybrid rule-based/LLM flow.

This is THE KEY NODE that determines whether to use:
- Fast rule-based path (no LLM, <100ms, $0 cost)
- Fallback LLM path (slower, $$$)

Philosophy:
- Use rules when confident (85%+ confidence)
- Use LLM only when rules fail
- This single decision saves 80-100% of LLM costs

Usage in graph:
    load_user_memory → should_use_llm → [rule_path | llm_path]
"""

import logging
from typing import Literal, Dict, Any, Tuple

from app.langgraph_flow.state import ConversationState, update_state
from services.local_intent_classifier import intent_classifier
from services.rule_based_extractor import rule_extractor

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIDENCE SCORING SYSTEM
# ============================================================================

def _calculate_overall_confidence(
    intent_confidence: float,
    extraction_confidence: float,
    state: ConversationState
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate overall confidence score from multiple signals.
    
    Signals:
    1. Intent classification confidence (0.0-1.0)
    2. Parameter extraction confidence (0.0-1.0)
    3. State completeness bonus (more filled slots = higher confidence)
    4. Conversation context (early turns = lower confidence)
    
    Returns:
        (overall_confidence, confidence_breakdown)
    """
    
    # Base confidence from classifiers
    base_confidence = (intent_confidence * 0.6 + extraction_confidence * 0.4)
    
    # State completeness bonus
    filled_slots = sum([
        1 if state.get("destination") else 0,
        1 if state.get("origin") else 0,
        1 if state.get("departure_date") else 0,
        1 if state.get("passengers") else 0,
    ])
    completeness_bonus = min(0.15, filled_slots * 0.04)  # Max +0.15
    
    # Conversation depth bonus (later turns = more context = higher confidence)
    turn_count = state.get("turn_count", 0)
    depth_bonus = min(0.10, turn_count * 0.02)  # Max +0.10
    
    # Calculate final confidence
    overall = min(1.0, base_confidence + completeness_bonus + depth_bonus)
    
    breakdown = {
        "intent_confidence": intent_confidence,
        "extraction_confidence": extraction_confidence,
        "base_confidence": base_confidence,
        "completeness_bonus": completeness_bonus,
        "depth_bonus": depth_bonus,
        "overall_confidence": overall,
    }
    
    return overall, breakdown


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================

async def should_use_llm_node(state: ConversationState) -> ConversationState:
    """
    Routing decision node - determines whether to use rule-based or LLM path.
    
    Decision flow:
    1. Run local intent classifier
    2. Run rule-based parameter extractor
    3. Calculate overall confidence
    4. If confidence >= 0.85 → use_rule_based_path = True
    5. If confidence < 0.85 → use_rule_based_path = False
    
    The graph will use 'use_rule_based_path' flag to route.
    
    Returns:
        Updated state with:
        - use_rule_based_path: bool
        - intent: str (if confident)
        - extracted parameters (if confident)
        - confidence_breakdown: dict (for debugging)
    """
    
    message = state.get("latest_user_message", "")
    
    logger.info(f"[ShouldUseLLM] Evaluating: '{message[:50]}...'")
    
    # ========================================
    # STEP 1: Run Rule-Based Intent Classifier
    # ========================================
    
    intent, intent_confidence = intent_classifier.classify(message, state)
    
    logger.info(f"[ShouldUseLLM] Intent: {intent} (confidence: {intent_confidence:.2f})")
    
    # ========================================
    # STEP 2: Run Rule-Based Parameter Extractor
    # ========================================
    
    extracted_params, extraction_confidence = rule_extractor.extract(message, state)
    
    logger.info(f"[ShouldUseLLM] Extracted: {list(extracted_params.keys())} (confidence: {extraction_confidence:.2f})")
    
    # ========================================
    # STEP 3: Calculate Overall Confidence
    # ========================================
    
    overall_confidence, breakdown = _calculate_overall_confidence(
        intent_confidence,
        extraction_confidence,
        state
    )
    
    logger.info(f"[ShouldUseLLM] Overall confidence: {overall_confidence:.2f}")
    
    # ========================================
    # STEP 4: Make Routing Decision
    # ========================================
    
    CONFIDENCE_THRESHOLD = 0.85
    use_rule_based = overall_confidence >= CONFIDENCE_THRESHOLD
    
    if use_rule_based:
        logger.info("✅ [ShouldUseLLM] HIGH confidence → RULE-BASED path (no LLM)")
    else:
        logger.info("⚠️ [ShouldUseLLM] LOW confidence → LLM path (fallback)")
    
    # ========================================
    # STEP 5: Update State
    # ========================================
    
    updates = {
        "use_rule_based_path": use_rule_based,
        "confidence_breakdown": breakdown,
    }
    
    # If confident, apply intent and extracted parameters immediately
    if use_rule_based:
        updates["intent"] = intent
        
        # Merge extracted parameters into state
        for key, value in extracted_params.items():
            if value is not None:
                updates[key] = value
        
        # Update derived fields
        if "destination" in extracted_params:
            dest = extracted_params["destination"]
            if dest and len(dest) == 3:
                updates["destination_airports"] = [dest.upper()]
        
        if "origin" in extracted_params:
            orig = extracted_params["origin"]
            if orig and len(orig) == 3:
                updates["origin_airports"] = [orig.upper()]
        
        if "return_date" in extracted_params:
            updates["is_round_trip"] = extracted_params["return_date"] is not None
    
    logger.info(f"[ShouldUseLLM] State updates: {list(updates.keys())}")
    
    return update_state(state, updates)


# ============================================================================
# GRAPH ROUTING HELPER (used in graph.py)
# ============================================================================

def route_based_on_confidence(
    state: ConversationState
) -> Literal["rule_based_path", "llm_path"]:
    """
    Routing function for LangGraph conditional edge.
    
    Called by graph.py to determine which path to take.
    
    Returns:
        "rule_based_path" if use_rule_based_path is True
        "llm_path" otherwise
    """
    use_rule_based = state.get("use_rule_based_path", False)
    
    if use_rule_based:
        logger.info("→ Routing to RULE-BASED path")
        return "rule_based_path"
    else:
        logger.info("→ Routing to LLM path")
        return "llm_path"