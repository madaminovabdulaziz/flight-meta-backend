"""
Rule-Based Ranking Node - Final Production Version
==================================================
Rank flights using cheap_ranker with zero LLM calls.

ENHANCEMENTS:
- Fixed field name validation (duration_minutes vs duration)
- Better error handling and validation
- Comprehensive logging
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from app.langgraph_flow.state import ConversationState, update_state
from services.cheap_ranker import cheap_ranker

logger = logging.getLogger(__name__)

# ==========================================
# CONFIGURATION
# ==========================================

# Required fields for flight validation
# Note: Using "duration_minutes" to match FlightService output format
REQUIRED_FLIGHT_FIELDS = ["price", "airline"]
OPTIONAL_FLIGHT_FIELDS = ["duration_minutes", "duration"]  # Accept either format

MAX_FLIGHTS_TO_RANK = 50


# ==========================================
# VALIDATION
# ==========================================

def _validate_flight(flight: Any) -> bool:
    """
    Validate that a flight dict has the minimum required fields.

    Accepts both "duration" and "duration_minutes" for flexibility.
    """
    if not isinstance(flight, dict):
        logger.debug(f"[RuleRanking] Flight validation failed: not a dict (type={type(flight)})")
        return False

    # Check required fields
    for field in REQUIRED_FLIGHT_FIELDS:
        if field not in flight:
            logger.debug(f"[RuleRanking] Flight validation failed: missing '{field}'")
            return False

        if field == "price" and not isinstance(flight[field], (int, float)):
            logger.debug(f"[RuleRanking] Flight validation failed: price is not numeric")
            return False

    # Check that at least one duration field exists
    has_duration = any(field in flight for field in OPTIONAL_FLIGHT_FIELDS)
    if not has_duration:
        logger.debug(f"[RuleRanking] Flight validation failed: no duration field found")
        return False

    # Validate duration if present
    if "duration_minutes" in flight:
        if not isinstance(flight["duration_minutes"], (int, float)):
            logger.debug(f"[RuleRanking] Flight validation failed: duration_minutes is not numeric")
            return False
    elif "duration" in flight:
        if not isinstance(flight["duration"], (int, float)):
            logger.debug(f"[RuleRanking] Flight validation failed: duration is not numeric")
            return False

    return True


def _normalize_flight(flight: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize flight data to standard format.

    Handles field name variations (e.g., duration_minutes → duration).
    """
    normalized = flight.copy()

    # Normalize duration field
    if "duration_minutes" in normalized and "duration" not in normalized:
        normalized["duration"] = normalized["duration_minutes"]
    elif "duration" not in normalized and "duration_minutes" in normalized:
        pass  # Already has duration_minutes, keep it

    return normalized


def _sanitize_flights(raw_flights: List[Any]) -> List[Dict[str, Any]]:
    """
    Validate and normalize flight list.

    Returns:
        List of valid, normalized flight dicts
    """
    if not isinstance(raw_flights, list):
        logger.error(f"[RuleRanking] raw_flights is not a list: {type(raw_flights)}")
        return []

    valid = []
    invalid_count = 0

    for i, flight in enumerate(raw_flights):
        if _validate_flight(flight):
            normalized = _normalize_flight(flight)
            valid.append(normalized)
        else:
            invalid_count += 1
            if invalid_count <= 3:  # Log first 3 failures for debugging
                logger.warning(
                    f"[RuleRanking] Skipping invalid flight at index {i}: "
                    f"type={type(flight)}, keys={list(flight.keys()) if isinstance(flight, dict) else 'N/A'}"
                )

    if invalid_count > 0:
        logger.info(f"[RuleRanking] Validated {len(valid)}/{len(raw_flights)} flights ({invalid_count} invalid)")
    else:
        logger.info(f"[RuleRanking] Validated {len(valid)}/{len(raw_flights)} flights (all valid)")

    if len(valid) > MAX_FLIGHTS_TO_RANK:
        logger.info(f"[RuleRanking] Limiting to {MAX_FLIGHTS_TO_RANK} flights (had {len(valid)})")
        valid = valid[:MAX_FLIGHTS_TO_RANK]

    return valid


def _generate_ranking_explanation(ranking_result: Dict[str, Any], flight_count: int) -> str:
    """Generate a human-readable explanation of the ranking."""
    if "summary_text" in ranking_result:
        return ranking_result["summary_text"]

    stats = ranking_result.get("summary_stats", {})
    total = stats.get("total_flights", flight_count)
    criteria = stats.get("criteria_used", [])

    if criteria:
        return f"Found {total} flights. Ranked by: {', '.join(criteria)}."
    return f"Found {total} flights. Showing top {flight_count} options."


# ==========================================
# GUARD CHECKER
# ==========================================

def should_skip_rule_ranking(state: ConversationState) -> bool:
    """Determine if rule-based ranking should be skipped."""
    if not getattr(state, "use_rule_based_path", False):
        return True

    raw_flights = getattr(state, "raw_flights", None)
    if not raw_flights:
        return True

    return False


# ==========================================
# MAIN NODE
# ==========================================

async def rule_based_ranking_node(state: ConversationState) -> ConversationState:
    """
    Rank flights using rule-based cheap_ranker (no LLM).

    This node processes flight results and ranks them using a deterministic
    algorithm based on price, duration, and other factors.
    """

    # ───────────────────────────────────────────
    # Guard 1 — only on rule-based path
    # ───────────────────────────────────────────
    if not getattr(state, "use_rule_based_path", False):
        logger.info("[RuleRanking] LLM path active → skip")
        return state

    # ───────────────────────────────────────────
    # Get flights
    # ───────────────────────────────────────────
    raw_flights = getattr(state, "raw_flights", None)

    if not raw_flights or not isinstance(raw_flights, list):
        logger.warning("[RuleRanking] No valid flights to rank")
        return update_state(state, {
            "ranked_flights": [],
            "ranking_explanation": "No valid flight options found for your search.",
            "smart_suggestions": [],
            "ranking_method": "rule_based",
            "ranking_confidence": 1.0,
            "updated_at": datetime.now(timezone.utc),
        })

    logger.info(f"[RuleRanking] Processing {len(raw_flights)} flights (rule-based, NO LLM)")

    # ───────────────────────────────────────────
    # Sanitize flights
    # ───────────────────────────────────────────
    valid_flights = _sanitize_flights(raw_flights)

    if not valid_flights:
        logger.error("[RuleRanking] No valid flights after sanitization")
        return update_state(state, {
            "ranked_flights": [],
            "ranking_explanation": (
                "Found flights but they don't have valid pricing or details. "
                "Please try adjusting your search criteria."
            ),
            "smart_suggestions": ["Change dates", "Try different airports", "Adjust budget"],
            "ranking_method": "rule_based",
            "ranking_confidence": 0.0,
            "updated_at": datetime.now(timezone.utc),
        })

    # ───────────────────────────────────────────
    # Ranking using cheap_ranker
    # ───────────────────────────────────────────
    try:
        ranking_result = cheap_ranker.rank_flights(valid_flights, state)

        ranked_flights = ranking_result.get("ranked_flights", [])
        suggestions = ranking_result.get("smart_suggestions", [])
        stats = ranking_result.get("summary_stats", {})

        logger.info(f"[RuleRanking] ✅ Ranked {len(ranked_flights)} flights")
    except Exception as e:
        logger.error(f"[RuleRanking] Ranking failed: {e}", exc_info=True)
        # Fallback: simple price sort
        ranked_flights = sorted(valid_flights, key=lambda x: x.get("price", float("inf")))
        suggestions = []
        stats = {"total_flights": len(ranked_flights)}
        logger.info(f"[RuleRanking] ⚠️ Fallback ranking applied ({len(ranked_flights)} flights)")

    # ───────────────────────────────────────────
    # Explanation — deterministic template
    # ───────────────────────────────────────────
    explanation = _generate_ranking_explanation(
        ranking_result if "ranking_result" in locals() else {},
        len(ranked_flights),
    )

    logger.info(f"[RuleRanking] Explanation: {explanation[:60]}...")

    # ───────────────────────────────────────────
    # Update state
    # ───────────────────────────────────────────
    updates = {
        "ranked_flights": ranked_flights,
        "ranking_explanation": explanation,
        "smart_suggestions": suggestions,
        "ranking_method": "rule_based",
        "ranking_confidence": 1.0,
        "ranking_stats": stats,
        "updated_at": datetime.now(timezone.utc),
    }

    return update_state(state, updates)


# ==========================================
# METRICS
# ==========================================

def get_ranking_metrics(state: ConversationState) -> Dict[str, Any]:
    """Get ranking performance metrics."""
    ranked = getattr(state, "ranked_flights", []) or []
    return {
        "ranking_method": getattr(state, "ranking_method", None),
        "ranking_confidence": getattr(state, "ranking_confidence", None),
        "total_ranked": len(ranked),
        "has_explanation": bool(getattr(state, "ranking_explanation", None)),
        "has_suggestions": bool(getattr(state, "smart_suggestions", [])),
        "suggestion_count": len(getattr(state, "smart_suggestions", []) or []),
        "ranking_stats": getattr(state, "ranking_stats", {}),
        "updated_at": getattr(state, "updated_at", None),
    }
