"""
Rule-Based Ranking Node - Final Production Version
==================================================
Rank flights using cheap_ranker with zero LLM calls.
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

REQUIRED_FLIGHT_FIELDS = ["price", "duration", "airline"]
MAX_FLIGHTS_TO_RANK = 50


# ==========================================
# VALIDATION
# ==========================================

def _validate_flight(flight: Any) -> bool:
    if not isinstance(flight, dict):
        return False

    for field in REQUIRED_FLIGHT_FIELDS:
        if field not in flight:
            return False

        if field == "price" and not isinstance(flight[field], (int, float)):
            return False

        if field == "duration" and not isinstance(flight[field], (int, float)):
            return False

    return True


def _sanitize_flights(raw_flights: List[Any]) -> List[Dict[str, Any]]:
    if not isinstance(raw_flights, list):
        logger.error(f"[RuleRanking] raw_flights is not a list: {type(raw_flights)}")
        return []

    valid = []
    for i, flight in enumerate(raw_flights):
        if _validate_flight(flight):
            valid.append(flight)
        else:
            logger.warning(f"[RuleRanking] Skipping invalid flight at index {i}: {type(flight)}")

    logger.info(f"[RuleRanking] Validated {len(valid)}/{len(raw_flights)} flights")

    if len(valid) > MAX_FLIGHTS_TO_RANK:
        logger.info(f"[RuleRanking] Limiting to {MAX_FLIGHTS_TO_RANK} flights (had {len(valid)})")
        valid = valid[:MAX_FLIGHTS_TO_RANK]

    return valid


def _generate_ranking_explanation(ranking_result: Dict[str, Any], flight_count: int) -> str:
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
            "smart_suggestions": [],
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
        ranked_flights = sorted(valid_flights, key=lambda x: x.get("price", float("inf")))
        suggestions = []
        stats = {"total_flights": len(ranked_flights)}

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
