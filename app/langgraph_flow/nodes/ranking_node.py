"""
Ranking Node - Final Production Version (LLM Fallback)
======================================================
LLM-powered ranking with explanations for low-confidence queries ONLY.
"""
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.langgraph_flow.state import ConversationState, update_state
from services.ranking_explanation import generate_batch_explanation

logger = logging.getLogger(__name__)


# ==========================================
# CONFIG
# ==========================================

MAX_FLIGHTS_TO_EXPLAIN = 5
REQUIRED_FIELDS = ["price", "duration", "airline"]


# ==========================================
# GUARD CONDITION
# ==========================================

def should_skip_llm_ranking(state: ConversationState) -> bool:
    """Skip LLM ranking when rule-based path or no flights."""
    if getattr(state, "use_rule_based_path", False):
        return True
    if not getattr(state, "raw_flights", []):
        return True
    return False


# ==========================================
# SCORING HELPERS
# ==========================================

def _calculate_price_score(flight, min_price, max_price, budget=None):
    price = flight.get("price", max_price)
    if max_price == min_price:
        return 1.0

    normalized = 1.0 - (price - min_price) / (max_price - min_price)

    if budget and price > budget:
        penalty = min(0.5, (price - budget) / budget)
        normalized *= (1.0 - penalty)

    return max(0.0, min(1.0, normalized))


def _calculate_duration_score(flight, min_duration, max_duration):
    duration = flight.get("duration", max_duration)
    if max_duration == min_duration:
        return 1.0
    normalized = 1.0 - (duration - min_duration) / (max_duration - min_duration)
    return max(0.0, min(1.0, normalized))


def _calculate_stops_score(flight):
    stops = flight.get("stops", 2)
    if stops == 0:
        return 1.0
    elif stops == 1:
        return 0.7
    return 0.4


def _calculate_airline_score(flight, preferred_airlines=None):
    if not preferred_airlines:
        return 0.5
    airline = flight.get("airline", "")
    return 1.0 if airline in preferred_airlines else 0.3


def _calculate_time_score(flight, preferred_time=None):
    return 0.5  # Simplified placeholder


def _calculate_class_score(flight, preferred_class=None):
    if not preferred_class:
        return 0.5
    return 1.0 if flight.get("travel_class", "Economy") == preferred_class else 0.4


def _score_single_flight(
    flight, min_price, max_price, min_duration, max_duration, state
):
    budget = getattr(state, "budget", None)
    preferred_airlines = getattr(state, "preferred_airlines", [])
    preferred_class = getattr(state, "travel_class", "Economy")

    price_score = _calculate_price_score(flight, min_price, max_price, budget)
    duration_score = _calculate_duration_score(flight, min_duration, max_duration)
    stops_score = _calculate_stops_score(flight)
    airline_score = _calculate_airline_score(flight, preferred_airlines)
    time_score = _calculate_time_score(flight)
    class_score = _calculate_class_score(flight, preferred_class)

    weights = {
        "price": 0.30,
        "duration": 0.25,
        "stops": 0.20,
        "airline": 0.10,
        "time": 0.10,
        "class": 0.05,
    }

    total_score = (
        price_score * weights["price"] +
        duration_score * weights["duration"] +
        stops_score * weights["stops"] +
        airline_score * weights["airline"] +
        time_score * weights["time"] +
        class_score * weights["class"]
    )

    return {
        **flight,
        "total_score": round(total_score, 3),
        "scores": {
            "price": round(price_score, 3),
            "duration": round(duration_score, 3),
            "stops": round(stops_score, 3),
            "airline": round(airline_score, 3),
            "time": round(time_score, 3),
            "class": round(class_score, 3),
        }
    }


def _score_and_rank_flights(flights, state):
    if not flights:
        return []

    prices = [f.get("price", 0) for f in flights if f.get("price")]
    durations = [f.get("duration", 0) for f in flights if f.get("duration")]

    if not prices or not durations:
        return flights

    min_price, max_price = min(prices), max(prices)
    min_duration, max_duration = min(durations), max(durations)

    scored = [
        _score_single_flight(f, min_price, max_price, min_duration, max_duration, state)
        for f in flights
    ]

    scored.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    return scored


# ==========================================
# LLM BATCH EXPLANATION
# ==========================================

async def _generate_batch_explanation(top_flights, state):
    try:
        summaries = [
            {
                "rank": i + 1,
                "airline": f.get("airline", "Unknown"),
                "price": f.get("price", 0),
                "duration": f.get("duration", 0),
                "stops": f.get("stops", 0),
                "scores": f.get("scores", {}),
                "total_score": f.get("total_score", 0),
            }
            for i, f in enumerate(top_flights)
        ]

        explanation = await generate_batch_explanation(
            flights=summaries,
            user_preferences={
                "budget": getattr(state, "budget", None),
                "travel_class": getattr(state, "travel_class", None),
                "preferred_airlines": getattr(state, "preferred_airlines", []),
            }
        )

        return explanation

    except Exception as e:
        logger.error(f"[LLMRanking] Batch explanation failed: {e}")
        return {
            "summary": f"Found {len(top_flights)} flights ranked by score.",
            "flight_labels": [f"Option {i+1}" for i in range(len(top_flights))],
            "tradeoffs": [],
            "suggestions": [],
        }


# ==========================================
# MAIN NODE
# ==========================================

async def ranking_node(state: ConversationState) -> ConversationState:

    if getattr(state, "use_rule_based_path", False):
        logger.info("[LLMRanking] Rule-based path → skip LLM ranking")
        return state

    raw_flights = getattr(state, "raw_flights", [])
    if not raw_flights:
        return update_state(state, {
            "ranked_flights": [],
            "ranking_explanation": "No flights found.",
            "smart_suggestions": ["Try different dates", "Try nearby airports"],
            "ranking_method": "llm",
            "ranking_confidence": 0.0,
            "updated_at": datetime.now(timezone.utc),
        })

    ranked = _score_and_rank_flights(raw_flights, state)
    top_flights = ranked[:MAX_FLIGHTS_TO_EXPLAIN]

    explanation = await _generate_batch_explanation(top_flights, state)

    updates = {
        "ranked_flights": ranked,
        "ranking_explanation": explanation.get(
            "summary",
            f"Top {len(top_flights)} flights ranked for you."
        ),
        "flight_labels": explanation.get(
            "flight_labels",
            [f"Option {i+1}" for i in range(len(top_flights))]
        ),
        "smart_suggestions": explanation.get("suggestions", []),
        "ranking_method": "llm",
        "ranking_confidence": 0.8,
        "updated_at": datetime.now(timezone.utc),
    }

    return update_state(state, updates)
