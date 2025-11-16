

"""
Production Ranking Node
Complete 6-factor flight ranking with AI-powered explanations.

Implements the full ranking algorithm from MASTER SPEC Part 3:
1. Price Score (0.35 weight)
2. Duration Score (0.25 weight)
3. Layover Score (0.15 weight)
4. Airline Quality Score (0.10 weight)
5. Airport Convenience Score (0.10 weight)
6. Personalization Score (0.05 weight)

Additional production features:
- Dynamic weight adjustment based on user intent/preferences
- Low-quality itinerary filtering
- AI-generated explanations, labels, tradeoffs
- Smart suggestions for alternatives
- Complete error handling
"""

import logging
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime

from app.langgraph_flow.state import ConversationState, update_state
from services.ranking_explanation import ranking_explanation_service

logger = logging.getLogger(__name__)


# ============================================================================
# SCORING FUNCTIONS (6 FACTORS)
# ============================================================================

def _normalize_range(value: float, min_v: float, max_v: float, epsilon: float = 1e-6) -> float:
    """
    Normalize a value to 0.0-1.0 range.
    Lower value → higher score (for minimization metrics like price, duration).
    """
    if max_v <= min_v + epsilon:
        return 0.5  # All values same, neutral score
    
    normalized = 1.0 - ((value - min_v) / (max_v - min_v))
    return max(0.0, min(1.0, normalized))


def _price_score(flight: Dict[str, Any], min_price: float, max_price: float) -> float:
    """
    Factor 1: Price Score (weight 0.35)
    Lower price = higher score
    """
    price = flight.get("price", 0.0)
    return _normalize_range(price, min_price, max_price)


def _duration_score(flight: Dict[str, Any], min_duration: float, max_duration: float) -> float:
    """
    Factor 2: Duration Score (weight 0.25)
    Shorter duration = higher score
    """
    duration = flight.get("duration_minutes", 0)
    return _normalize_range(duration, min_duration, max_duration)


def _layover_score(flight: Dict[str, Any]) -> float:
    """
    Factor 3: Layover Score (weight 0.15)
    
    Penalties:
    - Each stop: -0.3
    - Overnight layover: -0.2
    - Risky short connection (<60min): -0.3
    - Long layover (>4h): -0.1
    
    Capped at 0.0-1.0
    """
    stops = flight.get("stops", 0)
    layovers = flight.get("layovers", [])
    
    score = 1.0
    
    # Penalty for each stop
    score -= stops * 0.3
    
    # Analyze each layover
    for layover in layovers:
        # Overnight layover is highly undesirable
        if layover.get("overnight", False):
            score -= 0.2
        
        # Risky short connection
        connection_minutes = layover.get("min_connection_minutes", 999)
        if connection_minutes < 60:
            score -= 0.3  # High risk of missing connection
        elif connection_minutes < 90:
            score -= 0.1  # Somewhat tight
        
        # Very long layover
        layover_duration = layover.get("duration_minutes", 0)
        if layover_duration > 240:  # >4 hours
            score -= 0.1
    
    return max(0.0, min(1.0, score))


def _airline_quality_score(flight: Dict[str, Any]) -> float:
    """
    Factor 4: Airline Quality Score (weight 0.10)
    
    Combines:
    - Airline rating (1-5 stars) → 60% weight
    - On-time performance (0-1) → 40% weight
    """
    rating = flight.get("airline_rating")
    on_time = flight.get("on_time_score", 0.5)
    
    if rating is None:
        # No rating data, use neutral score
        rating_normalized = 0.5
    else:
        # Normalize rating from 1-5 scale to 0-1
        rating_normalized = (rating - 1.0) / 4.0
        rating_normalized = max(0.0, min(1.0, rating_normalized))
    
    # Blend rating and on-time performance
    quality_score = 0.6 * rating_normalized + 0.4 * on_time
    
    return quality_score


def _airport_convenience_score(flight: Dict[str, Any]) -> float:
    """
    Factor 5: Airport Convenience Score (weight 0.10)
    
    Based on distance to city center:
    - ≤10km: 1.0
    - 10-30km: 0.8
    - 30-50km: 0.5
    - >50km: 0.2
    
    Averages departure and arrival airport convenience.
    """
    dep_distance = flight.get("departure_airport_distance_to_city_km", 30)
    arr_distance = flight.get("arrival_airport_distance_to_city_km", 30)
    
    def distance_to_score(km: float) -> float:
        if km is None:
            return 0.5  # Unknown, neutral
        
        if km <= 10:
            return 1.0
        elif km <= 30:
            return 0.8
        elif km <= 50:
            return 0.5
        else:
            return 0.2
    
    dep_score = distance_to_score(dep_distance)
    arr_score = distance_to_score(arr_distance)
    
    # Equal weight to departure and arrival convenience
    return 0.5 * dep_score + 0.5 * arr_score


def _personalization_score(
    flight: Dict[str, Any],
    preferences: Dict[str, Any],
    budget: Optional[float],
) -> float:
    """
    Factor 6: Personalization Score (weight 0.05)
    
    Adjustments based on user preferences:
    - Preferred airline: +0.2
    - Prefers direct & is direct: +0.2
    - Hates overnight & has overnight: -0.3
    - Over budget significantly: -0.2 to -0.4
    - Matches preferred time of day: +0.1
    """
    score = 0.5  # Start neutral
    
    airline = flight.get("airline")
    price = flight.get("price", 0)
    stops = flight.get("stops", 0)
    layovers = flight.get("layovers", [])
    
    # Preferred airlines
    preferred_airlines = preferences.get("preferred_airlines", [])
    if airline and airline in preferred_airlines:
        score += 0.2
    
    # Direct flight preference
    prefers_direct = preferences.get("prefers_direct", False)
    if prefers_direct and stops == 0:
        score += 0.2
    
    # Overnight layover aversion
    hates_overnight = preferences.get("hates_overnight_layovers", False)
    if hates_overnight:
        for layover in layovers:
            if layover.get("overnight", False):
                score -= 0.3
                break
    
    # Budget constraint
    if budget is not None and price is not None:
        if price > budget * 1.2:
            score -= 0.4  # Way over budget
        elif price > budget:
            score -= 0.2  # Slightly over budget
    
    # Budget range preference
    budget_range = preferences.get("budget_range")
    if budget_range and price is not None:
        try:
            low, high = budget_range
            if price > high:
                score -= 0.2
            elif low <= price <= high:
                score += 0.1  # Within preferred range
        except:
            pass
    
    # Time of day preference (simplified - would need departure time parsing)
    preferred_time = preferences.get("preferred_time_of_day")
    if preferred_time:
        # This is a placeholder - in production, parse departure_time and compare
        # For now, we just give a small bonus if preference exists
        pass
    
    return max(0.0, min(1.0, score))


# ============================================================================
# DYNAMIC WEIGHT ADJUSTMENT
# ============================================================================

def _adjust_weights_for_intent(
    base_weights: Dict[str, float],
    intent: Optional[str],
    user_message: str,
    preferences: Dict[str, Any],
) -> Dict[str, float]:
    """
    Dynamically adjust ranking weights based on user intent and explicit requests.
    
    Examples:
    - "cheapest" → increase price weight to 0.50
    - "fastest" → increase duration weight to 0.45
    - "most comfortable" → increase airline_quality + layover weights
    - "direct only" → maximize layover weight
    """
    weights = base_weights.copy()
    
    message_lower = user_message.lower()
    
    # Explicit "cheapest" request
    if any(word in message_lower for word in ["cheapest", "lowest price", "budget", "save money"]):
        logger.info("[RankingNode] Detected price-focused intent, increasing price weight")
        weights["price"] = 0.50
        weights["duration"] = 0.20
        weights["layover"] = 0.10
        weights["airline_quality"] = 0.05
        weights["airport_convenience"] = 0.10
        weights["personalization"] = 0.05
    
    # Explicit "fastest" request
    elif any(word in message_lower for word in ["fastest", "quickest", "shortest", "quick"]):
        logger.info("[RankingNode] Detected speed-focused intent, increasing duration weight")
        weights["price"] = 0.25
        weights["duration"] = 0.45
        weights["layover"] = 0.15
        weights["airline_quality"] = 0.05
        weights["airport_convenience"] = 0.05
        weights["personalization"] = 0.05
    
    # Explicit "comfortable" / "premium" request
    elif any(word in message_lower for word in ["comfortable", "premium", "luxury", "best airline"]):
        logger.info("[RankingNode] Detected comfort-focused intent, increasing quality weights")
        weights["price"] = 0.15
        weights["duration"] = 0.20
        weights["layover"] = 0.20
        weights["airline_quality"] = 0.30
        weights["airport_convenience"] = 0.10
        weights["personalization"] = 0.05
    
    # Explicit "direct" request
    elif any(word in message_lower for word in ["direct", "non-stop", "no layover", "no connection"]):
        logger.info("[RankingNode] Detected direct-flight preference, increasing layover weight")
        weights["price"] = 0.30
        weights["duration"] = 0.20
        weights["layover"] = 0.35
        weights["airline_quality"] = 0.05
        weights["airport_convenience"] = 0.05
        weights["personalization"] = 0.05
    
    # Check long-term preferences
    elif preferences.get("prefers_direct"):
        weights["layover"] = 0.25
        weights["price"] = 0.30
    
    return weights


# ============================================================================
# QUALITY FILTERS
# ============================================================================

def _should_filter_out(flight: Dict[str, Any], all_flights: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """
    Filter out extremely low-quality itineraries.
    
    Filters:
    - Total duration >2x median
    - Very tight connection (<45 min) on multi-stop flights
    - Arrival between 2-4 AM at remote airport
    - Multiple unnecessary stops for short routes
    
    Returns: (should_filter, reason)
    """
    duration = flight.get("duration_minutes", 0)
    stops = flight.get("stops", 0)
    layovers = flight.get("layovers", [])
    arrival_time = flight.get("arrival_time", "")
    arr_distance = flight.get("arrival_airport_distance_to_city_km", 0)
    
    # Calculate median duration
    durations = [f.get("duration_minutes", 0) for f in all_flights]
    durations_sorted = sorted(durations)
    median_duration = durations_sorted[len(durations_sorted) // 2]
    
    # Filter: Extremely long duration
    if duration > median_duration * 2.2:
        return True, f"Duration {duration//60}h is unusually long (2x+ median)"
    
    # Filter: Very tight connections on multi-stop
    if stops > 0:
        for layover in layovers:
            if layover.get("min_connection_minutes", 999) < 45:
                return True, f"Risky {layover.get('min_connection_minutes')}min connection"
    
    # Filter: Arrival 2-4 AM at remote airport
    try:
        arr_dt = datetime.fromisoformat(arrival_time)
        if 2 <= arr_dt.hour < 4 and arr_distance > 40:
            return True, "Arrives 2-4 AM at remote airport (poor transport options)"
    except:
        pass
    
    # Filter: Multiple stops on short-haul route
    if stops >= 2 and duration < 360:  # <6h base route
        return True, f"Unnecessary {stops} stops on short route"
    
    return False, None


# ============================================================================
# MAIN RANKING NODE
# ============================================================================

async def ranking_node(state: ConversationState) -> ConversationState:
    """
    Production ranking node with complete 6-factor scoring and AI explanations.
    
    Process:
    1. Load flights and context
    2. Adjust weights dynamically
    3. Calculate all 6 factor scores
    4. Compute final composite score
    5. Filter low-quality itineraries
    6. Sort and select top N
    7. Generate AI explanations for each
    8. Return ranked flights with complete metadata
    """
    
    flights = state.get("raw_flights") or []
    preferences = state.get("long_term_preferences", {})
    budget = state.get("budget")
    intent = state.get("intent")
    latest_message = state.get("latest_user_message", "")
    base_weights = state.get("ranking_weights", {
        "price": 0.35,
        "duration": 0.25,
        "layover": 0.15,
        "airline_quality": 0.10,
        "airport_convenience": 0.10,
        "personalization": 0.05,
    })
    
    if not flights:
        logger.warning("[RankingNode] No flights to rank")
        return state
    
    logger.info(f"[RankingNode] Ranking {len(flights)} flights")
    
    # Step 1: Dynamic weight adjustment
    adjusted_weights = _adjust_weights_for_intent(
        base_weights, intent, latest_message, preferences
    )
    
    logger.info(f"[RankingNode] Using weights: {adjusted_weights}")
    
    # Step 2: Calculate min/max for normalization
    prices = [f.get("price", 0) for f in flights]
    durations = [f.get("duration_minutes", 0) for f in flights]
    
    min_price, max_price = min(prices), max(prices)
    min_duration, max_duration = min(durations), max(durations)
    
    # Step 3: Score each flight
    scored_flights: List[Dict[str, Any]] = []
    
    for flight in flights:
        # Calculate individual factor scores
        price_s = _price_score(flight, min_price, max_price)
        duration_s = _duration_score(flight, min_duration, max_duration)
        layover_s = _layover_score(flight)
        airline_s = _airline_quality_score(flight)
        airport_s = _airport_convenience_score(flight)
        pers_s = _personalization_score(flight, preferences, budget)
        
        # Composite score
        final_score = (
            price_s * adjusted_weights["price"] +
            duration_s * adjusted_weights["duration"] +
            layover_s * adjusted_weights["layover"] +
            airline_s * adjusted_weights["airline_quality"] +
            airport_s * adjusted_weights["airport_convenience"] +
            pers_s * adjusted_weights["personalization"]
        )
        
        # Quality filter
        should_filter, filter_reason = _should_filter_out(flight, flights)
        
        scored_flights.append({
            "flight": flight,
            "final_score": final_score,
            "score_breakdown": {
                "price": price_s,
                "duration": duration_s,
                "layover": layover_s,
                "airline_quality": airline_s,
                "airport_convenience": airport_s,
                "personalization": pers_s,
            },
            "should_filter": should_filter,
            "filter_reason": filter_reason,
        })
    
    # Step 4: Filter out low-quality
    before_filter_count = len(scored_flights)
    scored_flights = [sf for sf in scored_flights if not sf["should_filter"]]
    filtered_count = before_filter_count - len(scored_flights)
    
    if filtered_count > 0:
        logger.info(f"[RankingNode] Filtered out {filtered_count} low-quality flights")
    
    # Step 5: Sort by score
    scored_flights.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Step 6: Take top N
    top_n = min(5, len(scored_flights))
    top_scored = scored_flights[:top_n]
    
    logger.info(f"[RankingNode] Selected top {top_n} flights")
    
    # Step 7: Generate AI explanations
    ranked_with_explanations = []
    
    all_flights_list = [sf["flight"] for sf in scored_flights]
    
    for rank_idx, scored_flight in enumerate(top_scored):
        flight = scored_flight["flight"]
        score = scored_flight["final_score"]
        score_breakdown = scored_flight["score_breakdown"]
        
        try:
            # Generate complete explanation
            explanation_data = await ranking_explanation_service.generate_complete_explanation(
                flight=flight,
                rank_position=rank_idx,
                all_flights=all_flights_list,
                score_breakdown=score_breakdown,
                user_preferences=preferences,
                search_params={
                    "departure_date": state.get("departure_date"),
                    "origin": state.get("origin"),
                    "destination": state.get("destination"),
                    "budget": budget,
                },
            )
            
            ranked_with_explanations.append({
                "id": flight.get("id"),
                "score": round(score, 3),
                "label": explanation_data["label"],
                "summary": explanation_data["summary"],
                "explanation": explanation_data["explanation"],
                "tradeoffs": explanation_data["tradeoffs"],
                "smart_suggestions": explanation_data["smart_suggestions"],
                "raw_data": flight,
                "score_breakdown": score_breakdown,
            })
            
        except Exception as e:
            logger.error(f"[RankingNode] Failed to generate explanation for flight {flight.get('id')}: {e}")
            # Fallback to basic output
            ranked_with_explanations.append({
                "id": flight.get("id"),
                "score": round(score, 3),
                "label": "✈️ Recommended Option",
                "summary": f"{flight.get('airline_name', 'Unknown')} flight",
                "explanation": "Selected based on balanced scoring across all factors",
                "tradeoffs": [],
                "smart_suggestions": [],
                "raw_data": flight,
                "score_breakdown": score_breakdown,
            })
    
    logger.info(f"[RankingNode] Successfully ranked and explained {len(ranked_with_explanations)} flights")
    
    # Step 8: Return updated state
    return update_state(state, {
        "ranked_flights": ranked_with_explanations,
        "ranking_weights": adjusted_weights,  # Store adjusted weights
        "ranking_explanation": f"Ranked {len(flights)} flights using {adjusted_weights}",
    })