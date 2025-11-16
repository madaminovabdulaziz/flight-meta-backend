"""
Cheap Ranker
Fully deterministic flight ranking - ZERO LLM calls.

Implements complete 6-factor scoring + template-based outputs.
Uses template_engine for labels, summaries, tradeoffs.
Uses simple rules for smart suggestions.

This replaces expensive LLM-based ranking with deterministic logic,
achieving consistent quality at <50ms and $0 cost.

Usage:
    from services.cheap_ranker import cheap_ranker
    
    ranked = cheap_ranker.rank_flights(raw_flights, state)
    # Returns fully structured output ready for frontend
"""

import logging
from typing import List, Dict, Any, Optional

from services.template_engine import template_engine

logger = logging.getLogger(__name__)


class CheapRanker:
    """
    Deterministic flight ranking without LLM.
    
    Features:
    - 6-factor scoring (price, duration, layover, airline, airport, personalization)
    - Template-based labels, summaries, tradeoffs
    - Rule-based smart suggestions
    - Low-quality flight filtering
    """
    
    # ========================================
    # SCORING FUNCTIONS
    # ========================================
    
    @staticmethod
    def _normalize(value: float, min_v: float, max_v: float) -> float:
        """Normalize value to 0-1 range (lower value = higher score)."""
        if max_v <= min_v + 1e-6:
            return 0.5
        return max(0.0, min(1.0, 1.0 - (value - min_v) / (max_v - min_v)))
    
    @staticmethod
    def _price_score(flight: Dict[str, Any], min_price: float, max_price: float) -> float:
        """Factor 1: Price (weight 0.35)."""
        price = flight.get("price", 0.0)
        return CheapRanker._normalize(price, min_price, max_price)
    
    @staticmethod
    def _duration_score(flight: Dict[str, Any], min_dur: float, max_dur: float) -> float:
        """Factor 2: Duration (weight 0.25)."""
        duration = flight.get("duration_minutes", 0)
        return CheapRanker._normalize(duration, min_dur, max_dur)
    
    @staticmethod
    def _layover_score(flight: Dict[str, Any]) -> float:
        """
        Factor 3: Layover (weight 0.15).
        
        Penalties:
        - Each stop: -0.3
        - Overnight layover: -0.2
        - Risky connection (<60min): -0.3
        - Long layover (>4h): -0.1
        """
        stops = flight.get("stops", 0)
        layovers = flight.get("layovers", [])
        
        score = 1.0
        score -= stops * 0.3
        
        for layover in layovers:
            if layover.get("overnight"):
                score -= 0.2
            
            connection = layover.get("min_connection_minutes", 999)
            if connection < 60:
                score -= 0.3
            elif connection < 90:
                score -= 0.1
            
            duration = layover.get("duration_minutes", 0)
            if duration > 240:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def _airline_quality_score(flight: Dict[str, Any]) -> float:
        """
        Factor 4: Airline Quality (weight 0.10).
        
        Combines rating (60%) + on-time (40%).
        """
        rating = flight.get("airline_rating")
        on_time = flight.get("on_time_score", 0.5)
        
        if rating is None:
            rating_norm = 0.5
        else:
            rating_norm = (rating - 1.0) / 4.0  # 1-5 scale → 0-1
            rating_norm = max(0.0, min(1.0, rating_norm))
        
        return 0.6 * rating_norm + 0.4 * on_time
    
    @staticmethod
    def _airport_convenience_score(flight: Dict[str, Any]) -> float:
        """
        Factor 5: Airport Convenience (weight 0.10).
        
        Based on distance to city:
        - ≤10km: 1.0
        - 10-30km: 0.8
        - 30-50km: 0.5
        - >50km: 0.2
        """
        dep_dist = flight.get("departure_airport_distance_to_city_km", 30)
        arr_dist = flight.get("arrival_airport_distance_to_city_km", 30)
        
        def dist_to_score(km: Optional[float]) -> float:
            if km is None:
                return 0.5
            if km <= 10:
                return 1.0
            elif km <= 30:
                return 0.8
            elif km <= 50:
                return 0.5
            else:
                return 0.2
        
        return (dist_to_score(dep_dist) + dist_to_score(arr_dist)) / 2
    
    @staticmethod
    def _personalization_score(flight: Dict[str, Any], prefs: Dict[str, Any]) -> float:
        """
        Factor 6: Personalization (weight 0.05).
        
        Bonuses:
        - Preferred airline: +0.2
        - Within budget: +0.2
        - Preferred time of day: +0.1
        """
        score = 0.5  # Neutral baseline
        
        # Check airline preference
        airline_code = flight.get("airline", "")
        preferred_airlines = prefs.get("preferred_airlines", [])
        if airline_code in preferred_airlines:
            score += 0.2
        
        # Check budget
        price = flight.get("price", 0)
        max_budget = prefs.get("budget_range", [0, 999999])[1] if isinstance(prefs.get("budget_range"), list) else 999999
        if price <= max_budget:
            score += 0.2
        
        # Check time preference
        preferred_time = prefs.get("preferred_time_of_day", "")
        if preferred_time:
            dep_time = flight.get("departure_time", "")
            if "T" in dep_time:
                hour = int(dep_time.split("T")[1][:2])
                
                if preferred_time == "morning" and 6 <= hour < 12:
                    score += 0.1
                elif preferred_time == "afternoon" and 12 <= hour < 18:
                    score += 0.1
                elif preferred_time == "evening" and 18 <= hour < 22:
                    score += 0.1
        
        return min(1.0, score)
    
    # ========================================
    # MAIN RANKING FUNCTION
    # ========================================
    
    @staticmethod
    def rank_flights(
        raw_flights: List[Dict[str, Any]],
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Rank flights using 6-factor scoring + generate full output.
        
        Args:
            raw_flights: List of normalized flight objects
            state: Current conversation state
        
        Returns:
            {
                "ranked_flights": List of top flights with scores and explanations,
                "smart_suggestions": List of alternative recommendations,
                "summary_stats": Dict with price range, duration range, etc.
            }
        """
        
        if not raw_flights:
            logger.warning("[CheapRanker] No flights to rank")
            return {
                "ranked_flights": [],
                "smart_suggestions": [],
                "summary_stats": {}
            }
        
        logger.info(f"[CheapRanker] Ranking {len(raw_flights)} flights")
        
        # ========================================
        # STEP 1: Calculate Min/Max for Normalization
        # ========================================
        
        prices = [f.get("price", 0) for f in raw_flights]
        durations = [f.get("duration_minutes", 0) for f in raw_flights]
        
        min_price = min(prices)
        max_price = max(prices)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # ========================================
        # STEP 2: Get Weights (with dynamic adjustment)
        # ========================================
        
        weights = state.get("ranking_weights", {
            "price": 0.35,
            "duration": 0.25,
            "layover": 0.15,
            "airline_quality": 0.10,
            "airport_convenience": 0.10,
            "personalization": 0.05,
        })
        
        # ========================================
        # STEP 3: Score Each Flight
        # ========================================
        
        prefs = state.get("long_term_preferences", {})
        scored_flights = []
        
        for flight in raw_flights:
            # Calculate individual factor scores
            price_sc = CheapRanker._price_score(flight, min_price, max_price)
            duration_sc = CheapRanker._duration_score(flight, min_duration, max_duration)
            layover_sc = CheapRanker._layover_score(flight)
            airline_sc = CheapRanker._airline_quality_score(flight)
            airport_sc = CheapRanker._airport_convenience_score(flight)
            personal_sc = CheapRanker._personalization_score(flight, prefs)
            
            # Calculate final rank score
            rank = (
                price_sc * weights["price"] +
                duration_sc * weights["duration"] +
                layover_sc * weights["layover"] +
                airline_sc * weights["airline_quality"] +
                airport_sc * weights["airport_convenience"] +
                personal_sc * weights["personalization"]
            )
            
            scored_flights.append({
                "flight": flight,
                "rank": rank,
                "scores": {
                    "price": price_sc,
                    "duration": duration_sc,
                    "layover": layover_sc,
                    "airline_quality": airline_sc,
                    "airport_convenience": airport_sc,
                    "personalization": personal_sc,
                }
            })
        
        # ========================================
        # STEP 4: Sort and Filter
        # ========================================
        
        # Sort by rank (highest first)
        scored_flights.sort(key=lambda x: x["rank"], reverse=True)
        
        # Filter out low-quality options (rank < 0.3)
        scored_flights = [f for f in scored_flights if f["rank"] >= 0.3]
        
        # Keep top 5
        top_flights = scored_flights[:5]
        
        logger.info(f"[CheapRanker] Top {len(top_flights)} flights selected")
        
        # ========================================
        # STEP 5: Generate Labels, Summaries, Tradeoffs (Template-Based)
        # ========================================
        
        ranked_output = []
        
        for idx, scored in enumerate(top_flights):
            flight = scored["flight"]
            
            # Generate label
            label = template_engine.generate_label(flight, idx, raw_flights)
            
            # Generate summary
            summary = template_engine.generate_summary(flight)
            
            # Generate tradeoffs
            tradeoffs = template_engine.generate_tradeoffs(flight, raw_flights)
            
            ranked_output.append({
                "id": flight.get("id"),
                "rank_score": round(scored["rank"], 3),
                "label": label,
                "summary": summary,
                "tradeoffs": tradeoffs,
                "scores": scored["scores"],
                "raw_data": flight
            })
        
        # ========================================
        # STEP 6: Generate Smart Suggestions (Rule-Based)
        # ========================================
        
        suggestions = CheapRanker._generate_suggestions(raw_flights, state)
        
        # ========================================
        # STEP 7: Summary Stats
        # ========================================
        
        summary_stats = {
            "total_flights": len(raw_flights),
            "ranked_count": len(ranked_output),
            "price_range": [min_price, max_price],
            "duration_range": [min_duration, max_duration],
        }
        
        return {
            "ranked_flights": ranked_output,
            "smart_suggestions": suggestions,
            "summary_stats": summary_stats
        }
    
    @staticmethod
    def _generate_suggestions(
        raw_flights: List[Dict[str, Any]],
        state: Dict[str, Any]
    ) -> List[str]:
        """
        Generate smart alternative suggestions using rules.
        
        Examples:
        - "Fly 1 day earlier to save £50"
        - "Consider LGW instead of LHR for more options"
        - "Direct flights available for £40 more"
        """
        suggestions = []
        
        # Check for direct flight availability
        has_direct = any(f.get("stops", 0) == 0 for f in raw_flights)
        all_with_stops = all(f.get("stops", 0) > 0 for f in raw_flights[:5])
        
        if all_with_stops and not has_direct:
            suggestions.append("No direct flights found for these dates")
        
        # Check price variance
        prices = [f.get("price", 0) for f in raw_flights]
        if prices:
            price_range = max(prices) - min(prices)
            if price_range > 100:
                suggestions.append("Consider flexible dates to find better prices")
        
        # Check for overnight layovers
        has_overnight = any(
            any(l.get("overnight") for l in f.get("layovers", []))
            for f in raw_flights[:5]
        )
        if has_overnight:
            suggestions.append("Some options have overnight layovers - check times carefully")
        
        return suggestions[:3]  # Max 3 suggestions


# ============================================================
# SINGLETON INSTANCE
# ============================================================

cheap_ranker = CheapRanker()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def rank_flights_deterministic(
    raw_flights: List[Dict[str, Any]],
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """Convenience function for deterministic ranking."""
    return cheap_ranker.rank_flights(raw_flights, state)