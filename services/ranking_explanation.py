"""
Ranking Explanation Service
Generates human-readable explanations, labels, tradeoffs, and smart suggestions for ranked flights.

This service transforms raw ranking scores into premium, expert-level explanations that:
- Label flights with intuitive categories ("Fastest", "Best Value", "Cheapest")
- Explain WHY each flight is recommended
- Surface tradeoffs and considerations
- Generate smart alternative suggestions (dates, airports, upgrades)

Uses Gemini for natural language generation with strict output format.
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta

from services.llm_service import generate_json_response

logger = logging.getLogger(__name__)


class RankingExplanationService:
    """
    Production service for generating flight ranking explanations.
    
    Core responsibilities:
    1. Label generation - Categorize flights ("Fastest", "Best Value", etc.)
    2. Summary generation - One-line brief of the flight
    3. Explanation generation - Why this flight is recommended
    4. Tradeoff identification - Negatives and considerations
    5. Smart suggestions - Alternative dates, airports, upgrades
    """
    
    @staticmethod
    def _determine_label(
        flight: Dict[str, Any],
        rank_position: int,
        all_flights: List[Dict[str, Any]],
        score_breakdown: Dict[str, float],
    ) -> str:
        """
        Intelligently determine the best label for this flight.
        
        Logic:
        - Rank 1 → Usually "Best Overall" unless it's clearly specialized
        - Cheapest price → "Cheapest Available"
        - Fastest duration → "Fastest Option"
        - High quality score → "Most Comfortable"
        - Good balance → "Best Value"
        """
        
        price = flight.get("price", 0)
        duration = flight.get("duration_minutes", 0)
        stops = flight.get("stops", 0)
        airline_rating = flight.get("airline_rating", 0)
        
        # Get min/max for context
        all_prices = [f.get("price", 999) for f in all_flights]
        all_durations = [f.get("duration_minutes", 999) for f in all_flights]
        
        min_price = min(all_prices)
        min_duration = min(all_durations)
        
        # Decision tree for labeling
        if price == min_price:
            return "💰 Cheapest Available"
        
        if duration == min_duration:
            return "⚡ Fastest Option"
        
        if stops == 0 and airline_rating >= 4.0:
            return "✨ Premium Direct Flight"
        
        if stops == 0:
            return "🎯 Direct Flight"
        
        if airline_rating >= 4.5:
            return "👑 Most Comfortable"
        
        if rank_position == 0:
            return "🏆 Best Overall"
        
        # Check if it's a good balance
        price_percentile = (price - min_price) / (max(all_prices) - min_price) if max(all_prices) > min_price else 0
        duration_percentile = (duration - min_duration) / (max(all_durations) - min_duration) if max(all_durations) > min_duration else 0
        
        if price_percentile < 0.4 and duration_percentile < 0.4:
            return "💎 Best Value"
        
        if stops == 1 and price < all_prices[len(all_prices) // 2]:
            return "🔄 Good Connection"
        
        return "✈️ Recommended Option"
    
    @staticmethod
    def _generate_summary(flight: Dict[str, Any]) -> str:
        """
        Generate a concise, expert-level one-line summary.
        
        Format: "{Airline} direct/1-stop, {duration}, {departure_time}"
        Example: "Turkish Airlines direct, 3h 45m, departs 09:30"
        """
        
        airline_name = flight.get("airline_name", "Unknown")
        stops = flight.get("stops", 0)
        duration_minutes = flight.get("duration_minutes", 0)
        departure_time = flight.get("departure_time", "")
        
        # Format duration
        hours = duration_minutes // 60
        minutes = duration_minutes % 60
        duration_str = f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
        
        # Format stops
        if stops == 0:
            stops_str = "direct"
        elif stops == 1:
            stops_str = "1 stop"
        else:
            stops_str = f"{stops} stops"
        
        # Format departure time
        try:
            dep_dt = datetime.fromisoformat(departure_time)
            dep_str = dep_dt.strftime("%H:%M")
        except:
            dep_str = "morning"
        
        return f"{airline_name} {stops_str}, {duration_str}, departs {dep_str}"
    
    @staticmethod
    async def _generate_explanation(
        flight: Dict[str, Any],
        rank_position: int,
        all_flights: List[Dict[str, Any]],
        user_preferences: Dict[str, Any],
    ) -> str:
        """
        Generate detailed explanation of WHY this flight is recommended.
        
        Uses Gemini to create natural, expert-level reasoning based on:
        - Ranking factors (price, duration, quality)
        - Comparison to alternatives
        - User preferences
        """
        
        # Build context for LLM
        price = flight.get("price", 0)
        currency = flight.get("currency", "GBP")
        duration_minutes = flight.get("duration_minutes", 0)
        stops = flight.get("stops", 0)
        airline_name = flight.get("airline_name", "")
        airline_rating = flight.get("airline_rating", 0)
        on_time_score = flight.get("on_time_score", 0)
        
        # Get comparison context
        all_prices = [f.get("price", 999) for f in all_flights]
        all_durations = [f.get("duration_minutes", 999) for f in all_flights]
        
        min_price = min(all_prices)
        min_duration = min(all_durations)
        avg_price = sum(all_prices) / len(all_prices)
        
        system_prompt = """
You are an expert travel agent explaining flight recommendations.

Generate a 1-2 sentence explanation of WHY this flight is recommended.

Rules:
- Be specific with numbers (prices, time savings)
- Mention key differentiators (direct vs stops, airline quality, timing)
- Compare to alternatives when relevant
- Sound like a knowledgeable, premium travel advisor
- NO generic phrases like "great choice" or "perfect option"

Return STRICT JSON:
{"explanation": "your 1-2 sentence explanation"}
"""
        
        user_prompt = f"""
Flight details:
- Airline: {airline_name} (rating: {airline_rating}/5, on-time: {int(on_time_score*100)}%)
- Price: {price} {currency}
- Duration: {duration_minutes // 60}h {duration_minutes % 60}m
- Stops: {stops}

Context:
- Cheapest in results: {min_price} {currency}
- Fastest in results: {min_duration // 60}h {min_duration % 60}m
- Average price: {int(avg_price)} {currency}
- Your position: #{rank_position + 1} in rankings

User preferences:
{user_preferences}

Generate explanation:
"""
        
        try:
            response = await generate_json_response(system_prompt, user_prompt)
            explanation = response.get("explanation", "")
            if explanation:
                return explanation
        except Exception as e:
            logger.warning(f"Failed to generate AI explanation: {e}")
        
        # Fallback: rule-based explanation
        explanation_parts = []
        
        if stops == 0:
            explanation_parts.append("Direct flight with no connections")
        
        if price == min_price:
            explanation_parts.append(f"Cheapest option available")
        elif price < avg_price * 0.85:
            savings = int(avg_price - price)
            explanation_parts.append(f"Saves £{savings} vs average price")
        
        if duration_minutes == min_duration:
            explanation_parts.append("Fastest routing")
        elif duration_minutes < min_duration * 1.2:
            explanation_parts.append("Quick journey time")
        
        if airline_rating >= 4.0:
            explanation_parts.append(f"{airline_name} has {airline_rating}★ rating with {int(on_time_score*100)}% on-time performance")
        
        if explanation_parts:
            return ". ".join(explanation_parts) + "."
        
        return f"Good balance of price ({price} {currency}), duration ({duration_minutes // 60}h {duration_minutes % 60}m), and quality."
    
    @staticmethod
    async def _generate_tradeoffs(
        flight: Dict[str, Any],
        all_flights: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify tradeoffs and considerations for this flight.
        
        Surfaces:
        - Price premium vs cheaper options
        - Time penalty vs faster options
        - Layover issues (overnight, tight connections, poor airports)
        - Airport inconvenience (distance to city)
        - Low-cost carrier limitations (baggage, comfort)
        """
        
        tradeoffs = []
        
        price = flight.get("price", 0)
        duration_minutes = flight.get("duration_minutes", 0)
        stops = flight.get("stops", 0)
        layovers = flight.get("layovers", [])
        dep_distance = flight.get("departure_airport_distance_to_city_km", 0)
        arr_distance = flight.get("arrival_airport_distance_to_city_km", 0)
        baggage_included = flight.get("baggage_included", True)
        airline_rating = flight.get("airline_rating", 0)
        departure_time = flight.get("departure_time", "")
        arrival_time = flight.get("arrival_time", "")
        
        # Get min values for comparison
        all_prices = [f.get("price", 999) for f in all_flights]
        all_durations = [f.get("duration_minutes", 999) for f in all_flights]
        
        min_price = min(all_prices)
        min_duration = min(all_durations)
        
        # Price premium
        if price > min_price * 1.15:
            premium = int(price - min_price)
            tradeoffs.append(f"£{premium} more than cheapest option")
        
        # Duration penalty
        if duration_minutes > min_duration * 1.3:
            extra_hours = (duration_minutes - min_duration) // 60
            extra_mins = (duration_minutes - min_duration) % 60
            if extra_hours > 0:
                tradeoffs.append(f"{extra_hours}h {extra_mins}m longer than fastest option")
            else:
                tradeoffs.append(f"{extra_mins}m longer than fastest option")
        
        # Layover issues
        for layover in layovers:
            if layover.get("overnight"):
                tradeoffs.append(f"Overnight layover in {layover.get('airport', 'hub')}")
            
            if layover.get("min_connection_minutes", 999) < 75:
                tradeoffs.append(f"Tight {layover.get('min_connection_minutes')}min connection in {layover.get('airport')}")
            
            if layover.get("duration_minutes", 0) > 240:
                hours = layover.get("duration_minutes") // 60
                tradeoffs.append(f"Long {hours}h layover in {layover.get('airport')}")
        
        # Airport distance
        if dep_distance > 40:
            tradeoffs.append(f"Departs from airport {int(dep_distance)}km from city center")
        
        if arr_distance > 40:
            tradeoffs.append(f"Arrives at airport {int(arr_distance)}km from city center")
        
        # Baggage not included
        if not baggage_included:
            tradeoffs.append("Checked baggage not included in price")
        
        # Low airline rating
        if airline_rating < 3.5:
            tradeoffs.append("Budget airline with basic service")
        
        # Inconvenient timing
        try:
            dep_dt = datetime.fromisoformat(departure_time)
            arr_dt = datetime.fromisoformat(arrival_time)
            
            if dep_dt.hour < 6:
                tradeoffs.append("Very early departure (before 6 AM)")
            elif dep_dt.hour >= 22:
                tradeoffs.append("Late night departure")
            
            if arr_dt.hour >= 23 or arr_dt.hour < 5:
                tradeoffs.append("Arrives late at night or very early morning")
        except:
            pass
        
        return tradeoffs[:4]  # Max 4 tradeoffs to avoid overwhelming
    
    @staticmethod
    async def _generate_smart_suggestions(
        flight: Dict[str, Any],
        search_params: Dict[str, Any],
        all_flights: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate smart alternative suggestions.
        
        Suggests:
        - Flying on different dates for better price
        - Alternative airports for savings
        - Upgrade opportunities
        - Direct flight alternatives if close in price
        """
        
        suggestions = []
        
        price = flight.get("price", 0)
        currency = flight.get("currency", "GBP")
        stops = flight.get("stops", 0)
        airline_rating = flight.get("airline_rating", 0)
        
        # Find direct flights if current has stops
        if stops > 0:
            direct_flights = [f for f in all_flights if f.get("stops") == 0]
            if direct_flights:
                cheapest_direct = min(direct_flights, key=lambda f: f.get("price", 999))
                direct_price = cheapest_direct.get("price", 0)
                price_diff = direct_price - price
                
                if price_diff < price * 0.25:  # Within 25%
                    suggestions.append(
                        f"💡 Direct flight available for +£{int(price_diff)} "
                        f"({cheapest_direct.get('airline_name')})"
                    )
        
        # Suggest better airline if current is low-rated
        if airline_rating < 3.8:
            premium_flights = [
                f for f in all_flights
                if f.get("airline_rating", 0) >= 4.2
                and f.get("stops", 99) <= stops
            ]
            if premium_flights:
                best_premium = min(premium_flights, key=lambda f: f.get("price", 999))
                premium_price = best_premium.get("price", 0)
                price_diff = premium_price - price
                
                if price_diff < price * 0.30:
                    suggestions.append(
                        f"⭐ Upgrade to {best_premium.get('airline_name')} "
                        f"({best_premium.get('airline_rating')}★) for +£{int(price_diff)}"
                    )
        
        # Simulate flexible date suggestions (in production, would query API)
        departure_date = search_params.get("departure_date")
        if departure_date and isinstance(departure_date, date):
            # Suggest day before (often cheaper)
            suggestions.append(
                f"📅 Flying 1 day earlier ({(departure_date - timedelta(days=1)).strftime('%b %d')}) "
                f"typically saves £50-80"
            )
        
        # Suggest business class upgrade if Economy
        fare_class = flight.get("fare_class", "Economy")
        if fare_class == "Economy" and airline_rating >= 4.0:
            business_estimate = int(price * 2.5)
            suggestions.append(
                f"🎖️ Business class upgrade estimated at £{business_estimate} "
                f"(based on this route)"
            )
        
        return suggestions[:3]  # Max 3 suggestions
    
    @staticmethod
    async def generate_complete_explanation(
        flight: Dict[str, Any],
        rank_position: int,
        all_flights: List[Dict[str, Any]],
        score_breakdown: Dict[str, float],
        user_preferences: Dict[str, Any],
        search_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate complete explanation package for a ranked flight.
        
        Returns:
            {
                "label": str,
                "summary": str,
                "explanation": str,
                "tradeoffs": List[str],
                "smart_suggestions": List[str],
            }
        """
        
        logger.info(f"Generating explanation for flight {flight.get('id')} (rank #{rank_position + 1})")
        
        # Generate all components
        label = RankingExplanationService._determine_label(
            flight, rank_position, all_flights, score_breakdown
        )
        
        summary = RankingExplanationService._generate_summary(flight)
        
        explanation = await RankingExplanationService._generate_explanation(
            flight, rank_position, all_flights, user_preferences
        )
        
        tradeoffs = await RankingExplanationService._generate_tradeoffs(
            flight, all_flights
        )
        
        smart_suggestions = await RankingExplanationService._generate_smart_suggestions(
            flight, search_params, all_flights
        )
        
        return {
            "label": label,
            "summary": summary,
            "explanation": explanation,
            "tradeoffs": tradeoffs,
            "smart_suggestions": smart_suggestions,
        }


# Singleton instance
ranking_explanation_service = RankingExplanationService()