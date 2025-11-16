"""
Template Engine
Deterministic text generation - NO LLM needed for questions, labels, summaries, tradeoffs.

This module generates ALL non-personalized text using templates and rules,
achieving consistent quality with <1ms latency and zero cost.

Usage:
    from services.template_engine import template_engine
    
    question = template_engine.generate_question("destination", state)
    label = template_engine.generate_label(flight, rank, all_flights)
    summary = template_engine.generate_summary(flight)
    tradeoffs = template_engine.generate_tradeoffs(flight, all_flights)
"""

from typing import Dict, Any, List


class TemplateEngine:
    """
    Generate all deterministic text without LLM.
    
    Handles:
    - Questions for missing parameters
    - Flight labels (Best Overall, Cheapest, etc.)
    - Flight summaries (one-line descriptions)
    - Tradeoffs (honest considerations)
    """
    
    # ========================================
    # QUESTION TEMPLATES
    # ========================================
    
    @staticmethod
    def generate_question(missing_param: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate question for missing parameter using templates.
        
        Args:
            missing_param: Name of missing parameter
            state: Current conversation state
        
        Returns:
            Dict with: message, placeholder, suggestions
        
        Example:
            >>> generate_question("destination", {})
            {
                "message": "Great! Where would you like to fly to?",
                "placeholder": "Where are you flying to?",
                "suggestions": ["Istanbul", "Dubai", "London", "Paris"]
            }
        """
        
        # Get context from state
        destination = state.get("destination", "your destination")
        prefs = state.get("long_term_preferences", {})
        
        templates = {
            "destination": {
                "message": "Great! Where would you like to fly to?",
                "placeholder": "Where are you flying to?",
                "suggestions": TemplateEngine._get_popular_destinations()
            },
            
            "departure_date": {
                "message": f"Perfect! When would you like to fly to {destination}?",
                "placeholder": "When do you want to fly?",
                "suggestions": ["This month", "Next month", "Give exact dates"]
            },
            
            "origin": {
                "message": "Which city or airport are you flying from?",
                "placeholder": "Departure city or airport",
                "suggestions": TemplateEngine._get_airport_suggestions(prefs)
            },
            
            "return_date": {
                "message": "When would you like to return? (or say 'one-way')",
                "placeholder": "Return date",
                "suggestions": ["Same week", "One week later", "Two weeks later", "One-way"]
            },
            
            "passengers": {
                "message": "How many passengers?",
                "placeholder": "Number of passengers",
                "suggestions": ["1", "2", "3", "4+"]
            },
            
            "travel_class": {
                "message": "Which class would you prefer?",
                "placeholder": "Travel class",
                "suggestions": ["Economy", "Premium Economy", "Business"]
            },
            
            "budget": {
                "message": "What's your maximum budget for this trip?",
                "placeholder": "Maximum budget (GBP)",
                "suggestions": ["£200", "£300", "£400", "£500+"]
            },
            
            "flexibility": {
                "message": "How flexible are you with dates?",
                "placeholder": "Flexibility in days",
                "suggestions": ["±1 day", "±2 days", "±3 days", "Exact dates only"]
            },
        }
        
        return templates.get(missing_param, templates["destination"])
    
    @staticmethod
    def _get_popular_destinations() -> List[str]:
        """Get list of popular destinations for suggestions."""
        return ["Istanbul", "Dubai", "London", "Paris"]
    
    @staticmethod
    def _get_airport_suggestions(prefs: Dict[str, Any]) -> List[str]:
        """
        Get personalized airport suggestions based on user preferences.
        
        Uses memory to suggest preferred airports.
        """
        preferred_airports = prefs.get("preferred_airports", [])
        
        if "LGW" in preferred_airports:
            return [
                "London Gatwick (LGW)",
                "London Heathrow (LHR)",
                "Other London airport"
            ]
        elif "LHR" in preferred_airports:
            return [
                "London Heathrow (LHR)",
                "London Gatwick (LGW)",
                "Other London airport"
            ]
        else:
            # Default suggestions
            return [
                "London Heathrow (LHR)",
                "London Gatwick (LGW)",
                "London Stansted (STN)"
            ]
    
    # ========================================
    # FLIGHT LABELS
    # ========================================
    
    @staticmethod
    def generate_label(
        flight: Dict[str, Any],
        rank: int,
        all_flights: List[Dict[str, Any]]
    ) -> str:
        """
        Generate flight label based on properties.
        
        Uses deterministic rules to categorize flights.
        
        Args:
            flight: Flight data
            rank: Position in ranking (0-indexed)
            all_flights: All available flights for comparison
        
        Returns:
            Label string with emoji
        
        Example:
            "💰 Cheapest Available"
            "⚡ Fastest Option"
            "🏆 Best Overall"
        """
        
        price = flight.get("price", 0)
        duration = flight.get("duration_minutes", 0)
        stops = flight.get("stops", 0)
        rating = flight.get("airline_rating", 0)
        
        # Get min values for comparison
        prices = [f.get("price", 999999) for f in all_flights]
        durations = [f.get("duration_minutes", 999999) for f in all_flights]
        
        min_price = min(prices) if prices else 0
        min_duration = min(durations) if durations else 0
        
        # Rule-based labeling (priority order)
        
        # 1. Cheapest
        if price == min_price:
            return "💰 Cheapest Available"
        
        # 2. Fastest
        if duration == min_duration:
            return "⚡ Fastest Option"
        
        # 3. Premium direct
        if stops == 0 and rating >= 4.0:
            return "✨ Premium Direct Flight"
        
        # 4. Direct (any quality)
        if stops == 0:
            return "🎯 Direct Flight"
        
        # 5. Most comfortable
        if rating >= 4.5:
            return "👑 Most Comfortable"
        
        # 6. Best overall (rank #1 that doesn't fit above)
        if rank == 0:
            return "🏆 Best Overall"
        
        # 7. Good value (low price percentile + low duration percentile)
        price_percentile = (price - min_price) / (max(prices) - min_price) if max(prices) > min_price else 0
        duration_percentile = (duration - min_duration) / (max(durations) - min_duration) if max(durations) > min_duration else 0
        
        if price_percentile < 0.4 and duration_percentile < 0.4:
            return "💎 Best Value"
        
        # 8. Good connection
        if stops == 1 and price < sum(prices) / len(prices):
            return "🔄 Good Connection"
        
        # 9. Default
        return "✈️ Recommended Option"
    
    # ========================================
    # FLIGHT SUMMARIES
    # ========================================
    
    @staticmethod
    def generate_summary(flight: Dict[str, Any]) -> str:
        """
        Generate one-line flight summary.
        
        Format: "{Airline} {stops}, {duration}, departs {time}"
        
        Args:
            flight: Flight data
        
        Returns:
            Summary string
        
        Example:
            "Turkish Airlines direct, 3h 45m, departs 09:30"
        """
        
        airline = flight.get("airline_name", "Unknown")
        stops = flight.get("stops", 0)
        duration_minutes = flight.get("duration_minutes", 0)
        departure_time = flight.get("departure_time", "")
        
        # Format duration
        hours = duration_minutes // 60
        minutes = duration_minutes % 60
        duration_str = f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
        
        # Format stops
        if stops == 0:
            stop_str = "direct"
        elif stops == 1:
            stop_str = "1 stop"
        else:
            stop_str = f"{stops} stops"
        
        # Extract time from ISO datetime
        try:
            # departure_time format: "2025-12-15T09:30:00"
            dep_time = departure_time.split("T")[1][:5] if "T" in departure_time else "unknown"
        except:
            dep_time = "morning"
        
        return f"{airline} {stop_str}, {duration_str}, departs {dep_time}"
    
    # ========================================
    # TRADEOFFS
    # ========================================
    
    @staticmethod
    def generate_tradeoffs(
        flight: Dict[str, Any],
        all_flights: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify flight tradeoffs using deterministic rules.
        
        Returns honest considerations about:
        - Price premium
        - Duration penalty
        - Layover issues
        - Airport inconvenience
        - Baggage fees
        - Airline quality
        
        Args:
            flight: Flight data
            all_flights: All available flights for comparison
        
        Returns:
            List of tradeoff strings (max 4)
        
        Example:
            [
                "£40 more than cheapest option",
                "Departs from airport 45km from city",
                "Checked baggage not included"
            ]
        """
        
        tradeoffs = []
        
        # Extract flight data
        price = flight.get("price", 0)
        duration = flight.get("duration_minutes", 0)
        stops = flight.get("stops", 0)
        layovers = flight.get("layovers", [])
        dep_distance = flight.get("departure_airport_distance_to_city_km", 0)
        arr_distance = flight.get("arrival_airport_distance_to_city_km", 0)
        baggage = flight.get("baggage_included", True)
        rating = flight.get("airline_rating", 0)
        departure_time = flight.get("departure_time", "")
        arrival_time = flight.get("arrival_time", "")
        
        # Get comparison values
        prices = [f.get("price", 999999) for f in all_flights]
        durations = [f.get("duration_minutes", 999999) for f in all_flights]
        
        min_price = min(prices) if prices else 0
        min_duration = min(durations) if durations else 0
        
        # ========================================
        # RULE 1: Price Premium
        # ========================================
        
        if price > min_price * 1.15:  # More than 15% over cheapest
            premium = int(price - min_price)
            tradeoffs.append(f"£{premium} more than cheapest option")
        
        # ========================================
        # RULE 2: Duration Penalty
        # ========================================
        
        if duration > min_duration * 1.3:  # More than 30% slower
            extra_minutes = duration - min_duration
            extra_hours = extra_minutes // 60
            extra_mins = extra_minutes % 60
            
            if extra_hours > 0:
                tradeoffs.append(f"{extra_hours}h {extra_mins}m longer than fastest option")
            else:
                tradeoffs.append(f"{extra_mins}m longer than fastest option")
        
        # ========================================
        # RULE 3: Layover Issues
        # ========================================
        
        for layover in layovers:
            # Overnight layover
            if layover.get("overnight"):
                airport = layover.get("airport", "hub")
                tradeoffs.append(f"Overnight layover in {airport}")
            
            # Tight connection
            connection_time = layover.get("min_connection_minutes", 999)
            if connection_time < 75:
                airport = layover.get("airport", "hub")
                tradeoffs.append(f"Tight {connection_time}min connection in {airport}")
            
            # Very long layover
            layover_duration = layover.get("duration_minutes", 0)
            if layover_duration > 240:  # >4 hours
                layover_hours = layover_duration // 60
                airport = layover.get("airport", "hub")
                tradeoffs.append(f"Long {layover_hours}h layover in {airport}")
        
        # ========================================
        # RULE 4: Airport Distance
        # ========================================
        
        if dep_distance > 40:
            tradeoffs.append(f"Departs from airport {int(dep_distance)}km from city center")
        
        if arr_distance > 40:
            tradeoffs.append(f"Arrives at airport {int(arr_distance)}km from city center")
        
        # ========================================
        # RULE 5: Baggage Not Included
        # ========================================
        
        if not baggage:
            tradeoffs.append("Checked baggage not included in price")
        
        # ========================================
        # RULE 6: Low Airline Rating
        # ========================================
        
        if rating < 3.5 and rating > 0:
            tradeoffs.append("Budget airline with basic service")
        
        # ========================================
        # RULE 7: Inconvenient Timing
        # ========================================
        
        try:
            # Parse times
            if "T" in departure_time:
                dep_hour = int(departure_time.split("T")[1][:2])
                
                if dep_hour < 6:
                    tradeoffs.append("Very early departure (before 6 AM)")
                elif dep_hour >= 22:
                    tradeoffs.append("Late night departure")
            
            if "T" in arrival_time:
                arr_hour = int(arrival_time.split("T")[1][:2])
                
                if arr_hour >= 23 or arr_hour < 5:
                    tradeoffs.append("Arrives late at night or very early morning")
        
        except:
            # Time parsing failed - skip timing tradeoffs
            pass
        
        # Return max 4 tradeoffs (most important ones)
        return tradeoffs[:4]


# ============================================================
# SINGLETON INSTANCE
# ============================================================

template_engine = TemplateEngine()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def generate_question(missing_param: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate question for missing parameter."""
    return template_engine.generate_question(missing_param, state)


def generate_flight_label(flight: Dict[str, Any], rank: int, all_flights: List[Dict[str, Any]]) -> str:
    """Generate flight label."""
    return template_engine.generate_label(flight, rank, all_flights)


def generate_flight_summary(flight: Dict[str, Any]) -> str:
    """Generate flight summary."""
    return template_engine.generate_summary(flight)


def generate_flight_tradeoffs(flight: Dict[str, Any], all_flights: List[Dict[str, Any]]) -> List[str]:
    """Generate flight tradeoffs."""
    return template_engine.generate_tradeoffs(flight, all_flights)