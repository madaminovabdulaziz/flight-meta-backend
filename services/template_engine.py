"""
Template Engine (UI-Ready Version)
==================================
Generates context-aware questions and structured "Pills" for the frontend.
Matches the 'Travo' UX style (Label + Icon + Value).

Features:
1. Structured Suggestions (Pills) for React UI.
2. Context Awareness (Trip Reason, Group Type).
3. Aviation SME Logic (MCT checks, Red-eye detection).
"""

from typing import Dict, Any, List, Optional
from datetime import date, timedelta

class TemplateEngine:
    """
    Generate all deterministic text without LLM.
    """
    
    # ========================================
    # 1. CONTEXT-AWARE QUESTION GENERATION
    # ========================================
    
    @staticmethod
    def generate_question(missing_param: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a personalized question with UI-ready pills.
        
        Args:
            missing_param: The slot to fill (e.g., 'departure_date')
            state: Current conversation state
        
        Returns:
            Dict with 'message', 'placeholder', and 'suggestions' (List of Dicts).
        """
        
        # Extract Context
        dest = state.get("destination", "your destination")
        origin = state.get("origin", "your origin")
        
        # Context variables (populated by RuleBasedExtractor)
        reason = state.get("trip_reason")       # e.g., "culture", "business"
        group = state.get("travel_group")       # e.g., "friends", "family"
        prefs = state.get("long_term_preferences", {})

        # --- A. DEPARTURE DATE (Calendar Pills) ---
        if missing_param == "departure_date":
            if reason == "culture":
                msg = f"I bet {dest} has amazing museums! When are you planning to explore?"
            elif group == "friends":
                msg = f"A trip to {dest} with friends sounds fun! When is the group planning to fly?"
            elif reason == "business":
                msg = f"Understood. When do you need to be in {dest} for work?"
            else:
                msg = f"Perfect! When would you like to fly to {dest}?"

            return {
                "message": msg,
                "placeholder": "Select dates",
                "suggestions": [
                    {"label": "Next weekend", "value": "next_weekend", "icon": "📅", "type": "date"},
                    {"label": "Next month", "value": "next_month", "icon": "📅", "type": "date"},
                    {"label": "Flexible dates", "value": "flexible", "icon": "✨", "type": "action"}
                ]
            }

        # --- B. DESTINATION (Map Pin Pills) ---
        if missing_param == "destination":
            return {
                "message": "Where is your next adventure?",
                "placeholder": "Search destination...",
                "suggestions": [
                    {"label": "Istanbul", "value": "Istanbul", "icon": "🇹🇷", "type": "location"},
                    {"label": "Dubai", "value": "Dubai", "icon": "🇦🇪", "type": "location"},
                    {"label": "Paris", "value": "Paris", "icon": "🇫🇷", "type": "location"},
                    {"label": "Inspire me", "value": "anywhere", "icon": "🌍", "type": "action"}
                ]
            }

        # --- C. ORIGIN (Location Pills) ---
        if missing_param == "origin":
            # Use preferences if available
            history = prefs.get("preferred_airports", [])
            suggestions = []
            
            if history:
                for apt in history[:2]:
                    suggestions.append({"label": apt, "value": apt, "icon": "🛫", "type": "location"})
            else:
                suggestions = [
                    {"label": "London (LHR)", "value": "LHR", "icon": "🇬🇧", "type": "location"},
                    {"label": "New York (JFK)", "value": "JFK", "icon": "🇺🇸", "type": "location"}
                ]
                
            return {
                "message": f"Got it. Which city or airport are you flying from to get to {dest}?",
                "placeholder": "Departure city",
                "suggestions": suggestions
            }

        # --- D. PASSENGERS (User Icon Pills) ---
        if missing_param == "passengers":
            msg = "How many people are traveling?"
            if group == "friends": msg = "How many friends are joining you?"
            if group == "family": msg = "How many adults and children?"

            return {
                "message": msg,
                "placeholder": "Select travelers",
                "suggestions": [
                    {"label": "Solo Trip", "value": "1", "icon": "👤", "type": "pax"},
                    {"label": "Partner Trip", "value": "2", "icon": "💑", "type": "pax"},
                    {"label": "3 Friends", "value": "3", "icon": "👯", "type": "pax"},
                    {"label": "Family (4)", "value": "4", "icon": "👨‍👩‍👧‍👦", "type": "pax"}
                ]
            }

        # --- E. RETURN DATE ---
        if missing_param == "return_date":
            return {
                "message": "When would you like to return?",
                "placeholder": "Return date",
                "suggestions": [
                    {"label": "One week later", "value": "1_week", "icon": "📅", "type": "date"},
                    {"label": "Same weekend", "value": "same_weekend", "icon": "📅", "type": "date"},
                    {"label": "One-way", "value": "oneway", "icon": "➡️", "type": "action"}
                ]
            }
            
        # --- F. BUDGET ---
        if missing_param == "budget":
            return {
                "message": "Do you have a specific budget in mind?",
                "placeholder": "Max price",
                "suggestions": [
                    {"label": "Budget-friendly", "value": "300", "icon": "💸", "type": "budget"},
                    {"label": "Standard", "value": "600", "icon": "💳", "type": "budget"},
                    {"label": "Luxury", "value": "1500", "icon": "💎", "type": "budget"}
                ]
            }

        # Fallback
        return {
            "message": f"Could you please provide the {missing_param.replace('_', ' ')}?",
            "placeholder": "Type here...",
            "suggestions": []
        }


    # ========================================
    # 2. FLIGHT LABELING (UX Badges)
    # ========================================
    
    @staticmethod
    def generate_label(flight: Dict[str, Any], rank: int, all_flights: List[Dict[str, Any]]) -> str:
        """Generate an emoji badge for the flight card."""
        price = flight.get("price", 0)
        duration = flight.get("duration_minutes", 0)
        stops = flight.get("stops", 0)
        rating = flight.get("airline_rating", 0)
        
        # Context stats
        prices = [f.get("price", 999999) for f in all_flights]
        durations = [f.get("duration_minutes", 999999) for f in all_flights]
        min_price = min(prices) if prices else 0
        min_duration = min(durations) if durations else 0
        
        if price == min_price: return "💰 Cheapest"
        if duration == min_duration: return "⚡ Fastest"
        if stops == 0 and rating >= 4.5: return "👑 Premium Direct"
        
        # Best Value Logic
        max_p = max(prices) if prices else 1
        max_d = max(durations) if durations else 1
        p_score = (price - min_price) / (max_p - min_price) if max_p > min_price else 0
        d_score = (duration - min_duration) / (max_d - min_duration) if max_d > min_duration else 0
        
        if p_score < 0.3 and d_score < 0.3: return "💎 Best Value"
        if stops == 0: return "🎯 Direct"
        
        return "✈️ Recommended"


    # ========================================
    # 3. FLIGHT SUMMARIES & TRADEOFFS (SME Logic)
    # ========================================

    @staticmethod
    def generate_summary(flight: Dict[str, Any]) -> str:
        """Generate a one-line summary string."""
        airline = flight.get("airline_name", "Unknown")
        stops = flight.get("stops", 0)
        duration = flight.get("duration_minutes", 0)
        
        h, m = divmod(duration, 60)
        dur_str = f"{h}h {m}m" if m else f"{h}h"
        stop_str = "Direct" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"
        
        dep = flight.get("departure_time", "").split("T")[1][:5] if "T" in flight.get("departure_time", "") else "??"
        
        return f"{airline} • {stop_str} • {dur_str} • {dep}"

    @staticmethod
    def generate_tradeoffs(flight: Dict[str, Any], all_flights: List[Dict[str, Any]]) -> List[str]:
        """
        Generate honest tradeoffs (Cons) for the UI.
        Includes Aviation SME logic (MCT, red-eyes, etc.)
        """
        tradeoffs = []
        
        price = flight.get("price", 0)
        layovers = flight.get("layovers", [])
        dep_time = flight.get("departure_time", "")
        
        prices = [f.get("price", 0) for f in all_flights]
        min_price = min(prices) if prices else 0
        
        # 1. Price Premium
        if price > min_price * 1.20:
            diff = int(price - min_price)
            tradeoffs.append(f"£{diff} more expensive")
            
        # 2. Aviation SME: Risky Connections & Long Layovers
        for layover in layovers:
            dur = layover.get("duration_minutes", 0)
            loc = layover.get("airport", "HUB")
            
            # Minimum Connection Time Warning (< 60 mins is risky in big hubs)
            if dur < 60:
                tradeoffs.append(f"⚠️ Risky short connection in {loc} ({dur}m)")
            elif dur > 300:
                tradeoffs.append(f"Long layover in {loc} ({dur//60}h)")
                
        # 3. Red-Eye Flights
        if "T" in dep_time:
            try:
                hour = int(dep_time.split("T")[1][:2])
                if hour < 6:
                    tradeoffs.append("Early morning departure")
                elif hour >= 22:
                    tradeoffs.append("Late night departure")
            except:
                pass
                
        return tradeoffs[:3]


# ============================================================
# SINGLETON INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================

template_engine = TemplateEngine()

def generate_question(missing_param: str, state: Dict[str, Any]) -> Dict[str, Any]:
    return template_engine.generate_question(missing_param, state)

def generate_flight_label(flight: Dict[str, Any], rank: int, all_flights: List[Dict[str, Any]]) -> str:
    return template_engine.generate_label(flight, rank, all_flights)

def generate_flight_summary(flight: Dict[str, Any]) -> str:
    return template_engine.generate_summary(flight)

def generate_flight_tradeoffs(flight: Dict[str, Any], all_flights: List[Dict[str, Any]]) -> List[str]:
    return template_engine.generate_tradeoffs(flight, all_flights)