# app/conversation/suggestion_engine.py
"""
Suggestion Engine for Conversational Search
Generates contextual suggestion chips based on state and trip spec.
Primary mode: Rule-based (90% of cases)
Fallback: LLM-powered for complex cases
Enhanced with better suggestions and AI integration.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone

from app.conversation.models import (
    ConversationFlowState,
    ConversationContext,
    TripSpec,
    Suggestion,
    SuggestionType,
    CabinClass
)
from app.core.config import settings

logger = logging.getLogger(__name__)

# Tashkent timezone
TASHKENT_TZ = timezone(timedelta(hours=5))


class SuggestionEngine:
    """
    Generates suggestion chips for user interaction.
    Prioritizes rule-based suggestions for speed and reliability.
    Falls back to AI for complex/ambiguous cases.
    """
    
    def __init__(self, ai_adapter=None):
        """
        Initialize suggestion engine.
        
        Args:
            ai_adapter: Optional AI adapter for complex suggestions
        """
        self.ai_adapter = ai_adapter
        
        # Popular destinations by region
        self.popular_destinations = {
            "TAS": [  # From Tashkent
                {"code": "DXB", "name": "Dubai", "flag": "🇦🇪"},
                {"code": "IST", "name": "Istanbul", "flag": "🇹🇷"},
                {"code": "SVO", "name": "Moscow", "flag": "🇷🇺"},
                {"code": "DEL", "name": "Delhi", "flag": "🇮🇳"},
                {"code": "BKK", "name": "Bangkok", "flag": "🇹🇭"},
                {"code": "LHR", "name": "London", "flag": "🇬🇧"}
            ],
            "DXB": [  # From Dubai
                {"code": "LHR", "name": "London", "flag": "🇬🇧"},
                {"code": "JFK", "name": "New York", "flag": "🇺🇸"},
                {"code": "BKK", "name": "Bangkok", "flag": "🇹🇭"},
                {"code": "SIN", "name": "Singapore", "flag": "🇸🇬"},
                {"code": "CDG", "name": "Paris", "flag": "🇫🇷"},
                {"code": "HKG", "name": "Hong Kong", "flag": "🇭🇰"}
            ],
            "IST": [  # From Istanbul
                {"code": "LHR", "name": "London", "flag": "🇬🇧"},
                {"code": "CDG", "name": "Paris", "flag": "🇫🇷"},
                {"code": "DXB", "name": "Dubai", "flag": "🇦🇪"},
                {"code": "FRA", "name": "Frankfurt", "flag": "🇩🇪"},
                {"code": "AMS", "name": "Amsterdam", "flag": "🇳🇱"},
                {"code": "FCO", "name": "Rome", "flag": "🇮🇹"}
            ],
            "default": [  # Fallback
                {"code": "DXB", "name": "Dubai", "flag": "🇦🇪"},
                {"code": "IST", "name": "Istanbul", "flag": "🇹🇷"},
                {"code": "LHR", "name": "London", "flag": "🇬🇧"},
                {"code": "CDG", "name": "Paris", "flag": "🇫🇷"},
                {"code": "JFK", "name": "New York", "flag": "🇺🇸"}
            ]
        }
        
        logger.debug("✓ SuggestionEngine initialized")
    
    async def generate_suggestions(
        self,
        context: ConversationContext,
        user_input: Optional[str] = None
    ) -> List[Suggestion]:
        """
        Generate contextual suggestions based on current state.
        
        Args:
            context: Current conversation context
            user_input: Optional user input for context-aware suggestions
            
        Returns:
            List of suggestion chips
        """
        state = context.state
        trip_spec = context.trip_spec
        
        logger.debug(f"Generating suggestions for state: {state.value}")
        
        # Route to appropriate generator based on state
        if state == ConversationFlowState.INIT:
            return self._generate_init_suggestions(trip_spec, context.detected_origin)
        
        elif state == ConversationFlowState.CONFIRM_DESTINATION:
            return self._generate_confirmation_suggestions(trip_spec, context.detected_origin)
        
        elif state == ConversationFlowState.DATES:
            return self._generate_date_suggestions(trip_spec)
        
        elif state == ConversationFlowState.PASSENGERS:
            return self._generate_passenger_suggestions()
        
        elif state == ConversationFlowState.PREFERENCES:
            return self._generate_preference_suggestions(trip_spec)
        
        elif state == ConversationFlowState.CONFIRMATION:
            return self._generate_confirmation_action_suggestions()
        
        else:
            return []
    
    # ============================================================
    # STATE-SPECIFIC SUGGESTION GENERATORS
    # ============================================================
    
    def _generate_init_suggestions(
        self,
        trip_spec: TripSpec,
        detected_origin: Optional[str]
    ) -> List[Suggestion]:
        """Generate suggestions for INIT state"""
        suggestions = []
        
        # If we already have destination, skip popular destinations
        if trip_spec.destination:
            logger.debug("Destination already set, skipping init suggestions")
            return []
        
        # Get popular destinations based on detected origin
        origin_key = detected_origin if detected_origin in self.popular_destinations else "default"
        destinations = self.popular_destinations.get(origin_key, self.popular_destinations["default"])
        
        # Generate destination suggestions
        for dest in destinations[:5]:  # Top 5
            suggestions.append(Suggestion(
                type=SuggestionType.DESTINATION,
                label=f"{dest['flag']} {dest['name']}",
                value=dest["code"],
                metadata={"city": dest["name"], "code": dest["code"], "flag": dest["flag"]}
            ))
        
        logger.debug(f"Generated {len(suggestions)} init suggestions")
        return suggestions
    
    def _generate_confirmation_suggestions(
        self,
        trip_spec: TripSpec,
        detected_origin: Optional[str]
    ) -> List[Suggestion]:
        """Generate suggestions for CONFIRM_DESTINATION state"""
        suggestions = []
        
        # If we detected origin, offer to confirm or change
        if detected_origin and trip_spec.destination:
            suggestions.extend([
                Suggestion(
                    type=SuggestionType.CONFIRMATION,
                    label=f"✅ Yes, from {detected_origin}",
                    value={"action": "confirm", "origin": detected_origin},
                    metadata={"detected": True}
                ),
                Suggestion(
                    type=SuggestionType.CORRECTION,
                    label="🔄 Change origin",
                    value={"action": "change_origin"},
                    metadata={"show_input": True}
                )
            ])
        
        # Offer popular origins if no detection
        elif trip_spec.destination and not trip_spec.origin:
            # Major hub cities
            major_hubs = [
                {"code": "TAS", "name": "Tashkent", "flag": "🇺🇿"},
                {"code": "DXB", "name": "Dubai", "flag": "🇦🇪"},
                {"code": "IST", "name": "Istanbul", "flag": "🇹🇷"},
                {"code": "LHR", "name": "London", "flag": "🇬🇧"}
            ]
            
            for hub in major_hubs[:3]:
                suggestions.append(Suggestion(
                    type=SuggestionType.DESTINATION,
                    label=f"{hub['flag']} {hub['name']}",
                    value=hub["code"],
                    metadata={"city": hub["name"], "code": hub["code"]}
                ))
        
        logger.debug(f"Generated {len(suggestions)} confirmation suggestions")
        return suggestions
    
    def _generate_date_suggestions(self, trip_spec: TripSpec) -> List[Suggestion]:
        """Generate suggestions for DATES state"""
        today = datetime.now(TASHKENT_TZ).date()
        
        suggestions = [
            # Tomorrow
            Suggestion(
                type=SuggestionType.DATE_PRESET,
                label="🌅 Tomorrow",
                value={
                    "depart_date": (today + timedelta(days=1)).isoformat(),
                    "flexible_dates": False
                },
                metadata={"relative": "tomorrow"}
            ),
            
            # This weekend
            Suggestion(
                type=SuggestionType.DATE_PRESET,
                label="🎉 This weekend",
                value={
                    "depart_date": self._get_next_weekend().isoformat(),
                    "flexible_dates": False
                },
                metadata={"relative": "weekend"}
            ),
            
            # Next week
            Suggestion(
                type=SuggestionType.DATE_PRESET,
                label="📅 Next week",
                value={
                    "depart_date": (today + timedelta(days=7)).isoformat(),
                    "flexible_dates": True,
                    "flexibility_days": 3
                },
                metadata={"relative": "next_week"}
            ),
            
            # Next month
            Suggestion(
                type=SuggestionType.DATE_PRESET,
                label="🗓️ Next month",
                value={
                    "depart_date": (today + timedelta(days=30)).isoformat(),
                    "flexible_dates": True,
                    "flexibility_days": 7
                },
                metadata={"relative": "next_month"}
            ),
            
            # Flexible dates
            Suggestion(
                type=SuggestionType.QUICK_ACTION,
                label="🔀 I'm flexible",
                value={"action": "flexible_dates"},
                metadata={"show_calendar": True}
            )
        ]
        
        logger.debug(f"Generated {len(suggestions)} date suggestions")
        return suggestions
    
    def _generate_passenger_suggestions(self) -> List[Suggestion]:
        """Generate suggestions for PASSENGERS state"""
        suggestions = [
            Suggestion(
                type=SuggestionType.PASSENGER_COUNT,
                label="👤 Just me (1)",
                value=1,
                metadata={"type": "solo"}
            ),
            Suggestion(
                type=SuggestionType.PASSENGER_COUNT,
                label="👥 2 people",
                value=2,
                metadata={"type": "couple", "range": [2, 2]}
            ),
            Suggestion(
                type=SuggestionType.PASSENGER_COUNT,
                label="👨‍👩‍👧 3-4 people",
                value=3,
                metadata={"type": "family", "range": [3, 4]}
            ),
            Suggestion(
                type=SuggestionType.PASSENGER_COUNT,
                label="👨‍👩‍👧‍👦 5+ people",
                value=5,
                metadata={"type": "group", "range": [5, 9]}
            ),
        ]
        
        logger.debug(f"Generated {len(suggestions)} passenger suggestions")
        return suggestions
    
    def _generate_preference_suggestions(self, trip_spec: TripSpec) -> List[Suggestion]:
        """Generate suggestions for PREFERENCES state"""
        suggestions = [
            Suggestion(
                type=SuggestionType.PREFERENCE,
                label="💰 Cheapest options",
                value={"preferences": {"sort_by": "price"}},
                metadata={"priority": "price", "icon": "💰"}
            ),
            Suggestion(
                type=SuggestionType.PREFERENCE,
                label="✈️ Direct flights only",
                value={"preferences": {"direct_only": True, "max_stops": 0}},
                metadata={"filter": "direct", "icon": "✈️"}
            ),
            Suggestion(
                type=SuggestionType.PREFERENCE,
                label="🌅 Morning departures",
                value={"preferences": {"time_preference": "morning"}},
                metadata={"time": "morning", "icon": "🌅"}
            ),
            Suggestion(
                type=SuggestionType.PREFERENCE,
                label="💺 Business class",
                value={"preferences": {"cabin_class": "business"}},
                metadata={"cabin": "business", "icon": "💺"}
            ),
            Suggestion(
                type=SuggestionType.QUICK_ACTION,
                label="⏭️ Skip preferences",
                value={"action": "skip"},
                metadata={"skip": True}
            )
        ]
        
        logger.debug(f"Generated {len(suggestions)} preference suggestions")
        return suggestions
    
    def _generate_confirmation_action_suggestions(self) -> List[Suggestion]:
        """Generate suggestions for CONFIRMATION state"""
        suggestions = [
            Suggestion(
                type=SuggestionType.CONFIRMATION,
                label="✅ Search flights",
                value={"action": "confirm"},
                metadata={"primary": True}
            ),
            Suggestion(
                type=SuggestionType.CORRECTION,
                label="📅 Edit dates",
                value={"action": "edit", "field": "dates"},
                metadata={"edit_target": "dates"}
            ),
            Suggestion(
                type=SuggestionType.CORRECTION,
                label="👥 Edit passengers",
                value={"action": "edit", "field": "passengers"},
                metadata={"edit_target": "passengers"}
            ),
            Suggestion(
                type=SuggestionType.CORRECTION,
                label="⚙️ Edit preferences",
                value={"action": "edit", "field": "preferences"},
                metadata={"edit_target": "preferences"}
            )
        ]
        
        logger.debug(f"Generated {len(suggestions)} confirmation suggestions")
        return suggestions
    
    # ============================================================
    # LLM-POWERED SUGGESTIONS (FALLBACK)
    # ============================================================
    
    async def generate_ai_suggestions(
        self,
        context: ConversationContext,
        user_input: str
    ) -> List[Suggestion]:
        """
        Use LLM to generate suggestions for complex/ambiguous inputs.
        
        Args:
            context: Current conversation context
            user_input: User's complex input
            
        Returns:
            List of AI-generated suggestions
        """
        if not self.ai_adapter:
            logger.debug("AI adapter not available for suggestions")
            return []
        
        if not settings.ENABLE_SMART_SUGGESTIONS:
            logger.debug("Smart suggestions disabled in config")
            return []
        
        try:
            logger.info(f"Generating AI suggestions for: '{user_input[:50]}...'")
            
            # Use AI to understand intent and generate suggestions
            result = await self.ai_adapter.extract_preferences(user_input, context)
            
            suggestions = []
            for chip_label in result.get("chips", []):
                suggestions.append(Suggestion(
                    type=SuggestionType.QUICK_ACTION,
                    label=chip_label,
                    value={"action": chip_label.lower().replace(" ", "_")},
                    metadata={"ai_generated": True}
                ))
            
            logger.debug(f"Generated {len(suggestions)} AI suggestions")
            return suggestions
        
        except Exception as e:
            logger.error(f"AI suggestion generation failed: {e}", exc_info=True)
            return []
    
    # ============================================================
    # DYNAMIC SUGGESTION GENERATION
    # ============================================================
    
    def generate_destination_suggestions_for_origin(
        self,
        origin: str,
        limit: int = 5
    ) -> List[Suggestion]:
        """
        Generate destination suggestions based on specific origin.
        
        Args:
            origin: Origin airport code
            limit: Maximum number of suggestions
            
        Returns:
            List of destination suggestions
        """
        # Get destinations for this origin
        destinations = self.popular_destinations.get(
            origin,
            self.popular_destinations["default"]
        )
        
        suggestions = []
        for dest in destinations[:limit]:
            suggestions.append(Suggestion(
                type=SuggestionType.DESTINATION,
                label=f"{dest['flag']} {dest['name']}",
                value=dest["code"],
                metadata={"city": dest["name"], "code": dest["code"]}
            ))
        
        return suggestions
    
    def generate_round_trip_suggestions(
        self,
        depart_date: str,
        min_duration_days: int = 3,
        max_duration_days: int = 14
    ) -> List[Suggestion]:
        """
        Generate return date suggestions based on departure date.
        
        Args:
            depart_date: Departure date (ISO format)
            min_duration_days: Minimum trip duration
            max_duration_days: Maximum trip duration
            
        Returns:
            List of return date suggestions
        """
        try:
            depart = datetime.fromisoformat(depart_date).date()
        except ValueError:
            return []
        
        suggestions = []
        
        # Weekend trip (3 days)
        suggestions.append(Suggestion(
            type=SuggestionType.DATE_PRESET,
            label="🎉 Weekend (3 days)",
            value={"return_date": (depart + timedelta(days=3)).isoformat()},
            metadata={"duration_days": 3}
        ))
        
        # Week trip (7 days)
        suggestions.append(Suggestion(
            type=SuggestionType.DATE_PRESET,
            label="📅 One week",
            value={"return_date": (depart + timedelta(days=7)).isoformat()},
            metadata={"duration_days": 7}
        ))
        
        # Two weeks (14 days)
        suggestions.append(Suggestion(
            type=SuggestionType.DATE_PRESET,
            label="🗓️ Two weeks",
            value={"return_date": (depart + timedelta(days=14)).isoformat()},
            metadata={"duration_days": 14}
        ))
        
        return suggestions
    
    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================
    
    @staticmethod
    def _get_next_weekend() -> datetime.date:
        """
        Get the next Saturday date.
        
        Returns:
            Date of next Saturday
        """
        today = datetime.now(TASHKENT_TZ).date()
        days_ahead = 5 - today.weekday()  # Saturday = 5
        
        if days_ahead <= 0:
            days_ahead += 7
        
        return today + timedelta(days=days_ahead)
    
    def set_popular_destinations(self, destinations_by_origin: Dict[str, List[Dict[str, str]]]):
        """
        Update popular destinations list dynamically.
        
        Args:
            destinations_by_origin: Dict mapping origin codes to destination lists
        """
        self.popular_destinations.update(destinations_by_origin)
        logger.info(f"Updated popular destinations for {len(destinations_by_origin)} origins")
    
    def add_destination_for_origin(
        self,
        origin: str,
        destination: Dict[str, str]
    ) -> bool:
        """
        Add a single destination to an origin's popular list.
        
        Args:
            origin: Origin airport code
            destination: Destination dict with 'code', 'name', 'flag'
            
        Returns:
            True if added successfully
        """
        if origin not in self.popular_destinations:
            self.popular_destinations[origin] = []
        
        # Check if already exists
        for dest in self.popular_destinations[origin]:
            if dest["code"] == destination["code"]:
                logger.debug(f"Destination {destination['code']} already exists for {origin}")
                return False
        
        self.popular_destinations[origin].append(destination)
        logger.info(f"Added {destination['name']} to popular destinations for {origin}")
        return True


# ============================================================
# EXPORTS
# ============================================================

__all__ = ['SuggestionEngine']