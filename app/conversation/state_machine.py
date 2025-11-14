# app/conversation/state_machine.py
"""
State Machine for Conversational Flight Search
Implements optimized 5-6 step flow with smart skip logic.
Enhanced with logging, validation, and better state messages.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.conversation.models import (
    ConversationFlowState,
    ConversationContext,
    TripSpec,
    Suggestion,
    SuggestionType
)

logger = logging.getLogger(__name__)


class StateMachineError(Exception):
    """Raised when invalid state transition is attempted"""
    pass


class StateMachine:
    """
    Manages conversation flow state transitions.
    Enforces validation and skip logic for optimal UX.
    """
    
    def __init__(self):
        """Initialize state machine with valid transitions"""
        # Define valid state transitions
        self.transitions = {
            ConversationFlowState.INIT: [
                ConversationFlowState.CONFIRM_DESTINATION,
                ConversationFlowState.DATES,
                ConversationFlowState.ERROR
            ],
            ConversationFlowState.CONFIRM_DESTINATION: [
                ConversationFlowState.DATES,
                ConversationFlowState.ERROR
            ],
            ConversationFlowState.DATES: [
                ConversationFlowState.PASSENGERS,
                ConversationFlowState.PREFERENCES,
                ConversationFlowState.CONFIRMATION,
                ConversationFlowState.ERROR
            ],
            ConversationFlowState.PASSENGERS: [
                ConversationFlowState.PREFERENCES,
                ConversationFlowState.CONFIRMATION,
                ConversationFlowState.ERROR
            ],
            ConversationFlowState.PREFERENCES: [
                ConversationFlowState.CONFIRMATION,
                ConversationFlowState.ERROR
            ],
            ConversationFlowState.CONFIRMATION: [
                ConversationFlowState.SEARCH_READY,
                ConversationFlowState.DATES,
                ConversationFlowState.PASSENGERS,
                ConversationFlowState.PREFERENCES,
                ConversationFlowState.ERROR
            ],
            ConversationFlowState.SEARCH_READY: [
                ConversationFlowState.INIT,  # Start new search
                ConversationFlowState.ERROR
            ],
            ConversationFlowState.ERROR: [
                ConversationFlowState.INIT  # Recover by restarting
            ]
        }
        
        logger.debug("✓ StateMachine initialized")
    
    def get_next_state(
        self,
        current_state: ConversationFlowState,
        trip_spec: TripSpec,
        user_action: Optional[str] = None
    ) -> ConversationFlowState:
        """
        Determine next state based on current state, trip spec, and user action.
        
        Implements smart skip logic:
        - Skip CONFIRM_DESTINATION if both origin and destination are explicit
        - Skip PREFERENCES if user wants to proceed directly
        - Jump to CONFIRMATION once all required fields are present
        
        Args:
            current_state: Current conversation state
            trip_spec: Current trip specification
            user_action: Optional user action hint ("skip", "edit", etc.)
            
        Returns:
            Next conversation state
        """
        logger.debug(
            f"Determining next state: current={current_state}, "
            f"has_origin={bool(trip_spec.origin)}, "
            f"has_dest={bool(trip_spec.destination)}, "
            f"has_date={bool(trip_spec.depart_date)}, "
            f"action={user_action}"
        )
        
        # Handle user-requested actions
        if user_action == "skip":
            if current_state == ConversationFlowState.PREFERENCES:
                logger.info("User skipped preferences")
                return ConversationFlowState.CONFIRMATION
            elif current_state == ConversationFlowState.PASSENGERS:
                logger.info("User skipped passengers (using default)")
                return ConversationFlowState.PREFERENCES
        
        if user_action == "edit":
            # Return to appropriate edit state based on what's being edited
            logger.info(f"User requested edit from {current_state}")
            return current_state  # Handled by caller
        
        # State-specific transition logic
        if current_state == ConversationFlowState.INIT:
            # Check if we have both origin and destination
            if trip_spec.origin and trip_spec.destination:
                # Skip confirmation, go straight to dates
                logger.info("Fast-forwarding: origin and destination present")
                return ConversationFlowState.DATES
            elif trip_spec.destination:
                # Have destination, need to confirm/collect origin
                logger.info("Destination set, confirming origin")
                return ConversationFlowState.CONFIRM_DESTINATION
            else:
                # Need to collect destination
                return ConversationFlowState.CONFIRM_DESTINATION
        
        elif current_state == ConversationFlowState.CONFIRM_DESTINATION:
            # After confirming/correcting origin+destination, go to dates
            logger.info("Origin confirmed, moving to dates")
            return ConversationFlowState.DATES
        
        elif current_state == ConversationFlowState.DATES:
            # After collecting dates, check if we can skip to confirmation
            if self._has_minimum_required_fields(trip_spec):
                logger.info("All required fields present, going to passengers")
                return ConversationFlowState.PASSENGERS
            return ConversationFlowState.PASSENGERS
        
        elif current_state == ConversationFlowState.PASSENGERS:
            # Offer preferences but allow skip
            return ConversationFlowState.PREFERENCES
        
        elif current_state == ConversationFlowState.PREFERENCES:
            # After preferences (or skip), go to final confirmation
            return ConversationFlowState.CONFIRMATION
        
        elif current_state == ConversationFlowState.CONFIRMATION:
            # User confirmed, ready to search
            logger.info("User confirmed search parameters")
            return ConversationFlowState.SEARCH_READY
        
        elif current_state == ConversationFlowState.SEARCH_READY:
            # Reset to start new search
            logger.info("Search completed, resetting to INIT")
            return ConversationFlowState.INIT
        
        elif current_state == ConversationFlowState.ERROR:
            # Recover from error by restarting
            logger.info("Recovering from error state")
            return ConversationFlowState.INIT
        
        # Default: stay in current state
        logger.warning(f"No transition rule matched, staying in {current_state}")
        return current_state
    
    def can_transition(
        self,
        from_state: ConversationFlowState,
        to_state: ConversationFlowState
    ) -> bool:
        """
        Check if transition is valid.
        
        Args:
            from_state: Current state
            to_state: Desired next state
            
        Returns:
            True if transition is allowed
        """
        if from_state not in self.transitions:
            logger.warning(f"Unknown state: {from_state}")
            return False
        
        is_valid = to_state in self.transitions[from_state]
        
        if not is_valid:
            logger.warning(
                f"Invalid transition: {from_state} → {to_state}"
            )
        
        return is_valid
    
    def transition_to(
        self,
        context: ConversationContext,
        target_state: ConversationFlowState,
        force: bool = False
    ) -> bool:
        """
        Safely transition to a new state with validation.
        
        Args:
            context: Current conversation context
            target_state: Desired state
            force: If True, skip validation (use carefully)
            
        Returns:
            True if transition succeeded
            
        Raises:
            StateMachineError: If transition is invalid and not forced
        """
        current = context.state
        
        if not force and not self.can_transition(current, target_state):
            raise StateMachineError(
                f"Invalid transition: {current} → {target_state}"
            )
        
        # Log transition
        logger.info(f"State transition: {current} → {target_state}")
        
        context.state = target_state
        return True
    
    def get_state_message(
        self,
        state: ConversationFlowState,
        trip_spec: TripSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get the appropriate message for a given state.
        
        Args:
            state: Current conversation state
            trip_spec: Current trip specification
            context: Additional context for message personalization
            
        Returns:
            Message to display to user
        """
        context = context or {}
        
        messages = {
            ConversationFlowState.INIT: "Hi! Where would you like to fly? ✈️",
            
            ConversationFlowState.CONFIRM_DESTINATION: self._get_confirm_message(trip_spec, context),
            
            ConversationFlowState.DATES: self._get_dates_message(trip_spec),
            
            ConversationFlowState.PASSENGERS: "How many people are traveling?",
            
            ConversationFlowState.PREFERENCES: "Any preferences? (You can skip this)",
            
            ConversationFlowState.CONFIRMATION: self._get_confirmation_message(trip_spec),
            
            ConversationFlowState.SEARCH_READY: "Great! Searching for flights...",
            
            ConversationFlowState.ERROR: "Sorry, something went wrong. Let's start over."
        }
        
        return messages.get(state, "How can I help you?")
    
    def get_state_placeholder(self, state: ConversationFlowState) -> Optional[str]:
        """
        Get input placeholder text for a given state.
        
        Args:
            state: Current conversation state
            
        Returns:
            Placeholder text or None
        """
        placeholders = {
            ConversationFlowState.INIT: "e.g., Tokyo, London, Dubai",
            ConversationFlowState.CONFIRM_DESTINATION: "Confirm or type correct city",
            ConversationFlowState.DATES: "e.g., next weekend, Dec 25, tomorrow",
            ConversationFlowState.PASSENGERS: "e.g., 2 adults, 1 child, just me",
            ConversationFlowState.PREFERENCES: "e.g., direct flights only, morning departure",
            ConversationFlowState.CONFIRMATION: "Type 'confirm' or make changes",
        }
        
        return placeholders.get(state)
    
    def _has_minimum_required_fields(self, trip_spec: TripSpec) -> bool:
        """
        Check if trip spec has minimum required fields for search.
        
        Args:
            trip_spec: Trip specification to check
            
        Returns:
            True if all required fields are present
        """
        has_required = bool(
            trip_spec.origin and
            trip_spec.destination and
            trip_spec.depart_date
        )
        
        logger.debug(f"Has minimum fields: {has_required}")
        return has_required
    
    def _get_confirm_message(
        self,
        trip_spec: TripSpec,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate confirmation message for CONFIRM_DESTINATION state.
        
        Args:
            trip_spec: Current trip specification
            context: Additional context (detected_origin, etc.)
            
        Returns:
            Personalized confirmation message
        """
        detected_origin = context.get("detected_origin")
        destination = trip_spec.destination
        
        if detected_origin and destination:
            return f"Flying from {detected_origin} to {destination}?"
        elif detected_origin and not destination:
            return f"I detected you're near {detected_origin}. Where would you like to fly to?"
        elif destination and not trip_spec.origin:
            return f"You want to fly to {destination}. Where are you flying from?"
        else:
            return "Where would you like to fly to?"
    
    def _get_dates_message(self, trip_spec: TripSpec) -> str:
        """
        Generate date selection message.
        
        Args:
            trip_spec: Current trip specification
            
        Returns:
            Date prompt message
        """
        if trip_spec.destination:
            return f"When would you like to fly to {trip_spec.destination}?"
        else:
            return "When would you like to fly?"
    
    def _get_confirmation_message(self, trip_spec: TripSpec) -> str:
        """
        Generate final confirmation summary.
        
        Args:
            trip_spec: Complete trip specification
            
        Returns:
            Summary message
        """
        parts = []
        
        # Route
        if trip_spec.origin and trip_spec.destination:
            parts.append(f"✈️ {trip_spec.origin} → {trip_spec.destination}")
        
        # Dates
        if trip_spec.depart_date:
            date_str = f"📅 {trip_spec.depart_date}"
            if trip_spec.return_date:
                date_str += f" to {trip_spec.return_date}"
            parts.append(date_str)
        
        # Passengers
        if trip_spec.passengers > 1:
            parts.append(f"👥 {trip_spec.passengers} passengers")
        
        # Cabin class
        if trip_spec.cabin_class and trip_spec.cabin_class != "economy":
            parts.append(f"💺 {trip_spec.cabin_class.replace('_', ' ').title()}")
        
        # Preferences
        if trip_spec.preferences:
            pref_list = []
            if trip_spec.preferences.get("direct_only"):
                pref_list.append("direct flights")
            if trip_spec.preferences.get("time_preference"):
                pref_list.append(trip_spec.preferences["time_preference"])
            if pref_list:
                parts.append(f"⚙️ {', '.join(pref_list)}")
        
        summary = "\n".join(parts)
        return f"Ready to search?\n\n{summary}\n\nType 'confirm' to proceed or make changes."


# ============================================================
# HELPER FUNCTION
# ============================================================

def should_fast_forward_to_confirmation(trip_spec: TripSpec) -> bool:
    """
    Check if we have enough data to skip intermediate steps.
    
    Args:
        trip_spec: Current trip specification
        
    Returns:
        True if we can fast-forward to confirmation
    """
    can_fast_forward = bool(
        trip_spec.origin and
        trip_spec.destination and
        trip_spec.depart_date and
        trip_spec.origin != trip_spec.destination
    )
    
    if can_fast_forward:
        logger.info("Fast-forwarding to confirmation (all required fields present)")
    
    return can_fast_forward


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'StateMachine',
    'StateMachineError',
    'should_fast_forward_to_confirmation'
]