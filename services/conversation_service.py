# services/conversation_service.py
"""
Conversation Service - Main Orchestrator
Coordinates all components of the conversational flight search system.
Handles the complete conversation lifecycle from start to search execution.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta

from app.conversation.models import (
    ConversationContext,
    ConversationFlowState,
    TripSpec,
    Suggestion, 
    SearchParams,
    ValidationResult
)
from app.conversation.context_manager import ContextManager
from app.conversation.state_machine import StateMachine
from app.conversation.validators import StateValidator, InputSanitizer
from app.conversation.suggestion_engine import SuggestionEngine
from app.conversation.ai_adapter import AIAdapter, AIAdapterError
from app.infrastructure.geoip import IPGeolocationService
from app.core.config import settings

logger = logging.getLogger(__name__)

# Tashkent timezone (consistent with your Duffel code)
TASHKENT_TZ = timezone(timedelta(hours=5))


class ConversationService:
    """
    Main service orchestrating conversational flight search.
    Coordinates: AI extraction, state management, validation, suggestions, and persistence.
    """
    
    def __init__(
        self,
        context_manager: ContextManager,
        state_machine: StateMachine,
        validator: StateValidator,
        suggestion_engine: SuggestionEngine,
        ai_adapter: Optional[AIAdapter] = None,
        geoip_service: Optional[IPGeolocationService] = None
    ):
        """
        Initialize conversation service with all dependencies.
        
        Args:
            context_manager: Handles session persistence
            state_machine: Manages conversation flow states
            validator: Validates trip specifications
            suggestion_engine: Generates contextual suggestions
            ai_adapter: Optional AI adapter for NLP (uses Gemini)
            geoip_service: Optional IP geolocation service
        """
        self.context_manager = context_manager
        self.state_machine = state_machine
        self.validator = validator
        self.suggestion_engine = suggestion_engine
        self.ai_adapter = ai_adapter
        self.geoip_service = geoip_service
        
        logger.info("✓ ConversationService initialized")
    
    # ============================================================
    # PUBLIC API - MAIN CONVERSATION ENDPOINTS
    # ============================================================
    
    async def start_conversation(
        self,
        user_id: Optional[str] = None,
        initial_query: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new conversation session.
        
        Args:
            user_id: Optional user identifier
            initial_query: Optional initial query to process immediately
            client_ip: Client IP for geolocation
            
        Returns:
            Response dict with session info, message, and suggestions
        """
        logger.info(f"Starting new conversation (user={user_id}, ip={client_ip})")
        
        # Detect origin from IP if geolocation enabled
        detected_origin = None
        detected_origin_info = None
        
        if client_ip and self.geoip_service and settings.ENABLE_IP_GEOLOCATION:
            try:
                geoip_result = await self.geoip_service.get_nearest_airport_from_ip(client_ip)
                detected_origin = geoip_result["airport"]["iata"]
                detected_origin_info = geoip_result
                logger.info(f"✓ Detected origin: {detected_origin} from IP {client_ip}")
            except Exception as e:
                logger.warning(f"IP geolocation failed: {e}")
                detected_origin = "TAS"  # Fallback to Tashkent
        
        # Create new session
        context = await self.context_manager.create_session(
            user_id=user_id,
            detected_origin=detected_origin
        )
        
        # Set origin in trip_spec if detected
        if detected_origin:
            context.trip_spec.origin = detected_origin
            await self.context_manager.save_context(context)
        
        # Process initial query if provided
        if initial_query:
            logger.info(f"Processing initial query: '{initial_query}'")
            return await self.process_user_input(
                session_id=context.session_id,
                user_input=initial_query
            )
        
        # Generate initial message and suggestions
        message = self.state_machine.get_state_message(
            state=context.state,
            trip_spec=context.trip_spec,
            context={"detected_origin": detected_origin}
        )
        
        suggestions = await self.suggestion_engine.generate_suggestions(context)
        
        # Add assistant message to history
        await self.context_manager.add_message(
            session_id=context.session_id,
            role="assistant",
            content=message
        )
        
        response = {
            "session_id": context.session_id,
            "state": context.state,
            "message": message,
            "suggestions": [s.dict() for s in suggestions],
            "placeholder": self.state_machine.get_state_placeholder(context.state),
            "search_ready": False,
            "detected_origin": detected_origin_info
        }
        
        logger.info(f"✓ Started session {context.session_id} with state={context.state}")
        return response
    
    async def process_user_input(
        self,
        session_id: str,
        user_input: Optional[str] = None,
        selected_suggestion: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user input (text or suggestion click).
        Main conversation loop endpoint.
        
        Args:
            session_id: Session identifier
            user_input: Free-form text input
            selected_suggestion: Clicked suggestion data
            
        Returns:
            Response dict with updated state, message, and suggestions
        """
        logger.info(f"Processing input for session {session_id}")
        
        # Load conversation context
        context = await self.context_manager.get_context(session_id)
        if not context:
            logger.error(f"Session not found: {session_id}")
            return {
                "error": "Session not found",
                "message": "Your session has expired. Please start a new conversation."
            }
        
        # Add user message to history
        if user_input:
            await self.context_manager.add_message(
                session_id=session_id,
                role="user",
                content=user_input
            )
        
        # Process suggestion click or text input
        if selected_suggestion:
            logger.debug(f"Processing suggestion: {selected_suggestion}")
            await self._handle_suggestion(context, selected_suggestion)
        elif user_input:
            logger.debug(f"Processing text input: '{user_input}'")
            await self._handle_text_input(context, user_input)
        else:
            return {
                "error": "No input provided",
                "message": "Please provide either text input or select a suggestion."
            }
        
        # Validate current state
        validation = self.validator.validate_for_state(
            state=context.state,
            trip_spec=context.trip_spec
        )
        
        if not validation.is_valid:
            logger.warning(f"Validation failed: {validation.errors}")
            return await self._handle_validation_error(context, validation)
        
        # Determine next state
        next_state = self.state_machine.get_next_state(
            current_state=context.state,
            trip_spec=context.trip_spec
        )
        
        # Check if search is ready
        if next_state == ConversationFlowState.SEARCH_READY:
            return await self._finalize_search(context)
        
        # Transition to next state
        context.state = next_state
        await self.context_manager.save_context(context)
        
        # Generate response message
        message = self.state_machine.get_state_message(
            state=context.state,
            trip_spec=context.trip_spec,
            context={"detected_origin": context.detected_origin}
        )
        
        # Generate suggestions for next step
        suggestions = await self.suggestion_engine.generate_suggestions(context)
        
        # Add assistant message to history
        await self.context_manager.add_message(
            session_id=session_id,
            role="assistant",
            content=message
        )
        
        response = {
            "session_id": session_id,
            "state": context.state,
            "message": message,
            "suggestions": [s.dict() for s in suggestions],
            "placeholder": self.state_machine.get_state_placeholder(context.state),
            "search_ready": False,
            "trip_spec": context.trip_spec.dict() if context.trip_spec else None
        }
        
        logger.info(f"✓ Processed input for {session_id}, new state={context.state}")
        return response
    
    async def execute_search(self, session_id: str) -> Dict[str, Any]:
        """
        Execute flight search using collected conversation context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with search parameters ready for flight API
        """
        logger.info(f"Executing search for session {session_id}")
        
        # Load context
        context = await self.context_manager.get_context(session_id)
        if not context:
            return {
                "error": "Session not found",
                "message": "Your session has expired."
            }
        
        # Validate search readiness
        validation = self.validator.validate_search_ready(context.trip_spec)
        
        if not validation.is_valid:
            logger.warning(f"Search validation failed: {validation.errors}")
            return {
                "error": "validation_failed",
                "message": "Missing required information for search.",
                "validation_errors": validation.errors
            }
        
        # Build search parameters
        search_params = self._build_search_params(context.trip_spec)
        
        logger.info(
            f"✓ Search ready: {search_params['origin']} → {search_params['destination']} "
            f"on {search_params['depart_date']}"
        )
        
        return {
            "session_id": session_id,
            "search_params": search_params,
            "message": "Search parameters ready. Redirecting to results..."
        }
    
    async def get_conversation_history(self, session_id: str) -> Dict[str, Any]:
        """
        Retrieve full conversation history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with complete conversation context
        """
        context = await self.context_manager.get_context(session_id)
        
        if not context:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "state": context.state,
            "trip_spec": context.trip_spec.dict(),
            "messages": [m.dict() for m in context.messages],
            "detected_origin": context.detected_origin,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat()
        }
    
    async def reset_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Reset conversation to initial state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Response dict with fresh session
        """
        logger.info(f"Resetting conversation {session_id}")
        
        # Delete old session
        await self.context_manager.delete_session(session_id)
        
        # Create new session with same ID for continuity
        context = await self.context_manager.create_session()
        
        message = "Let's start fresh! Where would you like to fly?"
        
        await self.context_manager.add_message(
            session_id=context.session_id,
            role="assistant",
            content=message
        )
        
        suggestions = await self.suggestion_engine.generate_suggestions(context)
        
        return {
            "session_id": context.session_id,
            "state": context.state,
            "message": message,
            "suggestions": [s.dict() for s in suggestions],
            "search_ready": False
        }
    
    # ============================================================
    # PRIVATE HELPER METHODS
    # ============================================================
    
    async def _handle_text_input(
        self,
        context: ConversationContext,
        user_input: str
    ) -> None:
        """
        Process free-form text input using AI.
        
        Args:
            context: Current conversation context
            user_input: User's text input
        """
        # Sanitize input
        user_input = InputSanitizer.sanitize_text_input(user_input)
        
        if not self.ai_adapter:
            logger.warning("AI adapter not available, using fallback")
            await self._handle_text_fallback(context, user_input)
            return
        
        try:
            # Extract trip spec from text
            today = datetime.now(TASHKENT_TZ).date().isoformat()
            
            extracted_spec = await self.ai_adapter.extract_trip_spec_from_text(
                text=user_input,
                today=today,
                context=self._build_ai_context(context)
            )
            
            # Merge extracted data with existing trip_spec
            context.trip_spec = self._merge_trip_specs(
                base=context.trip_spec,
                updates=extracted_spec
            )
            
            await self.context_manager.save_context(context)
            
            logger.debug(f"✓ Extracted trip spec: {context.trip_spec.dict()}")
        
        except AIAdapterError as e:
            logger.error(f"AI extraction failed: {e}")
            await self._handle_text_fallback(context, user_input)
        
        except Exception as e:
            logger.exception(f"Unexpected error in text handling: {e}")
            await self._handle_text_fallback(context, user_input)
    
    async def _handle_text_fallback(
        self,
        context: ConversationContext,
        user_input: str
    ) -> None:
        """
        Fallback rule-based text processing when AI is unavailable.
        
        Args:
            context: Current conversation context
            user_input: User's text input
        """
        from app.conversation.ai_adapter import RuleBasedFallback
        
        logger.info("Using rule-based fallback for text processing")
        
        # Extract destination
        if not context.trip_spec.destination:
            destination = RuleBasedFallback.extract_destination(user_input)
            if destination:
                context.trip_spec.destination = destination
                logger.debug(f"✓ Extracted destination (fallback): {destination}")
        
        # Extract passengers
        passengers = RuleBasedFallback.extract_passengers(user_input)
        if passengers > 1:
            context.trip_spec.passengers = passengers
            logger.debug(f"✓ Extracted passengers (fallback): {passengers}")
        
        # Extract dates
        dates = RuleBasedFallback.extract_dates(user_input)
        if dates.get("depart_date"):
            context.trip_spec.depart_date = dates["depart_date"]
            logger.debug(f"✓ Extracted date (fallback): {dates['depart_date']}")
        
        await self.context_manager.save_context(context)
    
    async def _handle_suggestion(
        self,
        context: ConversationContext,
        suggestion: Dict[str, Any]
    ) -> None:
        """
        Process suggestion chip click.
        
        Args:
            context: Current conversation context
            suggestion: Suggestion data dict
        """
        suggestion_type = suggestion.get("type")
        value = suggestion.get("value")
        
        logger.debug(f"Handling suggestion: type={suggestion_type}, value={value}")
        
        # Apply suggestion value to trip_spec based on type
        if suggestion_type == "DESTINATION" and isinstance(value, str):
            context.trip_spec.destination = value
        
        elif suggestion_type == "DATE_PRESET" and isinstance(value, dict):
            if "depart_date" in value:
                context.trip_spec.depart_date = value["depart_date"]
            if "return_date" in value:
                context.trip_spec.return_date = value["return_date"]
            if "flexible_dates" in value:
                context.trip_spec.flexible_dates = value["flexible_dates"]
        
        elif suggestion_type == "PASSENGER_COUNT" and isinstance(value, int):
            context.trip_spec.passengers = value
        
        elif suggestion_type == "PREFERENCE" and isinstance(value, dict):
            # Merge preferences
            prefs = value.get("preferences", {})
            context.trip_spec.preferences.update(prefs)
        
        elif suggestion_type == "QUICK_ACTION":
            action = value.get("action") if isinstance(value, dict) else value
            if action == "skip":
                # User wants to skip preferences step
                pass
        
        await self.context_manager.save_context(context)
    
    async def _handle_validation_error(
        self,
        context: ConversationContext,
        validation: ValidationResult
    ) -> Dict[str, Any]:
        """
        Handle validation errors gracefully.
        
        Args:
            context: Current conversation context
            validation: Validation result with errors
            
        Returns:
            Response dict with error message
        """
        error_message = "I need to clarify a few things:\n" + "\n".join(
            f"• {error}" for error in validation.errors[:3]
        )
        
        await self.context_manager.add_message(
            session_id=context.session_id,
            role="assistant",
            content=error_message
        )
        
        suggestions = await self.suggestion_engine.generate_suggestions(context)
        
        return {
            "session_id": context.session_id,
            "state": context.state,
            "message": error_message,
            "validation_errors": validation.errors,
            "suggestions": [s.dict() for s in suggestions],
            "search_ready": False
        }
    
    async def _finalize_search(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Finalize search and prepare parameters.
        
        Args:
            context: Current conversation context
            
        Returns:
            Response dict with search parameters
        """
        # Update state
        context.state = ConversationFlowState.SEARCH_READY
        await self.context_manager.save_context(context)
        
        # Build search parameters
        search_params = self._build_search_params(context.trip_spec)
        
        # Generate confirmation message
        message = self.state_machine.get_state_message(
            state=context.state,
            trip_spec=context.trip_spec
        )
        
        await self.context_manager.add_message(
            session_id=context.session_id,
            role="assistant",
            content=message
        )
        
        return {
            "session_id": context.session_id,
            "state": context.state,
            "message": message,
            "search_ready": True,
            "search_params": search_params,
            "trip_spec": context.trip_spec.dict()
        }
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def _build_search_params(self, trip_spec: TripSpec) -> Dict[str, Any]:
        """
        Build search parameters for flight API.
        
        Args:
            trip_spec: Trip specification
            
        Returns:
            Dict with search parameters
        """
        params = {
            "origin": trip_spec.origin,
            "destination": trip_spec.destination,
            "depart_date": trip_spec.depart_date,
            "return_date": trip_spec.return_date,
            "passengers": trip_spec.passengers,
            "cabin_class": trip_spec.cabin_class,
            "flexible_dates": trip_spec.flexible_dates,
            "flexibility_days": trip_spec.flexibility_days,
            "currency": "USD",
            "locale": "en"
        }
        
        # Add preferences
        if trip_spec.preferences:
            params["preferences"] = trip_spec.preferences
        
        return params
    
    def _merge_trip_specs(self, base: TripSpec, updates: TripSpec) -> TripSpec:
        """
        Merge two trip specs (updates override base).
        
        Args:
            base: Base trip spec
            updates: Updates to apply
            
        Returns:
            Merged TripSpec
        """
        merged_data = base.dict()
        update_data = updates.dict(exclude_unset=True)
        
        for key, value in update_data.items():
            if value is not None:
                merged_data[key] = value
        
        return TripSpec(**merged_data)
    
    def _build_ai_context(self, context: ConversationContext) -> Optional[str]:
        """
        Build context string for AI from conversation history.
        
        Args:
            context: Conversation context
            
        Returns:
            Context string or None
        """
        if not context.messages:
            return None
        
        # Get recent messages for context
        recent = self.context_manager.get_recent_messages(context, limit=3)
        
        context_parts = []
        for msg in recent:
            context_parts.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(context_parts)


# ============================================================
# EXPORTS
# ============================================================

__all__ = ['ConversationService']