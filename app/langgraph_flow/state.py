"""
LangGraph State Object
Central state definition for AI Trip Planner conversation flow
"""

from typing import TypedDict, Optional, List, Dict, Any
from datetime import date, datetime


class ConversationState(TypedDict, total=False):
    """
    Central state object passed through all LangGraph nodes.
    Tracks conversation progress, collected parameters, and memory context.
    
    Philosophy:
    - Single source of truth for conversation state
    - Persisted to Redis (short-term) + MySQL (long-term)
    - Enriched with memory from Qdrant + UserPreference table
    - Updated by each node, never replaced entirely
    """
    
    # ==========================================
    # SESSION IDENTIFICATION
    # ==========================================
    
    session_id: str  # UUID for this conversation
    user_id: Optional[int]  # User ID if authenticated, None for anonymous
    session_token: str  # JWT or session token
    
    # ==========================================
    # CONVERSATION TRACKING
    # ==========================================
    
    latest_user_message: str  # Current user input
    conversation_history: List[Dict[str, str]]  # [{"role": "user"|"assistant", "content": "..."}]
    turn_count: int  # Number of exchanges
    
    # ==========================================
    # INTENT & FLOW CONTROL
    # ==========================================
    
    intent: Optional[str]  # Current message intent (travel_query, destination_provided, etc.)
    missing_parameter: Optional[str]  # Next slot to collect (destination, dates, origin, etc.)
    last_question: Optional[str]  # Last question asked to user
    flow_stage: str  # Current stage: collecting, searching, ranking, completed
    
    # ==========================================
    # FLIGHT SEARCH PARAMETERS (8 REQUIRED SLOTS)
    # ==========================================
    
    # Core parameters
    destination: Optional[str]  # Airport code (IST) or city (Istanbul)
    origin: Optional[str]  # Airport code (LGW) or city (London)
    departure_date: Optional[date]  # Parsed departure date
    return_date: Optional[date]  # Parsed return date (None for one-way)
    
    # Additional parameters
    passengers: int  # Number of passengers (default: 1)
    travel_class: Optional[str]  # Economy, Premium Economy, Business, First
    budget: Optional[float]  # Maximum budget or None
    flexibility: Optional[int]  # Days of flexibility (±N days)
    
    # Parsed metadata
    destination_airports: List[str]  # Resolved airport codes for destination
    origin_airports: List[str]  # Resolved airport codes for origin
    is_round_trip: bool  # True if return_date provided
    
    # ==========================================
    # MEMORY & PERSONALIZATION (3-LAYER SYSTEM)
    # ==========================================
    
    # Layer 1: Short-term session memory (current conversation)
    short_term_memory: Dict[str, Any]  # Temporary preferences, corrections
    
    # Layer 2: Structured long-term preferences (MySQL UserPreference)
    long_term_preferences: Dict[str, Any]  # {
    #     "preferred_airports": ["LGW", "STN"],
    #     "preferred_airlines": ["TK", "QR"],
    #     "budget_range": [200, 500],
    #     "prefers_direct": True,
    #     "hates_overnight_layovers": True,
    #     "preferred_time_of_day": "morning",
    # }
    
    # Layer 3: Semantic memory (Qdrant vectors)
    semantic_memories: List[Dict[str, Any]]  # Retrieved memory objects from Qdrant
    
    # ==========================================
    # FLIGHT ENGINE & RANKING
    # ==========================================
    
    # Flight search results
    raw_flights: List[Dict[str, Any]]  # Normalized flight data from APIs
    ranked_flights: List[Dict[str, Any]]  # Top N flights after ranking
    
    # Engine metadata
    search_executed: bool  # Has flight search been triggered?
    search_timestamp: Optional[datetime]  # When search was executed
    search_cache_key: Optional[str]  # For caching results
    
    # Ranking metadata
    ranking_weights: Dict[str, float]  # Dynamic weights based on user preferences
    ranking_explanation: Optional[str]  # Why these flights were chosen
    
    # ==========================================
    # RESPONSE GENERATION
    # ==========================================
    
    # Output to frontend
    next_placeholder: Optional[str]  # Placeholder text for input field
    suggestions: List[str]  # Suggestion buttons for user
    assistant_message: Optional[str]  # Text response from assistant
    
    # Completion tracking
    is_complete: bool  # All slots collected?
    ready_for_search: bool  # Ready to trigger flight engine?
    
    # ==========================================
    # ERROR HANDLING & DEBUGGING
    # ==========================================
    
    errors: List[str]  # Any errors encountered
    warnings: List[str]  # Non-fatal warnings
    debug_info: Dict[str, Any]  # Debug metadata
    
    # ==========================================
    # TIMESTAMPS
    # ==========================================
    
    created_at: datetime
    updated_at: datetime


# ==========================================
# STATE INITIALIZATION HELPER
# ==========================================

def create_initial_state(
    session_id: str,
    user_id: Optional[int] = None,
    latest_message: str = ""
) -> ConversationState:
    """
    Create a fresh state object for a new conversation.
    """
    now = datetime.utcnow()
    
    return ConversationState(
        # Session
        session_id=session_id,
        user_id=user_id,
        session_token="",  # Will be set by auth layer
        
        # Conversation
        latest_user_message=latest_message,
        conversation_history=[],
        turn_count=0,
        
        # Flow control
        intent=None,
        missing_parameter=None,
        last_question=None,
        flow_stage="collecting",
        
        # Flight parameters (8 slots)
        destination=None,
        origin=None,
        departure_date=None,
        return_date=None,
        passengers=1,
        travel_class=None,
        budget=None,
        flexibility=None,
        
        # Parsed metadata
        destination_airports=[],
        origin_airports=[],
        is_round_trip=False,
        
        # Memory (3 layers)
        short_term_memory={},
        long_term_preferences={},
        semantic_memories=[],
        
        # Flight engine
        raw_flights=[],
        ranked_flights=[],
        search_executed=False,
        search_timestamp=None,
        search_cache_key=None,
        
        # Ranking
        ranking_weights={
            "price": 0.35,
            "duration": 0.25,
            "layover": 0.15,
            "airline_quality": 0.10,
            "airport_convenience": 0.10,
            "personalization": 0.05,
        },
        ranking_explanation=None,
        
        # Response
        next_placeholder=None,
        suggestions=[],
        assistant_message=None,
        
        # Completion
        is_complete=False,
        ready_for_search=False,
        
        # Error tracking
        errors=[],
        warnings=[],
        debug_info={},
        
        # Timestamps
        created_at=now,
        updated_at=now,
    )


# ==========================================
# STATE UPDATE HELPER
# ==========================================

def update_state(state: ConversationState, updates: Dict[str, Any]) -> ConversationState:
    """
    Safely update state with new values.
    Always updates the 'updated_at' timestamp.
    """
    state.update(updates)
    state["updated_at"] = datetime.utcnow()
    return state


# ==========================================
# STATE VALIDATION
# ==========================================

def validate_required_slots(state: ConversationState) -> Optional[str]:
    """
    Check if all required slots are filled.
    Returns the name of the first missing required slot, or None if all filled.
    """
    required_slots = [
        ("destination", state.get("destination")),
        ("departure_date", state.get("departure_date")),
        ("origin", state.get("origin")),
    ]
    
    for slot_name, slot_value in required_slots:
        if not slot_value:
            return slot_name
    
    # All required slots filled
    return None


def is_ready_for_search(state: ConversationState) -> bool:
    """
    Determine if we have enough information to trigger flight search.
    """
    return (
        state.get("destination") is not None
        and state.get("origin") is not None
        and state.get("departure_date") is not None
        and len(state.get("destination_airports", [])) > 0
        and len(state.get("origin_airports", [])) > 0
    )