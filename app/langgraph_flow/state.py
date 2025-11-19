from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, date


# ============================================================
# Conversation State — LangGraph-Compatible Dataclass Version
# ============================================================

@dataclass
class ConversationState:
    # ==========================================
    # SESSION IDENTIFICATION
    # ==========================================
    session_id: str
    user_id: Optional[int] = None
    session_token: str = ""

    # ==========================================
    # CONVERSATION TRACKING
    # ==========================================
    latest_user_message: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    turn_count: int = 0

    # ==========================================
    # INTENT & FLOW CONTROL
    # ==========================================
    intent: Optional[str] = None
    missing_parameter: Optional[str] = None
    last_question: Optional[str] = None
    flow_stage: str = "collecting"

    # ==========================================
    # FLIGHT SEARCH PARAMETERS (8 REQUIRED SLOTS)
    # ==========================================
    destination: Optional[str] = None
    origin: Optional[str] = None
    departure_date: Optional[date] = None
    return_date: Optional[date] = None
    passengers: int = 1
    travel_class: Optional[str] = None
    budget: Optional[float] = None
    flexibility: Optional[int] = None

    # ==========================================
    # PARSED METADATA
    # ==========================================
    destination_airports: List[str] = field(default_factory=list)
    origin_airports: List[str] = field(default_factory=list)
    is_round_trip: bool = False

    # ==========================================
    # MEMORY LAYERS
    # ==========================================
    short_term_memory: Dict[str, Any] = field(default_factory=dict)
    long_term_preferences: Dict[str, Any] = field(default_factory=dict)
    semantic_memories: List[Dict[str, Any]] = field(default_factory=list)

    # ==========================================
    # FLIGHT ENGINE & RANKING
    # ==========================================
    raw_flights: List[Dict[str, Any]] = field(default_factory=list)
    ranked_flights: List[Dict[str, Any]] = field(default_factory=list)
    search_executed: bool = False
    search_timestamp: Optional[datetime] = None
    search_cache_key: Optional[str] = None

    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        "price": 0.35,
        "duration": 0.25,
        "layover": 0.15,
        "airline_quality": 0.10,
        "airport_convenience": 0.10,
        "personalization": 0.05,
    })
    ranking_explanation: Optional[str] = None

    # ==========================================
    # RESPONSE GENERATION
    # ==========================================
    next_placeholder: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    assistant_message: Optional[str] = None


        # Add after your existing fields, maybe in a "ROUTING" section:
    use_rule_based_path: bool = False
    routing_confidence: float = 0.0
    confidence_breakdown: Dict[str, Any] = field(default_factory=dict)
    extracted_params: Dict[str, Any] = field(default_factory=dict)

    # ==========================================
    # COMPLETION
    # ==========================================
    is_complete: bool = False
    ready_for_search: bool = False

    # ==========================================
    # ERROR HANDLING & DEBUGGING
    # ==========================================
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    debug_info: Dict[str, Any] = field(default_factory=dict)

    # ==========================================
    # TIMESTAMPS
    # ==========================================
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Used for DB + Redis serialization
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)



# ============================================================
# INITIAL STATE FACTORY
# ============================================================

def create_initial_state(
    session_id: str,
    user_id: Optional[int] = None,
    latest_message: str = "",
) -> ConversationState:
    """
    Create a brand-new state object for new session.
    """
    return ConversationState(
        session_id=session_id,
        user_id=user_id,
        latest_user_message=latest_message,
    )


# ============================================================
# STATE UPDATE HELPER
# ============================================================

def update_state(state: ConversationState, updates: Dict[str, Any]) -> ConversationState:
    """
    Safely update state — LangGraph works perfectly with dataclass mutation.
    """
    for key, value in updates.items():
        setattr(state, key, value)

    state.updated_at = datetime.utcnow()
    return state


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_required_slots(state: ConversationState) -> Optional[str]:
    required = [
        ("destination", state.destination),
        ("departure_date", state.departure_date),
        ("origin", state.origin),
    ]

    for name, value in required:
        if not value:
            return name

    return None


def is_ready_for_search(state: ConversationState) -> bool:
    return (
        state.destination is not None
        and state.origin is not None
        and state.departure_date is not None
        and len(state.destination_airports) > 0
        and len(state.origin_airports) > 0
    )


