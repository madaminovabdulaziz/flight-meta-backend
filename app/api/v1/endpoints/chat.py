"""
Chat Endpoint - Dual-Mode Architecture
======================================
Supports both structured (Mad Libs) and conversational input.

Routes:
- POST /api/v1/chat/message - Main chat endpoint (dual-mode)
- POST /api/v1/chat/search - Direct search (structured only, faster)
- GET /api/v1/chat/session/{session_id} - Get session state
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, date
import uuid

from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, Field

from app.langgraph_flow.graph import run_conversation_turn
from app.langgraph_flow.state import ConversationState
from app.db.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession

# Import search/ranking components directly
from app.langgraph_flow.nodes.flight_search_node import flight_search_node
from services.cheap_ranker import cheap_ranker
from services.local_intent_classifier import intent_classifier
from app.api.v1.endpoints.duffel_new import duffel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class StructuredParams(BaseModel):
    """Pre-validated flight search parameters from Mad Libs UI."""
    
    destination: str = Field(..., description="Destination city or airport code")
    origin: str = Field(..., description="Origin city or airport code")
    departure_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date for round trip")
    passengers: int = Field(1, ge=1, le=9, description="Number of passengers")
    travel_class: Optional[str] = Field("economy", description="Travel class")
    budget: Optional[float] = Field(None, description="Maximum budget")
    flexibility: Optional[int] = Field(0, description="Date flexibility in days")


class ChatMessageRequest(BaseModel):
    """Request body for chat message."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's message or natural language query",
        example="I want to fly to Istanbul next week"
    )
    
    # Session management
    session_id: Optional[str] = Field(None, description="Session ID")
    user_id: Optional[int] = Field(None, description="User ID (authenticated)")
    
    # NEW: Structured mode
    structured: bool = Field(
        False, 
        description="True if Mad Libs UI (pre-validated params)"
    )
    params: Optional[StructuredParams] = Field(
        None,
        description="Pre-extracted params (only if structured=true)"
    )


class ChatMessageResponse(BaseModel):
    """Response from chat endpoint."""
    
    session_id: str
    
    # For conversational mode (multi-turn)
    assistant_message: Optional[str] = None
    next_placeholder: Optional[str] = None
    suggestions: list[str] = Field(default_factory=list)
    
    # For both modes (results)
    ranked_flights: list[Dict[str, Any]] = Field(default_factory=list)
    
    # Metadata
    used_rule_based_path: bool
    confidence_score: Optional[float] = None
    latency_ms: int
    llm_calls: int
    
    # Flow control
    is_complete: bool = False
    flow_stage: str = "collecting"
    
    # For debugging
    path_taken: str = Field(..., description="'structured' or 'conversational'")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_client_ip(request: Request) -> str:
    """Extract client IP, handling proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host if request.client else "127.0.0.1"


def parse_date_safe(date_str: str) -> Optional[date]:
    """Parse date string safely."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None


# ============================================================================
# FAST PATH: STRUCTURED SEARCH
# ============================================================================

async def handle_structured_search(
    request: ChatMessageRequest,
    session_id: str,
    client_ip: str
) -> ChatMessageResponse:
    """
    Fast path for Mad Libs UI.
    
    Flow:
    1. Validate params (already done by Pydantic)
    2. Call Duffel API directly
    3. Rank with cheap_ranker (NO LLM)
    4. Return results
    
    Performance: 800-1500ms, 0 LLM calls, $0.00
    """
    
    logger.info(f"[Chat:Structured] Fast path for session {session_id}")
    start_time = datetime.utcnow()
    
    params = request.params
    
    # ========================================
    # Step 1: Parse dates
    # ========================================
    departure_date = parse_date_safe(params.departure_date)
    if not departure_date:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid departure_date: {params.departure_date}"
        )
    
    return_date = None
    if params.return_date:
        return_date = parse_date_safe(params.return_date)
    
    # ========================================
    # Step 2: Build Duffel payload
    # ========================================
    slices = [
        {
            "origin": params.origin.upper(),
            "destination": params.destination.upper(),
            "departure_date": params.departure_date
        }
    ]
    
    if return_date:
        slices.append({
            "origin": params.destination.upper(),
            "destination": params.origin.upper(),
            "departure_date": params.return_date
        })
    
    duffel_payload = {
        "slices": slices,
        "passengers": [{"type": "adult"} for _ in range(params.passengers)],
        "cabin_class": params.travel_class.lower() if params.travel_class else "economy",
    }
    
    # ========================================
    # Step 3: Search flights
    # ========================================
    try:
        logger.info(f"[Chat:Structured] Calling Duffel API...")
        
        response = await duffel.create_offer_request(
            duffel_payload,
            return_offers=True
        )
        
        raw_offers = response.get("data", {}).get("offers", [])
        logger.info(f"[Chat:Structured] Got {len(raw_offers)} offers from Duffel")
        
    except Exception as e:
        logger.error(f"[Chat:Structured] Duffel API failed: {e}", exc_info=True)
        
        # User-friendly error
        error_msg = str(e)
        if "destination_must_be_different" in error_msg:
            detail = "Origin and destination cannot be the same"
        elif "422" in error_msg:
            detail = "Invalid search parameters. Please check your inputs."
        else:
            detail = "Flight search temporarily unavailable. Please try again."
        
        raise HTTPException(status_code=400, detail=detail)
    
    # ========================================
    # Step 4: Normalize flights
    # ========================================
    from app.langgraph_flow.nodes.flight_search_node import _normalize_duffel_offer
    
    normalized_flights = []
    for offer in raw_offers:
        normalized = _normalize_duffel_offer(offer)
        if normalized:
            normalized_flights.append(normalized)
    
    logger.info(f"[Chat:Structured] Normalized {len(normalized_flights)} flights")
    
    # ========================================
    # Step 5: Rank flights (rule-based, NO LLM)
    # ========================================
    # Build minimal state for ranker
    ranking_state = {
        "budget": params.budget,
        "travel_class": params.travel_class,
        "long_term_preferences": {},  # TODO: Load from DB if user authenticated
    }
    
    ranking_result = cheap_ranker.rank_flights(normalized_flights, ranking_state)
    
    ranked_flights = ranking_result.get("ranked_flights", [])
    
    logger.info(f"[Chat:Structured] Ranked {len(ranked_flights)} flights")
    
    # ========================================
    # Step 6: Build response
    # ========================================
    latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    return ChatMessageResponse(
        session_id=session_id,
        assistant_message=None,  # No message needed for structured mode
        next_placeholder=None,
        suggestions=[],  # No suggestions needed
        ranked_flights=ranked_flights[:10],  # Top 10
        
        used_rule_based_path=True,
        confidence_score=1.0,
        latency_ms=latency_ms,
        llm_calls=0,
        
        is_complete=True,
        flow_stage="completed",
        path_taken="structured",
    )


# ============================================================================
# SLOW PATH: CONVERSATIONAL (Natural Language)
# ============================================================================

async def handle_conversational_search(
    request: ChatMessageRequest,
    session_id: str,
    client_ip: str,
    db: AsyncSession
) -> ChatMessageResponse:
    """
    Fallback path for natural language input.
    
    Flow:
    1. Classify intent (off-topic detection)
    2. Run LangGraph flow (existing)
    3. Return assistant message or results
    
    Performance: 2000-4000ms, 0-3 LLM calls, $0.00-0.003
    """
    
    logger.info(f"[Chat:Conversational] Natural language path for session {session_id}")
    start_time = datetime.utcnow()
    
    # ========================================
    # Step 1: Off-topic detection (fast)
    # ========================================
    intent, conf = intent_classifier.classify(request.message, {})
    
    if intent in ["irrelevant", "chitchat"]:
        logger.info(f"[Chat:Conversational] Off-topic detected: {intent}")
        
        return ChatMessageResponse(
            session_id=session_id,
            assistant_message="I'm here to help you find flights! Where would you like to go?",
            next_placeholder="Enter destination",
            suggestions=["Istanbul", "Dubai", "London", "Paris"],
            ranked_flights=[],
            
            used_rule_based_path=True,
            confidence_score=conf,
            latency_ms=50,
            llm_calls=0,
            
            is_complete=False,
            flow_stage="collecting",
            path_taken="conversational",
        )
    
    # ========================================
    # Step 2: Run LangGraph flow (existing)
    # ========================================
    try:
        result_state: ConversationState = await run_conversation_turn(
            session_id=session_id,
            user_id=request.user_id,
            user_message=request.message,
        )
        
        # Inject IP for debugging
        if not result_state.debug_info:
            result_state.debug_info = {}
        result_state.debug_info["client_ip"] = client_ip
        
    except Exception as e:
        logger.error(f"[Chat:Conversational] Graph execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Processing failed")
    
    # ========================================
    # Step 3: Calculate metrics
    # ========================================
    latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    used_rule_based = getattr(result_state, "use_rule_based_path", False)
    confidence = result_state.confidence_breakdown.get("overall") if result_state.confidence_breakdown else None
    llm_calls = 0 if used_rule_based else 3
    
    # ========================================
    # Step 4: Save state (async, don't block response)
    # ========================================
    try:
        from app.db.crud.session import save_session_state
        await save_session_state(db, session_id, result_state)
    except Exception as e:
        logger.error(f"[Chat:Conversational] State save failed: {e}")
    
    # ========================================
    # Step 5: Build response
    # ========================================
    return ChatMessageResponse(
        session_id=session_id,
        assistant_message=result_state.assistant_message,
        next_placeholder=result_state.next_placeholder,
        suggestions=result_state.suggestions or [],
        ranked_flights=result_state.ranked_flights or [],
        
        used_rule_based_path=used_rule_based,
        confidence_score=confidence,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        
        is_complete=result_state.is_complete,
        flow_stage=result_state.flow_stage,
        path_taken="conversational",
    )


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db)
) -> ChatMessageResponse:
    """
    Dual-mode chat endpoint.
    
    Modes:
    - Structured (structured=true): Fast path for Mad Libs UI
    - Conversational (structured=false): Slow path for natural language
    """
    
    # Generate session ID
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    client_ip = get_client_ip(http_request)
    
    logger.info(
        f"[Chat] Request: mode={'structured' if request.structured else 'conversational'}, "
        f"session={session_id}, ip={client_ip}"
    )
    
    # ========================================
    # Route to appropriate handler
    # ========================================
    
    if request.structured:
        # FAST PATH: Mad Libs UI
        if not request.params:
            raise HTTPException(
                status_code=400,
                detail="structured=true requires params field"
            )
        
        return await handle_structured_search(request, session_id, client_ip)
    
    else:
        # SLOW PATH: Natural language
        return await handle_conversational_search(request, session_id, client_ip, db)


# ============================================================================
# ALTERNATIVE: DEDICATED SEARCH ENDPOINT (Even Faster)
# ============================================================================

# ============================================================================
# SESSION MANAGEMENT (Keep existing endpoints)
# ============================================================================

@router.get("/session/{session_id}")
async def get_session_state(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get current session state."""
    
    from app.db.crud.session import load_session_state
    
    state = await load_session_state(db, session_id)
    
    if not state:
        raise HTTPException(404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "collected_parameters": {
            "destination": state.get("destination"),
            "origin": state.get("origin"),
            "departure_date": str(state.get("departure_date")) if state.get("departure_date") else None,
            "passengers": state.get("passengers"),
        },
        "turn_count": state.get("turn_count", 0),
        "flow_stage": state.get("flow_stage", "collecting"),
    }


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """Delete session."""
    
    from app.db.crud.session import delete_session_state
    
    deleted = await delete_session_state(db, session_id)
    
    if not deleted:
        raise HTTPException(404, detail="Session not found")
    
    return {"message": "Session deleted"}