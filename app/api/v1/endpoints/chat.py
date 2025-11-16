"""
Chat Endpoint
Main API endpoint for AI Trip Planner conversations with hybrid LLM/rule-based routing.

Endpoints:
- POST /api/v1/chat/message - Send a message, get response
- POST /api/v1/chat/session/new - Start new session
- GET /api/v1/chat/session/{session_id} - Get session state
- DELETE /api/v1/chat/session/{session_id} - Clear session
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

from app.langgraph_flow.graph import run_conversation_turn
from app.langgraph_flow.state import ConversationState
from app.db.database import get_db
from app.db.crud.session import save_session_state, load_session_state
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class ChatMessageRequest(BaseModel):
    """Request body for sending a chat message."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's message",
        example="I want to fly to Istanbul"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID (auto-generated if not provided)",
        example="sess_abc123"
    )
    user_id: Optional[int] = Field(
        None,
        description="User ID for authenticated users",
        example=42
    )


class ChatMessageResponse(BaseModel):
    """Response from chat endpoint."""
    
    session_id: str = Field(..., description="Session identifier")
    assistant_message: Optional[str] = Field(None, description="Assistant's text response")
    next_placeholder: Optional[str] = Field(None, description="Placeholder for input field")
    suggestions: list[str] = Field(default_factory=list, description="Suggestion buttons")
    ranked_flights: list[Dict[str, Any]] = Field(default_factory=list, description="Ranked flight results")
    
    # Hybrid system metadata
    used_rule_based_path: bool = Field(..., description="Whether rule-based path was used")
    confidence_score: Optional[float] = Field(None, description="Overall confidence score")
    latency_ms: int = Field(..., description="Response time in milliseconds")
    llm_calls: int = Field(..., description="Number of LLM calls made")
    
    # Flow control
    is_complete: bool = Field(False, description="Whether search is complete")
    flow_stage: str = Field("collecting", description="Current stage: collecting/searching/completed")


class NewSessionResponse(BaseModel):
    """Response when creating a new session."""
    
    session_id: str
    created_at: datetime
    message: str = "New session created"


class SessionStateResponse(BaseModel):
    """Response with current session state."""
    
    session_id: str
    user_id: Optional[int]
    flow_stage: str
    collected_parameters: Dict[str, Any]
    turn_count: int
    created_at: datetime
    updated_at: datetime


# ============================================================================
# MAIN CHAT ENDPOINT
# ============================================================================

@router.post("/message", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    db: AsyncSession = Depends(get_db)
) -> ChatMessageResponse:
    """
    Send a message and get a response from the AI Trip Planner.
    
    This endpoint runs the hybrid LangGraph flow:
    - High confidence requests use rule-based path (fast, free)
    - Low confidence requests use LLM path (fallback)
    
    The response includes metadata about which path was used and performance metrics.
    """
    
    start_time = datetime.utcnow()
    
    # Generate session ID if not provided
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    
    logger.info(f"[ChatEndpoint] Processing message for session {session_id}")
    logger.debug(f"[ChatEndpoint] Message: {request.message[:100]}...")
    
    try:
        # Run the hybrid conversation flow
        result_state: ConversationState = await run_conversation_turn(
            session_id=session_id,
            user_id=request.user_id,
            user_message=request.message
        )
        
        # Calculate latency
        end_time = datetime.utcnow()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Extract metadata
        used_rule_based = result_state.get("use_rule_based_path", False)
        confidence = result_state.get("confidence_breakdown", {}).get("overall_confidence")
        
        # Calculate LLM calls (0 if rule-based, 3 if LLM path)
        llm_calls = 0 if used_rule_based else 3
        
        # Log performance metrics
        confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        logger.info(
            f"[ChatEndpoint] Completed in {latency_ms}ms | "
            f"Path: {'RULE-BASED' if used_rule_based else 'LLM'} | "
            f"Confidence: {confidence_str} | "
            f"LLM calls: {llm_calls}"
        )
        
        # Save session state to database
        try:
            await save_session_state(db, session_id, result_state)
        except Exception as e:
            logger.error(f"[ChatEndpoint] Failed to save session: {e}")
            # Don't fail the request, just log
        
        # Build response
        response = ChatMessageResponse(
            session_id=session_id,
            assistant_message=result_state.get("assistant_message"),
            next_placeholder=result_state.get("next_placeholder"),
            suggestions=result_state.get("suggestions", []),
            ranked_flights=result_state.get("ranked_flights", []),
            
            # Metadata
            used_rule_based_path=used_rule_based,
            confidence_score=confidence,
            latency_ms=latency_ms,
            llm_calls=llm_calls,
            
            # Flow control
            is_complete=result_state.get("is_complete", False),
            flow_stage=result_state.get("flow_stage", "collecting"),
        )
        
        return response
    
    except Exception as e:
        logger.error(f"[ChatEndpoint] Error processing message: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/session/new", response_model=NewSessionResponse)
async def create_new_session(
    user_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
) -> NewSessionResponse:
    """
    Create a new conversation session.
    
    Returns a new session_id that can be used for subsequent messages.
    """
    
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    created_at = datetime.utcnow()
    
    logger.info(f"[ChatEndpoint] Created new session: {session_id}")
    
    # Initialize empty state in database
    from app.langgraph_flow.state import create_initial_state
    initial_state = create_initial_state(
        session_id=session_id,
        user_id=user_id,
        latest_message=""
    )
    
    try:
        await save_session_state(db, session_id, initial_state)
    except Exception as e:
        logger.error(f"[ChatEndpoint] Failed to create session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )
    
    return NewSessionResponse(
        session_id=session_id,
        created_at=created_at
    )


@router.get("/session/{session_id}", response_model=SessionStateResponse)
async def get_session_state(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> SessionStateResponse:
    """
    Get the current state of a conversation session.
    
    Useful for:
    - Resuming conversations
    - Debugging
    - Showing progress to user
    """
    
    logger.info(f"[ChatEndpoint] Retrieving session: {session_id}")
    
    try:
        state = await load_session_state(db, session_id)
        
        if not state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        # Extract collected parameters
        collected = {
            "destination": state.get("destination"),
            "origin": state.get("origin"),
            "departure_date": state.get("departure_date"),
            "return_date": state.get("return_date"),
            "passengers": state.get("passengers"),
            "travel_class": state.get("travel_class"),
            "budget": state.get("budget"),
            "flexibility": state.get("flexibility"),
        }
        
        # Remove None values
        collected = {k: v for k, v in collected.items() if v is not None}
        
        return SessionStateResponse(
            session_id=session_id,
            user_id=state.get("user_id"),
            flow_stage=state.get("flow_stage", "collecting"),
            collected_parameters=collected,
            turn_count=state.get("turn_count", 0),
            created_at=state.get("created_at", datetime.utcnow()),
            updated_at=state.get("updated_at", datetime.utcnow()),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ChatEndpoint] Error retrieving session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session"
        )


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete a conversation session.
    
    This clears all session data from the database.
    """
    
    logger.info(f"[ChatEndpoint] Deleting session: {session_id}")
    
    try:
        # Delete from database
        from app.db.crud.session import delete_session_state
        deleted = await delete_session_state(db, session_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found"
            )
        
        return {"message": f"Session {session_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ChatEndpoint] Error deleting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )


# ============================================================================
# DEBUG ENDPOINTS (Optional - remove in production)
# ============================================================================

@router.post("/debug/confidence")
async def debug_confidence(
    message: str
) -> Dict[str, Any]:
    """
    DEBUG ONLY: Check confidence score without running full flow.
    
    Useful for testing the routing logic.
    """
    
    from services.local_intent_classifier import intent_classifier
    from services.rule_based_extractor import rule_extractor
    from app.langgraph_flow.nodes.should_use_llm_node import _calculate_overall_confidence
    
    # Run classifiers
    intent, intent_conf = intent_classifier.classify(message, {})
    extracted, extract_conf = rule_extractor.extract(message, {})
    
    # Calculate confidence
    overall, breakdown = _calculate_overall_confidence(intent_conf, extract_conf, {})
    
    return {
        "message": message,
        "intent": intent,
        "intent_confidence": intent_conf,
        "extracted_parameters": extracted,
        "extraction_confidence": extract_conf,
        "overall_confidence": overall,
        "confidence_breakdown": breakdown,
        "would_use_rule_based": overall >= 0.85,
        "recommended_path": "RULE-BASED (fast, free)" if overall >= 0.85 else "LLM (fallback)"
    }