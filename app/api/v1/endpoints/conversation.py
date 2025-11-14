# app/api/v1/endpoints/conversation.py
"""
Conversational Search API Endpoints for SkySearch AI
Thin layer over ConversationService - handles HTTP concerns only.
Integrates with your existing FastAPI infrastructure.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, Depends, status
from pydantic import BaseModel, Field, validator
import logging

from services.conversation_service import ConversationService
from app.conversation.context_manager import ContextManager, get_context_manager
from app.conversation.state_machine import StateMachine
from app.conversation.validators import StateValidator
from app.conversation.suggestion_engine import SuggestionEngine
from app.conversation.ai_adapter import AIAdapter
from app.infrastructure.geoip import IPGeolocationService
from app.infrastructure.cache import RedisCache
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversation", tags=["conversation"])


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class StartConversationRequest(BaseModel):
    """Request body for starting a conversation"""
    initial_query: Optional[str] = Field(
        None,
        description="Optional initial query like 'I want to go to Dubai'",
        max_length=500
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier for tracking"
    )
    
    @validator('initial_query')
    def validate_query(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip() if v else None


class ProcessInputRequest(BaseModel):
    """Request body for processing user input"""
    session_id: str = Field(..., description="Conversation session ID")
    user_input: Optional[str] = Field(
        None,
        description="Free-form text input from user",
        max_length=500
    )
    selected_suggestion: Optional[Dict[str, Any]] = Field(
        None,
        description="Selected suggestion chip data"
    )
    
    @validator('user_input')
    def validate_input(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Input cannot be empty")
        return v.strip() if v else None
    
    @validator('selected_suggestion')
    def validate_input_provided(cls, v, values):
        # At least one of user_input or selected_suggestion must be provided
        if v is None and values.get('user_input') is None:
            raise ValueError("Either user_input or selected_suggestion must be provided")
        return v


class ExecuteSearchRequest(BaseModel):
    """Request body for executing search"""
    session_id: str = Field(..., description="Conversation session ID")


class ResetRequest(BaseModel):
    """Request body for resetting conversation"""
    session_id: str = Field(..., description="Conversation session ID")


# ============================================================
# DEPENDENCY INJECTION
# ============================================================

async def get_conversation_service() -> ConversationService:
    """
    Factory function for ConversationService with all dependencies.
    Uses your existing Redis infrastructure.
    """
    # Initialize cache with your Redis client
    cache = RedisCache()
    
    # Initialize core components
    context_manager = ContextManager(cache)
    state_machine = StateMachine()
    validator = StateValidator()
    suggestion_engine = SuggestionEngine()
    
    # Initialize AI adapter if API key available
    ai_adapter = None
    if settings.GEMINI_API_KEY and settings.ENABLE_CONVERSATIONAL_AI:
        try:
            ai_adapter = AIAdapter(
                model_name=settings.GEMINI_MODEL,
                api_key=settings.GEMINI_API_KEY,
                timeout=int(settings.GEMINI_TIMEOUT),
                max_retries=settings.GEMINI_MAX_RETRIES
            )
            logger.debug("✓ AI adapter initialized")
        except Exception as e:
            logger.warning(f"AI adapter initialization failed: {e}")
    else:
        logger.info("AI adapter disabled (no API key or feature flag off)")
    
    # Initialize GeoIP service (uses free ip-api.com, no key needed)
    geoip_service = None
    if settings.ENABLE_IP_GEOLOCATION:
        try:
            geoip_service = IPGeolocationService()
            logger.debug("✓ GeoIP service initialized")
        except Exception as e:
            logger.warning(f"GeoIP service initialization failed: {e}")
    
    return ConversationService(
        context_manager=context_manager,
        state_machine=state_machine,
        validator=validator,
        suggestion_engine=suggestion_engine,
        ai_adapter=ai_adapter,
        geoip_service=geoip_service
    )


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request headers.
    Handles proxy headers (X-Forwarded-For, X-Real-IP).
    """
    # Check for proxy headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client IP
    return request.client.host if request.client else "127.0.0.1"


# ============================================================
# ENDPOINTS
# ============================================================

@router.post("/start", response_model=Dict[str, Any])
async def start_conversation(
    request_body: StartConversationRequest,
    http_request: Request,
    service: ConversationService = Depends(get_conversation_service)
) -> Dict[str, Any]:
    """
    Start a new conversational search session.
    
    **Features:**
    - Auto-detects origin airport from IP
    - Optionally processes initial query immediately
    - Returns session ID, message, and suggestions
    
    **Example request:**
```json
    {
        "initial_query": "I want cheap flights to Tokyo next month",
        "user_id": "user_12345"
    }
```
    
    **Response includes:**
    - `session_id`: Unique session identifier
    - `state`: Current conversation state
    - `message`: Assistant's message to user
    - `suggestions`: List of suggestion chips
    - `search_ready`: Whether search can be executed
    - `detected_origin`: Airport detected from IP (if available)
    """
    try:
        logger.info(
            f"Starting conversation: user={request_body.user_id}, "
            f"has_query={bool(request_body.initial_query)}"
        )
        
        # Get client IP for geolocation
        client_ip = get_client_ip(http_request)
        logger.debug(f"Client IP: {client_ip}")
        
        # Start conversation
        result = await service.start_conversation(
            user_id=request_body.user_id,
            initial_query=request_body.initial_query,
            client_ip=client_ip
        )
        
        logger.info(f"✓ Conversation started: session={result.get('session_id')}")
        return result
    
    except Exception as e:
        logger.exception(f"Error starting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting conversation: {str(e)}"
        )


@router.post("/input", response_model=Dict[str, Any])
async def process_input(
    request_body: ProcessInputRequest,
    service: ConversationService = Depends(get_conversation_service)
) -> Dict[str, Any]:
    """
    Process user input (either free text or suggestion click).
    
    This is the main conversation loop endpoint.
    
    **Example request with text input:**
```json
    {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "user_input": "Next weekend"
    }
```
    
    **Example request with suggestion:**
```json
    {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "selected_suggestion": {
            "type": "DATE_PRESET",
            "value": {"depart_date": "2025-11-15"}
        }
    }
```
    
    **Response includes:**
    - `state`: Updated conversation state
    - `message`: Assistant's response
    - `suggestions`: Next step suggestions
    - `search_ready`: If true, call /search endpoint
    - `trip_spec`: Current trip specification
    """
    try:
        logger.info(
            f"Processing input: session={request_body.session_id}, "
            f"has_text={bool(request_body.user_input)}, "
            f"has_suggestion={bool(request_body.selected_suggestion)}"
        )
        
        result = await service.process_user_input(
            session_id=request_body.session_id,
            user_input=request_body.user_input,
            selected_suggestion=request_body.selected_suggestion
        )
        
        # Handle session not found
        if "error" in result and result["error"] == "Session not found":
            logger.warning(f"Session not found: {request_body.session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or expired. Please start a new conversation."
            )
        
        logger.info(
            f"✓ Input processed: session={request_body.session_id}, "
            f"state={result.get('state')}"
        )
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing input: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing input: {str(e)}"
        )


@router.post("/search", response_model=Dict[str, Any])
async def execute_search(
    request_body: ExecuteSearchRequest,
    service: ConversationService = Depends(get_conversation_service)
) -> Dict[str, Any]:
    """
    Execute a flight search using collected conversation context.
    
    Returns search parameters ready for your flight APIs (Duffel/Amadeus/TravelPayouts).
    
    **Example response:**
```json
    {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "search_params": {
            "origin": "TAS",
            "destination": "IST",
            "depart_date": "2025-11-15",
            "return_date": null,
            "passengers": 1,
            "cabin_class": "economy",
            "flexible_dates": false,
            "currency": "USD",
            "locale": "en"
        },
        "message": "Search parameters ready. Redirecting to results..."
    }
```
    """
    try:
        logger.info(f"Executing search: session={request_body.session_id}")
        
        result = await service.execute_search(request_body.session_id)
        
        # Handle errors
        if "error" in result:
            error_type = result.get("error")
            
            if error_type == "Session not found":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=result.get("message", "Session not found")
                )
            elif error_type == "validation_failed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=result.get("message", "Validation failed"),
                    headers={"X-Validation-Errors": str(result.get("validation_errors", []))}
                )
        
        logger.info(
            f"✓ Search executed: session={request_body.session_id}, "
            f"route={result['search_params']['origin']}→{result['search_params']['destination']}"
        )
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error executing search: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing search: {str(e)}"
        )


@router.get("/history/{session_id}", response_model=Dict[str, Any])
async def get_conversation_history(
    session_id: str,
    service: ConversationService = Depends(get_conversation_service)
) -> Dict[str, Any]:
    """
    Retrieve full conversation history and current context.
    
    **Response includes:**
    - `session_id`: Session identifier
    - `state`: Current conversation state
    - `trip_spec`: Current trip specification
    - `messages`: Full message history
    - `detected_origin`: Detected airport (if available)
    - `created_at`: Session creation timestamp
    - `updated_at`: Last update timestamp
    """
    try:
        logger.info(f"Retrieving history: session={session_id}")
        
        result = await service.get_conversation_history(session_id)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.post("/reset", response_model=Dict[str, Any])
async def reset_conversation(
    request_body: ResetRequest,
    service: ConversationService = Depends(get_conversation_service)
) -> Dict[str, Any]:
    """
    Reset an existing conversation session.
    
    Clears context and returns to initial state while keeping session association.
    
    **Use cases:**
    - User wants to start a new search
    - Clear incorrect information
    - Begin fresh conversation
    """
    try:
        logger.info(f"Resetting conversation: session={request_body.session_id}")
        
        result = await service.reset_conversation(request_body.session_id)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        logger.info(f"✓ Conversation reset: new_session={result['session_id']}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error resetting conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resetting conversation: {str(e)}"
        )


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    service: ConversationService = Depends(get_conversation_service)
) -> Dict[str, str]:
    """
    Delete a conversation session.
    
    **GDPR/Privacy compliance:** Users can request deletion of their conversation data.
    """
    try:
        logger.info(f"Deleting session: {session_id}")
        
        # Access context manager directly for deletion
        success = await service.context_manager.delete_session(session_id)
        
        if success:
            return {
                "message": "Session deleted successfully",
                "session_id": session_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting session: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Lightweight health check endpoint.
    
    **Response:**
```json
    {
        "status": "healthy",
        "service": "conversation",
        "version": "1.0.0",
        "features": {
            "ai_enabled": true,
            "geolocation_enabled": true,
            "smart_suggestions": true
        }
    }
```
    """
    return {
        "status": "healthy",
        "service": "conversation",
        "version": settings.VERSION,
        "features": {
            "ai_enabled": settings.ENABLE_CONVERSATIONAL_AI and bool(settings.GEMINI_API_KEY),
            "geolocation_enabled": settings.ENABLE_IP_GEOLOCATION,
            "smart_suggestions": settings.ENABLE_SMART_SUGGESTIONS,
            "personalization": settings.ENABLE_PERSONALIZATION
        },
        "config": {
            "gemini_model": settings.GEMINI_MODEL,
            "session_ttl": settings.CONVERSATION_SESSION_TTL,
            "max_turns": settings.MAX_CONVERSATION_TURNS
        }
    }


@router.get("/stats")
async def get_stats(
    service: ConversationService = Depends(get_conversation_service)
) -> Dict[str, Any]:
    """
    Get conversation system statistics.
    
    **Admin endpoint** - consider adding authentication in production.
    """
    try:
        stats = await service.context_manager.get_stats()
        
        return {
            "conversation_system": stats,
            "settings": {
                "session_ttl": settings.CONVERSATION_SESSION_TTL,
                "max_history": settings.MAX_CONVERSATION_HISTORY,
                "max_turns": settings.MAX_CONVERSATION_TURNS
            }
        }
    except Exception as e:
        logger.exception(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting statistics: {str(e)}"
        )

