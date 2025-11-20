"""
Session CRUD Operations - PRODUCTION VERSION
Database operations for managing conversation sessions with proper enum handling.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
from dataclasses import asdict, is_dataclass
from enum import Enum

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import AISession, SessionStatus
from app.langgraph_flow.state import ConversationState

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Helper: Serialize values for JSON storage
# ------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-compatible format."""
    
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    elif isinstance(value, Enum):
        return value.value
    elif is_dataclass(value):
        return asdict(value)
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, set):
        return list(value)
    elif value is None or isinstance(value, (str, int, float, bool)):
        return value
    else:
        try:
            return str(value)
        except:
            return None


def _serialize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively serialize all values in a dict."""
    return {key: _serialize_value(value) for key, value in data.items()}


# ------------------------------------------------------
# Helper: Convert ConversationState → serializable dict
# ------------------------------------------------------

def state_to_dict(state: ConversationState) -> Dict[str, Any]:
    """Safely convert ConversationState dataclass into a clean dict."""
    if hasattr(state, "to_dict") and callable(getattr(state, "to_dict")):
        try:
            result = state.to_dict()
            return _serialize_dict(result)
        except Exception as e:
            logger.warning(f"[StateConversion] to_dict() failed: {e}")
    
    if is_dataclass(state):
        try:
            result = asdict(state)
            return _serialize_dict(result)
        except Exception as e:
            logger.warning(f"[StateConversion] asdict() failed: {e}")
    
    result = {}
    try:
        for key, value in vars(state).items():
            result[key] = _serialize_value(value)
    except Exception as e:
        logger.error(f"[StateConversion] Manual conversion failed: {e}")
        raise
    
    return result


def _determine_session_status(state_dict: Dict[str, Any]) -> SessionStatus:
    """
    Intelligently determine session status from state.
    Returns the proper Enum object.
    """
    is_complete = state_dict.get("is_complete", False)
    search_executed = state_dict.get("search_executed", False)
    ready_for_search = state_dict.get("ready_for_search", False)
    
    if is_complete:
        return SessionStatus.COMPLETED
    elif search_executed:
        return SessionStatus.SEARCHED
    elif ready_for_search:
        return SessionStatus.ACTIVE
    else:
        return SessionStatus.ACTIVE


# ============================================================================
# SAVE SESSION STATE
# ============================================================================

async def save_session_state(
    db: AsyncSession,
    session_id: str,
    state: ConversationState
) -> bool:
    """Save or update session state in database."""

    try:
        # Convert state → pure dict → JSON
        state_dict = state_to_dict(state)
        state_json = json.dumps(state_dict)

        # Determine session status (Enum object)
        status_enum = _determine_session_status(state_dict)
        
        # 🛠️ CRITICAL FIX: Extract the string value ("active") explicitly.
        # This prevents SQLAlchemy from sending "ACTIVE" (Enum Name) which crashes MySQL.
        status_value = status_enum.value 

        # Check if session exists
        stmt = select(AISession).where(AISession.session_token == session_id)
        result = await db.execute(stmt)
        existing_session = result.scalar_one_or_none()

        if existing_session:
            # UPDATE existing session
            stmt = (
                update(AISession)
                .where(AISession.session_token == session_id)
                .values(
                    state_json=state_json,
                    status=status_value,  # <--- Use .value (string)
                    updated_at=datetime.utcnow(),
                )
            )
            await db.execute(stmt)
            logger.debug(f"[SessionCRUD] Updated session {session_id} (status: {status_value})")

        else:
            # CREATE new session
            user_id = state_dict.get("user_id")
            
            new_session = AISession(
                user_id=user_id if user_id and user_id > 0 else None,
                session_token=session_id,
                state_json=state_json,
                status=status_value,  # <--- Use .value (string)
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(new_session)
            logger.debug(f"[SessionCRUD] Created session {session_id} (user_id: {user_id}, status: {status_value})")

        await db.commit()
        logger.info(f"[SessionCRUD] ✅ Successfully saved session {session_id}")
        return True

    except Exception as e:
        await db.rollback()
        logger.error(f"[SessionCRUD] ❌ Error saving session {session_id}: {e}", exc_info=True)
        return False


# ============================================================================
# LOAD SESSION STATE
# ============================================================================

async def load_session_state(
    db: AsyncSession,
    session_id: str,
    user_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Load session state from database."""
    try:
        stmt = select(AISession).where(AISession.session_token == session_id)
        if user_id is not None:
            stmt = stmt.where(AISession.user_id == user_id)

        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            logger.debug(f"[SessionCRUD] Session {session_id} not found in DB")
            return None

        if not session.state_json:
            logger.warning(f"[SessionCRUD] Session {session_id} has no state_json")
            return None
            
        state = json.loads(session.state_json)
        return state

    except json.JSONDecodeError as e:
        logger.error(f"[SessionCRUD] JSON decode error for session {session_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"[SessionCRUD] Error loading session {session_id}: {e}", exc_info=True)
        return None


# ============================================================================
# DELETE SESSION STATE
# ============================================================================

async def delete_session_state(db: AsyncSession, session_id: str) -> bool:
    """Delete session from DB."""
    try:
        stmt = delete(AISession).where(AISession.session_token == session_id)
        result = await db.execute(stmt)
        await db.commit()
        
        deleted = result.rowcount > 0
        if deleted:
            logger.info(f"[SessionCRUD] ✅ Deleted session {session_id}")
        else:
            logger.debug(f"[SessionCRUD] Session {session_id} not found for deletion")
        return deleted
    except Exception as e:
        await db.rollback()
        logger.error(f"[SessionCRUD] ❌ Error deleting session {session_id}: {e}", exc_info=True)
        return False
