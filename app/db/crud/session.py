"""
Session CRUD Operations
Database operations for managing conversation sessions.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from dataclasses import asdict, is_dataclass

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import AISession
from app.langgraph_flow.state import ConversationState

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Helper: Serialize values for JSON storage
# ------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-compatible format."""
    
    if isinstance(value, (datetime, date)):
        return value.isoformat()
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
        # Try to convert to string as last resort
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
    """
    Safely convert ConversationState dataclass into a clean dict.
    Handles datetime, nested objects, missing fields, dataclasses, etc.
    """
    
    # Method 1: Use built-in to_dict if available
    if hasattr(state, "to_dict") and callable(getattr(state, "to_dict")):
        try:
            result = state.to_dict()
            return _serialize_dict(result)
        except Exception as e:
            logger.warning(f"[StateConversion] to_dict() failed: {e}, falling back")
    
    # Method 2: Use dataclasses.asdict for proper dataclass handling
    if is_dataclass(state):
        try:
            result = asdict(state)
            return _serialize_dict(result)
        except Exception as e:
            logger.warning(f"[StateConversion] asdict() failed: {e}, falling back")
    
    # Method 3: Fallback to manual conversion
    result = {}
    try:
        for key, value in vars(state).items():
            result[key] = _serialize_value(value)
    except Exception as e:
        logger.error(f"[StateConversion] Manual conversion failed: {e}")
        raise
    
    return result


# ============================================================================
# SAVE SESSION STATE
# ============================================================================

async def save_session_state(
    db: AsyncSession,
    session_id: str,
    state: ConversationState
) -> bool:
    """
    Save or update session state in database.
    
    Args:
        db: Database session
        session_id: Unique session identifier
        state: ConversationState dataclass to save
        
    Returns:
        True if successful, False otherwise
    """

    try:
        # Convert state → pure dict → JSON
        state_dict = state_to_dict(state)
        state_json = json.dumps(state_dict)

        # Determine session status
        is_complete = state_dict.get("is_complete", False)
        search_executed = state_dict.get("search_executed", False)
        
        if is_complete:
            status = "completed"
        elif search_executed:
            status = "searched"
        else:
            status = "active"

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
                    status=status,
                    updated_at=datetime.utcnow(),
                )
            )
            await db.execute(stmt)
            logger.debug(f"[SessionCRUD] Updated session {session_id} (status: {status})")

        else:
            # CREATE new session
            # Handle case where user_id might not exist in users table
            user_id = state_dict.get("user_id")
            
            # For test/debug scenarios, allow NULL user_id if foreign key would fail
            # In production, you should ensure valid users exist first
            new_session = AISession(
                user_id=user_id if user_id and user_id > 0 else None,
                session_token=session_id,
                state_json=state_json,
                status=status,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(new_session)
            logger.debug(f"[SessionCRUD] Created session {session_id} (user_id: {user_id}, status: {status})")

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
    """
    Load session state from database.
    
    Args:
        db: Database session
        session_id: Unique session identifier
        user_id: Optional user ID for additional filtering
        
    Returns:
        Dict representation of session state, or None if not found
    """

    try:
        stmt = select(AISession).where(AISession.session_token == session_id)

        if user_id is not None:
            stmt = stmt.where(AISession.user_id == user_id)

        result = await db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            logger.debug(f"[SessionCRUD] Session {session_id} not found in DB")
            return None

        # Convert JSON → dict
        if not session.state_json:
            logger.warning(f"[SessionCRUD] Session {session_id} has no state_json")
            return None
            
        state = json.loads(session.state_json)

        logger.debug(f"[SessionCRUD] ✅ Loaded session {session_id} ({len(state)} keys)")
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

async def delete_session_state(
    db: AsyncSession,
    session_id: str
) -> bool:
    """
    Delete session from DB.
    
    Args:
        db: Database session
        session_id: Unique session identifier
        
    Returns:
        True if deleted, False if not found or error
    """

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


# ============================================================================
# LIST USER SESSIONS
# ============================================================================

async def list_user_sessions(
    db: AsyncSession,
    user_id: int,
    limit: int = 10,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List recent sessions for a user (metadata only).
    
    Args:
        db: Database session
        user_id: User ID to filter by
        limit: Maximum number of sessions to return
        status: Optional status filter ('active', 'completed', 'searched')
        
    Returns:
        List of session metadata dicts
    """

    try:
        stmt = (
            select(AISession)
            .where(AISession.user_id == user_id)
            .order_by(AISession.updated_at.desc())
            .limit(limit)
        )
        
        if status:
            stmt = stmt.where(AISession.status == status)
            
        result = await db.execute(stmt)
        sessions = result.scalars().all()

        return [
            {
                "session_id": s.session_token,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ]

    except Exception as e:
        logger.error(f"[SessionCRUD] Error listing sessions for user {user_id}: {e}", exc_info=True)
        return []


# ============================================================================
# GET SESSION METADATA
# ============================================================================

async def get_session_metadata(
    db: AsyncSession,
    session_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get session metadata without loading full state.
    
    Args:
        db: Database session
        session_id: Unique session identifier
        
    Returns:
        Session metadata dict or None
    """
    
    try:
        stmt = select(AISession).where(AISession.session_token == session_id)
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            return None
            
        return {
            "session_id": session.session_token,
            "user_id": session.user_id,
            "status": session.status,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        }
        
    except Exception as e:
        logger.error(f"[SessionCRUD] Error getting metadata for {session_id}: {e}")
        return None


# ============================================================================
# CLEANUP OLD SESSIONS
# ============================================================================

async def cleanup_old_sessions(
    db: AsyncSession,
    days_old: int = 30
) -> int:
    """
    Delete sessions older than X days.
    
    Args:
        db: Database session
        days_old: Age threshold in days
        
    Returns:
        Number of sessions deleted
    """

    try:
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        stmt = delete(AISession).where(AISession.updated_at < cutoff_date)
        result = await db.execute(stmt)
        await db.commit()

        deleted_count = result.rowcount
        logger.info(f"[SessionCRUD] ✅ Cleaned {deleted_count} old sessions (>{days_old} days)")

        return deleted_count

    except Exception as e:
        await db.rollback()
        logger.error(f"[SessionCRUD] ❌ Error cleaning sessions: {e}", exc_info=True)
        return 0


# ============================================================================
# UPDATE SESSION STATUS
# ============================================================================

async def update_session_status(
    db: AsyncSession,
    session_id: str,
    status: str
) -> bool:
    """
    Update only the status field of a session.
    
    Args:
        db: Database session
        session_id: Unique session identifier
        status: New status ('active', 'completed', 'searched', 'expired')
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        stmt = (
            update(AISession)
            .where(AISession.session_token == session_id)
            .values(
                status=status,
                updated_at=datetime.utcnow()
            )
        )
        result = await db.execute(stmt)
        await db.commit()
        
        updated = result.rowcount > 0
        
        if updated:
            logger.info(f"[SessionCRUD] ✅ Updated session {session_id} status to '{status}'")
        else:
            logger.warning(f"[SessionCRUD] Session {session_id} not found for status update")
            
        return updated
        
    except Exception as e:
        await db.rollback()
        logger.error(f"[SessionCRUD] ❌ Error updating session status: {e}", exc_info=True)
        return False