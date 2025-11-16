"""
Session CRUD Operations
Database operations for managing conversation sessions.

This module provides async functions to:
- Save session state to MySQL
- Load session state from MySQL
- Delete sessions
- List user sessions
"""

import logging
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import AISession
from app.langgraph_flow.state import ConversationState

logger = logging.getLogger(__name__)


# ============================================================================
# SAVE AISession STATE
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
        session_id: Session identifier
        state: Complete conversation state
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Check if session exists
        stmt = select(AISession).where(AISession.session_token == session_id)
        result = await db.execute(stmt)
        existing_session = result.scalar_one_or_none()
        
        # Serialize state to JSON
        state_json = json.dumps(state, default=str)  # default=str handles datetime
        
        # Determine session status
        status = "completed" if state.get("is_complete") else "active"
        
        if existing_session:
            # Update existing session
            stmt = (
                update(AISession)
                .where(AISession.session_token == session_id)
                .values(
                    state_json=state_json,
                    status=status,
                    updated_at=datetime.utcnow()
                )
            )
            await db.execute(stmt)
            logger.debug(f"[SessionCRUD] Updated session {session_id}")
        
        else:
            # Create new session
            new_session = AISession(
                user_id=state.get("user_id"),
                session_token=session_id,
                state_json=state_json,
                status=status,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(new_session)
            logger.debug(f"[SessionCRUD] Created session {session_id}")
        
        await db.commit()
        return True
    
    except Exception as e:
        await db.rollback()
        logger.error(f"[SessionCRUD] Error saving session {session_id}: {e}", exc_info=True)
        return False


# ============================================================================
# LOAD SESSION STATE
# ============================================================================

async def load_session_state(
    db: AsyncSession,
    session_id: str,
    user_id: Optional[int] = None
) -> Optional[ConversationState]:
    """
    Load session state from database.
    
    Args:
        db: Database session
        session_id: Session identifier
        user_id: Optional user ID (for additional validation)
    
    Returns:
        ConversationState if found, None otherwise
    """
    
    try:
        # Build query
        stmt = select(AISession).where(AISession.session_token == session_id)
        
        # Optionally filter by user_id for security
        if user_id is not None:
            stmt = stmt.where(AISession.user_id == user_id)
        
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            logger.debug(f"[SessionCRUD] Session {session_id} not found")
            return None
        
        # Deserialize JSON to state
        state = json.loads(session.state_json)
        
        logger.debug(f"[SessionCRUD] Loaded session {session_id}")
        return state
    
    except Exception as e:
        logger.error(f"[SessionCRUD] Error loading session {session_id}: {e}", exc_info=True)
        return None


# ============================================================================
# DELETE SESSION
# ============================================================================

async def delete_session_state(
    db: AsyncSession,
    session_id: str
) -> bool:
    """
    Delete a session from database.
    
    Args:
        db: Database session
        session_id: Session identifier
    
    Returns:
        True if deleted, False if not found
    """
    
    try:
        stmt = delete(AISession).where(AISession.session_token == session_id)
        result = await db.execute(stmt)
        await db.commit()
        
        deleted = result.rowcount > 0
        
        if deleted:
            logger.info(f"[SessionCRUD] Deleted session {session_id}")
        else:
            logger.debug(f"[SessionCRUD] Session {session_id} not found for deletion")
        
        return deleted
    
    except Exception as e:
        await db.rollback()
        logger.error(f"[SessionCRUD] Error deleting session {session_id}: {e}", exc_info=True)
        return False


# ============================================================================
# LIST USER SESSIONS
# ============================================================================

async def list_user_sessions(
    db: AsyncSession,
    user_id: int,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    List recent sessions for a user.
    
    Args:
        db: Database session
        user_id: User identifier
        limit: Maximum number of sessions to return
    
    Returns:
        List of session metadata (without full state)
    """
    
    try:
        stmt = (
            select(AISession)
            .where(AISession.user_id == user_id)
            .order_by(AISession.updated_at.desc())
            .limit(limit)
        )
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        
        # Return metadata only (not full state)
        sessions_list = [
            {
                "session_id": s.session_token,
                "status": s.status,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in sessions
        ]
        
        logger.debug(f"[SessionCRUD] Found {len(sessions_list)} sessions for user {user_id}")
        return sessions_list
    
    except Exception as e:
        logger.error(f"[SessionCRUD] Error listing sessions for user {user_id}: {e}", exc_info=True)
        return []


# ============================================================================
# CLEANUP OLD SESSIONS
# ============================================================================

async def cleanup_old_sessions(
    db: AsyncSession,
    days_old: int = 30
) -> int:
    """
    Delete sessions older than specified days.
    
    Args:
        db: Database session
        days_old: Delete sessions older than this many days
    
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
        logger.info(f"[SessionCRUD] Cleaned up {deleted_count} old sessions (>{days_old} days)")
        
        return deleted_count
    
    except Exception as e:
        await db.rollback()
        logger.error(f"[SessionCRUD] Error cleaning up old sessions: {e}", exc_info=True)
        return 0