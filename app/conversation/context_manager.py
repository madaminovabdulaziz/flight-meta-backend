# app/conversation/context_manager.py
"""
Conversation Context Manager
Handles persistence and retrieval of conversation state using Redis.
Clean separation: no AI logic, no caching business rules.
Integrates with your existing Redis infrastructure.
"""

import json
import uuid
import logging
from typing import Optional, List
from datetime import datetime

from app.conversation.models import ConversationContext, Message
from app.infrastructure.cache import CacheAdapter, RedisCache
from app.core.config import settings

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages conversation context persistence using Redis.
    Each conversation session is stored with a TTL.
    """
    
    # Redis key prefixes
    KEY_PREFIX = "conversation:context:"
    USER_SESSIONS_PREFIX = "user:sessions:"
    
    # Configuration from settings
    CONTEXT_TTL = settings.CONVERSATION_SESSION_TTL  # 30 minutes
    MAX_MESSAGES = settings.MAX_CONVERSATION_HISTORY  # 30 messages
    MAX_USER_SESSIONS = 10  # Maximum sessions to track per user
    
    def __init__(self, cache: Optional[CacheAdapter] = None):
        """
        Initialize context manager.
        
        Args:
            cache: Cache adapter (defaults to RedisCache if not provided)
        """
        self.cache = cache or RedisCache()
        logger.debug(f"✓ ContextManager initialized with TTL={self.CONTEXT_TTL}s")
    
    # ============================================================
    # SESSION MANAGEMENT
    # ============================================================
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        detected_origin: Optional[str] = None
    ) -> ConversationContext:
        """
        Create a new conversation session and persist it.
        
        Args:
            user_id: Optional user identifier
            detected_origin: Optional detected origin from geolocation
            
        Returns:
            New ConversationContext
        """
        session_id = str(uuid.uuid4())
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            detected_origin=detected_origin
        )
        
        await self.save_context(context)
        
        # Track user sessions for history
        if user_id:
            await self._add_to_user_sessions(user_id, session_id)
        
        logger.info(f"✓ Created new session: {session_id} (user={user_id}, origin={detected_origin})")
        return context
    
    async def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Retrieve conversation context from cache.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext if found, None otherwise
        """
        key = f"{self.KEY_PREFIX}{session_id}"
        
        try:
            data = await self.cache.get(key)
            if not data:
                logger.debug(f"Session not found: {session_id}")
                return None
            
            # Deserialize JSON to dict
            context_dict = json.loads(data)
            
            # Create ConversationContext (Pydantic validators handle enum conversion)
            context = ConversationContext(**context_dict)
            
            # Now safe to access enum .value
            logger.debug(
                f"✓ Retrieved session {session_id}: "
                f"state={context.state}, "
                f"messages={len(context.messages)}, "
                f"updated={context.updated_at.isoformat()}"
            )
            return context
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in cached context for {session_id}: {e}")
            await self.cache.delete(key)
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving context for {session_id}: {e}", exc_info=True)
            return None
    
    async def save_context(self, context: ConversationContext) -> bool:
        """
        Persist conversation context in cache with TTL.
        
        Args:
            context: ConversationContext to save
            
        Returns:
            True if successful
        """
        key = f"{self.KEY_PREFIX}{context.session_id}"
        context.updated_at = datetime.utcnow()
        
        try:
            # Serialize using Pydantic's model_dump (v2) or dict() (v1)
            try:
                context_dict = context.model_dump()  # Pydantic v2
            except AttributeError:
                context_dict = context.dict()  # Pydantic v1 fallback
            
            context_json = json.dumps(
                context_dict,
                default=self._json_encoder
            )
            
            success = await self.cache.set(key, context_json, self.CONTEXT_TTL)
            
            if success:
                logger.debug(
                    f"✓ Saved session {context.session_id}: "
                    f"state={context.state.value if hasattr(context.state, 'value') else context.state}, "
                    f"messages={len(context.messages)}"
                )
            else:
                logger.warning(f"Failed to save session {context.session_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Error saving context for {context.session_id}: {e}", exc_info=True)
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete conversation session from cache.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        key = f"{self.KEY_PREFIX}{session_id}"
        
        try:
            success = await self.cache.delete(key)
            if success:
                logger.info(f"✓ Deleted session: {session_id}")
            return success
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    async def extend_session(self, session_id: str) -> bool:
        """
        Extend TTL for an active session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        # Read and re-save with fresh TTL
        context = await self.get_context(session_id)
        if context:
            logger.debug(f"Extending TTL for session: {session_id}")
            return await self.save_context(context)
        
        logger.warning(f"Cannot extend session {session_id} - not found")
        return False
    
    async def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists without loading full context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists
        """
        key = f"{self.KEY_PREFIX}{session_id}"
        return await self.cache.exists(key)
    
    # ============================================================
    # MESSAGE MANAGEMENT
    # ============================================================
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> Optional[ConversationContext]:
        """
        Append a message to the conversation history.
        Automatically trims history if it exceeds MAX_MESSAGES.
        
        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional metadata dict
            
        Returns:
            Updated ConversationContext if successful, None otherwise
        """
        context = await self.get_context(session_id)
        if not context:
            logger.warning(f"Cannot add message to non-existent session: {session_id}")
            return None
        
        # Create message
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        context.messages.append(message)
        
        # Trim history to prevent unbounded growth
        if len(context.messages) > self.MAX_MESSAGES:
            removed_count = len(context.messages) - self.MAX_MESSAGES
            context.messages = context.messages[-self.MAX_MESSAGES:]
            logger.debug(f"Trimmed {removed_count} old messages from session {session_id}")
        
        # Save updated context
        await self.save_context(context)
        
        logger.debug(f"✓ Added {role} message to session {session_id} (total={len(context.messages)})")
        return context
    
    def get_recent_messages(
        self,
        context: ConversationContext,
        limit: int = 10
    ) -> List[Message]:
        """
        Get recent messages from context.
        
        Args:
            context: ConversationContext
            limit: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        if not context.messages:
            return []
        
        return context.messages[-limit:]
    
    def get_conversation_summary(self, context: ConversationContext) -> dict:
        """
        Get a summary of the conversation state.
        
        Args:
            context: ConversationContext
            
        Returns:
            Dict with summary information
        """
        return {
            "session_id": context.session_id,
            "state": context.state.value,
            "message_count": len(context.messages),
            "has_origin": bool(context.trip_spec.origin or context.detected_origin),
            "has_destination": bool(context.trip_spec.destination),
            "has_dates": bool(context.trip_spec.depart_date),
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat(),
            "age_seconds": (datetime.utcnow() - context.created_at).total_seconds()
        }
    
    # ============================================================
    # USER SESSION TRACKING
    # ============================================================
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[ConversationContext]:
        """
        Get recent sessions for a specific user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of ConversationContexts
        """
        sessions = []
        
        try:
            # Get session IDs from user's session list
            session_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
            
            # Get list of session IDs (stored as list in Redis)
            session_ids = await self.cache.lrange(session_key, 0, limit - 1)
            
            # Load each session context
            for sid in session_ids:
                ctx = await self.get_context(sid)
                if ctx:
                    sessions.append(ctx)
            
            logger.debug(f"Retrieved {len(sessions)} sessions for user {user_id}")
        
        except Exception as e:
            logger.error(f"Error getting user sessions for {user_id}: {e}")
        
        return sessions
    
    async def _add_to_user_sessions(self, user_id: str, session_id: str) -> bool:
        """
        Add session to user's session list (internal).
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        try:
            session_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
            
            # Add to front of list
            await self.cache.lpush(session_key, session_id)
            
            # Keep only recent sessions
            await self.cache.ltrim(session_key, 0, self.MAX_USER_SESSIONS - 1)
            
            logger.debug(f"✓ Added session {session_id} to user {user_id} history")
            return True
        
        except Exception as e:
            logger.error(f"Error adding session to user list: {e}")
            return False
    
    # ============================================================
    # BULK OPERATIONS
    # ============================================================
    
    async def delete_user_sessions(self, user_id: str) -> int:
        """
        Delete all sessions for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of sessions deleted
        """
        try:
            session_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
            
            # Get all session IDs
            session_ids = await self.cache.lrange(session_key, 0, -1)
            
            # Delete each session
            deleted = 0
            for sid in session_ids:
                if await self.delete_session(sid):
                    deleted += 1
            
            # Delete user session list
            await self.cache.delete(session_key)
            
            logger.info(f"✓ Deleted {deleted} sessions for user {user_id}")
            return deleted
        
        except Exception as e:
            logger.error(f"Error deleting user sessions: {e}")
            return 0
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions (for maintenance).
        Redis TTL handles this automatically, but this can be used for stats.
        
        Returns:
            Number of sessions found (not necessarily deleted, Redis handles TTL)
        """
        try:
            # Count active sessions
            pattern = f"{self.KEY_PREFIX}*"
            
            # Note: We can't easily count with pattern without SCAN
            # This is a placeholder for future implementation
            logger.info("Session cleanup triggered (Redis TTL handles expiration)")
            return 0
        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    # ============================================================
    # UTILITIES
    # ============================================================
    
    @staticmethod
    def _json_encoder(obj):
        """
        Safe JSON encoder for datetime and enum serialization.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
            
        Raises:
            TypeError: If object is not serializable
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, 'value'):  # Enum
            return obj.value
        raise TypeError(f"Type {type(obj)} not serializable")
    
    # ============================================================
    # STATISTICS & MONITORING
    # ============================================================
    
    async def get_stats(self) -> dict:
        """
        Get statistics about conversation sessions.
        
        Returns:
            Dict with statistics
        """
        try:
            # This requires scanning keys, which can be slow
            # In production, consider maintaining separate counters
            
            stats = {
                "context_manager": "operational",
                "ttl_seconds": self.CONTEXT_TTL,
                "max_messages": self.MAX_MESSAGES,
                "max_user_sessions": self.MAX_USER_SESSIONS
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# ============================================================
# FACTORY FUNCTION
# ============================================================

def get_context_manager(cache: Optional[CacheAdapter] = None) -> ContextManager:
    """
    Factory function to get ContextManager instance.
    
    Args:
        cache: Optional cache adapter (defaults to RedisCache)
        
    Returns:
        ContextManager instance
    """
    return ContextManager(cache=cache)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'ContextManager',
    'get_context_manager'
]