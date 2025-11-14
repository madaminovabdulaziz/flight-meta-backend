"""
Conversation Module
AI-powered conversational flight search interface.
"""

from app.conversation.state_machine import (
    ConversationContext,
    Suggestion,
    SuggestionType,
    StateMachine
)
from app.conversation.context_manager import ContextManager, CacheAdapter
from app.conversation.suggestion_engine import SuggestionEngine

__all__ = [
    "ConversationContext",
    "Suggestion",
    "SuggestionType",
    "StateMachine",
    "SuggestionGenerator",
    "ContextManager",
    "ConversationCache",
    "AISuggestionEngine",
]