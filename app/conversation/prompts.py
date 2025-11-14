# app/conversation/prompts.py
"""
Centralized prompt templates for AI interactions.
Supports multiple languages and A/B testing of different prompt strategies.
Uses few-shot learning for improved accuracy.
"""

from typing import Dict, Optional
from string import Template
from enum import Enum
from datetime import datetime


class PromptVersion(str, Enum):
    """Versions for A/B testing prompts"""
    V1_BASIC = "v1_basic"
    V2_FEW_SHOT = "v2_few_shot"
    V3_CHAIN_OF_THOUGHT = "v3_cot"


class PromptTemplates:
    """
    Production-grade prompt templates with few-shot examples.
    All prompts optimized for JSON-only responses to prevent parsing errors.
    """
    
    # ============================================================
    # TRIP SPECIFICATION EXTRACTION
    # ============================================================
    
    EXTRACTION_FEW_SHOT = """Extract flight search parameters from the user's query.

Current date: ${today}
User locale: ${locale}

EXAMPLES:
Query: "I want to fly to Tokyo next weekend"
Output: {"destination": "TYO", "depart_date": "2025-11-16", "flexible_dates": true, "flexibility_days": 3}

Query: "Looking for 2 tickets from London to Dubai on Dec 25"
Output: {"origin": "LON", "destination": "DXB", "depart_date": "2025-12-25", "passengers": 2, "flexible_dates": false}

Query: "Cheap flights to Bali next month, I'm flexible"
Output: {"destination": "DPS", "depart_date": "2025-12-15", "flexible_dates": true, "flexibility_days": 7}

Query: "Paris round trip in February, 3 people"
Output: {"destination": "PAR", "depart_date": "2026-02-01", "return_date": "2026-02-08", "passengers": 3, "flexible_dates": true}

NOW EXTRACT FROM THIS QUERY:
User query: "${text}"
${context_info}

RULES:
- Return ONLY valid JSON, no explanations
- Use 3-letter IATA codes (NYC, LON, TYO, DXB, IST, PAR, etc.)
- Dates in YYYY-MM-DD format
- If user says "flexible" or "around", set flexible_dates: true
- Default passengers: 1
- If round trip mentioned, include return_date
- If one-way or return not mentioned, return_date: null

OUTPUT (JSON only):"""

    EXTRACTION_CHAIN_OF_THOUGHT = """You are an expert travel assistant. Analyze this flight search query carefully.

Current date: ${today}
User query: "${text}"
${context_info}

STEP-BY-STEP ANALYSIS:
1. Identify origin city/airport (if mentioned)
2. Identify destination city/airport (required)
3. Extract departure date (explicit like "Dec 25" or relative like "next week")
4. Check if return date mentioned (round trip vs one-way)
5. Count passengers (default 1)
6. Detect flexibility ("flexible", "around", "approximately")

After your analysis, output ONLY this JSON structure (no explanations before or after):
{
  "origin": "3-letter code or null",
  "destination": "3-letter code",
  "depart_date": "YYYY-MM-DD",
  "return_date": "YYYY-MM-DD or null",
  "passengers": 1,
  "flexible_dates": false,
  "flexibility_days": null
}"""

    # ============================================================
    # PREFERENCES EXTRACTION
    # ============================================================
    
    PREFERENCES_EXTRACTION = """Extract travel preferences from the user's statement.

${context_info}
User statement: "${text}"

EXAMPLES:
Input: "I prefer direct flights"
Output: {"preferences": {"direct_only": true}, "chips": ["direct flights", "any airline", "skip"]}

Input: "Morning departures only, business class"
Output: {"preferences": {"time_preference": "morning", "cabin_class": "business"}, "chips": ["morning", "afternoon", "skip"]}

Input: "Cheapest options please"
Output: {"preferences": {"sort_by": "price"}, "chips": ["cheapest", "fastest", "skip"]}

NOW EXTRACT:
Output ONLY valid JSON:
{
  "preferences": {
    "direct_only": false,
    "time_preference": null,
    "max_stops": null,
    "cabin_class": null,
    "sort_by": null
  },
  "chips": ["suggestion1", "suggestion2", "skip"]
}"""

    # ============================================================
    # CONVERSATIONAL REPLIES
    # ============================================================
    
    CONVERSATIONAL_REPLY = """You are a helpful flight search assistant for Travo.${trip_info}

User: ${user_message}

Instructions:
- Keep response under 50 words
- Be friendly and natural
- If user goes off-topic, gently guide them back to flight search
- Don't use markdown, emojis, or special formatting
- Don't repeat information already shown in the interface

Response:"""

    # ============================================================
    # CLARIFICATION PROMPTS
    # ============================================================
    
    CLARIFICATION_PROMPT = """The user's query is ambiguous. Ask ONE clarifying question.

User query: "${text}"
Issue: ${issue}

Examples:
Issue: "No destination mentioned"
Question: "Where would you like to fly to?"

Issue: "Date unclear"
Question: "When are you planning to travel?"

Issue: "Multiple destinations mentioned"
Question: "I see you mentioned Tokyo and Paris. Which one would you like to fly to?"

Generate a natural clarifying question (no explanations, just the question):"""

    # ============================================================
    # PROMPT BUILDER METHODS
    # ============================================================
    
    @classmethod
    def build_extraction_prompt(
        cls,
        text: str,
        today: str,
        locale: str = "en",
        context: Optional[str] = None,
        version: PromptVersion = PromptVersion.V2_FEW_SHOT
    ) -> str:
        """
        Build extraction prompt with specified version.
        
        Args:
            text: User's query
            today: Current date (YYYY-MM-DD)
            locale: User's locale (en, ru, uz)
            context: Optional conversation context
            version: Prompt version for A/B testing
            
        Returns:
            Formatted prompt string
        """
        context_info = ""
        if context:
            context_info = f"\nContext: {context}"
        
        if version == PromptVersion.V3_CHAIN_OF_THOUGHT:
            template = cls.EXTRACTION_CHAIN_OF_THOUGHT
        else:  # Default to few-shot
            template = cls.EXTRACTION_FEW_SHOT
        
        return Template(template).substitute(
            text=text,
            today=today,
            locale=locale,
            context_info=context_info
        )
    
    @classmethod
    def build_preferences_prompt(
        cls,
        text: str,
        context: Optional[str] = None
    ) -> str:
        """
        Build preferences extraction prompt.
        
        Args:
            text: User's preference statement
            context: Optional context (current trip details)
            
        Returns:
            Formatted prompt string
        """
        context_info = ""
        if context:
            context_info = f"Current trip: {context}\n"
        
        return Template(cls.PREFERENCES_EXTRACTION).substitute(
            text=text,
            context_info=context_info
        )
    
    @classmethod
    def build_reply_prompt(
        cls,
        user_message: str,
        trip_info: Optional[str] = None
    ) -> str:
        """
        Build conversational reply prompt.
        
        Args:
            user_message: User's message
            trip_info: Optional current trip information
            
        Returns:
            Formatted prompt string
        """
        trip_context = ""
        if trip_info:
            trip_context = f"\nCurrent trip planning: {trip_info}"
        
        return Template(cls.CONVERSATIONAL_REPLY).substitute(
            user_message=user_message,
            trip_info=trip_context
        )
    
    @classmethod
    def build_clarification_prompt(
        cls,
        text: str,
        issue: str
    ) -> str:
        """
        Build clarification question prompt.
        
        Args:
            text: User's unclear query
            issue: Description of what's unclear
            
        Returns:
            Formatted prompt string
        """
        return Template(cls.CLARIFICATION_PROMPT).substitute(
            text=text,
            issue=issue
        )


# ============================================================
# MULTILINGUAL SUPPORT
# ============================================================

class LocalizedMessages:
    """
    Localized messages for different languages.
    Used for system messages, not LLM prompts.
    """
    
    GREETINGS = {
        "en": "Hi! Where would you like to fly? ✈️",
        "ru": "Привет! Куда хотите полететь? ✈️",
        "uz": "Salom! Qayerga uchmoqchisiz? ✈️"
    }
    
    CONFIRM_DESTINATION = {
        "en": "Flying from {origin} to {destination}?",
        "ru": "Летим из {origin} в {destination}?",
        "uz": "{origin}dan {destination}ga uchmoqchimisiz?"
    }
    
    ASK_DATES = {
        "en": "When would you like to fly?",
        "ru": "Когда хотите полететь?",
        "uz": "Qachon uchmoqchisiz?"
    }
    
    ASK_PASSENGERS = {
        "en": "How many people are traveling?",
        "ru": "Сколько человек полетит?",
        "uz": "Necha kishi sayohat qiladi?"
    }
    
    ASK_PREFERENCES = {
        "en": "Any preferences? (You can skip this)",
        "ru": "Есть предпочтения? (Можете пропустить)",
        "uz": "Biror xohish bormi? (O'tkazib yuborishingiz mumkin)"
    }
    
    SEARCH_READY = {
        "en": "Great! Searching for flights...",
        "ru": "Отлично! Ищем рейсы...",
        "uz": "Ajoyib! Parvozlarni qidiryapmiz..."
    }
    
    ERROR_GENERIC = {
        "en": "Sorry, something went wrong. Let's start over.",
        "ru": "Извините, что-то пошло не так. Давайте начнем заново.",
        "uz": "Kechirasiz, xatolik yuz berdi. Qaytadan boshlaylik."
    }
    
    @classmethod
    def get(cls, key: str, locale: str = "en", **kwargs) -> str:
        """
        Get localized message.
        
        Args:
            key: Message key (e.g., 'GREETINGS')
            locale: Language code
            **kwargs: Format arguments
            
        Returns:
            Localized and formatted message
        """
        messages = getattr(cls, key, {})
        message = messages.get(locale, messages.get("en", ""))
        
        if kwargs:
            return message.format(**kwargs)
        return message


# ============================================================
# PROMPT REGISTRY FOR A/B TESTING
# ============================================================

class PromptRegistry:
    """
    Registry for tracking and switching between prompt versions.
    Useful for A/B testing to find best-performing prompts.
    """
    
    # Track which version to use (can be set via config/feature flags)
    ACTIVE_VERSION = PromptVersion.V2_FEW_SHOT
    
    # Performance tracking (to be implemented with analytics)
    VERSION_METRICS = {
        PromptVersion.V1_BASIC: {"accuracy": 0.0, "latency_ms": 0},
        PromptVersion.V2_FEW_SHOT: {"accuracy": 0.0, "latency_ms": 0},
        PromptVersion.V3_CHAIN_OF_THOUGHT: {"accuracy": 0.0, "latency_ms": 0},
    }
    
    @classmethod
    def get_active_version(cls) -> PromptVersion:
        """Get currently active prompt version"""
        return cls.ACTIVE_VERSION
    
    @classmethod
    def set_active_version(cls, version: PromptVersion):
        """Set active prompt version (for A/B testing)"""
        cls.ACTIVE_VERSION = version
    
    @classmethod
    def get_prompt_for_extraction(
        cls,
        text: str,
        today: str,
        locale: str = "en",
        context: Optional[str] = None,
        version: Optional[PromptVersion] = None
    ) -> str:
        """
        Get extraction prompt using active version.
        
        Args:
            text: User query
            today: Current date
            locale: Language
            context: Optional context
            version: Override active version (for testing)
            
        Returns:
            Formatted prompt
        """
        version = version or cls.ACTIVE_VERSION
        
        return PromptTemplates.build_extraction_prompt(
            text=text,
            today=today,
            locale=locale,
            context=context,
            version=version
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'PromptTemplates',
    'LocalizedMessages',
    'PromptRegistry',
    'PromptVersion'
]