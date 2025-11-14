# app/conversation/ai_adapter.py
"""
AI Adapter for Gemini 2.5 Flash Integration
Production-ready with retry logic, proper error handling, and JSON parsing.
Based on proven Gemini API patterns.
"""

import json
import asyncio
import logging
import random
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
import httpx

from app.conversation.models import TripSpec, ConversationContext
from app.conversation.prompts import PromptTemplates, PromptRegistry, PromptVersion

# Setup logging
logger = logging.getLogger(__name__)


class AIAdapterError(Exception):
    """Base exception for AI adapter errors"""
    pass


class GeminiAdapter:
    """
    Gemini 2.5 Flash adapter for conversational flight search.
    Uses production-proven retry logic and error handling.
    """
    
    # Gemini API configuration
    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    # Tashkent timezone for consistent date handling
    TASHKENT_TZ = timezone(timedelta(hours=5))
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        timeout: float = 60.0,
        max_retries: int = 4,
        base_delay: float = 0.5
    ):
        """
        Initialize Gemini adapter.
        
        Args:
            api_key: Gemini API key from environment
            model_name: Gemini model identifier
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts on failure
            base_delay: Base delay for exponential backoff
        """
        if not api_key:
            raise AIAdapterError("Gemini API key is required")
        
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = httpx.AsyncClient(timeout=timeout)
        
        # Construct API URL
        self.api_url = f"{self.GEMINI_API_BASE}/models/{self.model_name}:generateContent"
        
        logger.info(f"✓ Initialized GeminiAdapter with model={model_name}")
    
    # ============================================================
    # PUBLIC API - HIGH-LEVEL METHODS
    # ============================================================
    
    async def extract_trip_spec_from_text(
        self,
        text: str,
        today: Optional[str] = None,
        locale: str = "en",
        context: Optional[str] = None,
        prompt_version: Optional[PromptVersion] = None
    ) -> TripSpec:
        """
        Extract structured trip parameters from natural language text.
        
        Args:
            text: User's natural language query
            today: Current date in YYYY-MM-DD format
            locale: User's locale (en, ru, uz)
            context: Optional conversation context
            prompt_version: Prompt version to use (for A/B testing)
            
        Returns:
            TripSpec with extracted parameters
            
        Example:
            Input: "I want to fly to Tokyo next weekend for 2 people"
            Output: TripSpec(destination="TYO", passengers=2, flexible_dates=True)
        """
        today = today or self._get_today_tashkent()
        
        # Build prompt using registry
        prompt = PromptRegistry.get_prompt_for_extraction(
            text=text,
            today=today,
            locale=locale,
            context=context,
            version=prompt_version
        )
        
        try:
            # Call Gemini with retry logic
            response = await self._call_gemini(
                prompt=prompt,
                temperature=0.2,
                max_tokens=2048
            )
            
            # Parse JSON response
            trip_data = self._parse_json_response(response)
            
            # Validate and build TripSpec
            return TripSpec(
                origin=trip_data.get("origin"),
                destination=trip_data.get("destination"),
                depart_date=trip_data.get("depart_date"),
                return_date=trip_data.get("return_date"),
                passengers=trip_data.get("passengers", 1),
                flexible_dates=trip_data.get("flexible_dates", False),
                flexibility_days=trip_data.get("flexibility_days")
            )
        
        except Exception as e:
            logger.error(f"Error extracting trip spec: {e}", exc_info=True)
            # Return empty TripSpec on failure (fallback to rule-based)
            return TripSpec()
    
    async def extract_preferences(
        self,
        text: str,
        context: Optional[ConversationContext] = None
    ) -> Dict[str, Any]:
        """
        Extract travel preferences from user input.
        
        Args:
            text: User's preference statement
            context: Optional conversation context for better understanding
            
        Returns:
            Dict with preferences and suggested chips
            
        Example:
            Input: "I prefer direct flights in the morning"
            Output: {
                "preferences": {"direct_only": true, "time_preference": "morning"},
                "chips": ["direct flights", "morning departure", "skip"]
            }
        """
        # Build context string if available
        context_str = None
        if context and context.trip_spec.destination:
            context_str = f"Trip to {context.trip_spec.destination}"
        
        prompt = PromptTemplates.build_preferences_prompt(text, context_str)
        
        try:
            response = await self._call_gemini(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1024
            )
            return self._parse_json_response(response)
        
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}", exc_info=True)
            # Return empty preferences on failure
            return {"preferences": {}, "chips": ["skip"]}
    
    async def generate_reply(
        self,
        context: ConversationContext,
        user_message: str
    ) -> str:
        """
        Generate a natural conversational reply for edge cases.
        Use sparingly - most replies should be rule-based.
        
        Args:
            context: Current conversation context
            user_message: User's latest message
            
        Returns:
            Generated reply text
        """
        trip_info = None
        if context.trip_spec.destination:
            trip_info = f"{context.trip_spec.destination}"
        
        prompt = PromptTemplates.build_reply_prompt(user_message, trip_info)
        
        try:
            response = await self._call_gemini(
                prompt=prompt,
                temperature=0.7,
                max_tokens=512
            )
            return response.strip()
        
        except Exception as e:
            logger.error(f"Error generating reply: {e}", exc_info=True)
            return "I'm here to help you find flights. Could you tell me where you'd like to go?"
    
    async def generate_clarification_question(
        self,
        text: str,
        issue: str
    ) -> str:
        """
        Generate a clarifying question when user input is ambiguous.
        
        Args:
            text: User's unclear query
            issue: What's unclear (e.g., "No destination mentioned")
            
        Returns:
            Natural clarifying question
        """
        prompt = PromptTemplates.build_clarification_prompt(text, issue)
        
        try:
            response = await self._call_gemini(
                prompt=prompt,
                temperature=0.5,
                max_tokens=256
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating clarification: {e}", exc_info=True)
            return "Could you provide more details about your travel plans?"
    
    # ============================================================
    # GEMINI API CALL IMPLEMENTATION (Based on your code)
    # ============================================================
    
    async def _call_gemini(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """
        Call Gemini API with retry logic and exponential backoff.
        Based on production-proven pattern from duffel_ai_service.
        
        Args:
            prompt: Formatted prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum response tokens
            
        Returns:
            Raw response text
            
        Raises:
            AIAdapterError: If all retries fail
        """
        # Build request payload (following your pattern)
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "topK": 40,
                "topP": 0.8,
                "maxOutputTokens": max_tokens
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        last_exception = None
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Gemini API call attempt {attempt + 1}/{self.max_retries}")
                
                # Make API request
                response = await self.client.post(
                    f"{self.api_url}?key={self.api_key}",
                    json=payload,
                    headers=headers
                )
                
                # Check for HTTP errors
                if response.status_code != 200:
                    logger.error(f"Gemini API HTTP error {response.status_code}: {response.text}")
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response
                    )
                
                # Parse response
                data = response.json()
                
                # Extract text from nested structure
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    logger.debug(f"✓ Gemini response received ({len(text)} chars)")
                    return text.strip()
                
                except (KeyError, IndexError) as e:
                    logger.error(f"Unexpected Gemini response format: {data}")
                    raise AIAdapterError(f"Invalid Gemini API response structure: {e}")
            
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_exception = e
                logger.warning(f"Gemini API error on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter (from your code)
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    logger.info(f"Retrying after {delay:.2f}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")
                    raise AIAdapterError(
                        f"Gemini API failed after {self.max_retries} attempts: {last_exception}"
                    )
            
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}", exc_info=True)
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise AIAdapterError(f"Unexpected error calling Gemini: {e}")
        
        # Should never reach here, but just in case
        raise AIAdapterError(f"Failed after all retries: {last_exception}")
    
    # ============================================================
    # RESPONSE PARSING (Based on your JSON cleaning logic)
    # ============================================================
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse and validate JSON response from Gemini.
        Strips markdown code blocks if present (from your code pattern).
        
        Args:
            response: Raw Gemini response
            
        Returns:
            Parsed JSON dict
            
        Raises:
            AIAdapterError: If response is not valid JSON
        """
        try:
            # Strip whitespace
            text = response.strip()
            
            # Clean markdown code blocks (from your duffel code)
            if text.startswith("```json"):
                text = text[7:]  # Remove ```json
            elif text.startswith("```"):
                text = text[3:]  # Remove ```
            
            if text.endswith("```"):
                text = text[:-3]  # Remove trailing ```
            
            text = text.strip()
            
            # Try to find JSON object even if surrounded by text
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx + 1]
            
            # Parse JSON
            parsed = json.loads(text)
            
            logger.debug(f"✓ Successfully parsed JSON response")
            return parsed
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON. Response was: {response[:200]}...")
            raise AIAdapterError(f"Invalid JSON response from Gemini: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            raise AIAdapterError(f"Error parsing Gemini response: {e}")
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def _get_today_tashkent(self) -> str:
        """
        Get current date in Tashkent timezone (consistent with your code).
        
        Returns:
            Date string in YYYY-MM-DD format
        """
        return datetime.now(self.TASHKENT_TZ).date().isoformat()
    
    async def close(self):
        """Close HTTP client and cleanup resources"""
        await self.client.aclose()
        logger.info("✓ GeminiAdapter closed")


# ============================================================
# FACTORY FUNCTION (For backward compatibility with AIAdapter)
# ============================================================

class AIAdapter(GeminiAdapter):
    """
    Alias for GeminiAdapter to maintain backward compatibility.
    Since we're only using Gemini, this is just a passthrough.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 4
    ):
        """
        Initialize AI adapter (Gemini only).
        
        Args:
            model_name: Gemini model identifier
            api_key: API key (loaded from config if not provided)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        # Import config here to avoid circular imports
        try:
            from app.core.config import settings
            api_key = api_key or settings.GEMINI_API_KEY
        except ImportError:
            logger.warning("Could not import settings, using provided api_key")
        
        if not api_key:
            raise AIAdapterError("GEMINI_API_KEY not configured")
        
        # Initialize parent GeminiAdapter
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            timeout=float(timeout),
            max_retries=max_retries
        )


# ============================================================
# FALLBACK HANDLER (Rule-based extraction)
# ============================================================

class RuleBasedFallback:
    """
    Rule-based extraction fallback when Gemini is unavailable.
    Uses simple pattern matching for common queries.
    """
    
    @staticmethod
    def extract_destination(text: str) -> Optional[str]:
        """Extract destination using simple pattern matching"""
        import re
        
        patterns = [
            r"to\s+([A-Z]{3})\b",  # "to IST"
            r"to\s+(\w+)",  # "to Istanbul"
            r"fly(?:ing)?\s+(?:to\s+)?(\w+)",  # "flying Tokyo"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                destination = match.group(1).upper()
                # If it's 3 letters, assume IATA code
                if len(destination) == 3 and destination.isalpha():
                    return destination
                return destination
        
        return None
    
    @staticmethod
    def extract_passengers(text: str) -> int:
        """Extract passenger count using pattern matching"""
        import re
        
        # Look for numbers before "passenger", "people", "person"
        pattern = r"(\d+)\s*(?:passenger|people|person|pax|traveler)"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            count = int(match.group(1))
            # Clamp between 1 and 9
            return max(1, min(9, count))
        
        return 1  # Default
    
    @staticmethod
    def extract_dates(text: str) -> Dict[str, Optional[str]]:
        """Extract dates using pattern matching (basic)"""
        import re
        from datetime import datetime, timedelta
        
        result = {"depart_date": None, "return_date": None}
        
        # Look for relative dates
        if re.search(r"next\s+week", text, re.IGNORECASE):
            result["depart_date"] = (datetime.now() + timedelta(days=7)).date().isoformat()
        elif re.search(r"this\s+weekend", text, re.IGNORECASE):
            # Find next Saturday
            days_ahead = 5 - datetime.now().weekday()
            if days_ahead <= 0:
                days_ahead += 7
            result["depart_date"] = (datetime.now() + timedelta(days=days_ahead)).date().isoformat()
        elif re.search(r"tomorrow", text, re.IGNORECASE):
            result["depart_date"] = (datetime.now() + timedelta(days=1)).date().isoformat()
        
        return result


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'GeminiAdapter',
    'AIAdapter',
    'AIAdapterError',
    'RuleBasedFallback'
]