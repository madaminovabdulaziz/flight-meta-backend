# services/ai_service.py
"""
AI-Powered Flight Search Service - PRODUCTION READY
Converts natural language queries into structured flight search parameters.

Example:
    "I need a cheap flight to Dubai next Friday under $300"
    → {origin: "TAS", destination: "DXB", date: "2025-10-24", max_price: 300}
    
Fixes Applied:
- Updated to OpenAI tool_calls API (not deprecated function_call)
- Added IATA validation with clarification requests
- Fixed user_intent enum to match Amadeus expectations
- Enhanced error handling and logging
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio
import random

import openai
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai.api_key = settings.OPENAI_API_KEY


class AIFlightSearchService:
    """
    Converts conversational queries into structured flight search parameters.
    Uses OpenAI's function calling for reliable parameter extraction.
    """
    
    # Extended IATA Database (In production, use Redis cache + DB)
    _CITY_TO_IATA = {
        "tashkent": "TAS", "dubai": "DXB", "moscow": "SVO", "domodedovo": "DME",
        "istanbul": "IST", "london": "LHR", "heathrow": "LHR", "gatwick": "LGW",
        "seoul": "ICN", "incheon": "ICN", "new york": "JFK", "nyc": "JFK",
        "paris": "CDG", "singapore": "SIN", "tokyo": "NRT", "narita": "NRT",
        "beijing": "PEK", "shanghai": "PVG", "hong kong": "HKG",
        "los angeles": "LAX", "san francisco": "SFO", "chicago": "ORD",
        "miami": "MIA", "delhi": "DEL", "mumbai": "BOM", "bangkok": "BKK",
        "kuala lumpur": "KUL", "jakarta": "CGK", "manila": "MNL",
        "samarkand": "SKD", "bukhara": "BHK", "urgench": "UGC", "namangan": "NMA",
        "almaty": "ALA", "astana": "NQZ", "baku": "GYD", "tbilisi": "TBS",
        "yerevan": "EVN", "tehran": "IKA", "riyadh": "RUH", "jeddah": "JED",
        "cairo": "CAI", "casablanca": "CMN", "johannesburg": "JNB",
        "sydney": "SYD", "melbourne": "MEL", "auckland": "AKL"
    }
    
    def __init__(self):
        self.model = getattr(settings, 'OPENAI_MODEL', 'gpt-4o-mini')
        self.conversation_history: Dict[str, List[Dict]] = {}
        
    def _get_function_schema(self) -> List[Dict[str, Any]]:
        """
        Define the function schema for OpenAI to extract flight parameters.
        Updated to match Amadeus API expectations.
        """
        return [
            {
                "name": "search_flights",
                "description": "Search for flights based on user requirements",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {
                            "type": "string",
                            "description": "Origin city name (e.g., 'Tashkent'). If not specified, default to Tashkent."
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination city name (e.g., 'Dubai'). REQUIRED."
                        },
                        "departure_date": {
                            "type": "string",
                            "description": "Departure date in YYYY-MM-DD format. Calculate actual date from relative phrases like 'next Friday'."
                        },
                        "return_date": {
                            "type": "string",
                            "description": "Return date in YYYY-MM-DD format. Null for one-way trips."
                        },
                        "trip_type": {
                            "type": "string",
                            "enum": ["one-way", "round-trip"],
                            "description": "Type of trip. Default to round-trip unless clearly one-way."
                        },
                        "adults": {
                            "type": "integer",
                            "description": "Number of adult passengers (12+ years). Default: 1"
                        },
                        "children": {
                            "type": "integer",
                            "description": "Number of children (2-11 years). Default: 0"
                        },
                        "infants": {
                            "type": "integer",
                            "description": "Number of infants (under 2). Default: 0"
                        },
                        "cabin_class": {
                            "type": "string",
                            "enum": ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"],
                            "description": "Preferred cabin class. Default: ECONOMY"
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price in USD. Extract from 'under $300', 'cheap', 'budget'. Convert currencies."
                        },
                        "currency": {
                            "type": "string",
                            "enum": ["USD", "EUR", "RUB", "UZS"],
                            "description": "Currency code. Default: USD"
                        },
                        "flexible_dates": {
                            "type": "boolean",
                            "description": "True if user wants nearby dates. Keywords: 'flexible', 'around', 'sometime'."
                        },
                        "non_stop": {
                            "type": "boolean",
                            "description": "True for direct flights only. Default: false"
                        },
                        "user_intent": {
                            "type": "string",
                            "enum": ["find_cheapest", "find_fastest", "find_best_value", "compare_options"],
                            "description": "User's primary goal: cheapest price, fastest time, best value, or compare options"
                        }
                    },
                    "required": ["destination", "departure_date"]
                }
            },
            {
                "name": "ask_clarification",
                "description": "Ask user for clarification when information is missing or ambiguous",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The clarification question to ask"
                        },
                        "missing_field": {
                            "type": "string",
                            "description": "Which parameter needs clarification"
                        }
                    },
                    "required": ["question", "missing_field"]
                }
            }
        ]
    
    def _build_system_prompt(self) -> str:
        """System prompt for flight search understanding."""
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""You are an AI flight search assistant for SkySearch AI.

Today's date: {today}
Default origin: Tashkent (use if user doesn't specify)

Your job:
1. Extract flight parameters from natural language
2. Output CITY NAMES (not IATA codes) - backend handles conversion
3. Calculate exact dates from relative phrases ("next Friday" → actual date)
4. Understand price constraints and convert currencies
5. Infer missing info with smart defaults
6. Ask clarifying questions ONLY when absolutely necessary
7. Support multiple languages (English, Russian, Uzbek)

User Intent Detection:
- "cheapest", "budget", "lowest price" → find_cheapest
- "fastest", "quickest", "shortest time" → find_fastest
- "best deal", "good value" → find_best_value
- "show me options", "compare" → compare_options

Examples:
✓ "Cheap flight to Dubai next week" → destination: Dubai, dates: next week range, user_intent: find_cheapest
✓ "I need to be in Istanbul on the 25th" → destination: Istanbul, departure_date: 2025-10-25, trip_type: one-way
✓ "Weekend trip to Moscow" → round-trip, Friday-Sunday, destination: Moscow
✓ "2 tickets to Seoul in November under 500 bucks" → adults: 2, destination: Seoul, month: November, max_price: 500

Be helpful, accurate, and conversational."""
    
    async def _call_openai(self, **kwargs) -> Any:
        """Wrapper for OpenAI with retry logic."""
        retries = 3
        for attempt in range(retries):
            try:
                return await asyncio.to_thread(openai.chat.completions.create, **kwargs)
            except openai.OpenAIError as e:
                if attempt == retries - 1:
                    logger.error(f"OpenAI call failed after {retries} attempts: {e}")
                    raise
                delay = (2 ** attempt) + random.uniform(0, 0.2)
                logger.warning(f"OpenAI retry {attempt + 1}/{retries} after {e}, backoff {delay:.2f}s")
                await asyncio.sleep(delay)
        raise Exception("Max retries exceeded for OpenAI")
    
    async def parse_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language query into structured flight search parameters.
        
        Returns:
            Dict with either:
            - success=True, search_params: Structured parameters
            - success=False, needs_clarification: Question to ask user
        """
        try:
            # Build conversation context
            messages = [{"role": "system", "content": self._build_system_prompt()}]
            
            if user_id and user_id in self.conversation_history:
                messages.extend(self.conversation_history[user_id])
            elif conversation_context:
                messages.extend(conversation_context)
            
            messages.append({"role": "user", "content": query})
            
            logger.info(f"Parsing query: {query[:100]}...")
            
            # Call OpenAI with tool calling
            response = await self._call_openai(
                model=self.model,
                messages=messages,
                tools=[{"type": "function", "function": f} for f in self._get_function_schema()],
                tool_choice="auto",
                temperature=0.3
            )
            
            message = response.choices[0].message
            
            # Update conversation history
            if user_id:
                if user_id not in self.conversation_history:
                    self.conversation_history[user_id] = []
                
                self.conversation_history[user_id].append({"role": "user", "content": query})
                
                # Store assistant response
                if message.tool_calls:
                    self.conversation_history[user_id].append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in message.tool_calls
                        ]
                    })
                elif message.content:
                    self.conversation_history[user_id].append({
                        "role": "assistant",
                        "content": message.content
                    })
                
                # Keep last 20 messages
                self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
            
            # Process tool calls
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_flights":
                    # Normalize and validate parameters
                    try:
                        params = self._normalize_params(function_args)
                    except ValueError as e:
                        # IATA resolution failed - ask for clarification
                        return {
                            "success": False,
                            "needs_clarification": True,
                            "question": str(e),
                            "missing_field": "destination"
                        }
                    
                    return {
                        "success": True,
                        "search_params": params,
                        "ai_interpretation": message.content or "Understood your search request.",
                        "needs_clarification": False
                    }
                
                elif function_name == "ask_clarification":
                    return {
                        "success": False,
                        "needs_clarification": True,
                        "question": function_args["question"],
                        "missing_field": function_args["missing_field"]
                    }
            
            # Fallback: AI couldn't extract parameters
            return {
                "success": False,
                "error": "Could not extract flight parameters.",
                "ai_response": message.content or "Please try rephrasing your search query."
            }
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"AI service error: {str(e)}",
                "fallback_to_traditional_search": True
            }
        except Exception as e:
            logger.error(f"AI query parsing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "fallback_to_traditional_search": True
            }

    def _resolve_city_to_iata(self, city_name: str) -> Optional[str]:
        """
        Convert city name to IATA code with validation.
        Returns None if city cannot be resolved.
        """
        normalized_name = city_name.lower().strip()
        
        # Direct match
        iata_code = self._CITY_TO_IATA.get(normalized_name)
        if iata_code:
            return iata_code
        
        # Partial match
        for city, iata in self._CITY_TO_IATA.items():
            if normalized_name in city or city in normalized_name:
                logger.info(f"Partial match: '{city_name}' → {city} ({iata})")
                return iata
        
        logger.warning(f"Could not resolve city: '{city_name}'")
        return None

    def _normalize_params(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate extracted parameters.
        Raises ValueError if critical parameters cannot be resolved.
        """
        params = {}
        
        # Origin (with fallback)
        origin_city = raw_params.get("origin", "Tashkent")
        params["origin"] = self._resolve_city_to_iata(origin_city)
        if not params["origin"]:
            logger.warning(f"Origin '{origin_city}' not found, defaulting to TAS")
            params["origin"] = "TAS"
        
        # Destination (REQUIRED)
        destination_city = raw_params["destination"]
        params["destination"] = self._resolve_city_to_iata(destination_city)
        if not params["destination"]:
            raise ValueError(
                f"I couldn't find the airport for '{destination_city}'. "
                f"Could you specify a different city or provide the 3-letter airport code?"
            )
        
        # Dates
        params["departure_date"] = self._resolve_date(raw_params["departure_date"])
        params["return_date"] = self._resolve_date(raw_params.get("return_date"))
        
        # Trip type
        params["trip_type"] = raw_params.get("trip_type", "round-trip")
        if params["return_date"] is None:
            params["trip_type"] = "one-way"
        
        # Passengers
        params["adults"] = max(1, raw_params.get("adults", 1))
        params["children"] = max(0, raw_params.get("children", 0))
        params["infants"] = max(0, raw_params.get("infants", 0))
        
        # Cabin class
        params["cabin_class"] = raw_params.get("cabin_class", "ECONOMY")
        
        # Price with currency conversion
        currency = raw_params.get("currency", "USD")
        max_price = raw_params.get("max_price")
        if max_price:
            # Simplified conversion rates
            conversion_rates = {
                "EUR": 1.08, "RUB": 0.011, "UZS": 0.000083
            }
            if currency != "USD":
                max_price *= conversion_rates.get(currency, 1.0)
            params["max_price"] = round(max_price, 2)
        else:
            params["max_price"] = None
        
        params["currency"] = "USD"  # Always normalize to USD
        
        # Preferences
        params["flexible_dates"] = raw_params.get("flexible_dates", False)
        params["non_stop"] = raw_params.get("non_stop", False)
        
        # User intent (already in correct format from AI)
        params["user_intent"] = raw_params.get("user_intent", "find_best_value")
        
        return params
    
    def _resolve_date(self, date_str: Optional[str]) -> Optional[str]:
        """Resolve relative dates to YYYY-MM-DD format."""
        if not date_str:
            return None
        
        # Try parsing as YYYY-MM-DD
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            pass
        
        # Handle relative dates
        today = datetime.now()
        
        # Next weekday
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, name in enumerate(day_names):
            if f"next {name}" in date_str.lower():
                days_ahead = (i - today.weekday()) % 7
                if days_ahead <= 0:
                    days_ahead += 7
                return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # Next week/month
        if "next week" in date_str.lower():
            return (today + timedelta(days=7)).strftime("%Y-%m-%d")
        elif "next month" in date_str.lower():
            next_month = today.replace(day=1) + timedelta(days=32)
            return next_month.replace(day=1).strftime("%Y-%m-%d")
        
        # Fallback
        logger.warning(f"Could not resolve date '{date_str}', using today")
        return today.strftime("%Y-%m-%d")
    
    async def refine_search(
        self,
        original_query: str,
        refinement: str,
        previous_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle follow-up refinements like 'make it cheaper' or 'show business class'."""
        context = [
            {"role": "user", "content": original_query},
            {"role": "assistant", "content": f"Previous search: {json.dumps(previous_params)}"},
            {"role": "user", "content": refinement}
        ]
        
        result = await self.parse_query(refinement, conversation_context=context)
        
        if result.get("success"):
            # Merge old and new params
            updated_params = {**previous_params, **result["search_params"]}
            result["search_params"] = updated_params
        
        return result
    
    async def explain_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        user_intent: str = "find_best_value"
    ) -> str:
        """Generate natural language explanation of search results."""
        if not results:
            return "Sorry, I couldn't find any flights matching your criteria. Try adjusting your dates or preferences."
        
        try:
            cheapest = min(results, key=lambda x: x["price"])
            fastest = min(results, key=lambda x: x.get("duration_minutes", 999))
            
            result_summary = json.dumps(results[:5], indent=2)[:1000]  # Limit size
            
            prompt = f"""Summarize these flight search results conversationally.

User Query: {query}
User Intent: {user_intent}
Total Results: {len(results)}

Top Results (first 5):
{result_summary}

Key Stats:
- Cheapest: ${cheapest['price']:.2f}
- Fastest: {fastest.get('duration', 'N/A')}

Write a friendly 2-3 sentence summary highlighting the best options based on user intent. Be specific about prices."""
            
            response = await self._call_openai(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Result explanation failed: {e}")
            return f"Found {len(results)} flights starting from ${min(r['price'] for r in results):.2f}."


# Singleton instance
_ai_service = None

def get_ai_service() -> AIFlightSearchService:
    """Get or create AI service singleton."""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIFlightSearchService()
    return _ai_service