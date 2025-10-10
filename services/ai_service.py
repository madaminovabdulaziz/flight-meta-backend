# services/ai_service.py
"""
AI-Powered Flight Search Service
Converts natural language queries into structured flight search parameters.

Example:
    "I need a cheap flight to Dubai next Friday under $300"
    → {origin: "TAS", destination: "DXB", date: "2025-10-17", max_price: 300}
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio

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
    
    def __init__(self):
        self.model = "gpt-4o"  # Best for function calling
        self.conversation_history: Dict[str, List[Dict]] = {}
        
    def _get_function_schema(self) -> List[Dict[str, Any]]:
        """
        Define the function schema for OpenAI to extract flight parameters.
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
                            "description": "Origin airport IATA code (3 letters). If not specified, default to TAS (Tashkent)."
                        },
                        "destination": {
                            "type": "string",
                            "description": "Destination airport IATA code (3 letters). Required."
                        },
                        "departure_date": {
                            "type": "string",
                            "description": "Departure date in YYYY-MM-DD format. If user says 'next Friday', calculate the actual date."
                        },
                        "return_date": {
                            "type": "string",
                            "description": "Return date in YYYY-MM-DD format. Null for one-way trips."
                        },
                        "trip_type": {
                            "type": "string",
                            "enum": ["one-way", "round-trip"],
                            "description": "Type of trip. Default to round-trip unless specified."
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
                            "description": "Maximum price in USD. Extract from phrases like 'under $300', 'cheap', 'budget'."
                        },
                        "flexible_dates": {
                            "type": "boolean",
                            "description": "True if user wants to see prices for nearby dates. Keywords: 'flexible', 'around', 'sometime'."
                        },
                        "non_stop": {
                            "type": "boolean",
                            "description": "True if user prefers direct flights only. Default: false"
                        },
                        "user_intent": {
                            "type": "string",
                            "description": "The user's primary goal: 'find_cheapest', 'find_fastest', 'find_best_value', 'compare_options'"
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
                            "description": "The clarification question to ask the user"
                        },
                        "missing_field": {
                            "type": "string",
                            "description": "Which parameter needs clarification: origin, destination, date, passengers, etc."
                        }
                    },
                    "required": ["question", "missing_field"]
                }
            }
        ]
    
    def _build_system_prompt(self) -> str:
        """
        System prompt that teaches the AI how to understand flight queries.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        return f"""You are an AI flight search assistant for SkySearch AI, helping users find the perfect flights.

Today's date: {today}
Default origin: Tashkent (TAS) - use this if user doesn't specify origin

Your job:
1. Extract flight search parameters from natural language queries
2. Handle ambiguous dates ("next Friday" → calculate actual date)
3. Understand price constraints ("cheap", "under $300", "budget")
4. Infer missing information with smart defaults
5. Ask clarifying questions only when absolutely necessary

Examples:
- "Cheap flight to Dubai next week" → origin: TAS, destination: DXB, dates: next week range, max_price: infer "cheap"
- "I need to be in Istanbul on the 25th" → destination: IST, arrival date: 25th of current/next month
- "Weekend trip to Moscow" → round-trip, Friday-Sunday dates
- "2 tickets to Seoul in November" → adults: 2, destination: ICN, month: November

Language support: English, Russian, Uzbek
Be helpful, concise, and accurate."""
    
    async def parse_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        conversation_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Parse natural language query into structured flight search parameters.
        
        Args:
            query: User's natural language query
            user_id: Optional user ID for conversation tracking
            conversation_context: Previous messages in conversation
        
        Returns:
            Dict with either:
            - search_params: Structured parameters for flight search
            - clarification_needed: Question to ask user
        """
        try:
            # Build conversation context
            messages = [{"role": "system", "content": self._build_system_prompt()}]
            
            if conversation_context:
                messages.extend(conversation_context)
            
            messages.append({"role": "user", "content": query})
            
            # Call OpenAI with function calling
            logger.info(f"Parsing query: {query[:100]}...")
            
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model=self.model,
                messages=messages,
                functions=self._get_function_schema(),
                function_call="auto",
                temperature=0.3  # Lower temperature for more consistent extraction
            )
            
            message = response.choices[0].message
            
            # Check if AI wants to call a function
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                if function_name == "search_flights":
                    # Successfully extracted search parameters
                    params = self._normalize_params(function_args)
                    
                    return {
                        "success": True,
                        "search_params": params,
                        "ai_interpretation": message.content or "Understood!",
                        "needs_clarification": False
                    }
                
                elif function_name == "ask_clarification":
                    # AI needs more information
                    return {
                        "success": False,
                        "needs_clarification": True,
                        "question": function_args["question"],
                        "missing_field": function_args["missing_field"]
                    }
            
            # Fallback: AI couldn't extract parameters
            return {
                "success": False,
                "error": "Could not understand query. Please be more specific.",
                "ai_response": message.content
            }
            
        except Exception as e:
            logger.error(f"AI query parsing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"AI service error: {str(e)}",
                "fallback_to_traditional_search": True
            }
    
    def _normalize_params(self, raw_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate extracted parameters.
        Apply smart defaults and date calculations.
        """
        params = {}
        
        # IATA codes: ensure uppercase, 3 letters
        params["origin"] = raw_params.get("origin", "TAS").upper()[:3]
        params["destination"] = raw_params["destination"].upper()[:3]
        
        # Dates
        params["departure_date"] = raw_params["departure_date"]
        params["return_date"] = raw_params.get("return_date")
        
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
        
        # Price constraints
        if "max_price" in raw_params:
            params["max_price"] = raw_params["max_price"]
        
        # Preferences
        params["flexible_dates"] = raw_params.get("flexible_dates", False)
        params["non_stop"] = raw_params.get("non_stop", False)
        
        # User intent
        params["user_intent"] = raw_params.get("user_intent", "find_best_value")
        
        return params
    
    async def refine_search(
        self,
        original_query: str,
        refinement: str,
        previous_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle follow-up refinements like "make it cheaper" or "show business class".
        
        Example:
            Original: "Flights to Dubai next week"
            Refinement: "Actually, make it business class and non-stop only"
        """
        context = [
            {"role": "user", "content": original_query},
            {"role": "assistant", "content": f"Found flights with: {json.dumps(previous_params)}"},
            {"role": "user", "content": refinement}
        ]
        
        result = await self.parse_query(refinement, conversation_context=context)
        
        if result.get("success"):
            # Merge with previous params (new values override)
            updated_params = {**previous_params, **result["search_params"]}
            result["search_params"] = updated_params
        
        return result
    
    async def explain_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        user_intent: str = "find_best_value"
    ) -> str:
        """
        Generate natural language explanation of search results.
        
        Example:
            "I found 15 flights to Dubai. The cheapest is $245 on FlyDubai (1 stop).
            The fastest is $380 on Emirates (direct, 4h 30m). 
            Best value: Turkish Airlines at $298 (1 stop, great airline)."
        """
        if not results:
            return "Sorry, I couldn't find any flights matching your criteria."
        
        try:
            # Prepare summary for AI
            summary = {
                "query": query,
                "total_results": len(results),
                "cheapest": min(results, key=lambda x: x["price"]),
                "fastest": min(results, key=lambda x: x.get("duration_minutes", 999)),
                "user_intent": user_intent
            }
            
            prompt = f"""Based on this flight search, provide a concise, helpful summary:

Query: {query}
User wants: {user_intent}

Results:
- Total flights found: {summary['total_results']}
- Cheapest: ${summary['cheapest']['price']} ({summary['cheapest'].get('airline', 'Unknown')})
- Fastest: {summary['fastest'].get('duration', 'Unknown')} ({summary['fastest'].get('airline', 'Unknown')})

Write a friendly 2-3 sentence summary highlighting the best options based on what the user wants.
Be conversational and helpful."""
            
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o-mini",  # Faster model for summaries
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Result explanation failed: {e}")
            return f"Found {len(results)} flights. Sort by price or duration to see the best options."


# Singleton instance
_ai_service = None

def get_ai_service() -> AIFlightSearchService:
    """Get or create AI service singleton."""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIFlightSearchService()
    return _ai_service


# Example usage and testing
async def test_ai_service():
    """Test the AI service with sample queries."""
    ai = get_ai_service()
    
    test_queries = [
        "I need a cheap flight to Dubai next Friday",
        "Покажи мне рейсы в Москву на следующей неделе",
        "Istanbulga 2 ta chipta kerak",
        "Weekend trip to Seoul for 2 people in November",
        "Business class to London, leaving December 15th",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = await ai.parse_query(query)
        print(json.dumps(result, indent=2, ensure_ascii=False))
