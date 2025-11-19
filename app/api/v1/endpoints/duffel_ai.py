# # duffel_ai_service_final.py — Production AI with IP Geolocation + Same Airport Validation
# """
# Complete production-ready conversational AI for flight search with:
# - Automatic origin detection via IP geolocation
# - Same origin-destination validation (PREVENTS TAS → TAS searches)
# - Smart alternative airport suggestions
# - Exact Duffel API response format
# - Smart conversation flow
# """

# from __future__ import annotations

# import os
# import json
# import time
# import random
# import asyncio
# import logging
# import uuid
# from typing import List, Optional, Dict, Any, Tuple
# from datetime import datetime, timezone, timedelta
# from enum import Enum

# import httpx
# from fastapi import FastAPI, APIRouter, HTTPException, Header, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field, ValidationError, validator

# # Import IP geolocation service
# from services.ip_geolocation import IPGeolocationService

# # Optional Redis for session memory
# try:
#     import redis
#     REDIS_AVAILABLE = True
# except ImportError:
#     redis = None
#     REDIS_AVAILABLE = False

# # ============================================================================
# # Logging Configuration
# # ============================================================================
# LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
# logging.basicConfig(
#     level=LOG_LEVEL,
#     format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )
# logger = logging.getLogger(__name__)

# # ============================================================================
# # Configuration
# # ============================================================================
# class Config:
#     # Gemini API
#     GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
#     GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
#     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
#     # Timeouts and Retries
#     CLIENT_TIMEOUT = float(os.getenv("CLIENT_TIMEOUT", "60.0"))
#     MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
#     BASE_DELAY = float(os.getenv("BASE_DELAY", "0.5"))
    
#     # Session Management
#     SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
#     MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "20"))
    
#     # Rate Limiting
#     RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
    
#     # Redis
#     REDIS_URL = os.getenv("REDIS_URL")
    
#     # Features
#     ENABLE_PERSONALIZATION = os.getenv("ENABLE_PERSONALIZATION", "true").lower() == "true"
#     ENABLE_SMART_SUGGESTIONS = os.getenv("ENABLE_SMART_SUGGESTIONS", "true").lower() == "true"
#     ENABLE_IP_GEOLOCATION = os.getenv("ENABLE_IP_GEOLOCATION", "true").lower() == "true"

# # Timezone setup
# TASHKENT_TZ = timezone(timedelta(hours=5))

# def get_today_tashkent() -> datetime:
#     """Get current date in Tashkent timezone - always fresh!"""
#     return datetime.now(TASHKENT_TZ).date()

# # Initialize IP Geolocation Service
# ip_geo_service = IPGeolocationService()

# # ============================================================================
# # NEW: Alternative Airport Suggestions
# # ============================================================================
# ALTERNATIVE_AIRPORTS = {
#     "TAS": ["SKD", "BHK", "ALA"],  # Tashkent → Samarkand, Bukhara, Almaty
#     "SKD": ["TAS", "BHK", "DXB"],  # Samarkand → Tashkent, Bukhara, Dubai
#     "BHK": ["TAS", "SKD", "DXB"],  # Bukhara → Tashkent, Samarkand, Dubai
#     "IST": ["SAW", "AYT", "ESB"],  # Istanbul → Sabiha Gokcen, Antalya, Ankara
#     "SAW": ["IST", "AYT", "ESB"],  # Sabiha Gokcen → Istanbul, Antalya, Ankara
#     "DXB": ["AUH", "SHJ", "DOH"],  # Dubai → Abu Dhabi, Sharjah, Doha
#     "AUH": ["DXB", "SHJ", "DOH"],  # Abu Dhabi → Dubai, Sharjah, Doha
#     "LHR": ["LGW", "STN", "LCY"],  # London → Gatwick, Stansted, City
#     "LGW": ["LHR", "STN", "LCY"],  # Gatwick → Heathrow, Stansted, City
#     "JFK": ["EWR", "LGA", "PHL"],  # New York → Newark, LaGuardia, Philadelphia
#     "EWR": ["JFK", "LGA", "PHL"],  # Newark → JFK, LaGuardia, Philadelphia
#     "CDG": ["ORY", "BVA", "LHR"],  # Paris → Orly, Beauvais, London
#     "ORY": ["CDG", "BVA", "LHR"],  # Orly → CDG, Beauvais, London
#     "SVO": ["DME", "VKO", "IST"],  # Moscow → Domodedovo, Vnukovo, Istanbul
#     "DME": ["SVO", "VKO", "IST"],  # Domodedovo → Sheremetyevo, Vnukovo
# }

# AIRPORT_NAMES = {
#     "TAS": "Tashkent",
#     "SKD": "Samarkand",
#     "BHK": "Bukhara",
#     "ALA": "Almaty",
#     "IST": "Istanbul",
#     "SAW": "Istanbul Sabiha Gokcen",
#     "AYT": "Antalya",
#     "ESB": "Ankara",
#     "DXB": "Dubai",
#     "AUH": "Abu Dhabi",
#     "SHJ": "Sharjah",
#     "DOH": "Doha",
#     "LHR": "London Heathrow",
#     "LGW": "London Gatwick",
#     "STN": "London Stansted",
#     "LCY": "London City",
#     "JFK": "New York JFK",
#     "EWR": "New York Newark",
#     "LGA": "New York LaGuardia",
#     "PHL": "Philadelphia",
#     "CDG": "Paris CDG",
#     "ORY": "Paris Orly",
#     "BVA": "Paris Beauvais",
#     "SVO": "Moscow Sheremetyevo",
#     "DME": "Moscow Domodedovo",
#     "VKO": "Moscow Vnukovo",
# }

# def get_alternative_destinations(origin_iata: str) -> List[Dict[str, str]]:
#     """Get alternative destination suggestions when origin = destination"""
#     alternatives = ALTERNATIVE_AIRPORTS.get(origin_iata, ["DXB", "IST", "LHR"])
    
#     return [
#         {
#             "iata": alt,
#             "city": AIRPORT_NAMES.get(alt, alt),
#             "message": f"Try {AIRPORT_NAMES.get(alt, alt)} ({alt}) instead?"
#         }
#         for alt in alternatives[:3]
#     ]

# # ============================================================================
# # Enums
# # ============================================================================
# class PromptType(str, Enum):
#     TEXT = "TEXT"
#     DATE_PICKER = "DATE_PICKER"
#     DATE_RANGE_PICKER = "DATE_RANGE_PICKER"
#     CITY_SELECTOR = "CITY_SELECTOR"
#     PASSENGER_PICKER = "PASSENGER_PICKER"
#     CABIN_SELECTOR = "CABIN_SELECTOR"
#     CONNECTION_SELECTOR = "CONNECTION_SELECTOR"
#     TIME_RANGE_PICKER = "TIME_RANGE_PICKER"
#     MULTI_FIELD = "MULTI_FIELD"

# class ConversationStatus(str, Enum):
#     INCOMPLETE = "INCOMPLETE"
#     COMPLETE = "COMPLETE"
#     CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"
#     ERROR = "ERROR"
#     VALIDATION_ERROR = "VALIDATION_ERROR"  # NEW: For same origin-destination

# class IntentType(str, Enum):
#     SEARCH_FLIGHT = "SEARCH_FLIGHT"
#     MODIFY_SEARCH = "MODIFY_SEARCH"
#     ADD_INFO = "ADD_INFO"
#     QUESTION = "QUESTION"
#     RESET = "RESET"
#     CONFIRM = "CONFIRM"

# # ============================================================================
# # Models - EXACT Duffel Format
# # ============================================================================
# class TimeWindow(BaseModel):
#     """Time window for departure/arrival"""
#     from_: str = Field(..., alias="from", description="Start time HH:MM")
#     to: str = Field(..., description="End time HH:MM")
    
#     class Config:
#         populate_by_name = True

# class SliceOut(BaseModel):
#     """EXACT slice format for Duffel API"""
#     origin: str
#     destination: str
#     departure_date: str
#     departure_time: Optional[TimeWindow] = None
#     arrival_time: Optional[TimeWindow] = None
    
#     class Config:
#         populate_by_name = True

# class PassengerOut(BaseModel):
#     """EXACT passenger format for Duffel API"""
#     type: str  # "adult", "child", "infant_without_seat"
#     age: Optional[int] = None

# class DuffelSearchRequest(BaseModel):
#     """EXACT Duffel API request format"""
#     slices: List[SliceOut]
#     passengers: List[PassengerOut]
#     cabin_class: Optional[str] = None
#     max_connections: int = Field(1, ge=0, le=2)
#     supplier_timeout_ms: int = 10000
#     return_offers: bool = True
#     page_size: int = Field(50, ge=1, le=250)
#     currency: str = "USD"

# class ChatMessage(BaseModel):
#     role: str
#     content: str
#     timestamp: float = Field(default_factory=time.time)

# class ChatQuery(BaseModel):
#     query: str = Field(..., min_length=1, max_length=500)
#     session_id: Optional[str] = None
#     reset: bool = False
#     user_id: Optional[str] = None
    
#     @validator('query')
#     def validate_query(cls, v):
#         if not v or not v.strip():
#             raise ValueError("Query cannot be empty")
#         return v.strip()

# class SmartSuggestion(BaseModel):
#     text: str
#     action: str
#     data: Optional[Dict[str, Any]] = None

# class ConversationalResponse(BaseModel):
#     status: ConversationStatus
#     message: str
#     search_data: Optional[Dict[str, Any]] = None  # EXACT Duffel format when COMPLETE
#     missing_fields: Optional[List[str]] = None
#     prompt_type: Optional[PromptType] = None
#     suggestions: Optional[List[SmartSuggestion]] = None
#     session_id: str
#     progress: Optional[Dict[str, Any]] = None
#     detected_intent: Optional[IntentType] = None
#     conversation_turn: int = 0
#     detected_origin: Optional[Dict[str, Any]] = None  # IP-detected airport info
#     alternative_destinations: Optional[List[Dict[str, str]]] = None  # NEW

# class LLMResponse(BaseModel):
#     status: str
#     missing_fields: Optional[List[str]] = None
#     clarification_prompt: Optional[str] = None
#     search_payload: Optional[Dict[str, Any]] = None
#     detected_intent: Optional[str] = None
#     confidence: Optional[float] = None

# class ConversationHistory(BaseModel):
#     session_id: str
#     messages: List[ChatMessage] = Field(default_factory=list)
#     current_payload: Optional[Dict[str, Any]] = None
#     missing_fields: Optional[List[str]] = None
#     user_preferences: Dict[str, Any] = Field(default_factory=dict)
#     detected_origin_airport: Optional[str] = None  # Store detected airport
#     user_ip: Optional[str] = None
#     created_at: float = Field(default_factory=time.time)
#     updated_at: float = Field(default_factory=time.time)
#     turn_count: int = 0

# # ============================================================================
# # NEW: Validation Functions
# # ============================================================================
# def validate_origin_destination(origin: str, destination: str) -> Tuple[bool, Optional[str]]:
#     """
#     Validate that origin and destination are different airports
#     Returns: (is_valid, error_message)
#     """
#     if not origin or not destination:
#         return True, None
    
#     if origin.upper() == destination.upper():
#         origin_name = AIRPORT_NAMES.get(origin.upper(), origin)
#         return False, f"You can't fly from {origin_name} to {origin_name}! Please choose a different destination. ✈️"
    
#     return True, None

# def check_payload_for_same_airport(payload: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
#     """
#     Check if any slice has same origin and destination
#     Returns: (is_valid, error_message, conflicting_origin)
#     """
#     if not payload or "slices" not in payload:
#         return True, None, None
    
#     for slice_data in payload["slices"]:
#         origin = slice_data.get("origin", "").upper()
#         destination = slice_data.get("destination", "").upper()
        
#         if origin and destination and origin == destination:
#             origin_name = AIRPORT_NAMES.get(origin, origin)
#             error_msg = (
#                 f"Oops! You're trying to fly from {origin_name} to {origin_name}. "
#                 f"That's not quite a trip! 😅 Where would you actually like to go?"
#             )
#             return False, error_msg, origin
    
#     return True, None, None

# # ============================================================================
# # Session Memory
# # ============================================================================
# class MemoryABC:
#     async def get(self, key: str) -> Optional[ConversationHistory]: ...
#     async def set(self, key: str, history: ConversationHistory) -> None: ...
#     async def delete(self, key: str) -> None: ...
#     async def exists(self, key: str) -> bool: ...

# class InMemoryStore(MemoryABC):
#     def __init__(self, ttl_seconds: int = 1800, max_size: int = 1000):
#         from collections import OrderedDict
#         self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
#         self.ttl = ttl_seconds
#         self.max_size = max_size
#         self._lock = asyncio.Lock()

#     async def get(self, key: str) -> Optional[ConversationHistory]:
#         async with self._lock:
#             self._evict_expired()
#             item = self._store.get(key)
#             if not item:
#                 return None
#             self._store.move_to_end(key)
#             try:
#                 return ConversationHistory(**item['data'])
#             except Exception as e:
#                 logger.error(f"Error deserializing history for {key}: {e}")
#                 return None

#     async def set(self, key: str, history: ConversationHistory) -> None:
#         async with self._lock:
#             self._evict_expired()
#             if len(self._store) >= self.max_size and key not in self._store:
#                 self._store.popitem(last=False)
#             self._store[key] = {
#                 'data': history.dict(),
#                 'expires_at': time.time() + self.ttl
#             }
#             self._store.move_to_end(key)

#     async def delete(self, key: str) -> None:
#         async with self._lock:
#             self._store.pop(key, None)

#     async def exists(self, key: str) -> bool:
#         async with self._lock:
#             return key in self._store

#     def _evict_expired(self):
#         now = time.time()
#         expired = [k for k, v in self._store.items() if v['expires_at'] < now]
#         for k in expired:
#             self._store.pop(k, None)

# class RedisStore(MemoryABC):
#     def __init__(self, url: str, ttl_seconds: int = 1800):
#         if not REDIS_AVAILABLE:
#             raise RuntimeError("redis-py not installed")
#         self.r = redis.from_url(url, decode_responses=True)
#         self.ttl = ttl_seconds

#     async def get(self, key: str) -> Optional[ConversationHistory]:
#         try:
#             data = self.r.get(f"duffel_session:{key}")
#             if not data:
#                 return None
#             return ConversationHistory(**json.loads(data))
#         except Exception as e:
#             logger.error(f"Redis get error for {key}: {e}")
#             return None

#     async def set(self, key: str, history: ConversationHistory) -> None:
#         try:
#             self.r.setex(f"duffel_session:{key}", self.ttl, json.dumps(history.dict()))
#         except Exception as e:
#             logger.error(f"Redis set error for {key}: {e}")

#     async def delete(self, key: str) -> None:
#         try:
#             self.r.delete(f"duffel_session:{key}")
#         except Exception as e:
#             logger.error(f"Redis delete error for {key}: {e}")

#     async def exists(self, key: str) -> bool:
#         try:
#             return bool(self.r.exists(f"duffel_session:{key}"))
#         except Exception:
#             return False

# # Initialize memory store
# if Config.REDIS_URL and REDIS_AVAILABLE:
#     try:
#         memory: MemoryABC = RedisStore(Config.REDIS_URL, ttl_seconds=Config.SESSION_TTL_SECONDS)
#         logger.info("✓ Using Redis for session storage")
#     except Exception as e:
#         logger.warning(f"Redis connection failed: {e}, falling back to in-memory")
#         memory = InMemoryStore(ttl_seconds=Config.SESSION_TTL_SECONDS)
# else:
#     memory = InMemoryStore(ttl_seconds=Config.SESSION_TTL_SECONDS)
#     logger.info("✓ Using in-memory session storage")

# # ============================================================================
# # Rate Limiter
# # ============================================================================
# class RateLimiter:
#     def __init__(self, max_requests: int, window_seconds: int = 60):
#         self.max_requests = max_requests
#         self.window = window_seconds
#         self._requests: Dict[str, List[float]] = {}
#         self._lock = asyncio.Lock()

#     async def check_rate_limit(self, identifier: str) -> Tuple[bool, Optional[int]]:
#         async with self._lock:
#             now = time.time()
#             if identifier not in self._requests:
#                 self._requests[identifier] = []
            
#             self._requests[identifier] = [
#                 t for t in self._requests[identifier] 
#                 if now - t < self.window
#             ]
            
#             if len(self._requests[identifier]) >= self.max_requests:
#                 oldest = self._requests[identifier][0]
#                 retry_after = int(self.window - (now - oldest)) + 1
#                 return False, retry_after
            
#             self._requests[identifier].append(now)
#             return True, None

# rate_limiter = RateLimiter(max_requests=Config.RATE_LIMIT_PER_MINUTE, window_seconds=60)

# # ============================================================================
# # UPDATED: System Prompt with Validation Rules
# # ============================================================================
# def get_system_prompt(detected_origin: Optional[str] = None) -> str:
#     """Generate system prompt with detected origin and validation rules"""
    
#     today = get_today_tashkent()
#     tomorrow = today + timedelta(days=1)
    
#     origin_context = ""
#     if detected_origin:
#         origin_context = f"\n**USER'S DETECTED LOCATION**: The user is near {detected_origin} airport. Use this as the default origin unless they explicitly specify otherwise."
    
#     prompt = f"""You are an intelligent flight booking assistant.

# TODAY'S DATE: {today.isoformat()} (Asia/Tashkent timezone)
# {origin_context}

# CRITICAL RULES:
# 1. **Origin Airport**: 
#    - If user's origin is detected via IP geolocation, USE IT automatically
#    - DO NOT ask "where are you flying from?" if origin is already known
#    - Only ask about origin if it's not detected AND user doesn't mention it

# 2. **VALIDATION - Same Airport Check**:
#    - **NEVER** allow origin and destination to be the same airport
#    - If user says "I want to fly to Tashkent" and origin is already Tashkent, respond:
#      {{
#        "status": "INCOMPLETE",
#        "missing_fields": ["slices[0].destination"],
#        "clarification_prompt": "You're already in Tashkent! Where would you like to fly to? Perhaps Dubai, Istanbul, or Moscow?",
#        "detected_intent": "SEARCH_FLIGHT",
#        "confidence": 0.9
#      }}
#    - DO NOT include a search_payload with same origin and destination

# 3. **Time Fields**:
#    - ONLY include departure_time and arrival_time if user EXPLICITLY mentions time preferences
#    - If user says "morning flight", include: {{"from": "06:00", "to": "12:00"}}
#    - If user says "afternoon", include: {{"from": "12:00", "to": "18:00"}}
#    - If user says "evening", include: {{"from": "18:00", "to": "23:00"}}
#    - If user says "night", include: {{"from": "00:00", "to": "06:00"}}
#    - If NO time preference mentioned, EXCLUDE these fields entirely

# 4. **Output Format** (EXACT Duffel API format):
# {{
#   "status": "COMPLETE" | "INCOMPLETE",
#   "detected_intent": "SEARCH_FLIGHT" | "MODIFY_SEARCH" | "ADD_INFO",
#   "confidence": 0.0-1.0,
#   "missing_fields": ["field.path"],
#   "clarification_prompt": "friendly question",
#   "search_payload": {{
#     "slices": [{{
#       "origin": "IATA_CODE",
#       "destination": "IATA_CODE",  // MUST BE DIFFERENT FROM ORIGIN
#       "departure_date": "YYYY-MM-DD",
#       "departure_time": {{"from": "HH:MM", "to": "HH:MM"}},  // OPTIONAL
#       "arrival_time": {{"from": "HH:MM", "to": "HH:MM"}}     // OPTIONAL
#     }}],
#     "passengers": [{{"type": "adult|child|infant_without_seat", "age": number}}],
#     "cabin_class": "economy|premium_economy|business|first",
#     "max_connections": 0-2,
#     "supplier_timeout_ms": 10000,
#     "return_offers": true,
#     "page_size": 50,
#     "currency": "USD"
#   }}
# }}

# 5. **Date Interpretation**:
#    - "tomorrow" = {tomorrow.isoformat()}
#    - "next week" = 7 days from today
#    - "Christmas" = 2025-12-25
#    - "New Year" = 2026-01-01

# 6. **IATA Codes** (major airports):
#    - New York → JFK, London → LHR, Paris → CDG
#    - Dubai → DXB, Istanbul → IST, Moscow → SVO
#    - Tokyo → NRT, Hong Kong → HKG, Singapore → SIN
#    - Tashkent → TAS, Samarkand → SKD, Bukhara → BHK

# 7. **Default Values**:
#    - cabin_class: "economy"
#    - passengers: [{{"type": "adult"}}]
#    - max_connections: 1
#    - supplier_timeout_ms: 10000
#    - return_offers: true
#    - page_size: 50
#    - currency: "USD"

# 8. **Conversation Flow**:
#    - Be friendly and natural
#    - Only ask about missing REQUIRED fields: destination, departure_date
#    - Don't ask about optional fields unless user seems interested
#    - If origin = destination, suggest popular alternatives

# EXAMPLES:

# User: "I want to fly to Dubai tomorrow"
# [Origin detected: TAS]
# {{
#   "status": "COMPLETE",
#   "detected_intent": "SEARCH_FLIGHT",
#   "confidence": 0.95,
#   "search_payload": {{
#     "slices": [{{"origin": "TAS", "destination": "DXB", "departure_date": "{tomorrow.isoformat()}"}}],
#     "passengers": [{{"type": "adult"}}],
#     "cabin_class": "economy",
#     "max_connections": 1,
#     "supplier_timeout_ms": 10000,
#     "return_offers": true,
#     "page_size": 50,
#     "currency": "USD"
#   }}
# }}

# User: "I want to fly to Tashkent"
# [Origin detected: TAS]
# {{
#   "status": "INCOMPLETE",
#   "detected_intent": "SEARCH_FLIGHT",
#   "confidence": 0.8,
#   "missing_fields": ["slices[0].destination"],
#   "clarification_prompt": "You're already in Tashkent! Where would you like to travel to? Popular destinations include Dubai, Istanbul, Moscow, or London."
# }}

# User: "Business class to Paris"
# [Origin detected: TAS]
# {{
#   "status": "INCOMPLETE",
#   "detected_intent": "SEARCH_FLIGHT",
#   "confidence": 0.85,
#   "missing_fields": ["slices[0].departure_date"],
#   "clarification_prompt": "When would you like to fly to Paris?",
#   "search_payload": {{
#     "slices": [{{"origin": "TAS", "destination": "CDG"}}],
#     "passengers": [{{"type": "adult"}}],
#     "cabin_class": "business",
#     "max_connections": 1,
#     "supplier_timeout_ms": 10000,
#     "return_offers": true,
#     "page_size": 50,
#     "currency": "USD"
#   }}
# }}
# """
    
#     return prompt

# # ============================================================================
# # Gemini API Call
# # ============================================================================
# async def call_gemini_api(
#     user_query: str,
#     conversation_history: Optional[ConversationHistory] = None,
#     detected_origin: Optional[str] = None
# ) -> Dict[str, Any]:
#     """Call Gemini with context and detected origin"""
    
#     if not Config.GEMINI_API_KEY:
#         raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

#     # Build context
#     messages = []
#     if conversation_history and conversation_history.messages:
#         recent_messages = conversation_history.messages[-6:]
#         for msg in recent_messages:
#             messages.append(f"{msg.role.upper()}: {msg.content}")
    
#     messages.append(f"USER: {user_query}")
    
#     # Add known context
#     context_block = ""
#     if conversation_history and conversation_history.current_payload:
#         context_block = f"\n\nKnown Context (merge with new info):\n{json.dumps(conversation_history.current_payload, indent=2)}"
    
#     # Get system prompt with detected origin
#     system_prompt = get_system_prompt(detected_origin)
    
#     user_content = "\n".join(messages) + context_block
    
#     payload = {
#         "contents": [{
#             "parts": [{
#                 "text": f"{system_prompt}\n\n---\n\n{user_content}\n\nResponse (valid JSON only):"
#             }]
#         }],
#         "generationConfig": {
#             "temperature": 0.2,
#             "topK": 40,
#             "topP": 0.8,
#             "maxOutputTokens": 2048,
#         }
#     }

#     last_exception = None
#     for attempt in range(Config.MAX_RETRIES):
#         try:
#             async with httpx.AsyncClient(timeout=Config.CLIENT_TIMEOUT) as client:
#                 response = await client.post(
#                     f"{Config.GEMINI_API_URL}?key={Config.GEMINI_API_KEY}",
#                     json=payload,
#                     headers={"Content-Type": "application/json"}
#                 )
                
#                 if response.status_code != 200:
#                     logger.error(f"Gemini API error (attempt {attempt + 1}): {response.status_code}")
#                     raise httpx.HTTPStatusError(
#                         f"HTTP {response.status_code}",
#                         request=response.request,
#                         response=response
#                     )
                
#                 data = response.json()
#                 text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
                
#                 # Clean JSON
#                 text = text.strip()
#                 if text.startswith("```json"):
#                     text = text[7:]
#                 elif text.startswith("```"):
#                     text = text[3:]
#                 if text.endswith("```"):
#                     text = text[:-3]
#                 text = text.strip()
                
#                 parsed = json.loads(text)
#                 logger.info(f"✓ Gemini parsed query successfully")
#                 return parsed
                
#         except (httpx.HTTPError, json.JSONDecodeError) as e:
#             last_exception = e
#             if attempt < Config.MAX_RETRIES - 1:
#                 delay = Config.BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.1)
#                 logger.warning(f"Retry {attempt + 1}/{Config.MAX_RETRIES} after {delay:.2f}s")
#                 await asyncio.sleep(delay)
#             else:
#                 logger.error(f"Gemini API failed after {Config.MAX_RETRIES} attempts: {e}")
#                 raise HTTPException(
#                     status_code=503,
#                     detail="AI service temporarily unavailable. Please try again."
#                 )
    
#     raise HTTPException(status_code=500, detail=f"Unexpected error: {last_exception}")

# # ============================================================================
# # Helper Functions
# # ============================================================================

# def deep_merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
#     """Deep merge two dictionaries"""
#     if not base:
#         return updates or {}
#     if not updates:
#         return base
    
#     result = dict(base)
#     for key, value in updates.items():
#         if value is None:
#             continue
        
#         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
#             result[key] = deep_merge_dicts(result[key], value)
#         elif key in result and isinstance(result[key], list) and isinstance(value, list):
#             if value and isinstance(value[0], dict):
#                 for i, item in enumerate(value):
#                     if i < len(result[key]):
#                         result[key][i] = deep_merge_dicts(result[key][i], item)
#                     else:
#                         result[key].append(item)
#             else:
#                 result[key] = value
#         else:
#             result[key] = value
    
#     return result

# def compute_missing_fields(payload: Dict[str, Any]) -> List[str]:
#     """Identify missing required fields"""
#     missing = []
    
#     slices = payload.get("slices", [])
#     if not slices:
#         return ["destination", "departure_date"]  # Origin auto-detected
    
#     for i, slice_data in enumerate(slices):
#         if not slice_data.get("destination"):
#             missing.append(f"slices[{i}].destination")
#         if not slice_data.get("departure_date"):
#             missing.append(f"slices[{i}].departure_date")
    
#     return missing

# def is_payload_complete(payload: Dict[str, Any]) -> bool:
#     """Check if payload has all required fields"""
#     try:
#         slices = payload.get("slices", [])
#         if not slices:
#             return False
        
#         for s in slices:
#             # Required: origin, destination, departure_date
#             if not all([s.get("origin"), s.get("destination"), s.get("departure_date")]):
#                 return False
        
#         return True
#     except (ValidationError, KeyError, TypeError):
#         return False

# def determine_prompt_type(missing_fields: List[str]) -> PromptType:
#     """Determine which UI widget to show"""
#     if not missing_fields:
#         return PromptType.TEXT
    
#     date_fields = [f for f in missing_fields if "date" in f.lower()]
#     location_fields = [f for f in missing_fields if "destination" in f.lower()]
#     time_fields = [f for f in missing_fields if "time" in f.lower()]
    
#     if len(set([bool(date_fields), bool(location_fields), bool(time_fields)])) > 1:
#         return PromptType.MULTI_FIELD
    
#     if date_fields:
#         return PromptType.DATE_PICKER
#     elif location_fields:
#         return PromptType.CITY_SELECTOR
#     elif time_fields:
#         return PromptType.TIME_RANGE_PICKER
    
#     return PromptType.TEXT

# def generate_smart_suggestions(
#     payload: Dict[str, Any],
#     missing_fields: List[str],
#     detected_origin: Optional[str] = None
# ) -> List[SmartSuggestion]:
#     """Generate contextual suggestions"""
#     suggestions = []
    
#     slices = payload.get("slices", [])
    
#     # Popular destinations from detected origin
#     if slices and not slices[0].get("destination") and detected_origin:
#         popular_destinations = {
#             "TAS": [("Dubai", "DXB"), ("Istanbul", "IST"), ("Moscow", "SVO")],
#             "DXB": [("London", "LHR"), ("Paris", "CDG"), ("New York", "JFK")],
#             "IST": [("London", "LHR"), ("Paris", "CDG"), ("Dubai", "DXB")],
#         }
        
#         destinations = popular_destinations.get(detected_origin, [("Dubai", "DXB"), ("London", "LHR")])
#         for city, iata in destinations[:2]:
#             suggestions.append(SmartSuggestion(
#                 text=f"To {city}",
#                 action="fill",
#                 data={"slices[0].destination": iata}
#             ))
    
#     # Date suggestions
#     if "departure_date" in str(missing_fields):
#         today = get_today_tashkent()
#         tomorrow = (today + timedelta(days=1)).isoformat()
#         next_week = (today + timedelta(days=7)).isoformat()
        
#         suggestions.append(SmartSuggestion(
#             text="Tomorrow",
#             action="fill",
#             data={"slices[0].departure_date": tomorrow}
#         ))
#         suggestions.append(SmartSuggestion(
#             text="Next week",
#             action="fill",
#             data={"slices[0].departure_date": next_week}
#         ))
    
#     return suggestions[:3]

# def calculate_progress(payload: Dict[str, Any]) -> Dict[str, Any]:
#     """Calculate completion progress"""
#     required_fields = ["origin", "destination", "departure_date"]
    
#     slices = payload.get("slices", [])
#     if not slices:
#         return {"percentage": 0, "completed_fields": 0, "total_fields": 3, "required_remaining": 3}
    
#     first_slice = slices[0]
#     completed = sum(1 for field in required_fields if first_slice.get(field))
    
#     return {
#         "percentage": int((completed / len(required_fields)) * 100),
#         "completed_fields": completed,
#         "total_fields": len(required_fields),
#         "required_remaining": len(required_fields) - completed
#     }

# def personalize_message(
#     message: str,
#     history: Optional[ConversationHistory],
#     detected_intent: Optional[str]
# ) -> str:
#     """Add personality to messages"""
#     if not Config.ENABLE_PERSONALIZATION or not history:
#         return message
    
#     if history.turn_count == 0:
#         return f"👋 {message}"
    
#     if history.turn_count > 2 and detected_intent == "ADD_INFO":
#         prefixes = ["Great! ", "Perfect! ", "Excellent! ", "Awesome! "]
#         return random.choice(prefixes) + message
    
#     return message

# # ============================================================================
# # FastAPI Application
# # ============================================================================
# app = FastAPI(
#     title="Duffel AI Flight Search",
#     description="Conversational AI for flight search with IP geolocation + Same Airport Validation",
#     version="2.0.0"
# )

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# router = APIRouter(prefix="/api/v1", tags=["chat"])

# # ============================================================================
# # Main Conversational Endpoint with VALIDATION
# # ============================================================================
# @router.post("/chat", response_model=ConversationalResponse)
# async def conversational_search(
#     chat_query: ChatQuery,
#     request: Request,
#     x_user_id: Optional[str] = Header(None)
# ) -> ConversationalResponse:
#     """
#     Conversational flight search with AI + IP geolocation + SAME AIRPORT VALIDATION
    
#     NOW PREVENTS IMPOSSIBLE SEARCHES (e.g., TAS → TAS)
#     """
    
#     # Get client IP
#     client_ip = request.client.host if request.client else "127.0.0.1"
    
#     # Rate limiting
#     allowed, retry_after = await rate_limiter.check_rate_limit(client_ip)
#     if not allowed:
#         raise HTTPException(
#             status_code=429,
#             detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
#             headers={"Retry-After": str(retry_after)}
#         )
    
#     # Session ID management
#     if not chat_query.session_id:
#         chat_query.session_id = str(uuid.uuid4())
#         logger.info(f"Generated new session: {chat_query.session_id}")
    
#     session_id = chat_query.session_id
    
#     # Reset if requested
#     if chat_query.reset:
#         await memory.delete(session_id)
#         return ConversationalResponse(
#             status=ConversationStatus.INCOMPLETE,
#             message="Got it! Let's start fresh. Where would you like to fly?",
#             session_id=session_id,
#             prompt_type=PromptType.TEXT,
#             conversation_turn=0
#         )
    
#     # Load conversation history
#     history = await memory.get(session_id)
#     if not history:
#         history = ConversationHistory(session_id=session_id, user_ip=client_ip)
#         logger.info(f"Created new conversation for session: {session_id}")
    
#     # Detect origin airport from IP (once per session)
#     detected_origin_info = None
#     detected_origin_iata = history.detected_origin_airport
    
#     if not detected_origin_iata and Config.ENABLE_IP_GEOLOCATION:
#         try:
#             airport_info = await ip_geo_service.get_nearest_airport_from_ip(client_ip)
#             detected_origin_iata = airport_info["airport"]["iata"]
#             detected_origin_info = airport_info
#             history.detected_origin_airport = detected_origin_iata
#             logger.info(f"Detected origin airport: {detected_origin_iata} for IP {client_ip}")
#         except Exception as e:
#             logger.warning(f"IP geolocation failed: {e}, using default")
#             detected_origin_iata = "TAS"  # Default fallback
#             history.detected_origin_airport = detected_origin_iata
    
#     # Check conversation limits
#     if history.turn_count >= Config.MAX_CONVERSATION_TURNS:
#         await memory.delete(session_id)
#         raise HTTPException(
#             status_code=400,
#             detail="Conversation limit reached. Please start a new session."
#         )
    
#     # Add user message to history
#     history.messages.append(ChatMessage(role="user", content=chat_query.query))
#     history.turn_count += 1
#     history.updated_at = time.time()
    
#     try:
#         # Call Gemini AI with detected origin
#         llm_response_raw = await call_gemini_api(
#             chat_query.query, 
#             history, 
#             detected_origin_iata
#         )
#         llm_response = LLMResponse(**llm_response_raw)
        
#         logger.info(f"LLM Response - Status: {llm_response.status}, Intent: {llm_response.detected_intent}")
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception(f"Error parsing LLM response: {e}")
#         return ConversationalResponse(
#             status=ConversationStatus.ERROR,
#             message="I'm having trouble understanding. Could you rephrase your request?",
#             session_id=session_id,
#             prompt_type=PromptType.TEXT,
#             conversation_turn=history.turn_count
#         )
    
#     # Merge payloads
#     merged_payload = {}
#     if history.current_payload:
#         merged_payload = history.current_payload.copy()
    
#     if llm_response.search_payload:
#         merged_payload = deep_merge_dicts(merged_payload, llm_response.search_payload)
    
#     # Ensure origin is set from IP detection if not explicitly provided
#     if merged_payload.get("slices"):
#         for slice_data in merged_payload["slices"]:
#             if not slice_data.get("origin") and detected_origin_iata:
#                 slice_data["origin"] = detected_origin_iata
    
#     # ============================================================================
#     # NEW: VALIDATION CHECKPOINT 1 - Check for same origin-destination
#     # ============================================================================
#     is_valid, validation_error, conflicting_origin = check_payload_for_same_airport(merged_payload)
    
#     if not is_valid:
#         # Get alternative destination suggestions
#         alternatives = get_alternative_destinations(conflicting_origin)
        
#         # Create friendly suggestions
#         suggestions = [
#             SmartSuggestion(
#                 text=alt["message"],
#                 action="set_destination",
#                 data={"destination": alt["iata"]}
#             )
#             for alt in alternatives
#         ]
        
#         logger.warning(f"Same airport validation triggered: {conflicting_origin} → {conflicting_origin}")
        
#         history.messages.append(ChatMessage(role="assistant", content=validation_error))
#         await memory.set(session_id, history)
        
#         return ConversationalResponse(
#             status=ConversationStatus.VALIDATION_ERROR,
#             message=validation_error,
#             suggestions=suggestions,
#             alternative_destinations=alternatives,
#             session_id=session_id,
#             prompt_type=PromptType.CITY_SELECTOR,
#             conversation_turn=history.turn_count,
#             detected_origin=detected_origin_info
#         )
    
#     # ============================================================================
#     # Continue with normal flow
#     # ============================================================================
    
#     # Set defaults (EXACT Duffel format)
#     merged_payload.setdefault("currency", "USD")
#     merged_payload.setdefault("max_connections", 1)
#     merged_payload.setdefault("supplier_timeout_ms", 10000)
#     merged_payload.setdefault("return_offers", True)
#     merged_payload.setdefault("page_size", 50)
#     if "passengers" not in merged_payload:
#         merged_payload["passengers"] = [{"type": "adult"}]
#     if "cabin_class" not in merged_payload:
#         merged_payload["cabin_class"] = "economy"
    
#     # Update history
#     history.current_payload = merged_payload
    
#     # Check completeness
#     if merged_payload and is_payload_complete(merged_payload):
#         # Validate against exact Duffel format
#         try:
#             duffel_request = DuffelSearchRequest(**merged_payload)
#             final_payload = duffel_request.dict(by_alias=True)
            
#             # Remove None values for optional fields
#             if final_payload.get("slices"):
#                 for slice_data in final_payload["slices"]:
#                     if slice_data.get("departure_time") is None:
#                         del slice_data["departure_time"]
#                     if slice_data.get("arrival_time") is None:
#                         del slice_data["arrival_time"]
            
#             origin = final_payload["slices"][0]["origin"]
#             destination = final_payload["slices"][0]["destination"]
#             date = final_payload["slices"][0]["departure_date"]
#             cabin = final_payload.get("cabin_class", "economy")
            
#             # ============================================================================
#             # NEW: VALIDATION CHECKPOINT 2 - Final check before marking COMPLETE
#             # ============================================================================
#             is_valid, validation_error = validate_origin_destination(origin, destination)
#             if not is_valid:
#                 alternatives = get_alternative_destinations(origin)
#                 suggestions = [
#                     SmartSuggestion(
#                         text=alt["message"],
#                         action="set_destination",
#                         data={"destination": alt["iata"]}
#                     )
#                     for alt in alternatives
#                 ]
                
#                 logger.warning(f"Final validation failed: {origin} → {destination}")
                
#                 history.messages.append(ChatMessage(role="assistant", content=validation_error))
#                 await memory.set(session_id, history)
                
#                 return ConversationalResponse(
#                     status=ConversationStatus.VALIDATION_ERROR,
#                     message=validation_error,
#                     suggestions=suggestions,
#                     alternative_destinations=alternatives,
#                     session_id=session_id,
#                     prompt_type=PromptType.CITY_SELECTOR,
#                     conversation_turn=history.turn_count,
#                     detected_origin=detected_origin_info
#                 )
            
#             message = f"Perfect! I've prepared your search for {cabin} class flights from {origin} to {destination} on {date}. Ready to search!"
#             message = personalize_message(message, history, llm_response.detected_intent)
            
#             history.messages.append(ChatMessage(role="assistant", content=message))
#             await memory.set(session_id, history)
            
#             return ConversationalResponse(
#                 status=ConversationStatus.COMPLETE,
#                 message=message,
#                 search_data=final_payload,  # EXACT Duffel format
#                 session_id=session_id,
#                 progress=calculate_progress(merged_payload),
#                 detected_intent=llm_response.detected_intent,
#                 conversation_turn=history.turn_count,
#                 detected_origin=detected_origin_info
#             )
#         except ValidationError as ve:
#             logger.error(f"Validation error: {ve}")
#             # Continue to incomplete flow
    
#     # Incomplete - determine what's missing
#     missing = compute_missing_fields(merged_payload)
#     clarification = llm_response.clarification_prompt or "Could you provide more details about your trip?"
#     clarification = personalize_message(clarification, history, llm_response.detected_intent)
    
#     prompt_type = determine_prompt_type(missing)
#     suggestions = generate_smart_suggestions(merged_payload, missing, detected_origin_iata) if Config.ENABLE_SMART_SUGGESTIONS else None
    
#     history.messages.append(ChatMessage(role="assistant", content=clarification))
#     history.missing_fields = missing
#     await memory.set(session_id, history)
    
#     return ConversationalResponse(
#         status=ConversationStatus.INCOMPLETE,
#         message=clarification,
#         missing_fields=missing,
#         prompt_type=prompt_type,
#         suggestions=suggestions,
#         session_id=session_id,
#         progress=calculate_progress(merged_payload),
#         detected_intent=llm_response.detected_intent,
#         conversation_turn=history.turn_count,
#         detected_origin=detected_origin_info
#     )

# @router.get("/session/{session_id}")
# async def get_session(session_id: str):
#     """Retrieve conversation history for a session"""
#     history = await memory.get(session_id)
#     if not history:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     return {
#         "session_id": session_id,
#         "turn_count": history.turn_count,
#         "messages": history.messages,
#         "current_payload": history.current_payload,
#         "detected_origin": history.detected_origin_airport,
#         "created_at": datetime.fromtimestamp(history.created_at, TASHKENT_TZ).isoformat(),
#         "updated_at": datetime.fromtimestamp(history.updated_at, TASHKENT_TZ).isoformat()
#     }

# @router.delete("/session/{session_id}")
# async def delete_session(session_id: str):
#     """Delete a session"""
#     await memory.delete(session_id)
#     return {"message": "Session deleted", "session_id": session_id}

# @router.post("/session/new")
# async def create_session():
#     """Create a new session ID"""
#     session_id = str(uuid.uuid4())
#     return {"session_id": session_id}

# @router.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "service": "Duffel AI with Same Airport Validation",
#         "version": "2.0.0",
#         "features": {
#             "ip_geolocation": Config.ENABLE_IP_GEOLOCATION,
#             "same_airport_validation": True,
#             "smart_suggestions": Config.ENABLE_SMART_SUGGESTIONS,
#             "personalization": Config.ENABLE_PERSONALIZATION
#         },
#         "timestamp": datetime.now(TASHKENT_TZ).isoformat()
#     }

