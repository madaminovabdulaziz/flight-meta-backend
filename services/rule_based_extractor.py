"""
Rule-Based Parameter Extractor - CONTEXT AWARE FIXED VERSION
============================================================
Extracts travel parameters from natural language with actual intelligence.
Fixed to respect conversation context (prevents the "Amnesia Loop").
"""

import re
from datetime import date, datetime, timedelta
from typing import Dict, Any, Tuple, Optional
from dateparser.search import search_dates


class RuleBasedExtractor:
    
    # ========================================
    # CONSTANTS
    # ========================================
    
    NOISE_WORDS = {
        "want", "fly", "book", "going", "travel", "trip", "visit", 
        "need", "looking", "planning", "thinking", "flight", "flights"
    }
    
    CONTEXT_KEYWORDS = {
        "culture": "culture", "museum": "culture", "history": "culture",
        "business": "business", "meeting": "business", "work": "business",
        "romantic": "romantic", "couple": "romantic", "honeymoon": "romantic",
        "family": "family", "kids": "family",
        "friends": "friends", "party": "friends"
    }
    
    CITY_TO_AIRPORT = {
        "london": "LHR", "paris": "CDG", "new york": "JFK", "tokyo": "NRT",
        "istanbul": "IST", "tashkent": "TAS", "dubai": "DXB", "singapore": "SIN",
        "rome": "FCO", "barcelona": "BCN", "madrid": "MAD", "berlin": "BER",
        "amsterdam": "AMS", "frankfurt": "FRA", "munich": "MUC"
    }
    
    CLASS_PATTERNS = {
        "business": re.compile(r'\b(business|biz)\b', re.IGNORECASE),
        "first": re.compile(r'\bfirst\s*class\b', re.IGNORECASE),
        "premium economy": re.compile(r'\bpremium\s*economy\b', re.IGNORECASE),
        "economy": re.compile(r'\b(economy|coach)\b', re.IGNORECASE)
    }

    # ========================================
    # MAIN EXTRACTION
    # ========================================
    
    @staticmethod
    def extract(message: str, state: Any) -> Tuple[Dict[str, Any], float]:
        """
        Extract travel parameters with human-level understanding.
        Returns: (extracted_params, confidence)
        """
        extracted: Dict[str, Any] = {}
        confidence_scores = []
        
        clean_msg = RuleBasedExtractor._clean_message(message)
        
        # ---------------------------------------------------------
        # 0. CRITICAL FIX: CHECK CONTEXT FIRST
        # ---------------------------------------------------------
        missing_param = getattr(state, "missing_parameter", None)
        found_any_city = RuleBasedExtractor._find_any_city_in_message(message)

        # If we specifically asked for ORIGIN, and we found a city, FORCE it to be Origin
        if missing_param == "origin" and found_any_city:
            extracted["origin"] = found_any_city
            confidence_scores.append(1.0)
            # Prevent _extract_destination from grabbing it later
            clean_msg = clean_msg.replace(found_any_city.lower(), "") 

        # If we specifically asked for DESTINATION, and we found a city
        elif missing_param == "destination" and found_any_city:
            extracted["destination"] = found_any_city
            confidence_scores.append(1.0)
            
        # ---------------------------------------------------------
        
        # 1. ROUTE EXTRACTION (From X to Y)
        route = RuleBasedExtractor._extract_route(message)
        if route:
            extracted.update(route)
            confidence_scores.extend([0.95, 0.95])
        
        # 2. DESTINATION (To X)
        # Only run if we haven't already found it via context
        if "destination" not in extracted:
            dest = RuleBasedExtractor._extract_destination(clean_msg)
            if dest:
                extracted["destination"] = dest
                confidence_scores.append(0.90)
        
        # 3. ORIGIN (From X)
        if "origin" not in extracted:
            origin = RuleBasedExtractor._extract_origin(clean_msg)
            if origin:
                extracted["origin"] = origin
                confidence_scores.append(0.90)
        
        # 4. DATES (Smart parsing)
        dates = RuleBasedExtractor._extract_dates(message)
        if dates:
            extracted.update(dates)
            confidence_scores.extend([0.85] * len(dates))
        
        # 5. PASSENGERS (Infer correctly)
        passengers = RuleBasedExtractor._extract_passengers(message)
        if passengers:
            extracted["passengers"] = passengers
            confidence_scores.append(0.95)
        
        # 6. TRAVEL CLASS
        travel_class = RuleBasedExtractor._extract_class(message)
        if travel_class:
            extracted["travel_class"] = travel_class
            confidence_scores.append(1.0)
        
        # 7. CONTEXT (Trip reason)
        context = RuleBasedExtractor._extract_context(message)
        if context:
            extracted.update(context)
        
        # Calculate average confidence
        if not confidence_scores:
            return extracted, 0.0
        
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        return extracted, avg_conf
    
    # ========================================
    # EXTRACTION HELPERS
    # ========================================

    @staticmethod
    def _find_any_city_in_message(message: str) -> Optional[str]:
        """Helper to find ANY known city in the message without assigning intent."""
        for city, code in RuleBasedExtractor.CITY_TO_AIRPORT.items():
            if re.search(rf'\b{city}\b', message, re.IGNORECASE):
                return code
        return None

    @staticmethod
    def _clean_message(message: str) -> str:
        """Remove noise words that interfere with extraction."""
        words = message.split()
        cleaned = [w for w in words if w.lower() not in RuleBasedExtractor.NOISE_WORDS]
        return " ".join(cleaned)
    
    @staticmethod
    def _extract_route(message: str) -> Optional[Dict[str, str]]:
        """Extract full route: 'From London to Paris'"""
        pattern = re.compile(
            r'\bfrom\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+?)(?:\s|$|,|\.)',
            re.IGNORECASE
        )
        match = pattern.search(message)
        if match:
            origin = RuleBasedExtractor._resolve_location(match.group(1).strip())
            dest = RuleBasedExtractor._resolve_location(match.group(2).strip())
            if origin and dest:
                return {
                    "origin": origin,
                    "destination": dest,
                    "origin_airports": [origin] if len(origin) == 3 else [],
                    "destination_airports": [dest] if len(dest) == 3 else []
                }
        return None
    
    @staticmethod
    def _extract_destination(message: str) -> Optional[str]:
        """Extract destination: 'to Paris' OR fallback to standalone city"""
        # Explicit 'to' pattern
        pattern = re.compile(r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', re.IGNORECASE)
        match = pattern.search(message)
        if match:
            dest_raw = match.group(1).strip()
            if dest_raw.lower() not in ["fly", "go", "travel", "book"]:
                return RuleBasedExtractor._resolve_location(dest_raw)
        
        # Fallback: Standalone city names (ONLY if not caught by explicit 'from')
        return RuleBasedExtractor._find_any_city_in_message(message)
    
    @staticmethod
    def _extract_origin(message: str) -> Optional[str]:
        """Extract origin: 'from London'"""
        pattern = re.compile(r'\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', re.IGNORECASE)
        match = pattern.search(message)
        if match:
            return RuleBasedExtractor._resolve_location(match.group(1).strip())
        return None
    
    @staticmethod
    def _extract_dates(message: str) -> Dict[str, date]:
        dates = {}
        today = date.today()
        relative_patterns = {
            r'\bnext week\b': today + timedelta(days=7),
            r'\bthis weekend\b': RuleBasedExtractor._get_next_saturday(),
            r'\btomorrow\b': today + timedelta(days=1),
        }
        for pattern, target_date in relative_patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                dates["departure_date"] = target_date
                break
        
        if not dates:
            try:
                found_dates = search_dates(message, settings={'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': datetime.now()})
                if found_dates:
                    valid = [d[1].date() for d in found_dates if d[1].date() >= today]
                    if valid:
                        dates["departure_date"] = valid[0]
                        if len(valid) > 1: dates["return_date"] = valid[1]
            except Exception: pass
        return dates
    
    @staticmethod
    def _extract_passengers(message: str) -> Optional[int]:
        msg_lower = message.lower()
        match = re.search(r'with\s+(\d+)\s+(friend|colleague|people|person)', msg_lower)
        if match: return int(match.group(1)) + 1
        match = re.search(r'(\d+)\s+(passenger|adult|traveler|people)', msg_lower)
        if match: return int(match.group(1))
        return None
    
    @staticmethod
    def _extract_class(message: str) -> Optional[str]:
        for cls, pattern in RuleBasedExtractor.CLASS_PATTERNS.items():
            if pattern.search(message): return cls.title()
        return None
    
    @staticmethod
    def _extract_context(message: str) -> Dict[str, str]:
        context = {}
        msg_lower = message.lower()
        for keyword, category in RuleBasedExtractor.CONTEXT_KEYWORDS.items():
            if keyword in msg_lower:
                context["trip_reason"] = category
                break
        if "friends" in msg_lower: context["travel_group"] = "friends"
        elif "family" in msg_lower: context["travel_group"] = "family"
        elif "solo" in msg_lower: context["travel_group"] = "solo"
        return context
    
    @staticmethod
    def _resolve_location(text: str) -> Optional[str]:
        if not text: return None
        clean = text.strip().lower()
        if len(clean) == 3 and clean.isalpha(): return clean.upper()
        if clean in RuleBasedExtractor.CITY_TO_AIRPORT: return RuleBasedExtractor.CITY_TO_AIRPORT[clean]
        return text.strip().title()
    
    @staticmethod
    def _get_next_saturday() -> date:
        today = date.today()
        days_ahead = 5 - today.weekday()
        if days_ahead <= 0: days_ahead += 7
        return today + timedelta(days=days_ahead)


rule_extractor = RuleBasedExtractor()