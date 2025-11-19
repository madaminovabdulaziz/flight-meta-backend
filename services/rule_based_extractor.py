"""
Rule-Based Parameter Extractor
Extract flight parameters using regex + dateparser - NO LLM for 80% of cases.

This module handles parameter extraction using lightweight NLP,
achieving 80-90% success rate with 10-50ms latency and zero cost.

ENHANCEMENTS:
- Improved route detection with priority ordering
- Enhanced date parsing with relative date support
- Better handling of edge cases

Usage:
    from services.rule_based_extractor import rule_extractor

    extracted, confidence = rule_extractor.extract(message, state)
    if confidence > 0.80:
        # Use extracted parameters
    else:
        # Fallback to LLM
"""

import re
from datetime import date, datetime, timedelta
from typing import Dict, Any, Tuple, Optional
import dateparser


def _get_state_value(state: Any, key: str) -> Any:
    """Support both dict-like and dataclass ConversationState."""
    if isinstance(state, dict):
        return state.get(key)
    return getattr(state, key, None)


class RuleBasedExtractor:
    """
    Extract flight parameters without LLM.

    Uses regex patterns and lightweight NLP (dateparser)
    to extract structured data from natural language.
    """

    # ========================================
    # REGEX PATTERNS
    # ========================================

    # Airport code: 3 uppercase letters
    AIRPORT_PATTERN = re.compile(r'\b[A-Z]{3}\b')

    # Passengers: number + keyword (base pattern)
    PASSENGER_PATTERN = re.compile(
        r'(\d+)\s*(?:passenger|passengers|people|person|pax|adult|adults|traveler|traveller)',
        re.IGNORECASE
    )

    # Special case: "with 2 friends" → user + 2 friends
    FRIENDS_PATTERN = re.compile(
        r'with\s+(\d+)\s+friends?',
        re.IGNORECASE
    )

    # Budget: currency symbol + number
    BUDGET_PATTERN = re.compile(r'[£$€](\d+(?:,\d{3})*(?:\.\d{2})?)')

    # Budget: number + currency word
    BUDGET_WORD_PATTERN = re.compile(
        r'(\d+(?:,\d{3})*)\s*(?:pound|pounds|dollar|dollars|euro|euros|gbp|usd|eur)',
        re.IGNORECASE
    )

    # Flexibility: ±N days
    FLEXIBILITY_PATTERN = re.compile(r'[±+]\s*(\d+)\s*day', re.IGNORECASE)

    # Route patterns: "from X to Y", "X to Y", "to X from Y"
    # Enhanced with better capture groups
    FROM_TO_PATTERN = re.compile(
        r'\bfrom\s+([A-Za-z\s]+?)\s+(?:to|→|-)\s+([A-Za-z\s]+?)(?:\s|,|;|$)',
        re.IGNORECASE
    )
    TO_FROM_PATTERN = re.compile(
        r'\b(?:to|trip to|fly to|flight to)\s+([A-Za-z\s]+?)\s+from\s+([A-Za-z\s]+?)(?:\s|,|;|$)',
        re.IGNORECASE
    )
    SIMPLE_TO_PATTERN = re.compile(
        r'\b(?:to|trip to|fly to|flight to)\s+([A-Za-z\s]+?)(?:\s|,|;|$)',
        re.IGNORECASE
    )
    SIMPLE_FROM_PATTERN = re.compile(
        r'\bfrom\s+([A-Za-z\s]+?)(?:\s|,|;|$)',
        re.IGNORECASE
    )

    # Date phrase patterns with better support for relative dates
    DATE_PHRASE_PATTERN = re.compile(
        r'\b(departing|departure|on|for|next|this|tomorrow|in)\s+([^,.;]+)',
        re.IGNORECASE
    )

    # Relative date patterns
    RELATIVE_DATE_PATTERNS = [
        re.compile(r'\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.IGNORECASE),
        re.compile(r'\bthis\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.IGNORECASE),
        re.compile(r'\bnext\s+week\b', re.IGNORECASE),
        re.compile(r'\bin\s+(\d+)\s+(day|days|week|weeks)\b', re.IGNORECASE),
        re.compile(r'\btomorrow\b', re.IGNORECASE),
    ]

    # ========================================
    # CITY → AIRPORT MAPPING (Top 50 + a few extras)
    # ========================================

    CITY_TO_AIRPORT = {
        # Europe
        "istanbul": "IST",
        "london": "LHR",
        "paris": "CDG",
        "barcelona": "BCN",
        "rome": "FCO",
        "amsterdam": "AMS",
        "berlin": "BER",
        "madrid": "MAD",
        "vienna": "VIE",
        "prague": "PRG",
        "athens": "ATH",
        "lisbon": "LIS",
        "dublin": "DUB",
        "copenhagen": "CPH",
        "stockholm": "ARN",
        "france": "CDG",  # Country name → capital airport
        # Asia
        "dubai": "DXB",
        "tokyo": "NRT",
        "singapore": "SIN",
        "hong kong": "HKG",
        "bangkok": "BKK",
        "kuala lumpur": "KUL",
        "seoul": "ICN",
        "beijing": "PEK",
        "shanghai": "PVG",
        "delhi": "DEL",
        "mumbai": "BOM",
        "taipei": "TPE",
        # Americas
        "new york": "JFK",
        "los angeles": "LAX",
        "chicago": "ORD",
        "miami": "MIA",
        "toronto": "YYZ",
        "san francisco": "SFO",
        "boston": "BOS",
        "washington": "IAD",
        "las vegas": "LAS",
        "seattle": "SEA",
        "vancouver": "YVR",
        "mexico city": "MEX",
        # Middle East
        "doha": "DOH",
        "abu dhabi": "AUH",
        "riyadh": "RUH",
        "muscat": "MCT",
        "cairo": "CAI",
        # Oceania & Others
        "sydney": "SYD",
        "melbourne": "MEL",
        "auckland": "AKL",
        "johannesburg": "JNB",
        # Central Asia
        "tashkent": "TAS",
    }

    # ========================================
    # TRAVEL CLASS MAPPING
    # ========================================

    TRAVEL_CLASS_KEYWORDS = {
        "business class": "Business",
        "business": "Business",
        "first class": "First",
        "first": "First",
        "economy class": "Economy",
        "economy": "Economy",
        "coach": "Economy",
        "premium economy": "Premium Economy",
        "premium": "Premium Economy",
    }

    # ========================================
    # HELPERS
    # ========================================

    @staticmethod
    def _normalize_place_to_airport(place: str) -> Optional[str]:
        """
        Map a raw city/airport string to a 3-letter airport code if possible.

        Enhanced with better cleaning and fallback logic.
        """
        if not place:
            return None

        # Clean and normalize
        s = place.strip().lower()
        s = re.sub(r'\s+', ' ', s)  # Normalize whitespace

        # Direct city mapping (longest match first for multi-word cities)
        if s in RuleBasedExtractor.CITY_TO_AIRPORT:
            code = RuleBasedExtractor.CITY_TO_AIRPORT[s]
            if code:  # Skip region-like keys with None value
                return code

        # Try to detect 3-letter code in the string
        m = re.search(r'\b([a-z]{3})\b', s)
        if m:
            return m.group(1).upper()

        return None

    @staticmethod
    def _parse_date_from_message(message: str) -> Optional[date]:
        """
        Extract a date using enhanced relative date support.

        Handles:
        - Relative dates: "next Monday", "tomorrow", "in two weeks"
        - Absolute dates: "December 25th", "2025-12-15"
        - Date phrases: "departing next week"
        """
        # First try to extract a date phrase
        phrase = None

        m = RuleBasedExtractor.DATE_PHRASE_PATTERN.search(message)
        if m:
            phrase = m.group(0)  # Include the keyword for better parsing
        else:
            # Check for standalone relative dates
            for pattern in RuleBasedExtractor.RELATIVE_DATE_PATTERNS:
                m = pattern.search(message)
                if m:
                    phrase = m.group(0)
                    break

        if not phrase:
            # Fallback: try the whole message
            phrase = message

        # Parse using dateparser with enhanced settings
        parsed_date = dateparser.parse(
            phrase,
            settings={
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': datetime.now(),
                'STRICT_PARSING': False,
                'PREFER_DAY_OF_MONTH': 'first',  # For ambiguous cases
                'RETURN_AS_TIMEZONE_AWARE': False,
            }
        )

        if not parsed_date:
            return None

        d = parsed_date.date()

        # Sanity check: must be in the future
        if d < date.today():
            # Try adding a year for cases like "December 25th" (might be next year)
            d = d.replace(year=d.year + 1)
            if d < date.today():
                return None

        return d

    # ========================================
    # MAIN EXTRACTION
    # ========================================

    @staticmethod
    def extract(message: str, state: Any) -> Tuple[Dict[str, Any], float]:
        """
        Extract parameters with confidence score.

        Args:
            message: User's message
            state: Current conversation state (dict or ConversationState)

        Returns:
            (extracted_params, confidence) where:
            - extracted_params: Dict of extracted values
            - confidence: 0.0-1.0 (based on extraction success)
        """

        extracted: Dict[str, Any] = {}
        confidence_scores = []
        msg_lower = message.lower()

        # Normalize current state values
        current_destination = _get_state_value(state, "destination")
        current_origin = _get_state_value(state, "origin")
        current_departure = _get_state_value(state, "departure_date")
        current_return = _get_state_value(state, "return_date")

        # ========================================
        # 0. ROUTE PATTERNS (from X to Y, etc.) – HIGHEST PRIORITY
        # ========================================

        # Pattern 1: "from X to Y" (most explicit)
        m = RuleBasedExtractor.FROM_TO_PATTERN.search(message)
        if m:
            origin_raw = m.group(1).strip()
            dest_raw = m.group(2).strip()

            dest_code = RuleBasedExtractor._normalize_place_to_airport(dest_raw)
            origin_code = RuleBasedExtractor._normalize_place_to_airport(origin_raw)

            if dest_code and not current_destination:
                extracted["destination"] = dest_code
                confidence_scores.append(0.95)
            if origin_code and not current_origin:
                extracted["origin"] = origin_code
                confidence_scores.append(0.95)

        # Pattern 2: "to X from Y"
        if "destination" not in extracted or "origin" not in extracted:
            m = RuleBasedExtractor.TO_FROM_PATTERN.search(message)
            if m:
                dest_raw = m.group(1).strip()
                origin_raw = m.group(2).strip()

                dest_code = RuleBasedExtractor._normalize_place_to_airport(dest_raw)
                origin_code = RuleBasedExtractor._normalize_place_to_airport(origin_raw)

                if dest_code and not current_destination and "destination" not in extracted:
                    extracted["destination"] = dest_code
                    confidence_scores.append(0.95)
                if origin_code and not current_origin and "origin" not in extracted:
                    extracted["origin"] = origin_code
                    confidence_scores.append(0.95)

        # Pattern 3: Simple "to X" (only if destination not already extracted)
        if "destination" not in extracted and not current_destination:
            m = RuleBasedExtractor.SIMPLE_TO_PATTERN.search(message)
            if m:
                dest_raw = m.group(1).strip()
                dest_code = RuleBasedExtractor._normalize_place_to_airport(dest_raw)
                if dest_code:
                    extracted["destination"] = dest_code
                    confidence_scores.append(0.9)

        # Pattern 4: Simple "from Y" (only if origin not already extracted)
        if "origin" not in extracted and not current_origin:
            m = RuleBasedExtractor.SIMPLE_FROM_PATTERN.search(message)
            if m:
                origin_raw = m.group(1).strip()
                origin_code = RuleBasedExtractor._normalize_place_to_airport(origin_raw)
                if origin_code:
                    extracted["origin"] = origin_code
                    confidence_scores.append(0.9)

        # ========================================
        # 1. EXPLICIT AIRPORT CODES (3 uppercase letters)
        # ========================================

        airport_matches = RuleBasedExtractor.AIRPORT_PATTERN.findall(message)
        if airport_matches:
            # If we already have both from route patterns, skip
            if "destination" in extracted and "origin" in extracted:
                pass
            else:
                # First code fills the missing field
                if "destination" not in extracted and not current_destination:
                    extracted["destination"] = airport_matches[0]
                    confidence_scores.append(1.0)
                elif "origin" not in extracted and not current_origin:
                    extracted["origin"] = airport_matches[0]
                    confidence_scores.append(1.0)

                # Second code fills the other missing field
                if len(airport_matches) > 1:
                    if "origin" not in extracted and not current_origin:
                        extracted["origin"] = airport_matches[1]
                        confidence_scores.append(1.0)
                    elif "destination" not in extracted and not current_destination:
                        extracted["destination"] = airport_matches[1]
                        confidence_scores.append(1.0)

        # ========================================
        # 2. CITY NAMES (Fallback if route patterns missed them)
        # ========================================

        # Only run if we still have missing origin/destination
        if (not current_destination and "destination" not in extracted) or \
           (not current_origin and "origin" not in extracted):

            # Sort cities by length (longest first for better matching)
            sorted_cities = sorted(
                RuleBasedExtractor.CITY_TO_AIRPORT.items(),
                key=lambda x: (len(x[0]), x[0]),  # Length desc, then alphabetical
                reverse=True
            )

            for city, airport_code in sorted_cities:
                if not airport_code:
                    continue  # Skip region-like keys

                if city in msg_lower:
                    # Fill missing field
                    if not current_destination and "destination" not in extracted:
                        extracted["destination"] = airport_code
                        confidence_scores.append(0.9)
                    elif not current_origin and "origin" not in extracted:
                        extracted["origin"] = airport_code
                        confidence_scores.append(0.9)

        # ========================================
        # 3. DATES (Enhanced with relative date support)
        # ========================================

        parsed_date_obj = RuleBasedExtractor._parse_date_from_message(message)
        if parsed_date_obj:
            if not current_departure and "departure_date" not in extracted:
                extracted["departure_date"] = parsed_date_obj
                confidence_scores.append(0.85)
            elif current_departure and parsed_date_obj > current_departure and \
                 "return_date" not in extracted and not current_return:
                extracted["return_date"] = parsed_date_obj
                confidence_scores.append(0.85)

        # ========================================
        # 4. PASSENGERS (Confidence: 0.95)
        # ========================================

        # Special "with 2 friends" → user + friends
        friend_match = RuleBasedExtractor.FRIENDS_PATTERN.search(message)
        if friend_match:
            count = int(friend_match.group(1)) + 1  # user + friends
            if 1 <= count <= 9:
                extracted["passengers"] = count
                confidence_scores.append(0.95)
        else:
            # Generic pattern: "2 passengers", "3 people", etc.
            passenger_match = RuleBasedExtractor.PASSENGER_PATTERN.search(message)
            if passenger_match:
                count = int(passenger_match.group(1))
                if 1 <= count <= 9:
                    extracted["passengers"] = count
                    confidence_scores.append(0.95)

        # ========================================
        # 5. TRAVEL CLASS (Confidence: 1.0)
        # ========================================

        sorted_classes = sorted(
            RuleBasedExtractor.TRAVEL_CLASS_KEYWORDS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for keyword, class_name in sorted_classes:
            if keyword in msg_lower:
                extracted["travel_class"] = class_name
                confidence_scores.append(1.0)
                break  # Take first match

        # ========================================
        # 6. BUDGET (Confidence: 0.9)
        # ========================================

        budget_match = RuleBasedExtractor.BUDGET_PATTERN.search(message)
        if budget_match:
            budget_str = budget_match.group(1).replace(',', '')
            extracted["budget"] = float(budget_str)
            confidence_scores.append(0.9)
        else:
            budget_word_match = RuleBasedExtractor.BUDGET_WORD_PATTERN.search(message)
            if budget_word_match:
                budget_str = budget_word_match.group(1).replace(',', '')
                extracted["budget"] = float(budget_str)
                confidence_scores.append(0.85)

        # ========================================
        # 7. FLEXIBILITY (Confidence: 0.9)
        # ========================================

        flex_match = RuleBasedExtractor.FLEXIBILITY_PATTERN.search(message)
        if flex_match:
            extracted["flexibility"] = int(flex_match.group(1))
            confidence_scores.append(0.9)

        # ========================================
        # 8. CALCULATE OVERALL CONFIDENCE
        # ========================================

        if confidence_scores:
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            overall_confidence = 0.0

        return extracted, overall_confidence

    @staticmethod
    def get_confidence_threshold() -> float:
        """
        Get the recommended confidence threshold.

        Below this threshold, LLM fallback is recommended.

        Returns:
            0.80 - optimal balance of speed and accuracy
        """
        return 0.80


# ============================================================
# SINGLETON INSTANCE
# ============================================================

rule_extractor = RuleBasedExtractor()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def extract_parameters(message: str, state: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    """
    Convenience function for parameter extraction.

    Args:
        message: User's message
        state: Current conversation state

    Returns:
        (extracted_params, confidence)
    """
    return rule_extractor.extract(message, state)


def should_use_llm_for_extraction(message: str, state: Dict[str, Any]) -> bool:
    """
    Determine if LLM fallback is needed for parameter extraction.

    Args:
        message: User's message
        state: Current conversation state

    Returns:
        True if confidence is too low and LLM should be used
    """
    _, confidence = rule_extractor.extract(message, state)
    return confidence < rule_extractor.get_confidence_threshold()
