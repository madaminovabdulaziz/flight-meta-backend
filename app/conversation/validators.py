# app/conversation/validators.py
"""
Input Validation for Conversational Search
Ensures data quality at each state transition.
Enhanced with IATA code validation and better error messages.
"""

import re
import logging
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.conversation.models import (
    ConversationFlowState,
    TripSpec,
    TripType,
    CabinClass,
    ValidationResult
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(message)


# ============================================================
# IATA CODES DATABASE (Top airports worldwide)
# ============================================================

VALID_IATA_CODES = {
    # Central Asia
    "TAS", "SKD", "BHK", "ALA", "NQZ", "FRU", "DYU",
    
    # Middle East
    "DXB", "AUH", "SHJ", "DOH", "RUH", "JED", "DMM", "KWI", "BAH",
    "MCT", "CAI", "AMM", "BEY", "TLV", "IST", "SAW", "AYT", "ESB",
    
    # Europe
    "LHR", "LGW", "STN", "LCY", "LTN", "CDG", "ORY", "BVA",
    "FRA", "MUC", "AMS", "MAD", "BCN", "FCO", "MXP", "VCE",
    "VIE", "ZRH", "GVA", "CPH", "OSL", "ARN", "HEL", "WAW",
    "PRG", "BUD", "ATH", "LIS", "OPO", "DUB", "BRU", "LUX",
    
    # Russia & CIS
    "SVO", "DME", "VKO", "LED", "KZN", "SVX", "OVB", "KRR",
    
    # Asia-Pacific
    "PEK", "PVG", "CAN", "SZX", "HKG", "TPE", "ICN", "GMP",
    "NRT", "HND", "KIX", "NGO", "FUK", "SIN", "BKK", "DMK",
    "HAN", "SGN", "MNL", "CGK", "SUB", "KUL", "PEN", "DPS",
    "DEL", "BOM", "BLR", "MAA", "HYD", "CCU", "AMD", "COK",
    
    # Americas
    "JFK", "EWR", "LGA", "ORD", "LAX", "SFO", "SEA", "MIA",
    "ATL", "DFW", "IAH", "DEN", "LAS", "PHX", "MCO", "BOS",
    "YYZ", "YVR", "YUL", "MEX", "GDL", "CUN", "GRU", "GIG",
    "EZE", "BOG", "LIM", "SCL", "PTY",
    
    # Africa
    "JNB", "CPT", "DUR", "ADD", "NBO", "DAR", "CMN", "TUN",
    "CAI", "ALG", "LOS", "ACC",
    
    # Oceania
    "SYD", "MEL", "BNE", "PER", "AKL", "CHC", "WLG"
}


class FieldValidator:
    """Validates individual fields with enhanced IATA checking"""
    
    @staticmethod
    def validate_iata_code(code: str, allow_unknown: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Validate IATA airport/city code.
        
        Args:
            code: Airport code (e.g., 'LON', 'IST')
            allow_unknown: If True, allows codes not in database (for flexibility)
            
        Returns:
            (is_valid, error_message)
        """
        if not code:
            return False, "Airport code is required"
        
        if not isinstance(code, str):
            return False, "Airport code must be a string"
        
        code = code.strip().upper()
        
        if len(code) != 3:
            return False, f"Airport code must be 3 characters, got {len(code)}"
        
        if not code.isalpha():
            return False, f"Airport code must contain only letters, got '{code}'"
        
        # Check against known IATA codes
        if not allow_unknown and code not in VALID_IATA_CODES:
            logger.warning(f"Unknown IATA code: {code}")
            return False, f"Unknown airport code '{code}'. Please check and try again."
        
        return True, None
    
    @staticmethod
    def validate_date(
        date_str: str,
        allow_past: bool = False,
        max_future_days: int = 730
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate date string.
        
        Args:
            date_str: Date in ISO format (YYYY-MM-DD)
            allow_past: Whether past dates are allowed
            max_future_days: Maximum days in future (default 2 years)
            
        Returns:
            (is_valid, error_message)
        """
        if not date_str:
            return False, "Date is required"
        
        if not isinstance(date_str, str):
            return False, "Date must be a string"
        
        # Try parsing ISO format
        try:
            date_obj = datetime.fromisoformat(date_str)
        except ValueError:
            return False, f"Invalid date format '{date_str}'. Expected YYYY-MM-DD"
        
        # Check if date is not in the past
        if not allow_past:
            today = datetime.now().date()
            if date_obj.date() < today:
                return False, f"Date cannot be in the past. Got {date_str}"
        
        # Check if date is not too far in the future
        max_future = datetime.now() + timedelta(days=max_future_days)
        if date_obj > max_future:
            return False, f"Date too far in the future. Maximum is {max_future.date().isoformat()}"
        
        return True, None
    
    @staticmethod
    def validate_passenger_count(count: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate passenger count.
        
        Args:
            count: Number of passengers (int or convertible to int)
            
        Returns:
            (is_valid, error_message)
        """
        if count is None:
            return False, "Passenger count is required"
        
        try:
            count_int = int(count)
        except (ValueError, TypeError):
            return False, f"Passenger count must be a number, got '{count}'"
        
        if count_int < 1:
            return False, f"Must have at least 1 passenger, got {count_int}"
        
        if count_int > 9:
            return False, f"Maximum 9 passengers allowed, got {count_int}"
        
        return True, None
    
    @staticmethod
    def validate_trip_type(trip_type: str) -> Tuple[bool, Optional[str]]:
        """Validate trip type"""
        if not trip_type:
            return True, None  # Optional field
        
        try:
            TripType(trip_type.lower())
            return True, None
        except ValueError:
            valid_types = [t for t in TripType]
            return False, f"Invalid trip type '{trip_type}'. Valid: {', '.join(valid_types)}"
    
    @staticmethod
    def validate_cabin_class(cabin: str) -> Tuple[bool, Optional[str]]:
        """Validate cabin class"""
        if not cabin:
            return True, None  # Optional, defaults to economy
        
        try:
            CabinClass(cabin.lower())
            return True, None
        except ValueError:
            valid_classes = [c for c in CabinClass]
            return False, f"Invalid cabin class '{cabin}'. Valid: {', '.join(valid_classes)}"
    
    @staticmethod
    def validate_origin_destination_different(
        origin: Optional[str],
        destination: Optional[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that origin and destination are different.
        CRITICAL: Prevents TAS → TAS searches.
        
        Args:
            origin: Origin airport code
            destination: Destination airport code
            
        Returns:
            (is_valid, error_message)
        """
        if not origin or not destination:
            return True, None  # Can't validate if either is missing
        
        if origin.upper() == destination.upper():
            return False, f"Origin and destination cannot be the same ({origin})"
        
        return True, None


class StateValidator:
    """Validates data for specific conversation states"""
    
    def __init__(self):
        self.field_validator = FieldValidator()
    
    def validate_for_state(
        self,
        state: ConversationFlowState,
        trip_spec: TripSpec
    ) -> ValidationResult:
        """
        Validate trip_spec has required data for the given state.
        
        Args:
            state: Current conversation state
            trip_spec: Trip specification
            
        Returns:
            ValidationResult with is_valid flag and error list
        """
        errors = []
        
        state_value = state if hasattr(state, 'value') else state
        logger.debug(f"Validating state: {state_value}")
        
        if state == ConversationFlowState.CONFIRM_DESTINATION:
            # Should have valid destination at minimum
            if trip_spec.destination:
                is_valid, error = self.field_validator.validate_iata_code(
                    trip_spec.destination,
                    allow_unknown=True  # Be flexible during conversation
                )
                if not is_valid:
                    errors.append(f"destination: {error}")
            
            # Check origin if present
            if trip_spec.origin:
                is_valid, error = self.field_validator.validate_iata_code(
                    trip_spec.origin,
                    allow_unknown=True
                )
                if not is_valid:
                    errors.append(f"origin: {error}")
            
            # CRITICAL: Check origin != destination
            is_valid, error = self.field_validator.validate_origin_destination_different(
                trip_spec.origin,
                trip_spec.destination
            )
            if not is_valid:
                errors.append(error)
        
        elif state == ConversationFlowState.DATES:
            # Should have valid departure date
            if trip_spec.depart_date:
                is_valid, error = self.field_validator.validate_date(trip_spec.depart_date)
                if not is_valid:
                    errors.append(f"depart_date: {error}")
            
            # Validate return date if present
            if trip_spec.return_date:
                is_valid, error = self.field_validator.validate_date(trip_spec.return_date)
                if not is_valid:
                    errors.append(f"return_date: {error}")
                
                # Return date must be after departure
                if trip_spec.depart_date and trip_spec.return_date:
                    try:
                        depart = datetime.fromisoformat(trip_spec.depart_date)
                        ret = datetime.fromisoformat(trip_spec.return_date)
                        if ret <= depart:
                            errors.append("return_date: Must be after departure date")
                    except ValueError:
                        pass
        
        elif state == ConversationFlowState.PASSENGERS:
            # Should have valid passenger count
            if trip_spec.passengers:
                is_valid, error = self.field_validator.validate_passenger_count(trip_spec.passengers)
                if not is_valid:
                    errors.append(f"passengers: {error}")
        
        elif state == ConversationFlowState.PREFERENCES:
            # Validate optional preference fields
            if trip_spec.trip_type:
                is_valid, error = self.field_validator.validate_trip_type(trip_spec.trip_type)
                if not is_valid:
                    errors.append(f"trip_type: {error}")
            
            if trip_spec.cabin_class:
                is_valid, error = self.field_validator.validate_cabin_class(trip_spec.cabin_class)
                if not is_valid:
                    errors.append(f"cabin_class: {error}")
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Validation failed for {state}: {errors}")
        
        return ValidationResult(is_valid=is_valid, errors=errors)
    
    def validate_search_ready(self, trip_spec: TripSpec) -> ValidationResult:
        """
        Validate trip_spec is ready for flight search.
        
        Minimum required fields:
        - origin (valid IATA code)
        - destination (valid IATA code, different from origin)
        - depart_date (valid future date)
        - passengers (optional, defaults to 1)
        
        Args:
            trip_spec: Trip specification
            
        Returns:
            ValidationResult with is_valid flag and error list
        """
        errors = []
        
        logger.info("Validating search readiness")
        
        # 1. Validate origin
        if not trip_spec.origin:
            errors.append("origin: Required for search")
        else:
            is_valid, error = self.field_validator.validate_iata_code(
                trip_spec.origin,
                allow_unknown=False  # Strict validation for search
            )
            if not is_valid:
                errors.append(f"origin: {error}")
        
        # 2. Validate destination
        if not trip_spec.destination:
            errors.append("destination: Required for search")
        else:
            is_valid, error = self.field_validator.validate_iata_code(
                trip_spec.destination,
                allow_unknown=False  # Strict validation for search
            )
            if not is_valid:
                errors.append(f"destination: {error}")
        
        # 3. CRITICAL: Check origin != destination
        is_valid, error = self.field_validator.validate_origin_destination_different(
            trip_spec.origin,
            trip_spec.destination
        )
        if not is_valid:
            errors.append(f"CRITICAL: {error}")
            logger.error(f"Same airport search attempted: {trip_spec.origin} → {trip_spec.destination}")
        
        # 4. Validate departure date
        if not trip_spec.depart_date:
            errors.append("depart_date: Required for search")
        else:
            is_valid, error = self.field_validator.validate_date(trip_spec.depart_date)
            if not is_valid:
                errors.append(f"depart_date: {error}")
        
        # 5. Validate return date (if present)
        if trip_spec.return_date:
            is_valid, error = self.field_validator.validate_date(trip_spec.return_date)
            if not is_valid:
                errors.append(f"return_date: {error}")
            
            # Return must be after departure
            if trip_spec.depart_date and trip_spec.return_date:
                try:
                    depart = datetime.fromisoformat(trip_spec.depart_date)
                    ret = datetime.fromisoformat(trip_spec.return_date)
                    if ret <= depart:
                        errors.append("return_date: Must be after departure date")
                except ValueError:
                    pass
        
        # 6. Validate passenger count
        is_valid, error = self.field_validator.validate_passenger_count(trip_spec.passengers)
        if not is_valid:
            errors.append(f"passengers: {error}")
        
        # 7. Validate cabin class
        if trip_spec.cabin_class:
            is_valid, error = self.field_validator.validate_cabin_class(trip_spec.cabin_class)
            if not is_valid:
                errors.append(f"cabin_class: {error}")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("✓ Search validation passed")
        else:
            logger.error(f"✗ Search validation failed: {errors}")
        
        return ValidationResult(is_valid=is_valid, errors=errors)


class InputSanitizer:
    """Sanitizes and normalizes user input"""
    
    @staticmethod
    def sanitize_iata_code(code: str) -> str:
        """Normalize IATA code to uppercase 3-letter format"""
        if not code:
            return ""
        return code.strip().upper()[:3]
    
    @staticmethod
    def sanitize_date(date_str: str) -> Optional[str]:
        """Normalize date to ISO format"""
        if not date_str:
            return None
        
        try:
            # Try parsing various formats
            date_obj = datetime.fromisoformat(date_str)
            return date_obj.date().isoformat()
        except ValueError:
            return None
    
    @staticmethod
    def sanitize_passenger_count(count: Any) -> int:
        """Normalize passenger count to integer"""
        try:
            count_int = int(count)
            # Clamp between 1 and 9
            return max(1, min(9, count_int))
        except (ValueError, TypeError):
            return 1
    
    @staticmethod
    def sanitize_text_input(text: str, max_length: int = 500) -> str:
        """
        Sanitize free-form text input.
        Removes potentially harmful characters, limits length.
        
        Args:
            text: Raw user input
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Strip whitespace
        text = text.strip()
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Limit length
        text = text[:max_length]
        
        return text
    
    @staticmethod
    def sanitize_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize preference dictionary.
        
        Args:
            preferences: Raw preferences dict
            
        Returns:
            Sanitized preferences
        """
        sanitized = {}
        
        # Allowed preference keys with their types
        allowed = {
            "direct_only": bool,
            "time_preference": str,
            "max_stops": int,
            "cabin_class": str,
            "sort_by": str
        }
        
        for key, expected_type in allowed.items():
            if key in preferences:
                value = preferences[key]
                try:
                    # Type conversion
                    if expected_type == bool:
                        sanitized[key] = bool(value)
                    elif expected_type == int:
                        sanitized[key] = int(value)
                    elif expected_type == str:
                        sanitized[key] = str(value).strip()[:50]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid preference value for {key}: {value}")
        
        return sanitized


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_valid_iata_code(code: str) -> bool:
    """
    Quick check if IATA code is valid.
    
    Args:
        code: Airport code
        
    Returns:
        True if valid
    """
    if not code or len(code) != 3:
        return False
    return code.upper() in VALID_IATA_CODES


def add_custom_iata_code(code: str) -> bool:
    """
    Add custom IATA code to validation database.
    Useful for adding new airports dynamically.
    
    Args:
        code: 3-letter airport code
        
    Returns:
        True if added successfully
    """
    if not code or len(code) != 3 or not code.isalpha():
        return False
    
    VALID_IATA_CODES.add(code.upper())
    logger.info(f"Added custom IATA code: {code.upper()}")
    return True


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'FieldValidator',
    'StateValidator',
    'InputSanitizer',
    'ValidationError',
    'is_valid_iata_code',
    'add_custom_iata_code'
]