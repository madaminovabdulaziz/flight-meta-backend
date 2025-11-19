"""
Determine Missing Slot Node - Production-Ready
==============================================
Smart slot validation that respects hybrid routing and sets sensible defaults.
"""

import logging
from typing import Optional, List, Dict, Any

from app.langgraph_flow.state import (
    ConversationState,
    update_state,
)

logger = logging.getLogger(__name__)


# ==========================================
# CONFIGURATION
# ==========================================

PRIMARY_REQUIRED_SLOTS = [
    "destination",
    "origin",
    "departure_date",
]

OPTIONAL_SLOT_DEFAULTS = {
    "passengers": 1,
    "travel_class": "Economy",
    "flexibility": 0,
}

OPTIONAL_SLOTS = [
    "return_date",
    "budget",
    "flexibility",
]


# ==========================================
# SLOT VALIDATION
# ==========================================

def _get_missing_primary_slot(state: ConversationState) -> Optional[str]:
    for slot in PRIMARY_REQUIRED_SLOTS:
        value = getattr(state, slot, None)
        if value is None or value == "" or value == []:
            logger.debug(f"[DetermineMissingSlot] Primary slot '{slot}' is missing")
            return slot
    return None


def _apply_optional_defaults(state: ConversationState) -> Dict[str, Any]:
    defaults = {}
    
    for slot, default_value in OPTIONAL_SLOT_DEFAULTS.items():
        current_value = getattr(state, slot, None)

        # Apply default if empty
        if current_value in (None, "", 0):
            defaults[slot] = default_value
            logger.info(f"[DetermineMissingSlot] Applying default: {slot} = {default_value}")

    return defaults


def _is_ready_for_search(state: ConversationState) -> bool:
    for slot in PRIMARY_REQUIRED_SLOTS:
        value = getattr(state, slot, None)
        if value is None or value == "" or value == []:
            return False
    return True


# ==========================================
# SLOT VALIDATION WITH ROUTING CONTEXT
# ==========================================

def _validate_with_routing_context(
    state: ConversationState
) -> tuple[Optional[str], Dict[str, Any]]:

    use_rule_based = getattr(state, "use_rule_based_path", False)

    missing_slot = _get_missing_primary_slot(state)
    
    if missing_slot:
        logger.info(
            f"[DetermineMissingSlot] Missing primary slot: '{missing_slot}' "
            f"(path: {'rule-based' if use_rule_based else 'LLM'})"
        )
        return missing_slot, {}

    defaults = _apply_optional_defaults(state)
    
    logger.info(
        f"[DetermineMissingSlot] All primary slots present, "
        f"applied {len(defaults)} defaults"
    )
    
    return None, defaults


# ==========================================
# MAIN NODE
# ==========================================

async def determine_missing_slot_node(state: ConversationState) -> ConversationState:
    
    logger.info("[DetermineMissingSlot] Checking slot requirements")
    
    missing_slot, defaults_to_apply = _validate_with_routing_context(state)

    if missing_slot:
        logger.info(f"[DetermineMissingSlot] ❌ Not ready - missing '{missing_slot}'")
        return update_state(state, {
            "missing_parameter": missing_slot,
            "ready_for_search": False,
        })

    ready = _is_ready_for_search(state)

    logger.info(
        f"[DetermineMissingSlot] ✅ Ready for search "
        f"(destination: {getattr(state, 'destination', None)}, "
        f"origin: {getattr(state, 'origin', None)}, "
        f"date: {getattr(state, 'departure_date', None)})"
    )

    updates = {
        **defaults_to_apply,
        "missing_parameter": None,
        "ready_for_search": ready,
    }
    
    return update_state(state, updates)


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_slot_status(state: ConversationState) -> Dict[str, Any]:
    primary_status = {}
    for slot in PRIMARY_REQUIRED_SLOTS:
        value = getattr(state, slot, None)
        primary_status[slot] = {
            "filled": bool(value and value != "" and value != []),
            "value": value,
        }
    
    optional_status = {}
    for slot in OPTIONAL_SLOTS + list(OPTIONAL_SLOT_DEFAULTS.keys()):
        value = getattr(state, slot, None)
        optional_status[slot] = {
            "filled": bool(value and value not in (None, "", 0)),
            "value": value,
        }
    
    missing_primary = _get_missing_primary_slot(state)
    ready = _is_ready_for_search(state)
    
    return {
        "primary_slots": primary_status,
        "optional_slots": optional_status,
        "missing_primary_slot": missing_primary,
        "ready_for_search": ready,
        "needs_defaults": bool(_apply_optional_defaults(state)),
    }


def validate_slot_completeness(state: ConversationState) -> Dict[str, bool]:
    completeness = {}

    for slot in PRIMARY_REQUIRED_SLOTS:
        value = getattr(state, slot, None)
        completeness[slot] = bool(value and value != "" and value != [])

    for slot in OPTIONAL_SLOTS + list(OPTIONAL_SLOT_DEFAULTS.keys()):
        value = getattr(state, slot, None)
        completeness[slot] = bool(value and value not in (None, "", 0, []))
    
    return completeness


def get_next_question_context(state: ConversationState) -> Dict[str, Any]:
    missing_slot = _get_missing_primary_slot(state)
    
    if not missing_slot:
        return {
            "has_missing": False,
            "missing_slot": None,
            "question_context": None,
        }

    known_slots = {}
    for slot in PRIMARY_REQUIRED_SLOTS:
        if slot != missing_slot:
            value = getattr(state, slot, None)
            if value:
                known_slots[slot] = value
    
    return {
        "has_missing": True,
        "missing_slot": missing_slot,
        "known_slots": known_slots,
        "question_context": {
            "destination": getattr(state, "destination", None),
            "origin": getattr(state, "origin", None),
            "departure_date": getattr(state, "departure_date", None),
        },
    }


def calculate_slot_coverage(state: ConversationState) -> float:
    all_slots = PRIMARY_REQUIRED_SLOTS + list(OPTIONAL_SLOT_DEFAULTS.keys())
    filled = sum(
        1 for slot in all_slots
        if getattr(state, slot, None) not in (None, "", 0, [])
    )
    return filled / len(all_slots)
