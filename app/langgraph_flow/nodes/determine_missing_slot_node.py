# langgraph_flow/nodes/determine_missing_slot_node.py

import logging
from app.langgraph_flow.state import (
    ConversationState,
    update_state,
    validate_required_slots,
    is_ready_for_search,
)

logger = logging.getLogger(__name__)


async def determine_missing_slot_node(state: ConversationState) -> ConversationState:
    """
    Determine the next missing parameter (slot) required for a valid flight search.
    Sets:
      - missing_parameter
      - ready_for_search
    """

    logger.info("[DetermineMissingSlotNode] Checking required slots")

    # First-level required slots
    missing = validate_required_slots(state)

    if missing:
        logger.info(f"[DetermineMissingSlotNode] Missing primary slot: {missing}")
        return update_state(state, {
            "missing_parameter": missing,
            "ready_for_search": False,
        })

    # After primary slots are filled, check other important ones in desired order
    if state.get("passengers") in (None, 0):
        missing = "passengers"
    elif state.get("travel_class") is None:
        missing = "travel_class"
    elif state.get("budget") is None:
        missing = "budget"
    else:
        missing = None

    if missing:
        logger.info(f"[DetermineMissingSlotNode] Missing secondary slot: {missing}")
        return update_state(state, {
            "missing_parameter": missing,
            "ready_for_search": False,
        })

    # All required slots collected
    ready = is_ready_for_search(state)
    logger.info(f"[DetermineMissingSlotNode] All slots present. ready_for_search={ready}")

    return update_state(state, {
        "missing_parameter": None,
        "ready_for_search": ready,
    })
