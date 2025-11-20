"""
Confirmation Node - Production Version
======================================
Shows user a friendly summary before executing search.

Example output:
"Let me search for:
✈️  London → Istanbul
📅  Nov 26, 2025
👥  3 travelers
💺  Economy

Searching for the best options..."
"""

import logging
from datetime import datetime

from app.langgraph_flow.state import ConversationState, update_state

logger = logging.getLogger(__name__)


async def confirmation_node(state: ConversationState) -> ConversationState:
    """
    Generate a friendly confirmation message before search.
    
    This node runs AFTER all required slots are filled,
    BEFORE the actual flight search.
    
    ✅ SAFE: Only updates mutable UI fields (assistant_message, suggestions)
    """
    
    logger.info("[ConfirmationNode] Generating search summary")
    
    # Extract collected info (defensive access)
    origin = getattr(state, "origin", "Unknown")
    destination = getattr(state, "destination", "Unknown")
    departure_date = getattr(state, "departure_date", None)
    return_date = getattr(state, "return_date", None)
    passengers = getattr(state, "passengers", 1)
    travel_class = getattr(state, "travel_class", "Economy")
    
    # Build confirmation message
    lines = ["Let me search for:"]
    
    # Route
    lines.append(f"✈️  {origin} → {destination}")
    
    # Dates
    if departure_date:
        try:
            date_str = departure_date.strftime("%b %d, %Y")
            if return_date:
                return_str = return_date.strftime("%b %d")
                lines.append(f"📅  {date_str} - {return_str}")
            else:
                lines.append(f"📅  {date_str} (one-way)")
        except AttributeError:
            # departure_date might be string in some edge cases
            lines.append(f"📅  {departure_date}")
    
    # Passengers
    if passengers == 1:
        lines.append("👤  1 traveler")
    else:
        lines.append(f"👥  {passengers} travelers")
    
    # Class
    lines.append(f"💺  {travel_class}")
    
    # Footer
    lines.append("")
    lines.append("Searching for the best options...")
    
    message = "\n".join(lines)
    
    logger.info(f"[ConfirmationNode] Confirmation: {message[:100]}...")
    
    # ✅ SAFE: Only update mutable UI fields
    return update_state(state, {
        "assistant_message": message,
        "suggestions": [],
        "next_placeholder": None,
        "flow_stage": "searching",
    })


def should_show_confirmation(state: ConversationState) -> bool:
    """
    Check if we should show confirmation.
    
    Show confirmation when:
    1. All required slots are filled
    2. We haven't searched yet
    """
    ready = getattr(state, "ready_for_search", False)
    searched = getattr(state, "search_executed", False)
    
    return ready and not searched