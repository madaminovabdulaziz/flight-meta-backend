"""
Finalize Response Node - HUMAN-FRIENDLY VERSION
================================================
Makes flight results feel conversational and helpful.

Before: "Here are flights:"
After:  "Great news! I found 12 flights for you. Here are the top 3:"
"""

import logging
from datetime import datetime
from typing import Dict, Any

from app.langgraph_flow.state import ConversationState, update_state
from services.session_store import save_session_state_to_redis

logger = logging.getLogger(__name__)


def _format_flight_brief(flight: Dict[str, Any]) -> str:
    """
    Format a single flight into a readable string.
    
    Example: "Turkish Airlines: LHR→IST (Direct) - £280 | 09:30-14:15"
    """
    # Unwrap if nested
    f = flight.get("raw_data", flight)
    
    airline = f.get("airline_name") or f.get("airline") or "Unknown"
    origin = f.get("origin", "???")
    dest = f.get("destination", "???")
    price = f.get("price", 0)
    currency = f.get("currency", "GBP")
    stops = f.get("stops", 0)
    
    # Stops
    if stops == 0:
        stop_text = "Direct"
    elif stops == 1:
        stop_text = "1 stop"
    else:
        stop_text = f"{stops} stops"
    
    # Times
    dep_raw = f.get("departure_time", "")
    arr_raw = f.get("arrival_time", "")
    
    dep = dep_raw.split("T")[1][:5] if "T" in dep_raw else "morning"
    arr = arr_raw.split("T")[1][:5] if "T" in arr_raw else "evening"
    
    # Get label if available
    label = f.get("label", "")
    label_emoji = label.split()[0] if label else "✈️"
    
    return f"{label_emoji} {airline}: {origin}→{dest} ({stop_text}) - {currency}{price} | {dep}-{arr}"


async def finalize_response_node(state: ConversationState) -> ConversationState:
    """
    Generate final response with personality.
    """
    
    ranked = getattr(state, "ranked_flights", []) or []
    raw = getattr(state, "raw_flights", []) or []
    
    # If no ranked but have raw, use raw
    if not ranked and raw:
        ranked = raw
    
    assistant_message = getattr(state, "assistant_message", None)
    suggestions = getattr(state, "suggestions", []) or []
    is_complete = False
    
    # ========================================
    # CASE 1: We have flights! 🎉
    # ========================================
    if ranked:
        total_count = len(ranked)
        display_count = min(3, total_count)
        
        # Build enthusiastic header
        if total_count == 1:
            header = "I found 1 flight option for you:"
        elif total_count <= 5:
            header = f"Great news! I found {total_count} flight options. Here they are:"
        else:
            header = f"Excellent! I found {total_count} flights for you. Here are the top {display_count}:"
        
        lines = [header, ""]
        
        # Show top flights
        for i, flight in enumerate(ranked[:display_count], start=1):
            lines.append(f"{i}. {_format_flight_brief(flight)}")
        
        # Footer with personality
        lines.append("")
        
        if total_count > display_count:
            lines.append(f"💡 {total_count - display_count} more options available.")
        
        lines.append("Ready to book, or want to adjust your search?")
        
        assistant_message = "\n".join(lines)
        
        # Smart suggestions
        suggestions = [
            "Book option 1",
            "Show more options",
            "Change dates",
            "New search"
        ]
        
        is_complete = True
    
    # ========================================
    # CASE 2: Search executed but no results 😔
    # ========================================
    elif getattr(state, "search_executed", False):
        assistant_message = (
            "Hmm, I couldn't find any flights matching your exact criteria. "
            "Would you like to try:\n\n"
            "• Different dates\n"
            "• Nearby airports\n"
            "• Flexible travel class"
        )
        suggestions = [
            "Try different dates",
            "Change airports",
            "Start new search"
        ]
        is_complete = False
    
    # ========================================
    # CASE 3: Still collecting info
    # ========================================
    else:
        # Keep existing message if we have one
        if not assistant_message:
            assistant_message = "Let me help you find the perfect flight! What are you looking for?"
            suggestions = ["Search flights", "Help"]
    
    # ========================================
    # Save state to Redis
    # ========================================
    try:
        await save_session_state_to_redis(state.session_id, state.to_dict())
    except Exception as e:
        logger.error(f"[FinalizeResponseNode] Redis save failed: {e}")
    
    # ========================================
    # Update state
    # ========================================
    return update_state(state, {
        "assistant_message": assistant_message,
        "suggestions": suggestions,
        "is_complete": is_complete,
        "updated_at": datetime.utcnow(),
    })