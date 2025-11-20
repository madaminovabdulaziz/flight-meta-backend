"""
Learn Preferences Node - Production Version
===========================================
"The Listener" 👂
Analyzes user messages for long-term habits and persists them to the database.

Logic:
- Skips rule-based/short messages (efficiency)
- Extracts ONLY permanent preferences (e.g., "I hate layovers", "Only Star Alliance")
- Ignores specific trip details (e.g., "I'm going to Paris tomorrow")
- Writes to DB asynchronously
"""

import logging
from typing import Dict, Any

from app.langgraph_flow.state import ConversationState, update_state
from services.llm_service import generate_json_response
from services.preference_service import PreferenceService

logger = logging.getLogger(__name__)
pref_service = PreferenceService()

# ==========================================
# CONFIGURATION
# ==========================================

# Only run learner if message length > X chars to avoid analyzing "Yes", "No", "London"
MIN_MESSAGE_LENGTH = 15 

SYSTEM_PROMPT = """
You are a Memory Manager for a travel AI. 
Your goal is to extract LONG-TERM USER PREFERENCES from the conversation.

DISTINCTION IS CRITICAL:
- ❌ IGNORE Current Trip Details: "I want to go to Paris", "Next Tuesday", "2 people".
- ✅ CAPTURE Enduring Habits: "I hate Ryanair", "I prefer morning flights", "I only fly Business".

OUTPUT SCHEMA (JSON):
Return a flat dictionary of preferences found. Use standardized keys where possible:
- preferred_airlines (list of airline codes/names)
- avoided_airlines (list)
- preferred_airports (list)
- travel_class (Economy, Business, First)
- prefers_direct (boolean)
- max_layover_hours (int)
- preferred_time_of_day (morning, afternoon, evening, night)
- budget_sensitivity (low, medium, high)
- seat_preference (aisle, window)

EXAMPLE:
User: "I want to fly to London tomorrow, but please no EasyJet, I had a bad experience."
Output: {"avoided_airlines": ["EasyJet"]}

User: "Find me a flight to Dubai."
Output: {}

If no long-term preferences are found, return empty JSON: {}.
"""

# ==========================================
# MAIN NODE
# ==========================================

async def learn_preferences_node(state: ConversationState) -> ConversationState:
    
    # --------------------------------------
    # 1. Guard Rails (Efficiency)
    # --------------------------------------
    
    # Skip anonymous users (can't save to DB)
    if not state.user_id:
        return state

    # Skip rule-based path (usually simple slot-filling answers like "Tashkent")
    # We only want to analyze rich natural language.
    if state.use_rule_based_path:
        return state

    message = state.latest_user_message or ""
    
    # Skip short messages to save LLM tokens
    if len(message) < MIN_MESSAGE_LENGTH:
        return state

    logger.info(f"[Memory] 👂 Listening for preferences in: '{message[:50]}...'")

    # --------------------------------------
    # 2. LLM Extraction
    # --------------------------------------
    try:
        response = await generate_json_response(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=message,
            temperature=0.0  # Strict output
        )
        
        # The LLM might return the dict directly or wrapped in a key
        # Handle both {"found_preferences": {...}} and just {...}
        new_prefs = response.get("found_preferences", response)
        
        # Cleanup: Remove empty keys
        new_prefs = {k: v for k, v in new_prefs.items() if v not in [None, [], "", {}]}

        if not new_prefs:
            logger.debug("[Memory] No new preferences found.")
            return state

        logger.info(f"[Memory] 🧠 Learned new preferences: {new_prefs}")

    except Exception as e:
        logger.warning(f"[Memory] Extraction failed: {e}")
        return state

    # --------------------------------------
    # 3. Persist to Database
    # --------------------------------------
    try:
        # This updates the MySQL 'user_preferences' table
        await pref_service.bulk_update_preferences(state.user_id, new_prefs)
        logger.info("[Memory] ✅ Saved to Database")
    except Exception as e:
        logger.error(f"[Memory] DB Save failed: {e}", exc_info=True)
        # Don't return here, we still want to update the local state for this turn!

    # --------------------------------------
    # 4. Update Local State (Immediate Effect)
    # --------------------------------------
    # We merge the new preferences into the current state object
    # so the Ranker sees them *immediately* in this very request.
    
    current_prefs = state.long_term_preferences.copy() or {}
    current_prefs.update(new_prefs)
    
    return update_state(state, {"long_term_preferences": current_prefs})
