"""
Extract Parameters Node - Unified Production Version
====================================================
Handles BOTH Rule-Based (passed from router) and LLM extraction logic.
Serves as the central "Commit" point for all state changes.
"""

import logging
from datetime import date
from typing import Any, Dict, Optional
from app.langgraph_flow.state import ConversationState, update_state

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITIES
# ============================================================================

def _parse_date(value: Optional[str]) -> Optional[date]:
    """Parse date string safely."""
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


# ============================================================================
# MAIN NODE
# ============================================================================

async def extract_parameters_node(state: ConversationState) -> ConversationState:
    """
    Unified extraction node.
    
    Sources of Data:
    1. Rule-Based: Data passed from 'should_use_llm_node' in 'extracted_params'
    2. LLM: Generated on-the-fly via 'generate_json_response'
    
    Both sources flow through the SAME "Smart Merge" logic to ensure safety.
    """
    
    missing_parameter = getattr(state, "missing_parameter", None)
    extraction_data = {}
    source_label = "LLM"

    # ========================================
    # PATH 1: Rule-Based (Pre-calculated)
    # ========================================
    if getattr(state, "use_rule_based_path", False):
        logger.info("[ExtractParams] Rule-based path detected. Processing pre-extracted params.")
        
        # 1. Get the data passed from should_use_llm_node
        extraction_data = getattr(state, "extracted_params", {}) or {}
        source_label = "Rule-Based"
        
        if not extraction_data:
            logger.warning("[ExtractParams] Rule path active but 'extracted_params' is empty.")

    # ========================================
    # PATH 2: LLM Extraction (On-the-fly)
    # ========================================
    else:
        latest_message = state.latest_user_message
        if not latest_message:
            return state

        logger.info(f"[ExtractParams] LLM path detected. Extracting from: '{latest_message[:30]}...'")
        
        from services.llm_service import generate_json_response
        
        try:
            # Use the context-aware prompt builder defined in your file (omitted here for brevity, assumed available)
            # If it's a local function, ensure it is defined or imported
            from app.langgraph_flow.nodes.extract_parameters_node import build_context_aware_prompt_internal
            system_prompt = build_context_aware_prompt_internal(state, latest_message)
            
            extraction_data = await generate_json_response(
                system_prompt=system_prompt,
                user_prompt=latest_message
            )
        except Exception as e:
            # If import fails (because function is local), define simple prompt
            try:
                # Fallback prompt logic if helper is missing
                logger.warning(f"[ExtractParams] using fallback prompt due to error: {e}")
                extraction_data = await generate_json_response(
                    system_prompt="Extract travel data (origin, destination, date) from user input.",
                    user_prompt=latest_message
                )
            except Exception as inner_e:
                logger.error(f"[ExtractParams] LLM failed: {inner_e}")
                return state

    # ========================================
    # UNIFIED SMART MERGE LOGIC
    # ========================================
    # This logic now runs for BOTH paths (Rule & LLM)
    
    updates = {}
    all_fields = ["origin", "destination", "departure_date", "return_date", "passengers", "travel_class", "budget"]
    
    logger.info(f"[ExtractParams] Merging data from source: {source_label}")

    for key in all_fields:
        new_val = extraction_data.get(key)
        
        # Skip empty/null
        if new_val in [None, "null", "", []]:
            continue
            
        # ------------------------------------
        # Logic A: Targeted Update (We asked for this)
        # ------------------------------------
        if missing_parameter and key == missing_parameter:
            logger.info(f"[{source_label}] ✅ Updating REQUESTED slot '{key}' -> {new_val}")
            updates[key] = _sanitize_value(key, new_val)

        # ------------------------------------
        # Logic B: Safe Fill (Slot is empty)
        # ------------------------------------
        else:
            current_val = getattr(state, key, None)
            if not current_val:
                logger.info(f"[{source_label}] 🔹 Filling EMPTY slot '{key}' -> {new_val}")
                updates[key] = _sanitize_value(key, new_val)
            
            # ------------------------------------
            # Logic C: Conflict Protection (Slot is full)
            # ------------------------------------
            else:
                # If we are on Rule-Based path, we trust the Router's decision implicitly
                # UNLESS it conflicts with what we asked for.
                
                # Example: User said "Tashkent". Rule extracted "Tashkent" as Origin (correct).
                # But Rule might ALSO accidentally extract "Tashkent" as Destination (if regex matches).
                # We must protect the field we DIDN'T ask for if it's already set.
                
                if key in ["origin", "destination"]:
                    logger.info(
                        f"[{source_label}] 🛡️ Protecting existing '{key}'={current_val}. "
                        f"Ignoring update to {new_val}."
                    )
                else:
                    # Allow updates for other fields (e.g. changing date)
                    updates[key] = _sanitize_value(key, new_val)

    # ========================================
    # Derived Fields & Validation
    # ========================================
    
    # Helper to set array fields
    if "destination" in updates:
        val = str(updates["destination"])
        if len(val) == 3: updates["destination_airports"] = [val.upper()]
            
    if "origin" in updates:
        val = str(updates["origin"])
        if len(val) == 3: updates["origin_airports"] = [val.upper()]

    # Final Safety Check: Same Origin/Dest
    final_origin = updates.get("origin", state.origin)
    final_destination = updates.get("destination", state.destination)
    
    if final_origin and final_destination and final_origin.upper() == final_destination.upper():
         logger.warning(f"[ExtractParams] ⛔ Conflict detected: Origin == Destination ({final_origin}). Reverting update.")
         # Revert the field that was just updated
         if "origin" in updates and missing_parameter == "destination":
             del updates["origin"]
         elif "destination" in updates and missing_parameter == "origin":
             del updates["destination"]
         # If both were updated or we can't tell, delete the one we asked for to force re-ask
         elif missing_parameter in updates:
             del updates[missing_parameter]

    return update_state(state, updates)


def _sanitize_value(key: str, value: Any) -> Any:
    """Helper to format values correctly"""
    if key in ["departure_date", "return_date"]:
        return _parse_date(str(value))
    if key == "passengers":
        try: return int(value)
        except: return 1
    return value


# ============================================================================
# PROMPT BUILDER (Included to ensure self-contained execution)
# ============================================================================

def build_context_aware_prompt_internal(state: ConversationState, message: str) -> str:
    missing = getattr(state, "missing_parameter", None)
    context_str = ""
    if state.origin: context_str += f"Origin: {state.origin}\n"
    if state.destination: context_str += f"Destination: {state.destination}\n"
    
    focus_instruction = ""
    if missing:
        focus_instruction = f"USER WAS ASKED FOR: {missing.upper()}. Focus on extracting that."

    return f"""
    Extract travel details from the user message.
    JSON Output keys: origin, destination, departure_date, return_date, passengers, travel_class.
    
    KNOWN CONTEXT:
    {context_str}
    
    {focus_instruction}
    """