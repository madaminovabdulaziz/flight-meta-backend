"""
Entry Node - Session Initialization & IP Geolocation
"""
import logging
from app.langgraph_flow.state import ConversationState, update_state
from services.ip_geolocation import IPGeolocationService

logger = logging.getLogger(__name__)
geo_service = IPGeolocationService()

async def entry_node(state: ConversationState) -> ConversationState:
    """
    Initialize turn. 
    If this is the VERY first turn and Origin is unknown, detect it via IP.
    """
    
    # 1. Standard Init
    logger.info(f"[Entry] Processing session {state.session_id}")
    
    # 2. Auto-Detect Origin (Only on fresh sessions)
    if state.turn_count == 0 and not state.origin:
        try:
            # In a real HTTP request context, you'd pass the IP.
            # For this node, we might need to inject it or use a default.
            # Assuming we pass IP in state.debug_info or similar from API layer
            user_ip = state.debug_info.get("client_ip", "127.0.0.1")
            
            geo_data = await geo_service.get_nearest_airport_from_ip(user_ip)
            
            airport = geo_data.get("airport", {})
            origin_code = airport.get("iata")
            city_name = airport.get("city")
            
            if origin_code:
                logger.info(f"[Entry] Auto-detected origin: {origin_code} ({city_name})")
                
                # We update the state silently. 
                # The user doesn't know yet, but the 'missing_slot' logic will see it filled!
                return update_state(state, {
                    "origin": origin_code,
                    "origin_airports": [origin_code],
                    # Optional: Add a flag so we can tell the user "We assumed you're in..."
                    "debug_info": {**state.debug_info, "auto_origin": True}
                })
                
        except Exception as e:
            logger.warning(f"[Entry] Geolocation failed: {e}")

    return state