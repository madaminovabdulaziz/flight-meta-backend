"""
LangGraph Flow Definition - Production-Ready Hybrid Architecture
================================================================
Main orchestration graph with intelligent rule-based/LLM routing for AI Trip Planner.
"""

import logging
from typing import Literal, Optional, Dict, Any

from langgraph.graph import StateGraph, END

from app.langgraph_flow.state import ConversationState, create_initial_state
from app.langgraph_flow.nodes.entry_node import entry_node
from app.langgraph_flow.nodes.load_user_memory_node import load_user_memory_node
from app.langgraph_flow.nodes.should_use_llm_node import (
    should_use_llm_node,
    route_based_on_confidence,
)
from app.langgraph_flow.nodes.classify_intent_node import classify_intent_node
from app.langgraph_flow.nodes.extract_parameters_node import extract_parameters_node
from app.langgraph_flow.nodes.determine_missing_slot_node import determine_missing_slot_node
from app.langgraph_flow.nodes.ask_next_question_node import ask_next_question_node
from app.langgraph_flow.nodes.refusal_node import refusal_node
from app.langgraph_flow.nodes.flight_search_node import flight_search_node
from app.langgraph_flow.nodes.ranking_node import ranking_node
from app.langgraph_flow.nodes.finalize_response_node import finalize_response_node
from app.langgraph_flow.nodes.learn_preferences_node import learn_preferences_node


# Rule-based path nodes
from app.langgraph_flow.nodes.rule_based_question_node import rule_based_question_node
from app.langgraph_flow.nodes.rule_based_ranking_node import rule_based_ranking_node

logger = logging.getLogger(__name__)


# ==========================================
# CONDITIONAL EDGE FUNCTIONS - DECISION READERS ONLY
# ==========================================

def route_after_intent_classification(
    state: ConversationState,
) -> Literal["refusal", "extract_parameters"]:
    """
    Route based on intent classification (LLM path only).
    DECISION READER: Only reads intent from state.
    """
    intent = getattr(state, "intent", "") or ""

    if intent in ["irrelevant", "chitchat"]:
        logger.info(f"[Route] Intent '{intent}' → refusal_node")
        return "refusal"

    logger.info(f"[Route] Intent '{intent}' → extract_parameters")
    return "extract_parameters"


def route_after_missing_slot_check(
    state: ConversationState,
) -> Literal["ask_question", "rule_question", "flight_search"]:
    """
    Route after checking for missing parameters.
    DECISION READER: Only reads from state, NEVER recalculates.
    """
    missing = getattr(state, "missing_parameter", None)

    if not missing:
        logger.info("[Route] No missing parameters → flight_search")
        return "flight_search"

    use_rule_based = getattr(state, "use_rule_based_path", False)

    if use_rule_based:
        logger.info(
            f"[Route] Missing '{missing}' + rule-based path → rule_question (template)"
        )
        return "rule_question"

    logger.info(f"[Route] Missing '{missing}' + LLM path → ask_question (LLM)")
    return "ask_question"


def route_after_flight_search(
    state: ConversationState,
) -> Literal["rule_ranking", "llm_ranking"]:
    """
    Route to appropriate ranking node based on path.
    DECISION READER: Only reads from state, NEVER recalculates.
    """
    use_rule_based = getattr(state, "use_rule_based_path", False)

    if use_rule_based:
        logger.info(
            "[Route] Rule-based path → rule_ranking (cheap_ranker, 0 LLM calls)"
        )
        return "rule_ranking"

    logger.info("[Route] LLM path → llm_ranking (with explanations)")
    return "llm_ranking"


# ==========================================
# GRAPH CONSTRUCTION
# ==========================================

def create_trip_planner_graph() -> StateGraph:
    """
    Build the production-ready hybrid LangGraph with clean separation of concerns.
    
    CRITICAL: This graph does NOT handle session loading/saving.
    That is the responsibility of the caller (chat.py).
    """

    graph = StateGraph(ConversationState)

    # ------------------------
    # NODE REGISTRATION
    # ------------------------

    # Common entry nodes
    graph.add_node("entry", entry_node)
    # NOTE: load_session_state_node REMOVED - done outside graph
    graph.add_node("load_user_memory", load_user_memory_node)

    # Router
    graph.add_node("should_use_llm", should_use_llm_node)

    # LLM path nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("extract_parameters", extract_parameters_node)
    graph.add_node("ask_question", ask_next_question_node)
    graph.add_node("llm_ranking", ranking_node)

    # Rule-based path nodes
    graph.add_node("rule_question", rule_based_question_node)
    graph.add_node("rule_ranking", rule_based_ranking_node)
    graph.add_node("learn_preferences", learn_preferences_node)

    # Shared nodes
    graph.add_node("determine_missing_slot", determine_missing_slot_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("flight_search", flight_search_node)
    graph.add_node("finalize_response", finalize_response_node)


    # ------------------------
    # EDGE DEFINITIONS
    # ------------------------

    graph.set_entry_point("entry")

    # Common entry flow (UPDATED: no load_session_state)
    graph.add_edge("entry", "load_user_memory")
    graph.add_edge("load_user_memory", "should_use_llm")

    # Routing decision
    graph.add_conditional_edges(
        "should_use_llm",
        route_based_on_confidence,
        {
            "rule_based_path": "extract_parameters",
            "llm_path": "classify_intent",
        },
    )

    # LLM path
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent_classification,
        {
            "refusal": "refusal",
            "extract_parameters": "extract_parameters",
        },
    )

    graph.add_edge("extract_parameters", "learn_preferences")
    graph.add_edge("learn_preferences", "determine_missing_slot")
    graph.add_edge("refusal", END)

    # Missing slot → ask / rule / search
    graph.add_conditional_edges(
        "determine_missing_slot",
        route_after_missing_slot_check,
        {
            "ask_question": "ask_question",
            "rule_question": "rule_question",
            "flight_search": "flight_search",
        },
    )

    graph.add_edge("ask_question", END)
    graph.add_edge("rule_question", END)

    # Flight search → ranking
    graph.add_conditional_edges(
        "flight_search",
        route_after_flight_search,
        {
            "rule_ranking": "rule_ranking",
            "llm_ranking": "llm_ranking",
        },
    )

    graph.add_edge("rule_ranking", "finalize_response")
    graph.add_edge("llm_ranking", "finalize_response")
    graph.add_edge("finalize_response", END)

    logger.info("✅ Production-ready trip planner graph constructed")
    return graph


# ==========================================
# COMPILED GRAPH SINGLETON
# ==========================================

trip_planner_graph = create_trip_planner_graph().compile()

logger.info("✅ Trip planner graph compiled and ready")
logger.info("🚀 Performance target: 80-95% requests use rule-based path")


# ==========================================
# GRAPH EXECUTION - NEW SIMPLIFIED VERSION
# ==========================================

async def run_conversation_turn(
    session_id: str,
    user_id: Optional[int],
    user_message: str,
) -> ConversationState:
    """
    Execute one conversation turn with proper state persistence.
    
    NEW ARCHITECTURE:
    1. Load state ONCE (outside graph)
    2. Execute graph with fresh state
    3. Save state ONCE (outside graph)
    
    This ensures:
    - No double-load
    - No routing field corruption
    - Atomic save to both Redis and DB
    """
    from app.db.database import AsyncSessionLocal
    from app.db.crud.session import load_session_state, save_session_state
    from services.session_store import (
        get_session_state_from_redis,
        save_session_state_to_redis,
    )

    logger.info(f"🚀 Starting conversation turn for session {session_id}")

    # ========================================
    # STEP 1: LOAD STATE (Outside Graph)
    # ========================================
    
    loaded_state: Optional[ConversationState] = None
    
    # Try Redis first (fast cache)
    try:
        redis_data = await get_session_state_from_redis(session_id)
        
        if redis_data and isinstance(redis_data, dict):
            # Convert Redis dict to ConversationState
            # ONLY restore non-routing fields
            loaded_state = _restore_state_from_dict(redis_data, session_id)
            logger.info(f"📂 Loaded state from Redis (turn {loaded_state.turn_count or 0})")
            
    except Exception as e:
        logger.warning(f"⚠️ Redis load failed: {e}, trying DB...")

    # Fallback to DB if Redis failed
    if loaded_state is None:
        try:
            async with AsyncSessionLocal() as db:
                db_data = await load_session_state(db, session_id=session_id)
                
                if db_data:
                    loaded_state = _restore_state_from_dict(db_data, session_id)
                    logger.info(f"📂 Loaded state from DB (turn {loaded_state.turn_count or 0})")
                    
        except Exception as e:
            logger.warning(f"⚠️ DB load failed: {e}")

    # Create new state if nothing was loaded
    if loaded_state is None:
        logger.info("🆕 Creating new session")
        loaded_state = create_initial_state(
            session_id=session_id,
            user_id=user_id,
            latest_message=user_message,
        )

    # ========================================
    # STEP 2: PREPARE STATE FOR GRAPH
    # ========================================
    
    # Update state for current turn
    loaded_state.turn_count = (loaded_state.turn_count or 0) + 1
    loaded_state.latest_user_message = user_message
    
    # Append to conversation history
    if not loaded_state.conversation_history:
        loaded_state.conversation_history = []
    loaded_state.conversation_history.append({
        "role": "user",
        "content": user_message,
        "turn": loaded_state.turn_count,
    })

    logger.info(
        f"📝 Prepared state: turn {loaded_state.turn_count}, "
        f"session {session_id[:12]}..."
    )

    # ========================================
    # STEP 3: EXECUTE GRAPH
    # ========================================
    
    try:
        result = await trip_planner_graph.ainvoke(loaded_state)

        # Convert result to ConversationState if needed
        if isinstance(result, dict):
            # Graph returned dict - convert to state
            final_state = _convert_dict_to_state(result, session_id)
        elif isinstance(result, ConversationState):
            final_state = result
        else:
            logger.error(f"❌ Unexpected result type: {type(result)}")
            raise TypeError(f"Expected dict or ConversationState, got {type(result)}")

        # Add assistant response to conversation history
        if final_state.assistant_message:
            if not final_state.conversation_history:
                final_state.conversation_history = []
            final_state.conversation_history.append({
                "role": "assistant",
                "content": final_state.assistant_message,
                "turn": final_state.turn_count,
            })

        # Log path used
        used_rule_based = getattr(final_state, "use_rule_based_path", False)
        confidence = getattr(final_state, "routing_confidence", 0.0) or 0.0

        if used_rule_based:
            logger.info(
                f"✅ RULE-BASED path completed "
                f"(confidence: {confidence:.3f}, 0 LLM calls, ~300-800ms)"
            )
        else:
            logger.info(
                f"⚠️ LLM path completed "
                f"(confidence: {confidence:.3f}, ~3 LLM calls, ~2-4s)"
            )

    except Exception as e:
        logger.error(f"❌ Graph execution failed: {e}", exc_info=True)
        raise

    # ========================================
    # STEP 4: SAVE STATE (Outside Graph)
    # ========================================
    
    try:
        # Save to DB first (source of truth)
        async with AsyncSessionLocal() as db:
            await save_session_state(db, session_id=session_id, state=final_state)
            await db.commit()
        logger.debug("✅ State saved to DB")
        
        # Then save to Redis (cache) - only after DB succeeds
        await save_session_state_to_redis(session_id, final_state)
        logger.debug("✅ State saved to Redis")
        
    except Exception as e:
        logger.error(f"❌ Failed to save state: {e}", exc_info=True)
        # Don't fail the request if save fails
        # State is still returned to user and will be in memory

    return final_state


# ==========================================
# STATE RESTORATION HELPERS
# ==========================================

# Fields that should NEVER be restored from old state
ROUTING_FIELDS = {
    "use_rule_based_path",
    "routing_confidence",
    "confidence_breakdown",
    "intent",
    "llm_intent",
    "extracted_params",
}

# Fields that are only for current turn (don't restore)
TRANSIENT_FIELDS = {
    "latest_user_message",
    "assistant_message",
    "next_placeholder",
}


def _restore_state_from_dict(
    data: Dict[str, Any],
    session_id: str
) -> ConversationState:
    """
    Restore state from dict, excluding routing and transient fields.
    
    This ensures:
    - Routing decisions are made fresh each turn
    - Old assistant messages don't leak through
    - Session continuity for slots and history
    """
    if not isinstance(data, dict):
        logger.warning(f"Invalid state data type: {type(data)}")
        return create_initial_state(session_id=session_id, user_id=None, latest_message="")
    
    # Get all valid field names from ConversationState
    allowed_fields = set(ConversationState.__dataclass_fields__.keys())
    
    # Filter to only allowed fields, excluding routing and transient
    filtered = {}
    for k, v in data.items():
        if k in allowed_fields and k not in ROUTING_FIELDS and k not in TRANSIENT_FIELDS:
            filtered[k] = v
    
    # Ensure session_id is set
    filtered["session_id"] = session_id
    
    # Convert date strings back to date objects
    from datetime import datetime, date
    
    for date_field in ["departure_date", "return_date"]:
        if date_field in filtered and isinstance(filtered[date_field], str):
            try:
                filtered[date_field] = date.fromisoformat(filtered[date_field])
            except:
                filtered[date_field] = None
    
    for datetime_field in ["created_at", "updated_at", "search_timestamp"]:
        if datetime_field in filtered and isinstance(filtered[datetime_field], str):
            try:
                filtered[datetime_field] = datetime.fromisoformat(filtered[datetime_field])
            except:
                if datetime_field in ["created_at", "updated_at"]:
                    filtered[datetime_field] = datetime.utcnow()
                else:
                    filtered[datetime_field] = None
    
    try:
        return ConversationState(**filtered)
    except Exception as e:
        logger.error(f"Failed to restore state: {e}", exc_info=True)
        return create_initial_state(session_id=session_id, user_id=None, latest_message="")


def _convert_dict_to_state(result: dict, session_id: str) -> ConversationState:
    """
    Convert graph result dict to ConversationState.
    
    Used when graph returns dict instead of state object.
    """
    allowed_fields = set(ConversationState.__dataclass_fields__.keys())
    filtered = {k: v for k, v in result.items() if k in allowed_fields}
    filtered["session_id"] = session_id
    
    try:
        return ConversationState(**filtered)
    except Exception as e:
        logger.error(f"Failed to convert dict to state: {e}", exc_info=True)
        raise


# ==========================================
# GRAPH VISUALIZATION
# ==========================================

def get_graph_stats(state: ConversationState) -> dict:
    """
    Extract performance statistics from final state.
    """
    use_rule = getattr(state, "use_rule_based_path", False)
    confidence = getattr(state, "routing_confidence", None)
    breakdown = getattr(state, "confidence_breakdown", None)
    turn_count = getattr(state, "turn_count", None)
    intent = getattr(state, "intent", None)
    session_id = getattr(state, "session_id", None)

    return {
        "path_used": "rule_based" if use_rule else "llm",
        "routing_confidence": confidence,
        "confidence_breakdown": breakdown,
        "turn_count": turn_count,
        "intent": intent,
        "estimated_llm_calls": 0 if use_rule else 3,
        "session_id": session_id,
    }


# ==========================================
# HEALTH CHECK
# ==========================================

def validate_graph_health() -> bool:
    """
    Validate that graph is properly constructed.
    """
    try:
        if trip_planner_graph is None:
            logger.error("Graph is None")
            return False

        logger.info("✅ Graph health check passed")
        return True
    except Exception as e:
        logger.error(f"❌ Graph health check failed: {e}")
        return False
