"""
LangGraph Flow Definition - Production-Ready Hybrid Architecture
================================================================
Main orchestration graph with intelligent rule-based/LLM routing for AI Trip Planner.
"""

import logging
from typing import Literal, Optional

from langgraph.graph import StateGraph, END

from app.langgraph_flow.state import ConversationState, create_initial_state
from app.langgraph_flow.nodes.entry_node import entry_node
from app.langgraph_flow.nodes.load_session_state_node import load_session_state_node
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
    """

    graph = StateGraph(ConversationState)

    # ------------------------
    # NODE REGISTRATION
    # ------------------------

    # Common entry nodes
    graph.add_node("entry", entry_node)
    graph.add_node("load_session_state", load_session_state_node)
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

    # Shared nodes
    graph.add_node("determine_missing_slot", determine_missing_slot_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("flight_search", flight_search_node)
    graph.add_node("finalize_response", finalize_response_node)

    # ------------------------
    # EDGE DEFINITIONS
    # ------------------------

    graph.set_entry_point("entry")

    # Common entry flow
    graph.add_edge("entry", "load_session_state")
    graph.add_edge("load_session_state", "load_user_memory")
    graph.add_edge("load_user_memory", "should_use_llm")

    # Routing decision
    graph.add_conditional_edges(
        "should_use_llm",
        route_based_on_confidence,
        {
            "rule_based_path": "determine_missing_slot",
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

    graph.add_edge("extract_parameters", "determine_missing_slot")
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
# GRAPH EXECUTION
# ==========================================

async def run_conversation_turn(
    session_id: str,
    user_id: Optional[int],
    user_message: str,
) -> ConversationState:
    """
    Execute one conversation turn with proper state persistence.
    """
    from app.db.database import AsyncSessionLocal
    from app.db.crud.session import load_session_state, save_session_state
    from services.session_store import (
        get_session_state_from_redis,
        save_session_state_to_redis,
    )

    logger.info(f"🚀 Starting conversation turn for session {session_id}")

    # ------------------------
    # STEP 1: LOAD EXISTING STATE (Redis first, then DB)
    # ------------------------
    initial_state: Optional[ConversationState] = None
    
    try:
        # Try Redis first (faster)
        redis_state = await get_session_state_from_redis(session_id)
        
        if redis_state and isinstance(redis_state, dict):
            # Convert Redis dict to ConversationState
            allowed_fields = set(ConversationState.__dataclass_fields__.keys())
            filtered = {k: v for k, v in redis_state.items() if k in allowed_fields}
            initial_state = ConversationState(**filtered)
            logger.info(f"📂 Loaded state from Redis (turn {initial_state.turn_count or 0})")
            
    except Exception as e:
        logger.warning(f"⚠️ Redis load failed: {e}, trying DB...")

    # Fallback to DB if Redis failed
    if initial_state is None:
        try:
            async with AsyncSessionLocal() as db:
                db_state = await load_session_state(db, session_id=session_id)
                
                if db_state:
                    if isinstance(db_state, ConversationState):
                        initial_state = db_state
                    elif isinstance(db_state, dict):
                        allowed_fields = set(ConversationState.__dataclass_fields__.keys())
                        filtered = {k: v for k, v in db_state.items() if k in allowed_fields}
                        initial_state = ConversationState(**filtered)
                    
                    logger.info(f"📂 Loaded state from DB (turn {initial_state.turn_count or 0})")
                    
        except Exception as e:
            logger.warning(f"⚠️ DB load failed: {e}")

    # Create new state if nothing was loaded
    if initial_state is None:
        logger.info("🆕 Creating new session")
        initial_state = create_initial_state(
            session_id=session_id,
            user_id=user_id,
            latest_message=user_message,
        )
    else:
        # Update existing state for new turn
        logger.info(f"📝 Continuing existing session (turn {initial_state.turn_count + 1})")

    # Update state with current turn info
    initial_state.turn_count = (initial_state.turn_count or 0) + 1
    initial_state.latest_user_message = user_message
    
    # Append to conversation history
    if not initial_state.conversation_history:
        initial_state.conversation_history = []
    initial_state.conversation_history.append({
        "role": "user",
        "content": user_message,
        "turn": initial_state.turn_count,
    })

    # ------------------------
    # STEP 2: EXECUTE GRAPH
    # ------------------------
    try:
        result = await trip_planner_graph.ainvoke(initial_state)

        # Convert result to ConversationState if needed
        if isinstance(result, dict):
            allowed_fields = set(ConversationState.__dataclass_fields__.keys())
            filtered_result = {k: v for k, v in result.items() if k in allowed_fields}
            final_state = ConversationState(**filtered_result)
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

    # ------------------------
    # STEP 3: *** SAVE STATE *** (THE MISSING PIECE!)
    # ------------------------
    try:
        # Save to Redis (fast, temporary)
        await save_session_state_to_redis(session_id, final_state)
        logger.debug("✅ State saved to Redis")
        
        # Save to DB (persistent, slower) - do in background if possible
        async with AsyncSessionLocal() as db:
            await save_session_state(db, session_id=session_id, state=final_state)
            await db.commit()
        logger.debug("✅ State saved to DB")
        
    except Exception as e:
        logger.error(f"❌ Failed to save state: {e}", exc_info=True)
        # Don't fail the request if save fails, but log it prominently
        # The state is still returned to the user

    return final_state

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