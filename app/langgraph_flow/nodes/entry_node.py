import logging
from datetime import datetime
from app.langgraph_flow.state import ConversationState, update_state

logger = logging.getLogger(__name__)

async def entry_node(state: ConversationState) -> ConversationState:
    """
    First node in the LangGraph pipeline.
    Prepares state for the turn, initializes history, and increments counters.
    """

    logger.info("[EntryNode] Starting new turn")

    # Initialize conversation history
    history = state.conversation_history or []

    # Append latest user message
    latest_msg = state.latest_user_message
    if latest_msg:
        history.append({"role": "user", "content": latest_msg})

    # Increment turn count
    turn_count = state.turn_count + 1

    return update_state(state, {
        "conversation_history": history,
        "turn_count": turn_count,
        "updated_at": datetime.utcnow(),
    })
