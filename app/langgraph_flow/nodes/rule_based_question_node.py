"""
Rule-Based Question Node
Generate questions using templates - NO LLM needed.

Uses template_engine to generate questions with zero latency and cost.
This node is used in the rule-based path when confidence is high.

Latency: <5ms
Cost: $0
"""

import logging
from app.langgraph_flow.state import ConversationState, update_state
from services.template_engine import template_engine

logger = logging.getLogger(__name__)


async def rule_based_question_node(state: ConversationState) -> ConversationState:
    """
    Generate next question using templates (no LLM).
    
    Uses template_engine.generate_question() to create:
    - next_placeholder
    - suggestions
    - assistant_message
    
    This is the rule-based alternative to ask_next_question_node.
    """
    
    missing = state.get("missing_parameter")
    
    if not missing:
        logger.warning("[RuleBasedQuestion] No missing_parameter set")
        missing = "destination"
    
    logger.info(f"[RuleBasedQuestion] Generating question for: {missing}")
    
    # Use template engine (zero LLM cost)
    question_data = template_engine.generate_question(missing, state)
    
    logger.info(f"[RuleBasedQuestion] Generated template question (NO LLM)")
    
    return update_state(state, {
        "next_placeholder": question_data.get("placeholder"),
        "suggestions": question_data.get("suggestions", []),
        "assistant_message": question_data.get("message"),
        "last_question": question_data.get("message"),
    })