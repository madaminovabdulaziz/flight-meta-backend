"""
Rule-Based Ranking Node
Rank flights using cheap_ranker - NO LLM needed.

Uses cheap_ranker for complete 6-factor scoring with template-based outputs.
This node is used in the rule-based path when confidence is high.

Latency: <50ms
Cost: $0
"""

import logging
from app.langgraph_flow.state import ConversationState, update_state
from services.cheap_ranker import cheap_ranker

logger = logging.getLogger(__name__)


async def rule_based_ranking_node(state: ConversationState) -> ConversationState:
    """
    Rank flights using deterministic logic (no LLM).
    
    Uses cheap_ranker.rank_flights() to:
    - Calculate 6-factor scores
    - Generate labels using templates
    - Generate summaries using templates
    - Generate tradeoffs using rules
    - Generate smart suggestions using rules
    
    This is the rule-based alternative to ranking_node.
    """
    
    raw_flights = state.get("raw_flights", [])
    
    if not raw_flights:
        logger.warning("[RuleBasedRanking] No flights to rank")
        return update_state(state, {
            "ranked_flights": [],
            "ranking_explanation": "No flights found for your criteria."
        })
    
    logger.info(f"[RuleBasedRanking] Ranking {len(raw_flights)} flights (NO LLM)")
    
    # Use cheap ranker (zero LLM cost)
    ranking_result = cheap_ranker.rank_flights(raw_flights, state)
    
    ranked_flights = ranking_result.get("ranked_flights", [])
    suggestions = ranking_result.get("smart_suggestions", [])
    stats = ranking_result.get("summary_stats", {})
    
    logger.info(f"[RuleBasedRanking] Ranked {len(ranked_flights)} flights using templates (NO LLM)")
    
    # Create simple explanation
    explanation = f"Found {stats.get('total_flights', 0)} flights. " \
                  f"Showing top {len(ranked_flights)} options ranked by price, duration, and quality."
    
    return update_state(state, {
        "ranked_flights": ranked_flights,
        "ranking_explanation": explanation,
        "smart_suggestions": suggestions,
    })