# app/conversation/react_agent.py
"""
ReAct Agent - Production Layer 5
Implements the ReAct (Reason + Act) pattern for intelligent agentic behavior.

The Game-Changer:
Instead of just extracting data, the AI THINKS and ACTS like a human travel agent:

BEFORE (Simple Extraction):
User: "Flights to Tokyo next month"
AI: Searches Tokyo next month → Returns results

AFTER (ReAct Agent):
User: "Flights to Tokyo next month"
Thought: "User wants Tokyo, prices will be high in peak season"
Action: Check flight API for next month
Observation: "$1200 for direct flights"
Thought: "That's expensive! Let me check flexible dates to save them money"
Action: Check flight API with ±3 days flexibility
Observation: "$800 if they leave 2 days earlier"
Final Response: "I found flights for $1200, BUT if you leave 2 days earlier, I found one for $800. Would you like to see both options?"

This is WHY users will switch from Google Flights to your product.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# AGENT ACTION TYPES
# ============================================================

class ActionType(str, Enum):
    """Types of actions the agent can take"""
    SEARCH_FLIGHTS = "search_flights"
    CHECK_FLEXIBLE_DATES = "check_flexible_dates"
    CHECK_NEARBY_AIRPORTS = "check_nearby_airports"
    QUERY_KNOWLEDGE_BASE = "query_knowledge_base"
    COMPARE_PRICES = "compare_prices"
    GET_VISA_INFO = "get_visa_info"
    GET_WEATHER_INFO = "get_weather_info"
    FINAL_ANSWER = "final_answer"


class AgentAction(BaseModel):
    """Action to be executed by the agent"""
    action_type: ActionType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str


class AgentObservation(BaseModel):
    """Observation from executing an action"""
    action: AgentAction
    result: Any
    success: bool
    error: Optional[str] = None


class AgentThought(BaseModel):
    """Agent's reasoning process"""
    thought: str
    next_action: Optional[ActionType] = None


# ============================================================
# REACT CYCLE
# ============================================================

class ReActCycle(BaseModel):
    """Single cycle of Reason → Act → Observe"""
    step: int
    thought: AgentThought
    action: Optional[AgentAction] = None
    observation: Optional[AgentObservation] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# REACT AGENT
# ============================================================

class ReActAgent:
    """
    Intelligent agent using ReAct pattern for flight search.
    Makes autonomous decisions to provide best value to users.
    """

    MAX_ITERATIONS = 5  # Prevent infinite loops
    PRICE_IMPROVEMENT_THRESHOLD = 0.15  # 15% cheaper = worth suggesting

    def __init__(
        self,
        llm_gateway: Optional[Any] = None,
        rag_engine: Optional[Any] = None,
        flight_api: Optional[Any] = None
    ):
        """
        Initialize ReAct agent.

        Args:
            llm_gateway: LLM gateway for reasoning
            rag_engine: RAG engine for knowledge retrieval
            flight_api: Flight search API interface
        """
        self.llm_gateway = llm_gateway
        self.rag_engine = rag_engine
        self.flight_api = flight_api

        logger.info("✓ ReActAgent initialized")

    async def run(
        self,
        user_query: str,
        context: Dict[str, Any],
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run ReAct agent to process user query.

        Args:
            user_query: User's query
            context: Conversation context
            max_iterations: Maximum ReAct iterations

        Returns:
            Agent response with reasoning chain
        """
        max_iter = max_iterations or self.MAX_ITERATIONS

        logger.info(f"🤖 ReAct Agent starting: '{user_query[:50]}...'")

        # Initialize reasoning chain
        react_cycles: List[ReActCycle] = []
        final_answer = None

        for step in range(1, max_iter + 1):
            logger.info(f"--- ReAct Cycle {step}/{max_iter} ---")

            # Step 1: THOUGHT (Reason)
            thought = await self._generate_thought(
                query=user_query,
                context=context,
                previous_cycles=react_cycles
            )

            logger.info(f"💭 THOUGHT: {thought.thought}")

            cycle = ReActCycle(
                step=step,
                thought=thought
            )

            # Check if agent wants to give final answer
            if thought.next_action == ActionType.FINAL_ANSWER:
                final_answer = await self._generate_final_answer(
                    query=user_query,
                    context=context,
                    reasoning_chain=react_cycles
                )
                logger.info(f"✅ FINAL ANSWER: {final_answer[:100]}...")
                break

            # Step 2: ACTION (Act)
            if thought.next_action:
                action = await self._plan_action(
                    thought=thought,
                    context=context
                )

                cycle.action = action
                logger.info(
                    f"⚡ ACTION: {action.action_type.value} "
                    f"(params={list(action.parameters.keys())})"
                )

                # Step 3: OBSERVATION (Observe)
                observation = await self._execute_action(action, context)
                cycle.observation = observation

                if observation.success:
                    logger.info(f"👁️ OBSERVATION: Success - {observation.result}")
                else:
                    logger.warning(f"👁️ OBSERVATION: Failed - {observation.error}")

            react_cycles.append(cycle)

            # Check if we should stop
            if self._should_terminate(react_cycles):
                logger.info("Agent terminating based on heuristics")
                final_answer = await self._generate_final_answer(
                    query=user_query,
                    context=context,
                    reasoning_chain=react_cycles
                )
                break

        # Fallback if no final answer generated
        if final_answer is None:
            logger.warning("Max iterations reached without final answer")
            final_answer = "I've analyzed your request. Let me search for flights based on what we discussed."

        return {
            "final_answer": final_answer,
            "reasoning_chain": [cycle.dict() for cycle in react_cycles],
            "total_steps": len(react_cycles),
            "query": user_query
        }

    async def _generate_thought(
        self,
        query: str,
        context: Dict[str, Any],
        previous_cycles: List[ReActCycle]
    ) -> AgentThought:
        """
        Generate agent's reasoning (thought).

        Args:
            query: User query
            context: Current context
            previous_cycles: Previous ReAct cycles

        Returns:
            AgentThought
        """
        # Build prompt for reasoning
        prompt = self._build_reasoning_prompt(
            query=query,
            context=context,
            previous_cycles=previous_cycles
        )

        try:
            if self.llm_gateway:
                from app.conversation.llm_gateway import get_llm_gateway
                gateway = get_llm_gateway()

                result = await gateway.call(
                    prompt=prompt,
                    task_type="reasoning",
                    temperature=0.3,
                    max_tokens=256
                )

                # Parse thought from response
                thought_text = result["response"].strip()

                # Extract next action (simple parsing)
                next_action = self._extract_action_from_thought(thought_text)

                return AgentThought(
                    thought=thought_text,
                    next_action=next_action
                )
            else:
                # Fallback heuristic reasoning
                return self._heuristic_reasoning(query, context, previous_cycles)

        except Exception as e:
            logger.error(f"Error generating thought: {e}")
            # Fallback to heuristic
            return self._heuristic_reasoning(query, context, previous_cycles)

    def _heuristic_reasoning(
        self,
        query: str,
        context: Dict[str, Any],
        previous_cycles: List[ReActCycle]
    ) -> AgentThought:
        """
        Fallback heuristic-based reasoning when LLM unavailable.

        Args:
            query: User query
            context: Context
            previous_cycles: Previous cycles

        Returns:
            AgentThought
        """
        step_count = len(previous_cycles)

        # Step 0: Initial search
        if step_count == 0:
            return AgentThought(
                thought="User wants flight search. Let me search for basic options first.",
                next_action=ActionType.SEARCH_FLIGHTS
            )

        # Step 1: Check if we can save money with flexible dates
        elif step_count == 1:
            prev_obs = previous_cycles[0].observation

            if prev_obs and prev_obs.success:
                # Check if there are results
                results = prev_obs.result
                if isinstance(results, dict) and results.get("offers"):
                    cheapest_price = results.get("cheapest_price", 999999)

                    if cheapest_price > 500:  # Arbitrary threshold
                        return AgentThought(
                            thought=(
                                f"Found flights for ${cheapest_price}, but that seems expensive. "
                                "Let me check flexible dates to find better deals."
                            ),
                            next_action=ActionType.CHECK_FLEXIBLE_DATES
                        )

        # Step 2: Check nearby airports
        elif step_count == 2:
            return AgentThought(
                thought="Let me check nearby airports for potentially cheaper options.",
                next_action=ActionType.CHECK_NEARBY_AIRPORTS
            )

        # Step 3: Final answer
        return AgentThought(
            thought="I have enough information to provide a comprehensive answer.",
            next_action=ActionType.FINAL_ANSWER
        )

    async def _plan_action(
        self,
        thought: AgentThought,
        context: Dict[str, Any]
    ) -> AgentAction:
        """
        Plan the action based on thought.

        Args:
            thought: Agent's thought
            context: Current context

        Returns:
            AgentAction
        """
        action_type = thought.next_action

        if not action_type:
            action_type = ActionType.FINAL_ANSWER

        # Build parameters based on action type
        parameters = {}

        if action_type == ActionType.SEARCH_FLIGHTS:
            parameters = {
                "origin": context.get("origin", "TAS"),
                "destination": context.get("destination"),
                "depart_date": context.get("depart_date"),
                "passengers": context.get("passengers", 1)
            }

        elif action_type == ActionType.CHECK_FLEXIBLE_DATES:
            parameters = {
                "origin": context.get("origin"),
                "destination": context.get("destination"),
                "depart_date": context.get("depart_date"),
                "flexibility_days": 3,
                "passengers": context.get("passengers", 1)
            }

        elif action_type == ActionType.CHECK_NEARBY_AIRPORTS:
            parameters = {
                "origin": context.get("origin"),
                "destination": context.get("destination"),
                "include_nearby": True,
                "depart_date": context.get("depart_date")
            }

        elif action_type == ActionType.QUERY_KNOWLEDGE_BASE:
            parameters = {
                "query": context.get("destination", "travel tips"),
                "doc_type": "destination_guide"
            }

        return AgentAction(
            action_type=action_type,
            parameters=parameters,
            reasoning=thought.thought
        )

    async def _execute_action(
        self,
        action: AgentAction,
        context: Dict[str, Any]
    ) -> AgentObservation:
        """
        Execute the planned action.

        Args:
            action: Action to execute
            context: Current context

        Returns:
            AgentObservation
        """
        logger.debug(f"Executing action: {action.action_type.value}")

        try:
            if action.action_type == ActionType.SEARCH_FLIGHTS:
                result = await self._search_flights(action.parameters)

            elif action.action_type == ActionType.CHECK_FLEXIBLE_DATES:
                result = await self._check_flexible_dates(action.parameters)

            elif action.action_type == ActionType.CHECK_NEARBY_AIRPORTS:
                result = await self._check_nearby_airports(action.parameters)

            elif action.action_type == ActionType.QUERY_KNOWLEDGE_BASE:
                result = await self._query_knowledge_base(action.parameters)

            else:
                result = {"message": "Action type not implemented yet"}

            return AgentObservation(
                action=action,
                result=result,
                success=True
            )

        except Exception as e:
            logger.error(f"Action execution failed: {e}")

            return AgentObservation(
                action=action,
                result=None,
                success=False,
                error=str(e)
            )

    async def _search_flights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for flights"""
        # Mock implementation (replace with actual flight API)
        logger.info(f"Searching flights: {params}")

        # Simulate flight search results
        return {
            "offers": [
                {
                    "price": 850,
                    "airline": "Turkish Airlines",
                    "duration": "12h 30m",
                    "stops": 0
                },
                {
                    "price": 920,
                    "airline": "Emirates",
                    "duration": "14h 15m",
                    "stops": 1
                }
            ],
            "cheapest_price": 850,
            "average_price": 885
        }

    async def _check_flexible_dates(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check flexible dates for better prices"""
        logger.info(f"Checking flexible dates: {params}")

        # Simulate finding cheaper option on different dates
        return {
            "original_price": 850,
            "flexible_options": [
                {
                    "depart_date": "2025-11-20",
                    "price": 680,
                    "savings": 170,
                    "savings_percent": 20
                },
                {
                    "depart_date": "2025-11-23",
                    "price": 720,
                    "savings": 130,
                    "savings_percent": 15
                }
            ],
            "best_deal": {
                "depart_date": "2025-11-20",
                "price": 680,
                "savings": 170
            }
        }

    async def _check_nearby_airports(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check nearby airports for alternatives"""
        logger.info(f"Checking nearby airports: {params}")

        return {
            "alternative_routes": [
                {
                    "origin": "SAW",  # Sabiha Gökçen (alternative to IST)
                    "price": 780,
                    "savings": 70
                }
            ]
        }

    async def _query_knowledge_base(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query RAG knowledge base"""
        logger.info(f"Querying knowledge base: {params}")

        if self.rag_engine:
            try:
                from app.conversation.rag_engine import get_rag_engine
                rag = await get_rag_engine()

                result = await rag.query(
                    query=params.get("query", ""),
                    doc_type=params.get("doc_type"),
                    top_k=2,
                    generate_answer=False
                )

                return result

            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                return {"retrieved_docs": []}

        return {"retrieved_docs": []}

    def _should_terminate(self, cycles: List[ReActCycle]) -> bool:
        """
        Check if agent should terminate based on heuristics.

        Args:
            cycles: ReAct cycles so far

        Returns:
            True if should terminate
        """
        # Terminate if we have good results after 2 steps
        if len(cycles) >= 2:
            last_cycle = cycles[-1]

            if last_cycle.observation and last_cycle.observation.success:
                result = last_cycle.observation.result

                # Check if we found significant savings
                if isinstance(result, dict):
                    if "flexible_options" in result:
                        best_deal = result.get("best_deal", {})
                        savings_percent = best_deal.get("savings_percent", 0)

                        if savings_percent >= 15:
                            logger.info(
                                f"Found good deal ({savings_percent}% savings), terminating"
                            )
                            return True

        return False

    async def _generate_final_answer(
        self,
        query: str,
        context: Dict[str, Any],
        reasoning_chain: List[ReActCycle]
    ) -> str:
        """
        Generate final answer based on reasoning chain.

        Args:
            query: Original query
            context: Context
            reasoning_chain: All ReAct cycles

        Returns:
            Final answer text
        """
        # Build comprehensive summary from reasoning chain
        summary_parts = []

        for cycle in reasoning_chain:
            if cycle.observation and cycle.observation.success:
                result = cycle.observation.result

                if isinstance(result, dict):
                    # Extract key information
                    if "offers" in result:
                        cheapest = result.get("cheapest_price")
                        summary_parts.append(
                            f"I found flights starting at ${cheapest}."
                        )

                    if "flexible_options" in result:
                        best_deal = result.get("best_deal", {})
                        savings = best_deal.get("savings")
                        new_date = best_deal.get("depart_date")
                        new_price = best_deal.get("price")

                        if savings and savings > 50:
                            summary_parts.append(
                                f"BUT if you fly on {new_date}, I found flights for "
                                f"${new_price} - that's ${savings} cheaper! "
                                f"Would you like to see both options?"
                            )

        if summary_parts:
            return " ".join(summary_parts)
        else:
            return "Let me search for flights based on your requirements."

    @staticmethod
    def _extract_action_from_thought(thought: str) -> Optional[ActionType]:
        """
        Extract action type from thought text.

        Args:
            thought: Thought text

        Returns:
            ActionType or None
        """
        thought_lower = thought.lower()

        # Simple keyword matching
        if "search" in thought_lower and "flight" in thought_lower:
            return ActionType.SEARCH_FLIGHTS
        elif "flexible" in thought_lower or "date" in thought_lower:
            return ActionType.CHECK_FLEXIBLE_DATES
        elif "nearby" in thought_lower or "alternative" in thought_lower:
            return ActionType.CHECK_NEARBY_AIRPORTS
        elif "knowledge" in thought_lower or "visa" in thought_lower:
            return ActionType.QUERY_KNOWLEDGE_BASE
        elif "enough" in thought_lower or "final" in thought_lower:
            return ActionType.FINAL_ANSWER

        return None

    @staticmethod
    def _build_reasoning_prompt(
        query: str,
        context: Dict[str, Any],
        previous_cycles: List[ReActCycle]
    ) -> str:
        """
        Build prompt for LLM reasoning.

        Args:
            query: User query
            context: Context
            previous_cycles: Previous cycles

        Returns:
            Prompt string
        """
        prompt = f"""You are an intelligent flight search agent. Your goal is to find the BEST flight deals for users, not just the first results.

User Query: "{query}"
Context: {json.dumps(context, indent=2)}

Previous Actions:
"""

        for i, cycle in enumerate(previous_cycles, 1):
            prompt += f"\nStep {i}:\n"
            prompt += f"  Thought: {cycle.thought.thought}\n"

            if cycle.action:
                prompt += f"  Action: {cycle.action.action_type.value}\n"

            if cycle.observation:
                prompt += f"  Observation: {cycle.observation.success}\n"

        prompt += """
What should you do next? Think step by step:
1. Do we have enough information to help the user?
2. Can we save them money by checking flexible dates or nearby airports?
3. Should we provide the final answer now?

Your thought (be concise):"""

        return prompt


# ============================================================
# FACTORY FUNCTION
# ============================================================

_react_agent_instance: Optional[ReActAgent] = None

def get_react_agent() -> ReActAgent:
    """
    Get singleton ReAct agent instance.

    Returns:
        ReActAgent instance
    """
    global _react_agent_instance

    if _react_agent_instance is None:
        _react_agent_instance = ReActAgent()

    return _react_agent_instance


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ReActAgent",
    "ActionType",
    "AgentAction",
    "AgentObservation",
    "AgentThought",
    "ReActCycle",
    "get_react_agent"
]
