"""
Session Persistence Tests
==========================
Verify that state is correctly saved and loaded across turns.
"""

import asyncio
import pytest
from datetime import date, datetime

from app.langgraph_flow.graph import run_conversation_turn
from app.langgraph_flow.state import ConversationState


@pytest.mark.asyncio
async def test_single_turn_with_complete_info():
    """
    Test: Single turn with all information provided.
    
    Expected:
    - State created
    - Routing decision made
    - Parameters extracted
    - State saved
    """
    session_id = "test_single_001"
    user_id = 999
    message = "I want to fly from London to Istanbul on December 25th, 2 passengers, economy class"
    
    # Execute turn
    state = await run_conversation_turn(
        session_id=session_id,
        user_id=user_id,
        user_message=message
    )
    
    # Verify state
    assert state.session_id == session_id
    assert state.user_id == user_id
    assert state.turn_count == 1
    assert state.latest_user_message == message
    
    # Verify routing decision was made
    assert hasattr(state, "use_rule_based_path")
    assert hasattr(state, "routing_confidence")
    assert state.routing_confidence > 0
    
    # Verify parameters extracted (if rule-based path)
    if state.use_rule_based_path:
        assert state.origin is not None or state.destination is not None
    
    # Verify history
    assert len(state.conversation_history) >= 1
    assert state.conversation_history[0]["role"] == "user"
    assert state.conversation_history[0]["content"] == message
    
    print(f"✅ Single turn test passed")
    print(f"   - Path: {'rule-based' if state.use_rule_based_path else 'LLM'}")
    print(f"   - Confidence: {state.routing_confidence:.3f}")
    print(f"   - Origin: {state.origin}")
    print(f"   - Destination: {state.destination}")


@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """
    Test: Multi-turn conversation preserves state.
    
    Expected:
    - Turn 1: State created, slots partially filled
    - Turn 2: State loaded, more slots filled
    - Turn 3: State loaded, ready for search
    - Routing decisions independent per turn
    """
    session_id = "test_multi_002"
    user_id = 999
    
    # Turn 1: Initial query
    state1 = await run_conversation_turn(
        session_id=session_id,
        user_id=user_id,
        user_message="I want to book a flight"
    )
    
    assert state1.turn_count == 1
    assert len(state1.conversation_history) >= 1
    confidence1 = state1.routing_confidence
    
    print(f"\n📊 Turn 1:")
    print(f"   - Confidence: {confidence1:.3f}")
    print(f"   - Missing: {state1.missing_parameter}")
    
    # Small delay to ensure distinct timestamps
    await asyncio.sleep(0.1)
    
    # Turn 2: Provide destination
    state2 = await run_conversation_turn(
        session_id=session_id,
        user_id=user_id,
        user_message="To Paris"
    )
    
    assert state2.turn_count == 2  # Incremented
    assert len(state2.conversation_history) >= 2  # History preserved
    confidence2 = state2.routing_confidence
    
    # Verify routing decision was recalculated (not restored)
    # Confidence might be different between turns
    assert confidence2 > 0
    
    print(f"\n📊 Turn 2:")
    print(f"   - Confidence: {confidence2:.3f}")
    print(f"   - Destination: {state2.destination}")
    print(f"   - Missing: {state2.missing_parameter}")
    
    await asyncio.sleep(0.1)
    
    # Turn 3: Provide origin
    state3 = await run_conversation_turn(
        session_id=session_id,
        user_id=user_id,
        user_message="From London"
    )
    
    assert state3.turn_count == 3
    assert len(state3.conversation_history) >= 3
    confidence3 = state3.routing_confidence
    
    # Verify both slots preserved
    assert state3.destination is not None  # From turn 2
    assert state3.origin is not None  # From turn 3
    
    print(f"\n📊 Turn 3:")
    print(f"   - Confidence: {confidence3:.3f}")
    print(f"   - Origin: {state3.origin}")
    print(f"   - Destination: {state3.destination}")
    print(f"   - Missing: {state3.missing_parameter}")
    
    print(f"\n✅ Multi-turn test passed")
    print(f"   - History length: {len(state3.conversation_history)}")
    print(f"   - Slots preserved across turns: ✓")


@pytest.mark.asyncio
async def test_routing_independence():
    """
    Test: Routing decision is recalculated each turn, not restored.
    
    Expected:
    - Turn 1: Low confidence (vague query) → LLM path
    - Turn 2: High confidence (specific query) → Rule-based path
    - Proves routing is recalculated, not cached
    """
    session_id = "test_routing_003"
    user_id = 999
    
    # Turn 1: Vague query (should be low confidence)
    state1 = await run_conversation_turn(
        session_id=session_id,
        user_id=user_id,
        user_message="I want to travel"
    )
    
    confidence1 = state1.routing_confidence
    path1 = state1.use_rule_based_path
    
    print(f"\n📊 Turn 1 (vague):")
    print(f"   - Confidence: {confidence1:.3f}")
    print(f"   - Path: {'rule-based' if path1 else 'LLM'}")
    
    await asyncio.sleep(0.1)
    
    # Turn 2: Specific query (should be high confidence)
    state2 = await run_conversation_turn(
        session_id=session_id,
        user_id=user_id,
        user_message="LHR to CDG on 2025-12-25"
    )
    
    confidence2 = state2.routing_confidence
    path2 = state2.use_rule_based_path
    
    print(f"\n📊 Turn 2 (specific):")
    print(f"   - Confidence: {confidence2:.3f}")
    print(f"   - Path: {'rule-based' if path2 else 'LLM'}")
    
    # Verify routing was recalculated
    assert confidence2 != confidence1 or path2 != path1
    
    # Specific query should have higher confidence
    assert confidence2 > confidence1
    
    print(f"\n✅ Routing independence test passed")
    print(f"   - Confidence changed: {confidence1:.3f} → {confidence2:.3f}")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_single_turn_with_complete_info())
    asyncio.run(test_multi_turn_conversation())
    asyncio.run(test_routing_independence())