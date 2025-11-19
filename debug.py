# tests/comprehensive_test.py

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from app.langgraph_flow.graph import run_conversation_turn, get_graph_stats
from app.langgraph_flow.state import ConversationState

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# TEST RESULT TRACKING
# ============================================================================

class TestResult:
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.turns: List[Dict[str, Any]] = []
        self.total_time = 0.0
        self.passed = False
        self.error: Optional[str] = None
        
    def add_turn(self, turn_data: Dict[str, Any]):
        self.turns.append(turn_data)
        
    def summary(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "error": self.error,
            "total_time": round(self.total_time, 3),
            "turn_count": len(self.turns),
            "avg_turn_time": round(self.total_time / len(self.turns), 3) if self.turns else 0,
        }


# ============================================================================
# CONVERSATION TURN EXECUTOR WITH METRICS
# ============================================================================

async def execute_turn(
    session_id: str,
    user_id: int,
    user_message: str,
    turn_number: int
) -> Dict[str, Any]:
    """Execute a single conversation turn and collect detailed metrics."""
    
    start_time = time.time()
    
    try:
        # Execute the turn
        state = await run_conversation_turn(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Extract graph statistics
        stats = get_graph_stats(state)
        
        # Build comprehensive turn data
        turn_data = {
            "turn_number": turn_number,
            "user_message": user_message,
            "latency_seconds": round(latency, 3),
            
            # Routing information
            "path_used": stats.get("path_used"),
            "routing_confidence": stats.get("routing_confidence"),
            "intent": stats.get("intent"),
            "estimated_llm_calls": stats.get("estimated_llm_calls"),
            
            # State information
            "slots": {
                "origin": state.origin,
                "destination": state.destination,
                "departure_date": str(state.departure_date) if state.departure_date else None,
                "return_date": str(state.return_date) if state.return_date else None,
                "passengers": state.passengers,
                "travel_class": state.travel_class,
                "budget": state.budget,
            },
            
            # Flow control
            "ready_for_search": state.ready_for_search,
            "missing_parameter": state.missing_parameter,
            "search_executed": state.search_executed,
            
            # Response
            "assistant_message": state.assistant_message,
            "suggestions": state.suggestions or [],
            "ranked_flights": len(state.ranked_flights or []),
            
            # Errors/Warnings
            "errors": state.errors or [],
            "warnings": state.warnings or [],
        }
        
        return turn_data
        
    except Exception as e:
        logger.error(f"Turn {turn_number} failed: {e}", exc_info=True)
        return {
            "turn_number": turn_number,
            "user_message": user_message,
            "error": str(e),
            "latency_seconds": time.time() - start_time,
        }


# ============================================================================
# MULTI-TURN CONVERSATION TEST
# ============================================================================

async def test_multi_turn_conversation(
    test_name: str,
    session_id: str,
    user_id: int,
    messages: List[str],
    expected_outcomes: Dict[str, Any]
) -> TestResult:
    """
    Test a complete multi-turn conversation flow.
    
    Args:
        test_name: Name of the test
        session_id: Unique session ID for this test
        user_id: User ID
        messages: List of user messages to send in sequence
        expected_outcomes: Expected final state
    """
    
    result = TestResult(test_name)
    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}\n")
    
    try:
        for i, message in enumerate(messages, 1):
            print(f"\n--- Turn {i}/{len(messages)} ---")
            print(f"User: {message}")
            
            turn_data = await execute_turn(session_id, user_id, message, i)
            result.add_turn(turn_data)
            
            # Print turn summary
            if "error" in turn_data:
                print(f"❌ ERROR: {turn_data['error']}")
            else:
                print(f"⏱️  Latency: {turn_data['latency_seconds']}s")
                print(f"🔀 Path: {turn_data['path_used']} (conf: {turn_data.get('routing_confidence', 'N/A')})")
                print(f"🎯 Intent: {turn_data['intent']}")
                print(f"📦 Slots filled: {sum(1 for v in turn_data['slots'].values() if v)}/8")
                print(f"🤖 Assistant: {turn_data['assistant_message'][:100]}...")
                
                if turn_data['missing_parameter']:
                    print(f"❓ Asking for: {turn_data['missing_parameter']}")
                
                if turn_data['ready_for_search']:
                    print(f"✅ Ready to search!")
                    
                if turn_data['search_executed']:
                    print(f"🔍 Found {turn_data['ranked_flights']} flights")
            
            # Small delay to avoid rate limits
            await asyncio.sleep(2)
        
        # Validate final state
        final_turn = result.turns[-1] if result.turns else {}
        
        validation_passed = True
        validation_errors = []
        
        # Check expected outcomes
        if expected_outcomes.get("should_complete"):
            if not final_turn.get("ready_for_search") and not final_turn.get("search_executed"):
                validation_errors.append("Expected conversation to complete but it didn't")
                validation_passed = False
        
        if expected_outcomes.get("min_filled_slots"):
            filled_slots = sum(1 for v in final_turn.get("slots", {}).values() if v)
            if filled_slots < expected_outcomes["min_filled_slots"]:
                validation_errors.append(
                    f"Expected at least {expected_outcomes['min_filled_slots']} slots filled, got {filled_slots}"
                )
                validation_passed = False
        
        if expected_outcomes.get("should_find_flights"):
            if final_turn.get("ranked_flights", 0) == 0:
                validation_errors.append("Expected to find flights but none were found")
                validation_passed = False
        
        result.passed = validation_passed
        if validation_errors:
            result.error = "; ".join(validation_errors)
        
    except Exception as e:
        result.passed = False
        result.error = str(e)
        logger.error(f"Test {test_name} failed: {e}", exc_info=True)
    
    result.total_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TEST RESULT: {test_name}")
    print(f"{'='*80}")
    summary = result.summary()
    print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Total time: {summary['total_time']}s")
    print(f"Turns: {summary['turn_count']}")
    print(f"Avg turn time: {summary['avg_turn_time']}s")
    print(f"{'='*80}\n")
    
    return result


# ============================================================================
# SINGLE-TURN COMPLETE REQUEST TEST
# ============================================================================

async def test_single_turn_complete(
    test_name: str,
    user_message: str,
    expected_slots: Dict[str, Any]
) -> TestResult:
    """Test a single message that contains all required information."""
    
    result = TestResult(test_name)
    session_id = f"test_single_{int(time.time())}"
    
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        turn_data = await execute_turn(session_id, 999, user_message, 1)
        result.add_turn(turn_data)
        
        # Print detailed results
        print(f"User: {user_message}")
        print(f"\n⏱️  Latency: {turn_data['latency_seconds']}s")
        print(f"🔀 Path: {turn_data['path_used']} (confidence: {turn_data.get('routing_confidence', 'N/A')})")
        print(f"🎯 Intent: {turn_data['intent']}")
        
        print(f"\n📦 Extracted Slots:")
        for slot, value in turn_data['slots'].items():
            expected = expected_slots.get(slot)
            if expected:
                match = "✅" if value else "❌"
                print(f"  {match} {slot}: {value} (expected: {expected})")
            elif value:
                print(f"  ℹ️  {slot}: {value}")
        
        print(f"\n🤖 Assistant Response:")
        print(f"  {turn_data['assistant_message']}")
        
        if turn_data['suggestions']:
            print(f"\n💡 Suggestions: {', '.join(turn_data['suggestions'])}")
        
        # Validation
        validation_passed = True
        errors = []
        
        for slot, expected_value in expected_slots.items():
            actual_value = turn_data['slots'].get(slot)
            if not actual_value:
                errors.append(f"Missing expected slot: {slot}")
                validation_passed = False
        
        # Should NOT be asking questions if all info provided
        if turn_data.get('missing_parameter') and len(expected_slots) >= 3:
            errors.append(f"Shouldn't be asking for {turn_data['missing_parameter']} when all info was provided")
            validation_passed = False
        
        result.passed = validation_passed
        if errors:
            result.error = "; ".join(errors)
            
    except Exception as e:
        result.passed = False
        result.error = str(e)
        logger.error(f"Test failed: {e}", exc_info=True)
    
    result.total_time = time.time() - start_time
    
    # Print result
    print(f"\n{'='*80}")
    print(f"Result: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    if result.error:
        print(f"Errors: {result.error}")
    print(f"Time: {result.total_time:.3f}s")
    print(f"{'='*80}\n")
    
    return result


# ============================================================================
# TEST SUITE
# ============================================================================

async def run_comprehensive_tests():
    """Run all comprehensive tests."""
    
    results: List[TestResult] = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TRIP PLANNER TEST SUITE")
    print("="*80)
    
    # ========================================
    # TEST 1: Complete information in one message
    # ========================================
    result1 = await test_single_turn_complete(
        test_name="Single Turn - Complete Info",
        user_message="I want to trip to London with 2 friends to explore culture next week from Tashkent",
        expected_slots={
            "origin": "Tashkent",
            "destination": "London",
            "passengers": 3,  # User + 2 friends
        }
    )
    results.append(result1)
    
    await asyncio.sleep(5)  # Rate limit protection
    
    # ========================================
    # TEST 2: Clear multi-city trip
    # ========================================
    result2 = await test_single_turn_complete(
        test_name="Single Turn - City to City",
        user_message="Flight from New York to Tokyo, departing December 15th, 2 passengers, economy class",
        expected_slots={
            "origin": "New York",
            "destination": "Tokyo",
            "passengers": 2,
            "travel_class": "economy",
        }
    )
    results.append(result2)
    
    await asyncio.sleep(5)
    
    # ========================================
    # TEST 3: Multi-turn conversation
    # ========================================
    result3 = await test_multi_turn_conversation(
        test_name="Multi-Turn - Progressive Information",
        session_id=f"test_multi_{int(time.time())}",
        user_id=123,
        messages=[
            "I want to book a flight",
            "To Paris",
            "From London",
            "Next Monday",
            "2 passengers"
        ],
        expected_outcomes={
            "should_complete": True,
            "min_filled_slots": 4,
        }
    )
    results.append(result3)
    
    await asyncio.sleep(5)
    
    # ========================================
    # TEST 4: Vague query that needs clarification
    # ========================================
    result4 = await test_multi_turn_conversation(
        test_name="Multi-Turn - Vague Initial Query",
        session_id=f"test_vague_{int(time.time())}",
        user_id=124,
        messages=[
            "I want to travel somewhere nice",
            "Maybe Europe",
            "France sounds good",
            "From Berlin",
            "In two weeks"
        ],
        expected_outcomes={
            "min_filled_slots": 3,
        }
    )
    results.append(result4)
    
    await asyncio.sleep(5)
    
    # ========================================
    # TEST 5: High confidence rule-based path
    # ========================================
    result5 = await test_single_turn_complete(
        test_name="Rule-Based Path - High Confidence",
        user_message="Tashkent to Istanbul, December 25th, 2 passengers, business class",
        expected_slots={
            "origin": "Tashkent",
            "destination": "Istanbul",
            "passengers": 2,
            "travel_class": "business",
        }
    )
    results.append(result5)
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Pass Rate: {(passed/len(results)*100):.1f}%\n")
    
    total_time = sum(r.total_time for r in results)
    total_turns = sum(len(r.turns) for r in results)
    
    print(f"Total Execution Time: {total_time:.2f}s")
    print(f"Total Conversation Turns: {total_turns}")
    print(f"Average Turn Time: {(total_time/total_turns):.3f}s\n")
    
    print("Detailed Results:")
    print("-" * 80)
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        print(f"{status} | {r.test_name:45} | {r.total_time:.2f}s | {len(r.turns)} turns")
        if r.error:
            print(f"     └─ Error: {r.error}")
    
    print("="*80 + "\n")
    
    # Save detailed results to JSON
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed/len(results)*100, 1),
            "total_time": round(total_time, 2),
            "total_turns": total_turns,
            "avg_turn_time": round(total_time/total_turns, 3),
        },
        "tests": [
            {
                "name": r.test_name,
                "passed": r.passed,
                "error": r.error,
                "time": r.total_time,
                "turns": r.turns,
            }
            for r in results
        ]
    }
    
    with open("test_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print("📄 Detailed results saved to: test_results.json\n")
    
    return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())