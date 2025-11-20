# """
# Comprehensive Test Suite - Production Version
# =============================================
# Tests with proper error handling and defensive extraction.
# """

# import asyncio
# import logging
# import time
# from typing import Dict, Any, List, Optional
# from datetime import datetime
# import json

# from app.langgraph_flow.graph import run_conversation_turn, get_graph_stats
# from app.langgraph_flow.state import ConversationState

# # Configure detailed logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
# )

# logger = logging.getLogger(__name__)


# # ============================================================================
# # TEST RESULT TRACKING
# # ============================================================================

# class TestResult:
#     def __init__(self, test_name: str):
#         self.test_name = test_name
#         self.turns: List[Dict[str, Any]] = []
#         self.total_time = 0.0
#         self.passed = False
#         self.error: Optional[str] = None
        
#     def add_turn(self, turn_data: Dict[str, Any]):
#         self.turns.append(turn_data)
        
#     def summary(self) -> Dict[str, Any]:
#         return {
#             "test_name": self.test_name,
#             "passed": self.passed,
#             "error": self.error,
#             "total_time": round(self.total_time, 3),
#             "turn_count": len(self.turns),
#             "avg_turn_time": round(self.total_time / len(self.turns), 3) if self.turns else 0,
#         }


# # ============================================================================
# # CONVERSATION TURN EXECUTOR (FIXED - DEFENSIVE EXTRACTION)
# # ============================================================================

# async def execute_turn(
#     session_id: str,
#     user_id: Optional[int],  # ✅ Now accepts None
#     user_message: str,
#     turn_number: int
# ) -> Dict[str, Any]:
#     """
#     Execute a single conversation turn with defensive error handling.
    
#     ✅ FIXED: Handles None user_id
#     ✅ FIXED: Defensive extraction with .get() fallbacks
#     ✅ FIXED: Error-safe return structure
#     """
    
#     start_time = time.time()
    
#     try:
#         # Execute the turn
#         state = await run_conversation_turn(
#             session_id=session_id,
#             user_id=user_id,  # ✅ Can be None now
#             user_message=user_message,
#         )
        
#         end_time = time.time()
#         latency = end_time - start_time
        
#         # Extract graph statistics (defensive)
#         try:
#             stats = get_graph_stats(state)
#         except Exception as e:
#             logger.warning(f"[ExecuteTurn] Failed to get graph stats: {e}")
#             stats = {}
        
#         # ✅ DEFENSIVE: Use .get() with fallbacks
#         turn_data = {
#             "turn_number": turn_number,
#             "user_message": user_message,
#             "latency_seconds": round(latency, 3),
            
#             # Routing information (with fallbacks)
#             "path_used": stats.get("path_used", "unknown"),
#             "routing_confidence": stats.get("routing_confidence", 0.0),
#             "intent": stats.get("intent", "unknown"),
#             "estimated_llm_calls": stats.get("estimated_llm_calls", 0),
            
#             # State information (defensive access)
#             "slots": {
#                 "origin": getattr(state, "origin", None),
#                 "destination": getattr(state, "destination", None),
#                 "departure_date": str(getattr(state, "departure_date", None)) if getattr(state, "departure_date", None) else None,
#                 "return_date": str(getattr(state, "return_date", None)) if getattr(state, "return_date", None) else None,
#                 "passengers": getattr(state, "passengers", 1),
#                 "travel_class": getattr(state, "travel_class", None),
#                 "budget": getattr(state, "budget", None),
#             },
            
#             # Flow control
#             "ready_for_search": getattr(state, "ready_for_search", False),
#             "missing_parameter": getattr(state, "missing_parameter", None),
#             "search_executed": getattr(state, "search_executed", False),
            
#             # Response
#             "assistant_message": getattr(state, "assistant_message", ""),
#             "suggestions": getattr(state, "suggestions", []) or [],
#             "ranked_flights": len(getattr(state, "ranked_flights", []) or []),
            
#             # Errors/Warnings
#             "errors": getattr(state, "errors", []) or [],
#             "warnings": getattr(state, "warnings", []) or [],
#         }
        
#         return turn_data
        
#     except Exception as e:
#         logger.error(f"Turn {turn_number} failed: {e}", exc_info=True)
        
#         # ✅ ERROR-SAFE: Return structured error response
#         return {
#             "turn_number": turn_number,
#             "user_message": user_message,
#             "error": str(e),
#             "latency_seconds": time.time() - start_time,
#             "path_used": "error",
#             "routing_confidence": 0.0,
#             "intent": "error",
#             "estimated_llm_calls": 0,
#             "slots": {},
#             "ready_for_search": False,
#             "missing_parameter": None,
#             "search_executed": False,
#             "assistant_message": "",
#             "suggestions": [],
#             "ranked_flights": 0,
#             "errors": [str(e)],
#             "warnings": [],
#         }


# # ============================================================================
# # SINGLE-TURN COMPLETE REQUEST TEST (FIXED)
# # ============================================================================

# async def test_single_turn_complete(
#     test_name: str,
#     user_message: str,
#     expected_slots: Dict[str, Any]
# ) -> TestResult:
#     """
#     Test a single message that contains all required information.
    
#     ✅ FIXED: Uses None for user_id (no DB foreign key issues)
#     """
    
#     result = TestResult(test_name)
#     session_id = f"test_single_{int(time.time())}"
    
#     print(f"\n{'='*80}")
#     print(f"TEST: {test_name}")
#     print(f"{'='*80}\n")
    
#     start_time = time.time()
    
#     try:
#         # ✅ FIXED: Use None for anonymous testing
#         turn_data = await execute_turn(session_id, None, user_message, 1)
#         result.add_turn(turn_data)
        
#         # Print detailed results
#         print(f"User: {user_message}")
#         print(f"\n⏱️  Latency: {turn_data['latency_seconds']}s")
#         print(f"🔀 Path: {turn_data['path_used']} (confidence: {turn_data.get('routing_confidence', 'N/A')})")
#         print(f"🎯 Intent: {turn_data['intent']}")
        
#         print(f"\n📦 Extracted Slots:")
#         for slot, value in turn_data['slots'].items():
#             expected = expected_slots.get(slot)
#             if expected:
#                 match = "✅" if value else "❌"
#                 print(f"  {match} {slot}: {value} (expected: {expected})")
#             elif value:
#                 print(f"  ℹ️  {slot}: {value}")
        
#         print(f"\n🤖 Assistant Response:")
#         print(f"  {turn_data['assistant_message'][:200]}...")
        
#         if turn_data['suggestions']:
#             print(f"\n💡 Suggestions: {', '.join(turn_data['suggestions'][:3])}")
        
#         # Validation
#         validation_passed = True
#         errors = []
        
#         # Check for execution errors
#         if "error" in turn_data and turn_data["error"]:
#             errors.append(f"Execution error: {turn_data['error']}")
#             validation_passed = False
        
#         # Check expected slots
#         for slot, expected_value in expected_slots.items():
#             actual_value = turn_data['slots'].get(slot)
#             if not actual_value:
#                 errors.append(f"Missing expected slot: {slot}")
#                 validation_passed = False
        
#         result.passed = validation_passed
#         if errors:
#             result.error = "; ".join(errors)
            
#     except Exception as e:
#         result.passed = False
#         result.error = str(e)
#         logger.error(f"Test failed: {e}", exc_info=True)
    
#     result.total_time = time.time() - start_time
    
#     # Print result
#     print(f"\n{'='*80}")
#     print(f"Result: {'✅ PASSED' if result.passed else '❌ FAILED'}")
#     if result.error:
#         print(f"Errors: {result.error}")
#     print(f"Time: {result.total_time:.3f}s")
#     print(f"{'='*80}\n")
    
#     return result


# # ============================================================================
# # MULTI-TURN CONVERSATION TEST (FIXED)
# # ============================================================================

# async def test_multi_turn_conversation(
#     test_name: str,
#     session_id: str,
#     user_id: Optional[int],  # ✅ Now accepts None
#     messages: List[str],
#     expected_outcomes: Dict[str, Any]
# ) -> TestResult:
#     """
#     Test a complete multi-turn conversation flow.
    
#     ✅ FIXED: Accepts None for user_id
#     """
    
#     result = TestResult(test_name)
#     start_time = time.time()
    
#     print(f"\n{'='*80}")
#     print(f"TEST: {test_name}")
#     print(f"{'='*80}\n")
    
#     try:
#         for i, message in enumerate(messages, 1):
#             print(f"\n--- Turn {i}/{len(messages)} ---")
#             print(f"User: {message}")
            
#             turn_data = await execute_turn(session_id, user_id, message, i)
#             result.add_turn(turn_data)
            
#             # Print turn summary
#             if turn_data.get("error"):
#                 print(f"❌ ERROR: {turn_data['error']}")
#             else:
#                 print(f"⏱️  Latency: {turn_data['latency_seconds']}s")
#                 print(f"🔀 Path: {turn_data['path_used']} (conf: {turn_data.get('routing_confidence', 'N/A')})")
#                 print(f"🎯 Intent: {turn_data['intent']}")
#                 print(f"📦 Slots filled: {sum(1 for v in turn_data['slots'].values() if v)}/7")
#                 print(f"🤖 Assistant: {turn_data['assistant_message'][:100]}...")
                
#                 if turn_data['missing_parameter']:
#                     print(f"❓ Asking for: {turn_data['missing_parameter']}")
                
#                 if turn_data['ready_for_search']:
#                     print(f"✅ Ready to search!")
                    
#                 if turn_data['search_executed']:
#                     print(f"🔍 Found {turn_data['ranked_flights']} flights")
            
#             # Small delay to avoid rate limits
#             await asyncio.sleep(2)
        
#         # Validate final state
#         final_turn = result.turns[-1] if result.turns else {}
        
#         validation_passed = True
#         validation_errors = []
        
#         # Check expected outcomes
#         if expected_outcomes.get("should_complete"):
#             if not final_turn.get("ready_for_search") and not final_turn.get("search_executed"):
#                 validation_errors.append("Expected conversation to complete but it didn't")
#                 validation_passed = False
        
#         if expected_outcomes.get("min_filled_slots"):
#             filled_slots = sum(1 for v in final_turn.get("slots", {}).values() if v)
#             if filled_slots < expected_outcomes["min_filled_slots"]:
#                 validation_errors.append(
#                     f"Expected at least {expected_outcomes['min_filled_slots']} slots filled, got {filled_slots}"
#                 )
#                 validation_passed = False
        
#         result.passed = validation_passed
#         if validation_errors:
#             result.error = "; ".join(validation_errors)
        
#     except Exception as e:
#         result.passed = False
#         result.error = str(e)
#         logger.error(f"Test {test_name} failed: {e}", exc_info=True)
    
#     result.total_time = time.time() - start_time
    
#     # Print summary
#     print(f"\n{'='*80}")
#     print(f"TEST RESULT: {test_name}")
#     print(f"{'='*80}")
#     summary = result.summary()
#     print(f"Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
#     if result.error:
#         print(f"Error: {result.error}")
#     print(f"Total time: {summary['total_time']}s")
#     print(f"Turns: {summary['turn_count']}")
#     print(f"Avg turn time: {summary['avg_turn_time']}s")
#     print(f"{'='*80}\n")
    
#     return result


# # ============================================================================
# # TEST SUITE (FIXED)
# # ============================================================================

# async def run_comprehensive_tests():
#     """Run all comprehensive tests with fixed user_id handling."""
    
#     results: List[TestResult] = []
    
#     print("\n" + "="*80)
#     print("COMPREHENSIVE TRIP PLANNER TEST SUITE")
#     print("="*80)
    
#     # ========================================
#     # TEST 1: Complete information in one message
#     # ========================================
#     result1 = await test_single_turn_complete(
#         test_name="Single Turn - Complete Info",
#         user_message="I want to fly to London with 2 friends next week from Tashkent",
#         expected_slots={
#             "origin": "Tashkent",
#             "destination": "London",
#             "passengers": 3,
#         }
#     )
#     results.append(result1)
    
#     await asyncio.sleep(3)
    
#     # ========================================
#     # TEST 2: Multi-turn conversation (✅ FIXED: user_id=None)
#     # ========================================
#     result2 = await test_multi_turn_conversation(
#         test_name="Multi-Turn - Progressive Information",
#         session_id=f"test_multi_{int(time.time())}",
#         user_id=None,  # ✅ FIXED: Use None instead of 123
#         messages=[
#             "I want to book a flight",
#             "To Istanbul",
#             "From London",
#             "December 25th"
#         ],
#         expected_outcomes={
#             "min_filled_slots": 3,
#         }
#     )
#     results.append(result2)
    
#     # ========================================
#     # FINAL SUMMARY
#     # ========================================
#     print("\n" + "="*80)
#     print("FINAL TEST SUMMARY")
#     print("="*80)
    
#     passed = sum(1 for r in results if r.passed)
#     failed = len(results) - passed
    
#     print(f"\nTotal Tests: {len(results)}")
#     print(f"✅ Passed: {passed}")
#     print(f"❌ Failed: {failed}")
    
#     if results:
#         print(f"Pass Rate: {(passed/len(results)*100):.1f}%\n")
    
#     total_time = sum(r.total_time for r in results)
#     total_turns = sum(len(r.turns) for r in results)
    
#     print(f"Total Execution Time: {total_time:.2f}s")
#     print(f"Total Conversation Turns: {total_turns}")
#     if total_turns > 0:
#         print(f"Average Turn Time: {(total_time/total_turns):.3f}s\n")
    
#     print("Detailed Results:")
#     print("-" * 80)
#     for r in results:
#         status = "✅ PASS" if r.passed else "❌ FAIL"
#         print(f"{status} | {r.test_name:45} | {r.total_time:.2f}s | {len(r.turns)} turns")
#         if r.error:
#             print(f"     └─ Error: {r.error}")
    
#     print("="*80 + "\n")
    
#     return results


# # ============================================================================
# # MAIN ENTRY POINT
# # ============================================================================

# if __name__ == "__main__":
#     asyncio.run(run_comprehensive_tests())






"""
Complete Conversation Flow Test
Shows state at every step to diagnose bugs
"""

import asyncio
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


async def test_two_turn_conversation():
    """
    Test the exact scenario that's failing:
    Turn 1: "I want to fly to Istanbul next week"
    Turn 2: "Tashkent"
    """
    
    from app.langgraph_flow.graph import run_conversation_turn
    
    print("\n" + "="*80)
    print("TEST: Two-Turn Conversation Flow")
    print("="*80)
    
    session_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # ========================================
    # TURN 1: Initial query
    # ========================================
    print("\n" + "-"*80)
    print("TURN 1: User says 'I want to fly to Istanbul next week'")
    print("-"*80)
    
    state_1 = await run_conversation_turn(
        session_id=session_id,
        user_id=999,
        user_message="I want to fly to Istanbul next week"
    )
    
    print("\n📊 STATE AFTER TURN 1:")
    print(f"  destination: {state_1.destination}")
    print(f"  destination_airports: {state_1.destination_airports}")
    print(f"  origin: {state_1.origin}")
    print(f"  origin_airports: {state_1.origin_airports}")
    print(f"  departure_date: {state_1.departure_date}")
    print(f"  missing_parameter: {state_1.missing_parameter}")
    print(f"  ready_for_search: {state_1.ready_for_search}")
    
    print(f"\n🤖 ASSISTANT RESPONSE:")
    print(f"  {state_1.assistant_message}")
    
    print(f"\n💡 SUGGESTIONS:")
    for s in (state_1.suggestions or [])[:3]:
        print(f"  - {s}")
    
    # Validate Turn 1
    assert state_1.destination in ["IST", "Istanbul"], f"❌ Destination should be Istanbul, got: {state_1.destination}"
    assert state_1.origin is None or state_1.origin == "", f"❌ Origin should be empty, got: {state_1.origin}"
    assert state_1.missing_parameter == "origin", f"❌ Should be asking for origin, got: {state_1.missing_parameter}"
    
    print("\n✅ Turn 1 validation passed")
    
    # ========================================
    # TURN 2: Answer the question
    # ========================================
    print("\n" + "-"*80)
    print("TURN 2: User says 'Tashkent' (answering the origin question)")
    print("-"*80)
    
    # Add a small delay to ensure state is persisted
    await asyncio.sleep(1)
    
    state_2 = await run_conversation_turn(
        session_id=session_id,
        user_id=999,
        user_message="Tashkent"
    )
    
    print("\n📊 STATE AFTER TURN 2:")
    print(f"  destination: {state_2.destination}")
    print(f"  destination_airports: {state_2.destination_airports}")
    print(f"  origin: {state_2.origin}")
    print(f"  origin_airports: {state_2.origin_airports}")
    print(f"  departure_date: {state_2.departure_date}")
    print(f"  missing_parameter: {state_2.missing_parameter}")
    print(f"  ready_for_search: {state_2.ready_for_search}")
    
    print(f"\n🤖 ASSISTANT RESPONSE:")
    print(f"  {state_2.assistant_message}")
    
    if state_2.ranked_flights:
        print(f"\n✈️ FOUND {len(state_2.ranked_flights)} FLIGHTS")
    
    # ========================================
    # VALIDATION
    # ========================================
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    errors = []
    
    # Check destination wasn't overwritten
    if state_2.destination not in ["IST", "Istanbul"]:
        errors.append(f"❌ FAIL: Destination was overwritten! Expected IST/Istanbul, got: {state_2.destination}")
    else:
        print(f"✅ PASS: Destination preserved ({state_2.destination})")
    
    # Check origin was set
    if not state_2.origin or state_2.origin not in ["TAS", "Tashkent"]:
        errors.append(f"❌ FAIL: Origin not set correctly! Expected TAS/Tashkent, got: {state_2.origin}")
    else:
        print(f"✅ PASS: Origin set correctly ({state_2.origin})")
    
    # Check they're not the same
    origin_code = state_2.origin_airports[0] if state_2.origin_airports else state_2.origin
    dest_code = state_2.destination_airports[0] if state_2.destination_airports else state_2.destination
    
    if origin_code and dest_code and origin_code.upper() == dest_code.upper():
        errors.append(f"❌ FAIL: Origin and destination are the same! ({origin_code})")
    else:
        print(f"✅ PASS: Origin ≠ Destination")
    
    # Check missing_parameter was cleared or set correctly
    if state_2.ready_for_search:
        if state_2.missing_parameter is not None:
            errors.append(f"❌ FAIL: Ready for search but missing_parameter not cleared! ({state_2.missing_parameter})")
        else:
            print(f"✅ PASS: Ready for search, missing_parameter cleared")
    else:
        if state_2.missing_parameter == "origin":
            errors.append(f"❌ FAIL: Still asking for origin after user provided it!")
        else:
            print(f"✅ PASS: Correctly asking for next parameter ({state_2.missing_parameter})")
    
    # ========================================
    # FINAL REPORT
    # ========================================
    print("\n" + "="*80)
    if errors:
        print("❌ TEST FAILED")
        print("="*80)
        for error in errors:
            print(error)
        return False
    else:
        print("✅ TEST PASSED - All validations successful!")
        print("="*80)
        return True


async def test_with_detailed_node_logging():
    """
    Same test but with manual node execution to see intermediate states
    """
    
    print("\n" + "="*80)
    print("DETAILED NODE-BY-NODE EXECUTION")
    print("="*80)
    
    from app.langgraph_flow.state import create_initial_state, update_state
    from app.langgraph_flow.nodes.extract_parameters_node import extract_parameters_node
    from app.langgraph_flow.nodes.determine_missing_slot_node import determine_missing_slot_node
    from app.langgraph_flow.nodes.rule_based_question_node import rule_based_question_node
    
    # ========================================
    # Simulate Turn 2 (after Turn 1 already set destination)
    # ========================================
    print("\n📝 SIMULATING STATE AFTER TURN 1:")
    
    state = create_initial_state(
        session_id="test_manual",
        user_id=999,
        latest_message="Tashkent"
    )
    
    # Manually set state as it would be after Turn 1
    state = update_state(state, {
        "destination": "IST",
        "destination_airports": ["IST"],
        "origin": None,
        "origin_airports": [],
        "departure_date": "2025-12-01",
        "missing_parameter": "origin",
        "use_rule_based_path": False,  # Using LLM path for this test
    })
    
    print(f"  destination: {state.destination}")
    print(f"  origin: {state.origin}")
    print(f"  missing_parameter: {state.missing_parameter}")
    
    # ========================================
    # Node 1: Extract Parameters
    # ========================================
    print("\n🔧 NODE 1: extract_parameters_node")
    print("-"*80)
    
    state = await extract_parameters_node(state)
    
    print(f"AFTER EXTRACTION:")
    print(f"  destination: {state.destination}")
    print(f"  origin: {state.origin}")
    print(f"  latest_user_message: {state.latest_user_message}")
    
    if state.destination == state.origin:
        print(f"  ⚠️  WARNING: Destination == Origin ({state.destination})")
    
    # ========================================
    # Node 2: Determine Missing Slot
    # ========================================
    print("\n🔧 NODE 2: determine_missing_slot_node")
    print("-"*80)
    
    state = await determine_missing_slot_node(state)
    
    print(f"AFTER SLOT CHECK:")
    print(f"  destination: {state.destination}")
    print(f"  origin: {state.origin}")
    print(f"  missing_parameter: {state.missing_parameter}")
    print(f"  ready_for_search: {state.ready_for_search}")
    
    # ========================================
    # Node 3: Generate Question (if needed)
    # ========================================
    if state.missing_parameter and not state.ready_for_search:
        print("\n🔧 NODE 3: rule_based_question_node")
        print("-"*80)
        
        state = await rule_based_question_node(state)
        
        print(f"GENERATED QUESTION:")
        print(f"  {state.assistant_message}")
    
    # ========================================
    # Final Check
    # ========================================
    print("\n" + "="*80)
    print("FINAL STATE CHECK")
    print("="*80)
    
    if state.destination == "IST" and state.origin in ["TAS", "Tashkent"]:
        print("✅ PASS: State is correct!")
        return True
    else:
        print(f"❌ FAIL: State is corrupted!")
        print(f"   Expected: destination=IST, origin=TAS")
        print(f"   Got: destination={state.destination}, origin={state.origin}")
        return False


async def run_all_tests():
    """Run all test scenarios"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE CONVERSATION FLOW TESTING")
    print("="*80)
    
    results = []
    
    # Test 1: Full conversation flow
    try:
        result_1 = await test_two_turn_conversation()
        results.append(("Full Flow Test", result_1))
    except Exception as e:
        print(f"\n❌ Full flow test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Full Flow Test", False))
    
    await asyncio.sleep(2)
    
    # Test 2: Detailed node execution
    try:
        result_2 = await test_with_detailed_node_logging()
        results.append(("Node-by-Node Test", result_2))
    except Exception as e:
        print(f"\n❌ Node test crashed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Node-by-Node Test", False))
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Check logs above for details")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(run_all_tests())