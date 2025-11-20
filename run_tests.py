"""
Refactored Test Runner - All tests from /tests + results.txt logging
====================================================================
All pytest runs are executed from tests/ and results are logged.
"""

import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

RESULTS_FILE = "results.txt"


def write_to_results(text: str):
    """Append a line to results.txt."""
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def run_pytest(pytest_args: list, description: str = ""):
    """Run pytest with given args, print output, and save to results.txt."""

    header = f"\n{'='*70}\n  {description}\n{'='*70}\n"
    print(header)
    write_to_results(header)

    cmd = ["pytest", "tests", "-v", "--tb=short"] + pytest_args

    # Capture output so we can log it
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Output to console
    print(result.stdout)
    print(result.stderr)

    # Save to results.txt
    write_to_results(result.stdout)
    write_to_results(result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run AI Trip Planner tests")

    # Test suite selections
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true")
    parser.add_argument("--integration", action="store_true")
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--mad-libs", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--quick", action="store_true")

    # Options
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("-k", "--keyword")
    parser.add_argument("--failfast", action="store_true")

    args = parser.parse_args()

    # Default to all if nothing chosen
    if not any([
        args.all, args.unit, args.integration, args.e2e, args.api,
        args.mad_libs, args.performance, args.quick
    ]):
        args.all = True

    # Clear old results
    open(RESULTS_FILE, "w").close()

    results = []

    # ===========================================
    # HEADER
    # ===========================================
    header = (
        "\n" + "="*70 + "\n"
        " AI TRIP PLANNER - COMPREHENSIVE TEST SUITE\n"
        + "="*70 + "\n"
        f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "="*70
    )

    print(header)
    write_to_results(header)

    # Build common pytest args
    base_pytest_args = []

    if args.coverage:
        base_pytest_args += ["--cov=app", "--cov-report=term-missing"]
        if args.html:
            base_pytest_args.append("--cov-report=html")

    if args.parallel:
        base_pytest_args += ["-n", "auto"]

    if args.keyword:
        base_pytest_args += ["-k", args.keyword]

    if args.failfast:
        base_pytest_args += ["--maxfail=1"]

    # ===========================================
    # QUICK TESTS
    # ===========================================
    if args.quick:
        quick_targets = [
            "test_destination_intent",
            "test_extract_destination",
            "test_send_message_creates_session"
        ]

        success = run_pytest(
            ["-k", " or ".join(quick_targets)] + base_pytest_args,
            "🔥 Quick Smoke Tests (3 essential tests)"
        )
        results.append(("Quick Smoke Tests", success))

    # ===========================================
    # UNIT TESTS
    # ===========================================
    if args.unit or args.all:
        success = run_pytest(
            ["-k", "unit"] + base_pytest_args,
            "🧪 Unit Tests - Core Components"
        )
        results.append(("Unit Tests", success))

    # ===========================================
    # INTEGRATION TESTS
    # ===========================================
    if args.integration or args.all:
        success = run_pytest(
            ["-k", "integration"] + base_pytest_args,
            "🔗 Integration Tests - Node Interactions"
        )
        results.append(("Integration Tests", success))

    # ===========================================
    # MAD LIBS UI TESTS
    # ===========================================
    if args.mad_libs or args.e2e or args.all:
        ml_path = Path("tests/e2e/test_mad_libs_ui.py")
        if ml_path.exists():
            success = run_pytest(
                [str(ml_path)] + base_pytest_args,
                "⭐ Mad Libs UI Tests - Progressive Slot Filling"
            )
        else:
            success = False
        results.append(("Mad Libs UI Tests", success))

    # ===========================================
    # E2E TESTS
    # ===========================================
    if args.e2e or args.all:
        success = run_pytest(
            ["-k", "e2e"] + base_pytest_args,
            "🎯 End-to-End Tests"
        )
        results.append(("E2E Tests", success))

    # ===========================================
    # API TESTS
    # ===========================================
    if args.api or args.all:
        success = run_pytest(
            ["-k", "api"] + base_pytest_args,
            "🌐 API Tests - FastAPI Endpoints"
        )
        results.append(("API Tests", success))

    # ===========================================
    # PERFORMANCE TESTS
    # ===========================================
    if args.performance or args.all:
        success = run_pytest(
            ["-k", "performance", "--durations=10"] + base_pytest_args,
            "⚡ Performance Tests"
        )
        results.append(("Performance Tests", success))

    # ===========================================
    # SUMMARY
    # ===========================================
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed

    summary_lines = [
        "\n" + "="*70,
        " TEST EXECUTION SUMMARY",
        "="*70,
    ]

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        summary_lines.append(f"  {status}  {name}")

    summary_lines += [
        "="*70,
        f"  Total: {len(results)} suites",
        f"  Passed: {passed}",
        f"  Failed: {failed}",
        f"  Pass Rate: {(passed/len(results)*100):.1f}%",
        "="*70,
        f" End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*70
    ]

    # Print + save summary
    print("\n".join(summary_lines))
    for line in summary_lines:
        write_to_results(line)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
