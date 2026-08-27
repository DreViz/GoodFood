#!/usr/bin/env python
"""
End-to-end smoke test for the agent on the local model.

Drives one scripted search -> availability -> booking conversation through
`process_user_query` and asserts, turn by turn, that the planner produces valid
decisions (not the mock fallback), the right tools fire, and a real reservation
row lands in the database.

Usage:
    python -m scripts.smoke_test                 # uses OLLAMA_MODEL / default
    python -m scripts.smoke_test --model qwen3:4b
    python -m scripts.smoke_test --model qwen3:8b

Exits 0 on PASS, 1 on FAIL. Cleans up its own test reservation so it stays
re-runnable.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

TEST_EMAIL = "smoke@test.com"
TEST_RESTAURANT = "GoodFoods Bistro"   # Italian, zone East, seeded location_id 30
TEST_PARTY = 2
MOCK_FALLBACK_MARKER = "GoodFoods Concierge. How can I help"

SEARCH_ACTIONS = {"search_restaurants_by_filters", "recommend_venues"}


def _c(code, text):
    return f"\033[{code}m{text}\033[0m"


def green(t): return _c("92", t)
def red(t): return _c("91", t)
def yellow(t): return _c("93", t)
def dim(t): return _c("2", t)


def pick_available_slot(get_available_slots, location_id):
    """Find the first future date (within a week) that has an available slot."""
    for offset in range(1, 8):
        date = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
        data = get_available_slots(location_id, date, TEST_PARTY)
        slots = (data or {}).get("available_slots") or []
        if slots:
            return date, slots[0]["time"]
    return None, None


def cleanup(SessionLocal, Reservation, Customer, date):
    """Remove any prior smoke-test reservation so the run is repeatable."""
    session = SessionLocal()
    try:
        cust = session.query(Customer).filter_by(email=TEST_EMAIL).first()
        if cust:
            (session.query(Reservation)
                    .filter_by(customer_id=cust.id)
                    .delete(synchronize_session=False))
            session.commit()
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        session.rollback()
        print(dim(f"  (cleanup warning: {exc})"))
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="GoodFoods Phase 1 smoke test")
    parser.add_argument("--model", default=None,
                        help="Override OLLAMA_MODEL for this run (e.g. qwen3:4b, qwen3:8b)")
    args = parser.parse_args()

    # Set the model override BEFORE importing anything that reads settings,
    # so the cached get_settings() picks it up.
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    from app.config import get_settings
    settings = get_settings()

    from app.agent import agent as agent_module
    from app.agent.planner_agent import memory
    from app.api.utils.slot_manager import get_available_slots
    from app.data.db_connection import SessionLocal
    from app.data.db_models import Reservation, Customer

    print("=" * 68)
    print(f" GoodFoods smoke test  |  model={settings.ollama_model}  think={settings.ollama_think}")
    print(f" endpoint={settings.ollama_generate_url}")
    print("=" * 68)

    # Find a genuinely available slot for the target restaurant.
    date, slot = pick_available_slot(get_available_slots, 30)
    if not slot:
        print(red("FAIL: no available slot found for the test restaurant in the next 7 days."))
        return 1
    print(f" Target: {TEST_RESTAURANT}  date={date}  slot={slot}  party={TEST_PARTY}\n")

    cleanup(SessionLocal, Reservation, Customer, date)

    memory.reset()
    agent_module.recent_results.clear()

    turns = [
        "Any Italian restaurants in the East zone?",
        f"Let's go with {TEST_RESTAURANT}.",
        f"I'd like a table for {TEST_PARTY} on {date} at {slot}.",
        f"My email is {TEST_EMAIL}.",
        "Yes, please confirm the reservation.",
    ]

    saw_search = False
    saw_availability = False
    booking_result = None
    mock_hits = 0

    for i, user_text in enumerate(turns, 1):
        t0 = time.time()
        result = agent_module.process_user_query(user_text)
        dt = time.time() - t0

        reply = (result.get("reply") or "").strip()
        tool_output = result.get("tool_output") or {}
        action = tool_output.get("action")
        tool_result = tool_output.get("result") or {}

        # Detect planner falling back to the mock greeting.
        is_mock = MOCK_FALLBACK_MARKER in reply and action is None
        if is_mock:
            mock_hits += 1

        if action in SEARCH_ACTIONS:
            saw_search = True
        if action == "check_availability":
            saw_availability = True
        if action == "create_reservation":
            booking_result = tool_result

        label = action if action else "reply"
        tag = red("MOCK") if is_mock else green(label)
        print(f" Turn {i} [{dt:5.1f}s] {tag}")
        print(dim(f"   user : {user_text}"))
        print(dim(f"   reply: {reply[:120]}"))
        if action:
            print(dim(f"   args : {tool_output.get('args')}"))
            if isinstance(tool_result, dict) and "ok" in tool_result:
                print(dim(f"   ok   : {tool_result.get('ok')}  {tool_result.get('error','')}"))
        print()

    session = SessionLocal()
    try:
        cust = session.query(Customer).filter_by(email=TEST_EMAIL).first()
        db_rows = (session.query(Reservation).filter_by(customer_id=cust.id).count()
                   if cust else 0)
    finally:
        session.close()

    print("-" * 68)
    booked_ok = bool(booking_result and booking_result.get("ok"))
    checks = [
        ("planner produced valid decisions (no mock fallback)", mock_hits == 0),
        ("search tool fired in discovery", saw_search),
        ("create_reservation returned ok:True", booked_ok),
        ("reservation row present in DB", db_rows > 0),
    ]
    for name, ok in checks:
        print(f"  [{green('PASS') if ok else red('FAIL')}] {name}")

    # availability is informational — some flows skip straight to booking.
    print(f"  [{green('seen') if saw_availability else yellow('skip')}] check_availability observed (informational)")

    passed = all(ok for _, ok in checks)
    print("-" * 68)
    print((green(" SMOKE TEST PASSED") if passed else red(" SMOKE TEST FAILED")))

    cleanup(SessionLocal, Reservation, Customer, date)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
