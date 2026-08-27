"""Schema guard for tests/eval/heldout_conversations.yaml.

The held-out set is run on a different machine, hours into a fine-tuning
session, so a typo there costs a whole eval cycle. Validates schema, tool
names, restaurant names, coverage, and independence from the 45-conversation
benchmark. Runs offline — no DB, no model, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HELDOUT = REPO_ROOT / "tests" / "eval" / "heldout_conversations.yaml"
BASELINE = REPO_ROOT / "tests" / "eval" / "conversations.yaml"
SEED_DATA = REPO_ROOT / "app" / "data" / "goodfoods_locations_unique_50.json"

# scorer.score_outcome's dispatch table.
VALID_OUTCOMES = {
    "no_action",
    "phase_reached_availability",
    "phase_reached_booking",
    "booking_created",
    "booking_cancelled",
    "booking_modified",
    "error_presented",
}

# evaluate.py --categories values.
VALID_CATEGORIES = {"search", "availability", "booking", "edge", "cancel_modify"}

EXPECTED_COVERAGE = {"search": 2, "availability": 3, "booking": 3, "edge": 2}


@pytest.fixture(scope="module")
def heldout():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def restaurant_names():
    rows = json.loads(SEED_DATA.read_text(encoding="utf-8"))
    return {r["unit_name"] for r in rows}


def test_parses_to_ten_conversations(heldout):
    assert isinstance(heldout, list)
    assert len(heldout) == 10


def test_ids_are_unique_and_do_not_collide_with_the_baseline(heldout):
    yaml = pytest.importorskip("yaml")
    ids = [c["id"] for c in heldout]
    assert len(ids) == len(set(ids))
    baseline_ids = {c["id"] for c in yaml.safe_load(BASELINE.read_text(encoding="utf-8"))}
    assert not (set(ids) & baseline_ids)


def test_coverage_matches_the_phase_8_spec(heldout):
    counts: dict = {}
    for convo in heldout:
        counts[convo["category"]] = counts.get(convo["category"], 0) + 1
    assert counts == EXPECTED_COVERAGE


def test_every_conversation_has_the_required_fields(heldout):
    for convo in heldout:
        assert convo["category"] in VALID_CATEGORIES, convo["id"]
        assert convo["expected_outcome"] in VALID_OUTCOMES, convo["id"]
        assert convo.get("description", "").strip(), convo["id"]
        assert convo.get("turns"), convo["id"]
        for turn in convo["turns"]:
            assert turn.get("user", "").strip(), convo["id"]
            expect = turn.get("expect") or {}
            assert expect.get("plan") in ("reply", "execute"), convo["id"]
            if expect["plan"] == "execute":
                assert expect.get("action"), convo["id"]
            else:
                assert "action" not in expect, convo["id"]
            for key in expect:
                assert key in {"plan", "action", "args_subset",
                               "reply_contains_any", "memory_after"}, (convo["id"], key)


def test_actions_exist_in_the_tool_layer(heldout):
    tool_calls = pytest.importorskip(
        "app.agent.tool_calls", reason="tool_calls needs sqlalchemy",
    )
    for convo in heldout:
        for turn in convo["turns"]:
            action = (turn["expect"] or {}).get("action")
            if action:
                assert action in tool_calls.TOOL_SPEC, (convo["id"], action)
                assert action in tool_calls.TOOL_FUNCTIONS, (convo["id"], action)


def test_referenced_restaurants_exist_in_the_seed_data(heldout, restaurant_names):
    import re

    pattern = re.compile(r"GoodFoods(?:\s+[A-Z][A-Za-z]+)+")
    for convo in heldout:
        for turn in convo["turns"]:
            for mentioned in pattern.findall(turn["user"]):
                assert mentioned in restaurant_names, (convo["id"], mentioned)
            expected = (turn["expect"].get("args_subset") or {}).get("restaurant")
            if expected:
                assert expected in restaurant_names, (convo["id"], expected)


def test_booking_outcomes_are_reachable(heldout):
    """A conversation expecting a booking must actually expect the call."""
    for convo in heldout:
        actions = [(t["expect"] or {}).get("action") for t in convo["turns"]]
        if convo["expected_outcome"] == "booking_created":
            assert "create_reservation" in actions, convo["id"]
        if convo["expected_outcome"] == "no_action":
            assert "create_reservation" not in actions, convo["id"]
            assert "cancel_reservation" not in actions, convo["id"]
            assert "modify_reservation" not in actions, convo["id"]


def test_no_turn_is_shared_with_the_baseline_benchmark(heldout):
    """The two fixture sets must stay independent of each other."""
    yaml = pytest.importorskip("yaml")
    baseline = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    baseline_turns = {
        (t.get("user") or "").strip().lower()
        for convo in baseline for t in (convo.get("turns") or [])
    }
    for convo in heldout:
        for turn in convo["turns"]:
            assert turn["user"].strip().lower() not in baseline_turns, (
                convo["id"], turn["user"],
            )


def test_no_turn_is_shared_with_the_training_generator():
    """3000 generated samples must not reproduce a held-out utterance."""
    yaml = pytest.importorskip("yaml")
    from scripts import generate_training_data as gen

    heldout = yaml.safe_load(HELDOUT.read_text(encoding="utf-8"))
    held_turns = {
        (t.get("user") or "").strip().lower()
        for convo in heldout for t in (convo.get("turns") or [])
    }
    train, val = gen.generate_dataset(n=3000, seed=42, val_split=0.0)
    generated = {
        rec["messages"][1]["content"].split("\n", 1)[0].strip().lower()
        for rec in train + val
    }
    assert not (generated & held_turns)
