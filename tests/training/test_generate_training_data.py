"""
Phase 8 — contract tests for the QLoRA training-data generator.

No GPU, no network, no DB. Everything here runs on stdlib + PyYAML; the checks
that need the live agent (the planner's JSON validator, date/time
normalisation) import it lazily and skip if its deps are missing.

Four things are being defended:

  1. FORMAT FIDELITY  — every sample matches what call_planner_llm sends at
     inference time, byte for byte.
  2. THE INVARIANTS   — create_reservation only with an email + a confirmation;
     a time in the availability phase always means check_availability. These
     are the 1.7B's measured failure modes.
  3. REPRODUCIBILITY  — same seed, same bytes; no user message spans the
     train/val boundary.
  4. INTEGRITY        — no utterance is lifted from the eval fixtures.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import generate_training_data as gen  # noqa: E402

FIXTURE_PATHS = [
    REPO_ROOT / "tests" / "eval" / "conversations.yaml",
    REPO_ROOT / "tests" / "eval" / "heldout_conversations.yaml",
]

# A generated utterance may coincide with a fixture utterance ONLY when it is a
# bare affirmative/decline token that the runtime itself enumerates —
# planner_agent._strict_affirmatives (line ~529) and _CONFIRM_START/
# _DECLINE_WORDS. Those came from the agent code, not from the fixtures, and
# there is no other way to phrase "yes". Anything else overlapping means the
# template bank drifted toward the benchmark and must be rewritten.
ALLOWED_FIXTURE_COLLISIONS = {
    "yes",
    "yep",
    "go ahead",
    "book it",
    "do it",
    "confirm it",
    "yes please",
    "please confirm",
    "sounds good",
}


# ---------------------------------------------------------------------------
# Session-scoped corpora
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def samples():
    """500 seeded samples — the corpus the property tests run against."""
    train, val = gen.generate_dataset(n=500, seed=1234, val_split=0.0)
    assert not val
    return train


@pytest.fixture(scope="module")
def by_case(samples):
    out = {}
    for rec in samples:
        out.setdefault(rec["meta"]["case"], []).append(rec)
    return out


def assistant_json(record):
    return json.loads(record["messages"][2]["content"])


def user_text(record):
    """The raw user utterance — the first line, before the context block."""
    return record["messages"][1]["content"].split("\n", 1)[0]


# ---------------------------------------------------------------------------
# 1. Shape and format fidelity
# ---------------------------------------------------------------------------

def test_every_assistant_message_is_valid_planner_json(samples):
    for rec in samples:
        plan = assistant_json(rec)
        assert plan["plan"] in ("reply", "execute"), rec["meta"]


def test_no_think_preamble_or_markdown_in_targets(samples):
    """Reasoning is disabled and stripped at runtime; targets must be bare JSON."""
    for rec in samples:
        content = rec["messages"][2]["content"]
        assert content.startswith("{") and content.endswith("}")
        assert "<think>" not in content and "</think>" not in content
        assert "```" not in content


def test_role_sequence_is_system_user_assistant(samples):
    for rec in samples:
        assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]


def test_system_prompt_is_the_verbatim_phase_file(samples):
    """call_planner_llm does `system_content = PHASE_PROMPT` — no substitution."""
    for rec in samples:
        phase = rec["meta"]["phase"]
        expected = (REPO_ROOT / "app" / "agent" / gen.PHASE_PROMPT_FILES[phase]).read_text(
            encoding="utf-8"
        )
        assert rec["messages"][0]["content"] == expected


def test_user_content_matches_the_runtime_template(samples):
    """Reproduce the runtime f-string and require an exact match."""
    for rec in samples:
        content = rec["messages"][1]["content"]
        head, sep, tail = content.partition("\n\n--- Context (do not echo back) ---\n")
        assert sep, "context block missing"
        lines = tail.split("\n")
        assert lines[0].startswith("Memory: ")
        assert lines[1].startswith("Recent results: ")
        assert lines[2].startswith("Customer profile: ")
        assert lines[3] == "Respond ONLY with one valid JSON object."
        assert len(lines) == 4
        # Each context blob must be single-line JSON (pretty-printed JSON makes
        # qwen3 echo the template back — see the comment in call_planner_llm).
        memory = json.loads(lines[0][len("Memory: "):])
        json.loads(lines[1][len("Recent results: "):])
        assert json.loads(lines[2][len("Customer profile: "):]) == {}
        assert memory["phase"] == rec["meta"]["phase"]
        assert "intent" not in memory  # hidden from the LLM by merge_into_context
        assert None not in memory.values()
        # And the whole thing must rebuild exactly.
        assert content == gen.build_user_content(head, memory, json.loads(
            lines[1][len("Recent results: "):]))


def test_memory_serialisation_matches_conversation_memory():
    """The memory blob is produced by the real ConversationMemory class."""
    from app.agent.conversation_memory import ConversationMemory

    state = {"phase": "booking", "restaurant": "GoodFoods Grill", "party_size": 4,
             "customer_email": None, "intent": "manage"}
    mem = ConversationMemory()
    mem.update_from_planner(dict(state))
    expected = json.dumps(mem.merge_into_context({})["memory"], ensure_ascii=False)
    assert gen.memory_inline(state) == expected
    assert "intent" not in expected and "customer_email" not in expected


# ---------------------------------------------------------------------------
# 2. Action whitelist — checked against the live validator, not a copy
# ---------------------------------------------------------------------------

def test_generated_actions_are_in_the_planner_whitelist(samples):
    for rec in samples:
        plan = assistant_json(rec)
        if plan["plan"] == "execute":
            assert plan["action"] in gen.ALLOWED_ACTIONS, rec["meta"]


def test_allowed_actions_copy_still_matches_the_live_validator():
    """gen.ALLOWED_ACTIONS mirrors a function-local set in planner_agent.

    Probe the real validator so the copy cannot silently drift.
    """
    planner = pytest.importorskip(
        "app.agent.planner_agent",
        reason="planner_agent needs requests + pydantic-settings",
    )
    for action in gen.ALLOWED_ACTIONS:
        probe = json.dumps({"plan": "execute", "action": action, "args": {}})
        assert planner.validate_planner_json(probe) is not None, action
    for bogus in ("book_table", "search", "delete_reservation", ""):
        probe = json.dumps({"plan": "execute", "action": bogus, "args": {}})
        assert planner.validate_planner_json(probe) is None, bogus


def test_every_target_passes_the_live_planner_validator(samples):
    """The end-to-end gate: a target the validator rejects is a wasted sample."""
    planner = pytest.importorskip(
        "app.agent.planner_agent",
        reason="planner_agent needs requests + pydantic-settings",
    )
    for rec in samples:
        assert planner.validate_planner_json(rec["messages"][2]["content"]) is not None, (
            rec["meta"], rec["messages"][2]["content"],
        )


def test_reply_lengths_are_within_prompt_and_validator_bounds(samples):
    for rec in samples:
        plan = assistant_json(rec)
        if plan["plan"] == "reply":
            assert gen.REPLY_MIN_CHARS <= len(plan["reply"]) <= gen.REPLY_MAX_CHARS


# ---------------------------------------------------------------------------
# 3. The invariants this dataset exists to teach
# ---------------------------------------------------------------------------

def test_create_reservation_requires_email_and_confirmation(samples):
    """No booking without an email on file AND an explicit user confirmation."""
    fired = 0
    for rec in samples:
        plan = assistant_json(rec)
        if plan.get("action") != "create_reservation":
            continue
        fired += 1
        memory = json.loads(
            rec["messages"][1]["content"].split("Memory: ", 1)[1].split("\n", 1)[0]
        )
        assert memory.get("customer_email"), f"booked with no email in memory: {memory}"
        assert plan["args"].get("customer_email") == memory["customer_email"]
        assert memory["phase"] == "booking"
        assert user_text(rec) in gen.BOOKING_CONFIRM_TEMPLATES, user_text(rec)
    assert fired > 0, "no create_reservation samples generated — corpus too small?"


def test_create_reservation_confirmations_satisfy_the_runtime_classifier():
    """Every confirmation phrase must also read as a confirmation to Python.

    If the model and planner_agent._is_booking_confirmation disagree, the
    deterministic short-circuit and the model fight each other.
    """
    planner = pytest.importorskip(
        "app.agent.planner_agent",
        reason="planner_agent needs requests + pydantic-settings",
    )
    for phrase in gen.BOOKING_CONFIRM_TEMPLATES:
        assert planner._is_booking_confirmation(phrase), phrase
    for phrase in gen.BOOKING_DECLINE_TEMPLATES:
        assert not planner._is_booking_confirmation(phrase), phrase


_TIME_TOKEN = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\b\d{1,2}:\d{2}\b", re.I)


def test_time_in_availability_phase_always_means_check_availability(samples):
    """The anti-premature-booking invariant.

    Exception by design: e_ambiguous_time ("7pm or 8pm") carries time tokens but
    cannot be checked — it must ask the user to pick one.
    """
    checked = 0
    for rec in samples:
        meta = rec["meta"]
        if meta["phase"] != "availability" or meta["case"] == "e_ambiguous_time":
            continue
        if not _TIME_TOKEN.search(user_text(rec)):
            continue
        checked += 1
        plan = assistant_json(rec)
        assert plan["plan"] == "execute", (meta, user_text(rec))
        assert plan["action"] == "check_availability", (meta, user_text(rec))
        assert plan["args"].get("time"), "a stated time must reach the tool"
    assert checked > 0


def test_availability_phase_never_books(samples):
    for rec in samples:
        if rec["meta"]["phase"] == "availability":
            assert assistant_json(rec).get("action") != "create_reservation"


def test_discovery_phase_never_books_or_checks(samples):
    for rec in samples:
        if rec["meta"]["phase"] == "discovery":
            assert assistant_json(rec).get("action") not in (
                "create_reservation", "check_availability",
            )


def test_invalid_party_sizes_are_never_stored_or_executed(by_case):
    for name in ("e_party_zero_or_negative", "e_party_too_large"):
        for rec in by_case.get(name, []):
            plan = assistant_json(rec)
            assert plan["plan"] == "reply"
            assert "guest" in plan["reply"].lower() or "how many" in plan["reply"].lower()


def test_declines_never_fire_a_tool(by_case):
    for rec in by_case.get("b_declined", []):
        assert assistant_json(rec)["plan"] == "reply"


def test_check_availability_args_are_complete(by_case):
    """Every check_availability target carries restaurant + date + party_size."""
    cases = ("a_date_party_time", "a_date_party_no_time", "a_time_completes",
             "a_party_completes", "a_date_completes")
    for name in cases:
        for rec in by_case.get(name, []):
            args = assistant_json(rec)["args"]
            assert args["restaurant"]
            assert args["date"]
            assert isinstance(args["party_size"], int) and args["party_size"] >= 1


def test_time_values_in_args_are_24h_or_null(samples):
    for rec in samples:
        plan = assistant_json(rec)
        if plan["plan"] != "execute":
            continue
        value = plan.get("args", {}).get("time")
        if value is not None:
            assert re.fullmatch(r"[0-2]\d:[0-5]\d", value), value


def test_memory_sourced_args_are_copied_verbatim(samples):
    """When a slot is already in memory the target must echo it, not re-derive."""
    for rec in samples:
        plan = assistant_json(rec)
        if plan["plan"] != "execute":
            continue
        memory = json.loads(
            rec["messages"][1]["content"].split("Memory: ", 1)[1].split("\n", 1)[0]
        )
        args = plan.get("args", {})
        for slot in ("restaurant", "party_size"):
            if slot in memory and slot in args:
                assert args[slot] == memory[slot], (rec["meta"], slot)
        # A date already in memory is ISO and must be reused as-is; a date the
        # user just stated stays raw (Python normalises it downstream).
        if "date" in memory and "date" in args and rec["meta"]["case"] in (
            "a_time_completes", "a_party_completes",
        ):
            assert args["date"] == memory["date"]


# ---------------------------------------------------------------------------
# 4. Template banks agree with the runtime parsers
# ---------------------------------------------------------------------------

def test_date_phrases_all_normalise():
    date_utils = pytest.importorskip(
        "app.api.utils.date_utils", reason="date_utils needs python-dateutil",
    )
    for phrase in gen.DATE_PHRASES:
        assert date_utils.normalize_date_to_iso(phrase) is not None, phrase


def test_iso_date_pool_round_trips():
    date_utils = pytest.importorskip(
        "app.api.utils.date_utils", reason="date_utils needs python-dateutil",
    )
    for iso in gen.ISO_DATE_POOL:
        assert date_utils.normalize_date_to_iso(iso) == iso


def test_time_phrase_mapping_matches_normalize_time():
    date_utils = pytest.importorskip(
        "app.api.utils.date_utils", reason="date_utils needs python-dateutil",
    )
    for phrase, expected in gen.TIME_PHRASES:
        assert date_utils.normalize_time(phrase) == expected, phrase


def test_party_phrases_match_the_runtime_extractor():
    planner = pytest.importorskip(
        "app.agent.planner_agent",
        reason="planner_agent needs requests + pydantic-settings",
    )
    for phrase, expected in gen.PARTY_PHRASES:
        assert planner.extract_party_size_from_text(phrase) == expected, phrase


def test_invalid_emails_contain_no_parseable_address():
    """The re-ask cases must not hide an address the runtime regex would grab."""
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    for bad in gen.INVALID_EMAILS:
        assert not email_re.search(bad), bad


def test_seating_phrases_canonicalise():
    planner = pytest.importorskip(
        "app.agent.planner_agent",
        reason="planner_agent needs requests + pydantic-settings",
    )
    for phrase, expected in gen.SEATING_PHRASES:
        assert planner.extract_seating_pref(phrase) == expected, phrase


# ---------------------------------------------------------------------------
# 5. Reproducibility, split hygiene, manifest
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_output():
    a_train, a_val = gen.generate_dataset(n=200, seed=7, val_split=0.05)
    b_train, b_val = gen.generate_dataset(n=200, seed=7, val_split=0.05)
    assert [json.dumps(r, sort_keys=True) for r in a_train] == \
           [json.dumps(r, sort_keys=True) for r in b_train]
    assert [json.dumps(r, sort_keys=True) for r in a_val] == \
           [json.dumps(r, sort_keys=True) for r in b_val]


def test_different_seeds_produce_different_output():
    a_train, _ = gen.generate_dataset(n=200, seed=7, val_split=0.0)
    b_train, _ = gen.generate_dataset(n=200, seed=8, val_split=0.0)
    assert a_train != b_train


def test_val_split_shares_no_user_message_with_train():
    train, val = gen.generate_dataset(n=800, seed=99, val_split=0.05)
    assert val, "val split came back empty"
    train_users = {r["messages"][1]["content"] for r in train}
    val_users = {r["messages"][1]["content"] for r in val}
    assert not (train_users & val_users)


def test_group_allocation_sums_to_n():
    for n in (1, 7, 20, 50, 999, 3000):
        counts = gen.allocate_group_counts(n)
        assert sum(counts.values()) == n
        assert set(counts) == set(gen.GROUP_WEIGHTS)


def test_manifest_totals_add_up(tmp_path):
    out = tmp_path / "planner_train.jsonl"
    assert gen.main(["--n", "300", "--out", str(out), "--seed", "5"]) == 0

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    train_lines = out.read_text(encoding="utf-8").strip().splitlines()
    val_lines = (tmp_path / "planner_train_val.jsonl").read_text(
        encoding="utf-8").strip().splitlines()

    assert manifest["total"] == 300 == len(train_lines) + len(val_lines)
    assert manifest["train"] == len(train_lines)
    assert manifest["val"] == len(val_lines)
    assert sum(manifest["by_phase"].values()) == 300
    assert sum(manifest["by_group"].values()) == 300
    assert sum(manifest["by_case"].values()) == 300
    assert sum(manifest["by_action"].values()) == 300
    assert manifest["plan_counts"]["reply"] + manifest["plan_counts"]["execute"] == 300


def test_balance_targets_are_roughly_met():
    train, val = gen.generate_dataset(n=2000, seed=3, val_split=0.0)
    records = train + val
    groups = {}
    for rec in records:
        groups[rec["meta"]["group"]] = groups.get(rec["meta"]["group"], 0) + 1
    total = len(records)
    for group, target in gen.GROUP_WEIGHTS.items():
        assert abs(groups[group] / total - target) < 0.02, (group, groups[group])

    replies = sum(1 for r in records if r["meta"]["plan"] == "reply")
    assert 0.25 <= replies / total <= 0.40, replies / total


def test_smoke_mode_runs(tmp_path):
    """The documented smoke invocation must work and stay tiny."""
    out = tmp_path / "smoke.jsonl"
    assert gen.main(["--n", "20", "--out", str(out), "--seed", "42"]) == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert 0 < len(lines) <= 20
    for line in lines:
        json.loads(line)


def test_files_are_utf8_with_unix_newlines(tmp_path):
    out = tmp_path / "planner_train.jsonl"
    assert gen.main(["--n", "40", "--out", str(out), "--seed", "11"]) == 0
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    raw.decode("utf-8")


# ---------------------------------------------------------------------------
# 6. INTEGRITY — the training set must not know the benchmark
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _fixture_user_messages():
    yaml = pytest.importorskip("yaml")
    messages = set()
    for path in FIXTURE_PATHS:
        if not path.exists():
            continue
        for convo in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            for turn in convo.get("turns") or []:
                if turn.get("user"):
                    messages.add(_normalise(turn["user"]))
    return messages


def test_no_generated_utterance_is_lifted_from_the_eval_fixtures():
    fixture_messages = _fixture_user_messages()
    assert fixture_messages, "no fixture utterances found — check FIXTURE_PATHS"

    train, val = gen.generate_dataset(n=3000, seed=42, val_split=0.0)
    generated = {_normalise(user_text(rec)) for rec in train + val}

    overlap = {m for m in generated & fixture_messages if m}
    unexplained = overlap - {_normalise(x) for x in ALLOWED_FIXTURE_COLLISIONS}
    assert not unexplained, (
        "generated utterances match eval fixtures verbatim: "
        f"{sorted(unexplained)} — rewrite those templates"
    )


def test_generator_does_not_read_the_fixture_files():
    """Static check: no executable code in the generator touches tests/eval/*.

    Docstrings and comments may *discuss* the fixtures (the integrity note
    does); what must never appear is a live reference that could read them.
    """
    import ast

    source = (REPO_ROOT / "scripts" / "generate_training_data.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstring_nodes.add(id(body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in docstring_nodes:
            assert "conversations.yaml" not in node.value, node.value
            assert "tests/eval" not in node.value.replace("\\", "/"), node.value
