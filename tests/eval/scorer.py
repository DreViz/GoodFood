# tests/eval/scorer.py
"""
Pure scoring functions for the Phase 3 evaluation harness.

These functions take an `expect` block (from conversations.yaml) and an actual
response captured by the runner, and return a CheckResult / TurnResult. They
do NO I/O and import nothing from `app.*` — keep them pure so they can be
unit-tested in isolation.

Response shape expected from the runner:

    {
        "reply":   str,                            # text shown to the user
        "tool_output": Optional[Dict],             # {action, args, result} when plan=execute
        "phase":   str,                            # memory.state["phase"] after the turn
        "memory":  Dict[str, Any],                 # full memory.state snapshot after the turn
    }

Conversation-outcome scoring receives a ConversationContext with everything
the per-conversation checks need (final memory, all turn results, DB-derived
booking facts, seeded-reservation snapshot, detected customer_email).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single atomic assertion."""
    name: str
    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # convenient `if result:` usage
        return self.passed


@dataclass
class TurnResult:
    """Aggregate result for one turn across all expectations."""
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    reason: str = ""

    @property
    def failed_check(self) -> Optional[CheckResult]:
        return next((c for c in self.checks if not c.passed), None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(value: Any) -> Any:
    """Normalise a scalar for case-insensitive comparison.

    Strings are lowercased and stripped. Non-strings pass through unchanged.
    """
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _is_match(expected: Any, actual: Any) -> bool:
    """Compare two scalars. Strings compared case-insensitively; numbers by
    equality after int/float coercion; everything else by equality."""
    if isinstance(expected, str) and isinstance(actual, str):
        return _norm(expected) == _norm(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual)
    if isinstance(expected, (int, float)) and isinstance(actual, str):
        try:
            return float(expected) == float(actual)
        except ValueError:
            return False
    if isinstance(expected, str) and isinstance(actual, (int, float)):
        return _norm(expected) == _norm(str(actual))
    return _norm(expected) == _norm(actual)


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


# ---------------------------------------------------------------------------
# Atomic per-turn scorers
# ---------------------------------------------------------------------------

def score_plan(expect: Dict[str, Any], response: Dict[str, Any]) -> CheckResult:
    """Verify the planner chose the expected high-level plan."""
    expected_plan = expect.get("plan")
    if expected_plan not in ("reply", "execute"):
        return CheckResult("plan", False, f"invalid expected plan: {expected_plan!r}")

    tool_output = response.get("tool_output")
    # The agent returns tool_output=None when the planner emitted plan=reply,
    # and a dict when plan=execute.
    actual_plan = "execute" if isinstance(tool_output, dict) else "reply"

    if actual_plan == expected_plan:
        return CheckResult("plan", True, f"plan={actual_plan}")
    return CheckResult(
        "plan",
        False,
        f"expected plan={expected_plan} but observed plan={actual_plan}",
    )


def score_action(expect: Dict[str, Any], response: Dict[str, Any]) -> CheckResult:
    """Verify the action name when plan=execute."""
    expected_action = expect.get("action")
    tool_output = response.get("tool_output") or {}

    if expected_action is None:
        # Caller asked for plan=reply; nothing to check here.
        return CheckResult("action", True, "no action expected")

    actual_action = tool_output.get("action") if isinstance(tool_output, dict) else None
    if actual_action == expected_action:
        return CheckResult("action", True, f"action={actual_action}")
    return CheckResult(
        "action",
        False,
        f"expected action={expected_action!r} observed={actual_action!r}",
    )


def score_args_subset(
    expect: Dict[str, Any],
    response: Dict[str, Any],
) -> CheckResult:
    """Subset-match args. Only the keys in `args_subset` are checked; extra
    args produced by the planner are ignored. String values match
    case-insensitively."""
    expected_subset = expect.get("args_subset")
    if expected_subset is None:
        return CheckResult("args_subset", True, "no args subset declared")

    if not isinstance(expected_subset, dict):
        return CheckResult("args_subset", False, "args_subset must be a dict")

    tool_output = response.get("tool_output") or {}
    actual_args = tool_output.get("args", {}) if isinstance(tool_output, dict) else {}
    if not isinstance(actual_args, dict):
        actual_args = {}

    mismatches = []
    for key, expected_val in expected_subset.items():
        actual_val = actual_args.get(key)
        if not _is_match(expected_val, actual_val):
            mismatches.append(
                f"{key}: expected {expected_val!r} observed {actual_val!r}"
            )

    if not mismatches:
        return CheckResult(
            "args_subset",
            True,
            f"all {len(expected_subset)} keys match",
        )
    return CheckResult(
        "args_subset",
        False,
        "; ".join(mismatches),
    )


def score_reply_contains_any(
    expect: Dict[str, Any],
    response: Dict[str, Any],
) -> CheckResult:
    """Pass if reply contains at least one candidate substring (ci)."""
    candidates: Sequence[str] = expect.get("reply_contains_any") or []
    if not candidates:
        return CheckResult("reply_contains_any", True, "no candidates declared")

    reply = response.get("reply") or ""
    reply_lc = reply.lower()
    for cand in candidates:
        if cand.lower() in reply_lc:
            return CheckResult(
                "reply_contains_any",
                True,
                f"matched {cand!r}",
            )
    return CheckResult(
        "reply_contains_any",
        False,
        f"reply matched none of {list(candidates)}; reply={reply!r}",
    )


def score_memory_after(
    expect: Dict[str, Any],
    response: Dict[str, Any],
) -> CheckResult:
    """Subset-match memory_after against the post-turn memory snapshot.

    The runner is expected to attach `memory` (full memory.state) to the
    response. If only `phase` is available, this scorer degrades gracefully
    and only checks phase."""
    expected_mem = expect.get("memory_after")
    if expected_mem is None:
        return CheckResult("memory_after", True, "no memory_after declared")

    if not isinstance(expected_mem, dict):
        return CheckResult("memory_after", False, "memory_after must be a dict")

    actual_mem = response.get("memory") or {}
    # Fall back to phase-only if the runner didn't attach full memory.
    if not actual_mem and "phase" in response:
        actual_mem = {"phase": response.get("phase")}

    mismatches = []
    for key, expected_val in expected_mem.items():
        actual_val = actual_mem.get(key)
        if not _is_match(expected_val, actual_val):
            mismatches.append(
                f"{key}: expected {expected_val!r} observed {actual_val!r}"
            )

    if not mismatches:
        return CheckResult(
            "memory_after",
            True,
            f"all {len(expected_mem)} keys match",
        )
    return CheckResult(
        "memory_after",
        False,
        "; ".join(mismatches),
    )


# ---------------------------------------------------------------------------
# Per-turn composite scorer
# ---------------------------------------------------------------------------

def score_turn(expect: Dict[str, Any], response: Dict[str, Any]) -> TurnResult:
    """Run all relevant per-turn checks for one turn and aggregate."""
    checks: List[CheckResult] = []

    # 1. plan is always required.
    checks.append(score_plan(expect, response))

    expected_plan = expect.get("plan")

    # 2. action only if execute.
    if expected_plan == "execute":
        checks.append(score_action(expect, response))
        checks.append(score_args_subset(expect, response))

    # 3. reply_contains_any is optional; checked whenever declared.
    if expect.get("reply_contains_any"):
        checks.append(score_reply_contains_any(expect, response))

    # 4. memory_after is optional; checked whenever declared.
    if expect.get("memory_after"):
        checks.append(score_memory_after(expect, response))

    failed = next((c for c in checks if not c.passed), None)
    return TurnResult(
        passed=failed is None,
        checks=checks,
        reason="" if failed is None else f"{failed.name}: {failed.reason}",
    )


# ---------------------------------------------------------------------------
# Per-conversation outcome scorer
# ---------------------------------------------------------------------------

# Actions that mutate a reservation row. Used for `no_action` and
# `booking_created` reasoning.
RESERVATION_MUTATING_ACTIONS = {
    "create_reservation",
    "cancel_reservation",
    "modify_reservation",
}

# Heuristics for surfacing errors via the user-visible reply text.
ERROR_REPLY_TOKENS = (
    "error",
    "sorry",
    "couldn't",
    "could not",
    "unable",
    "not found",
    "invalid",
    "missing",
)


@dataclass
class ConversationContext:
    """Bundle of facts the outcome scorer needs, gathered by the runner."""

    expected_outcome: str
    turns: List[Dict[str, Any]]              # list of per-turn responses
    turn_results: List[TurnResult]
    final_memory: Dict[str, Any]
    customer_email: Optional[str]            # best-effort detection
    seeded_reservation: Optional[Dict[str, Any]] = None  # {id, original:{...}}
    booking_created_during: Optional[bool] = None        # DB-derived (when relevant)
    seeded_now_cancelled: Optional[bool] = None          # DB-derived
    seeded_now_modified: Optional[bool] = None           # DB-derived


def _actions_fired(turns: List[Dict[str, Any]]) -> List[str]:
    out = []
    for t in turns:
        to = t.get("tool_output")
        if isinstance(to, dict) and to.get("action"):
            out.append(to["action"])
    return out


def score_outcome(ctx: ConversationContext) -> CheckResult:
    """Verify the conversation-level expected_outcome."""
    expected = ctx.expected_outcome
    actions = _actions_fired(ctx.turns)

    if expected == "no_action":
        mutators = [a for a in actions if a in RESERVATION_MUTATING_ACTIONS]
        if not mutators:
            return CheckResult("outcome", True, "no reservation-mutating tool fired")
        return CheckResult(
            "outcome",
            False,
            f"expected no_action but fired {mutators}",
        )

    if expected == "phase_reached_availability":
        phase = (ctx.final_memory or {}).get("phase")
        if phase == "availability":
            return CheckResult("outcome", True, "final phase=availability")
        return CheckResult("outcome", False, f"expected phase=availability observed={phase!r}")

    if expected == "phase_reached_booking":
        phase = (ctx.final_memory or {}).get("phase")
        if phase == "booking":
            return CheckResult("outcome", True, "final phase=booking")
        return CheckResult("outcome", False, f"expected phase=booking observed={phase!r}")

    if expected == "booking_created":
        if "create_reservation" not in actions:
            return CheckResult(
                "outcome",
                False,
                "expected booking_created but create_reservation never fired",
            )
        if ctx.booking_created_during is False:
            return CheckResult(
                "outcome",
                False,
                f"create_reservation fired but no confirmed row in DB for {ctx.customer_email!r}",
            )
        return CheckResult("outcome", True, "booking created in DB")

    if expected == "booking_cancelled":
        if "cancel_reservation" not in actions:
            return CheckResult(
                "outcome",
                False,
                "expected booking_cancelled but cancel_reservation never fired",
            )
        if ctx.seeded_reservation is None:
            return CheckResult(
                "outcome",
                False,
                "expected booking_cancelled but no seeded reservation to verify",
            )
        if ctx.seeded_now_cancelled:
            return CheckResult("outcome", True, "seeded reservation status=cancelled")
        return CheckResult(
            "outcome",
            False,
            "cancel_reservation fired but seeded reservation status unchanged",
        )

    if expected == "booking_modified":
        if "modify_reservation" not in actions:
            return CheckResult(
                "outcome",
                False,
                "expected booking_modified but modify_reservation never fired",
            )
        if ctx.seeded_reservation is None:
            return CheckResult(
                "outcome",
                False,
                "expected booking_modified but no seeded reservation to verify",
            )
        if ctx.seeded_now_modified:
            return CheckResult("outcome", True, "seeded reservation fields changed")
        return CheckResult(
            "outcome",
            False,
            "modify_reservation fired but seeded reservation fields unchanged",
        )

    if expected == "error_presented":
        # Surface errors via tool result OR via the reply text.
        for turn in ctx.turns:
            to = turn.get("tool_output")
            if isinstance(to, dict):
                result = to.get("result")
                if isinstance(result, dict) and result.get("ok") is False:
                    return CheckResult(
                        "outcome",
                        True,
                        f"tool {to.get('action')} returned ok=False",
                    )
        for turn in ctx.turns:
            reply_lc = (turn.get("reply") or "").lower()
            if any(tok in reply_lc for tok in ERROR_REPLY_TOKENS):
                return CheckResult(
                    "outcome",
                    True,
                    "reply contained error token",
                )
        return CheckResult(
            "outcome",
            False,
            "expected error_presented but no error surfaced",
        )

    return CheckResult("outcome", False, f"unknown expected_outcome: {expected!r}")


# ---------------------------------------------------------------------------
# Convenience: extract customer_email from a conversation
# ---------------------------------------------------------------------------

def detect_customer_email(turns: List[Dict[str, Any]], user_messages: List[str]) -> Optional[str]:
    """Best-effort: prefer an email that appears in tool args, else scan user
    messages. Returns the last detected email (most recent wins)."""
    email: Optional[str] = None
    for turn in turns:
        to = turn.get("tool_output")
        if isinstance(to, dict):
            args = to.get("args") or {}
            cand = args.get("customer_email")
            if isinstance(cand, str) and "@" in cand:
                email = cand.strip()
    if email is not None:
        return email
    for msg in user_messages:
        m = _EMAIL_RE.search(msg or "")
        if m:
            email = m.group(0)
    return email
