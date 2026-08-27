#!/usr/bin/env python
"""Evaluation harness for the GoodFoods agent.

Drives every conversation in tests/eval/conversations.yaml through the agent
(in-process, the same way scripts/smoke_test.py does), scores each turn with
tests/eval/scorer.py, verifies expected_outcome against live DB state, and
writes a JSON + Markdown report into reports/.

    python -m scripts.evaluate --model qwen3:4b [--categories search,booking]

Requires a reachable Postgres and Ollama (with the model pulled), plus pyyaml.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the repo root importable no matter where this is invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GoodFoods Phase 3 evaluation runner")
    p.add_argument(
        "--model",
        default=None,
        help="Override OLLAMA_MODEL for this run (e.g. qwen3:4b, qwen3:8b).",
    )
    p.add_argument(
        "--categories",
        default=None,
        help="Comma-separated subset of categories to run "
             "(search, availability, booking, edge, cancel_modify). "
             "Default: all.",
    )
    p.add_argument(
        "--conversations",
        default=None,
        help="Comma-separated conversation ids to run (e.g. A01,B06). "
             "Overrides --categories.",
    )
    p.add_argument(
        "--reset-db",
        action="store_true",
        help="Re-run reset_db + load_restaurants + add_opening_hours before eval.",
    )
    p.add_argument(
        "--report-dir",
        default=str(REPO_ROOT / "reports"),
        help="Where to write the JSON + Markdown reports.",
    )
    p.add_argument(
        "--fixture",
        default=str(REPO_ROOT / "tests" / "eval" / "conversations.yaml"),
        help="Path to the conversations YAML fixture.",
    )
    p.add_argument(
        "--timeout-per-turn",
        type=float,
        default=240.0,
        help="Soft watchdog — log a warning if any single turn exceeds this many seconds.",
    )
    return p.parse_args(argv)


def _c(code, text):
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _c("92", t)
def red(t):    return _c("91", t)
def yellow(t): return _c("93", t)
def dim(t):    return _c("2", t)
def bold(t):   return _c("1", t)


def _resolve_restaurant(session, Restaurant, name: str):
    """Mirror dispatch_tool's restaurant resolution (ilike substring match)."""
    return (
        session.query(Restaurant)
        .filter(Restaurant.unit_name.ilike(f"%{name}%"))
        .first()
    )


def seed_reservation(seed_data: dict, models, SessionLocal) -> dict:
    """Insert a Customer + Reservation row so cancel/modify flows have
    something to operate on. Returns a handle the runner uses for outcome
    checks and cleanup."""
    Customer, Restaurant, Reservation = models["Customer"], models["Restaurant"], models["Reservation"]
    email = seed_data["customer_email"]
    restaurant_name = seed_data["restaurant"]
    date = seed_data["date"]
    time_ = seed_data["time"]
    party_size = int(seed_data["party_size"])

    session = SessionLocal()
    try:
        cust = session.query(Customer).filter_by(email=email).first()
        if cust is None:
            cust = Customer(name=email.split("@")[0].title(), email=email)
            session.add(cust)
            session.flush()

        rest = _resolve_restaurant(session, Restaurant, restaurant_name)
        if rest is None:
            raise RuntimeError(f"seed restaurant not found: {restaurant_name!r}")

        res = Reservation(
            customer_id=cust.id,
            restaurant_id=rest.id,
            date=date,
            time=time_,
            party_size=party_size,
            status="confirmed",
        )
        session.add(res)
        session.commit()
        session.refresh(res)
        return {
            "reservation_id": res.id,
            "customer_id": cust.id,
            "customer_email": email,
            "original": {
                "date": res.date,
                "time": res.time,
                "party_size": res.party_size,
                "status": res.status,
            },
        }
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def cleanup_seeded(handle: dict, models, SessionLocal) -> None:
    """Delete the seeded reservation. Best-effort; never raises."""
    if not handle:
        return
    Reservation = models["Reservation"]
    session = SessionLocal()
    try:
        res = session.query(Reservation).filter_by(id=handle["reservation_id"]).first()
        if res is not None:
            session.delete(res)
            session.commit()
    except Exception as exc:  # pragma: no cover - best-effort
        session.rollback()
        print(dim(f"  (seeded cleanup warning: {exc})"))
    finally:
        session.close()


def cleanup_created_reservations(email: str, started_at_iso: str, models, SessionLocal) -> int:
    """Remove Reservation rows for `email` created during the run so the eval
    stays re-runnable. Returns the count deleted."""
    Reservation = models["Reservation"]
    session = SessionLocal()
    try:
        rows = (
            session.query(Reservation)
            .filter(
                Reservation.customer.has(email=email),
                Reservation.created_at >= started_at_iso,
            )
            .all()
        )
        for r in rows:
            session.delete(r)
        session.commit()
        return len(rows)
    except Exception as exc:  # pragma: no cover - best-effort
        session.rollback()
        print(dim(f"  (created cleanup warning: {exc})"))
        return 0
    finally:
        session.close()


def probe_booking_created(email: str, started_at_iso: str, models, SessionLocal) -> bool:
    """True if a confirmed reservation for `email` exists with created_at >=
    the conversation start timestamp."""
    if not email:
        return False
    Reservation = models["Reservation"]
    session = SessionLocal()
    try:
        return (
            session.query(Reservation)
            .filter(
                Reservation.customer.has(email=email),
                Reservation.created_at >= started_at_iso,
                Reservation.status == "confirmed",
            )
            .count()
            > 0
        )
    finally:
        session.close()


def probe_seeded_cancelled(handle: dict, models, SessionLocal) -> bool:
    """True if the seeded reservation is now in 'cancelled' status."""
    Reservation = models["Reservation"]
    session = SessionLocal()
    try:
        r = session.query(Reservation).filter_by(id=handle["reservation_id"]).first()
        return bool(r and r.status == "cancelled")
    finally:
        session.close()


def probe_seeded_modified(handle: dict, models, SessionLocal) -> bool:
    """True if any seeded field (date / time / party_size) differs from the
    original snapshot."""
    Reservation = models["Reservation"]
    session = SessionLocal()
    try:
        r = session.query(Reservation).filter_by(id=handle["reservation_id"]).first()
        if r is None:
            return False
        orig = handle["original"]
        return (
            r.date != orig["date"]
            or r.time != orig["time"]
            or r.party_size != orig["party_size"]
        )
    finally:
        session.close()


def run_conversation(convo: dict, agent_module, memory, models, SessionLocal,
                     timeout_per_turn: float) -> dict:
    """Run one conversation end-to-end; return a structured result record."""
    from tests.eval.scorer import (
        TurnResult,
        ConversationContext,
        detect_customer_email,
        score_outcome,
        score_turn,
    )

    convo_id = convo.get("id", "<no-id>")
    description = (convo.get("description") or "").strip()
    turns = convo.get("turns") or []
    expected_outcome = convo.get("expected_outcome")

    # Reset agent state so prior conversations don't bleed in.
    memory.reset()
    agent_module.recent_results.clear()

    seeded_handle = None
    seed = convo.get("seed") or {}
    if seed.get("reservation"):
        try:
            seeded_handle = seed_reservation(seed["reservation"], models, SessionLocal)
        except Exception as exc:
            return {
                "id": convo_id,
                "category": convo.get("category"),
                "description": description,
                "seed_error": str(exc),
                "turns": [],
                "expected_outcome": expected_outcome,
                "outcome_result": {"passed": False, "reason": f"seed failed: {exc}"},
                "passed": False,
            }

    started_at_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    turn_responses = []
    turn_results: list[TurnResult] = []
    user_messages = []
    conversation_aborted = False

    for idx, turn in enumerate(turns, start=1):
        user_text = turn.get("user", "")
        expect = turn.get("expect") or {}
        user_messages.append(user_text)

        t0 = time.time()
        try:
            output = agent_module.process_user_query(user_text)
        except Exception as exc:
            conversation_aborted = True
            turn_responses.append({
                "reply": f"[runner exception] {exc}",
                "tool_output": None,
                "phase": memory.state.get("phase"),
                "memory": dict(memory.state),
                "elapsed_s": round(time.time() - t0, 2),
            })
            turn_results.append(TurnResult(
                passed=False,
                reason=f"runner exception: {exc}",
            ))
            break
        elapsed = time.time() - t0

        reply = (output.get("reply") or "").strip()
        tool_output = output.get("tool_output")
        phase = memory.state.get("phase") or "discovery"

        response = {
            "reply": reply,
            "tool_output": tool_output,
            "phase": phase,
            "memory": dict(memory.state),
            "elapsed_s": round(elapsed, 2),
        }
        turn_responses.append(response)

        turn_result = score_turn(expect, response)
        turn_results.append(turn_result)

        if elapsed > timeout_per_turn:
            print(yellow(f"    [{convo_id} T{idx}] slow turn: {elapsed:.1f}s"))

    final_memory = dict(memory.state)
    detected_email = detect_customer_email(turn_responses, user_messages)

    booking_created_during = None
    seeded_now_cancelled = None
    seeded_now_modified = None

    if expected_outcome == "booking_created":
        booking_created_during = probe_booking_created(
            detected_email or "", started_at_iso, models, SessionLocal,
        )
    if expected_outcome == "booking_cancelled" and seeded_handle:
        seeded_now_cancelled = probe_seeded_cancelled(seeded_handle, models, SessionLocal)
    if expected_outcome == "booking_modified" and seeded_handle:
        seeded_now_modified = probe_seeded_modified(seeded_handle, models, SessionLocal)

    ctx = ConversationContext(
        expected_outcome=expected_outcome,
        turns=turn_responses,
        turn_results=turn_results,
        final_memory=final_memory,
        customer_email=detected_email,
        seeded_reservation=seeded_handle,
        booking_created_during=booking_created_during,
        seeded_now_cancelled=seeded_now_cancelled,
        seeded_now_modified=seeded_now_modified,
    )
    outcome_result = score_outcome(ctx)

    if detected_email and expected_outcome == "booking_created":
        cleanup_created_reservations(detected_email, started_at_iso, models, SessionLocal)
    if seeded_handle:
        cleanup_seeded(seeded_handle, models, SessionLocal)

    all_turns_passed = all(t.passed for t in turn_results) and not conversation_aborted
    passed = all_turns_passed and bool(outcome_result) and not conversation_aborted

    return {
        "id": convo_id,
        "category": convo.get("category"),
        "description": description,
        "expected_outcome": expected_outcome,
        "turns": [
            {
                "index": i + 1,
                "user": user_messages[i] if i < len(user_messages) else "",
                "expect": (turns[i].get("expect") if i < len(turns) else None),
                "response": {k: v for k, v in r.items() if k != "memory"},
                "memory_after": r.get("memory"),
                "passed": turn_results[i].passed,
                "reason": turn_results[i].reason,
                "checks": [
                    {"name": c.name, "passed": c.passed, "reason": c.reason}
                    for c in turn_results[i].checks
                ],
            }
            for i, r in enumerate(turn_responses)
        ],
        "outcome_result": {
            "passed": outcome_result.passed,
            "reason": outcome_result.reason,
        },
        "passed": passed,
        "customer_email": detected_email,
        "aborted": conversation_aborted,
    }


def aggregate_summary(records: list[dict]) -> dict:
    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    by_category: dict[str, dict[str, int]] = {}
    for r in records:
        cat = r.get("category") or "unknown"
        bucket = by_category.setdefault(cat, {"total": 0, "passed": 0})
        bucket["total"] += 1
        if r["passed"]:
            bucket["passed"] += 1

    # Turn-level accuracy (excludes aborted conversations' implicit fail).
    turn_total = 0
    turn_passed = 0
    for r in records:
        for t in r.get("turns", []):
            turn_total += 1
            if t["passed"]:
                turn_passed += 1

    return {
        "total_conversations": total,
        "passed_conversations": passed,
        "conversation_pass_rate": round(passed / total, 4) if total else 0.0,
        "total_turns": turn_total,
        "passed_turns": turn_passed,
        "turn_pass_rate": round(turn_passed / turn_total, 4) if turn_total else 0.0,
        "by_category": {
            cat: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
            }
            for cat, v in by_category.items()
        },
    }


def write_reports(records: list[dict], summary: dict, model: str,
                  report_dir: Path, timestamp: str) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{model.replace(':', '_')}_{timestamp}"

    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"

    json_payload = {
        "model": model,
        "generated_at": timestamp,
        "summary": summary,
        "conversations": records,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# GoodFoods eval — `{model}`")
    lines.append("")
    lines.append(f"_Generated {timestamp} (UTC)._")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Conversations: **{summary['passed_conversations']}/{summary['total_conversations']}** "
                 f"({summary['conversation_pass_rate']*100:.1f}%)")
    lines.append(f"- Turns: **{summary['passed_turns']}/{summary['total_turns']}** "
                 f"({summary['turn_pass_rate']*100:.1f}%)")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| Category | Pass | Total | Pass rate |")
    lines.append("|---|---:|---:|---:|")
    for cat, v in sorted(summary["by_category"].items()):
        lines.append(f"| {cat} | {v['passed']} | {v['total']} | {v['pass_rate']*100:.1f}% |")
    lines.append("")
    lines.append("## Per-conversation results")
    lines.append("")
    for r in records:
        mark = green("PASS") if r["passed"] else red("FAIL")
        lines.append(f"### {r['id']} — {mark}  ({r.get('category') or ''})")
        lines.append("")
        if r.get("description"):
            lines.append(f"_{r['description']}_")
            lines.append("")
        if r.get("aborted"):
            lines.append("> Conversation aborted mid-run (runner exception).")
            lines.append("")
        for t in r.get("turns", []):
            tmark = "[x]" if t["passed"] else "[ ]"
            lines.append(f"- {tmark} **T{t['index']}** `{t['user']}`")
            if not t["passed"]:
                lines.append(f"  - reason: {t['reason']}")
                resp = t.get("response") or {}
                to = resp.get("tool_output")
                if isinstance(to, dict):
                    lines.append(f"  - observed action: `{to.get('action')}` args=`{json.dumps(to.get('args', {}), default=str)}`")
                reply = (resp.get("reply") or "").strip()
                if reply:
                    snippet = reply if len(reply) <= 200 else reply[:200] + "..."
                    lines.append(f"  - reply: {snippet!r}")
        outcome = r.get("outcome_result") or {}
        if not outcome.get("passed"):
            lines.append("")
            lines.append(f"- outcome **{r.get('expected_outcome')}**: {outcome.get('reason')}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def select_conversations(all_convos: list[dict], args) -> list[dict]:
    if args.conversations:
        wanted = {c.strip() for c in args.conversations.split(",") if c.strip()}
        return [c for c in all_convos if c.get("id") in wanted]
    if args.categories:
        wanted_cats = {c.strip() for c in args.categories.split(",") if c.strip()}
        return [c for c in all_convos if c.get("category") in wanted_cats]
    return list(all_convos)


def reset_db_sequence() -> None:
    """Re-run the project's own seed scripts in order."""
    print(bold("Resetting DB (reset_db -> load_restaurants -> add_opening_hours)..."))
    for step in ("scripts.reset_db", "scripts.load_restaurants", "scripts.add_opening_hours"):
        print(dim(f"  -> python -m {step}"))
        result = subprocess.run(
            [sys.executable, "-m", step],
            cwd=str(REPO_ROOT),
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{step} exited {result.returncode}")


def main(argv=None) -> int:
    args = parse_args(argv)

    # Set model override BEFORE any app imports so get_settings() picks it up.
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    # Lazy YAML import — pyyaml may not be installed yet, and we want a clean
    # error pointing the user at requirements.txt.
    try:
        import yaml
    except ImportError:
        print(red("pyyaml is not installed. Run: pip install -r requirements.txt"))
        return 2

    if args.reset_db:
        reset_db_sequence()

    # App imports happen after the model env var is set.
    from app.config import get_settings
    from app.agent import agent as agent_module
    from app.agent.planner_agent import memory
    from app.data.db_connection import SessionLocal
    from app.data.db_models import Customer, Restaurant, Reservation

    models = {"Customer": Customer, "Restaurant": Restaurant, "Reservation": Reservation}
    settings = get_settings()
    model_tag = settings.ollama_model

    with open(args.fixture, "r", encoding="utf-8") as fh:
        fixture = yaml.safe_load(fh)
    if not isinstance(fixture, list):
        print(red(f"Fixture {args.fixture} did not parse to a list of conversations."))
        return 2

    selected = select_conversations(fixture, args)
    if not selected:
        print(yellow("No conversations matched the given filters."))
        return 0

    print("=" * 72)
    print(f" GoodFoods eval  |  model={model_tag}  |  conversations={len(selected)}")
    print(f" endpoint={settings.ollama_generate_url}")
    print("=" * 72)

    records: list[dict] = []
    overall_t0 = time.time()
    for convo in selected:
        cid = convo.get("id", "<no-id>")
        print(bold(f"\n[{cid}] {convo.get('category', '')}"), flush=True)
        record = run_conversation(
            convo, agent_module, memory, models, SessionLocal,
            timeout_per_turn=args.timeout_per_turn,
        )
        records.append(record)
        mark = green("PASS") if record["passed"] else red("FAIL")
        failed_turns = [t["index"] for t in record.get("turns", []) if not t["passed"]]
        outcome_ok = (record.get("outcome_result") or {}).get("passed", False)
        suffix = ""
        if failed_turns:
            suffix += f"  failed turns: {failed_turns}"
        if not outcome_ok:
            suffix += f"  outcome FAIL: {(record.get('outcome_result') or {}).get('reason', '')}"
        print(f"  {mark}{suffix}")

    overall_dt = time.time() - overall_t0
    summary = aggregate_summary(records)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = write_reports(
        records, summary, model_tag, Path(args.report_dir), timestamp,
    )

    print("\n" + "=" * 72)
    print(f" Conversations: {summary['passed_conversations']}/{summary['total_conversations']} "
          f"({summary['conversation_pass_rate']*100:.1f}%)")
    print(f" Turns:         {summary['passed_turns']}/{summary['total_turns']} "
          f"({summary['turn_pass_rate']*100:.1f}%)")
    for cat, v in sorted(summary["by_category"].items()):
        print(f"   {cat:<16} {v['passed']}/{v['total']}  ({v['pass_rate']*100:.1f}%)")
    print(dim(f"\n elapsed: {overall_dt:.1f}s"))
    print(bold(f" reports:\n  {json_path}\n  {md_path}"))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
