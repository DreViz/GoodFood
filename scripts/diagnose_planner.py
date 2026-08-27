"""
Controlled A/B/C/D diagnostic for the planner decision regression.

When the integrated planner call started emitting safe "Could you clarify"
replies for inputs the isolated test handled correctly, the likely culprit was
structural noise in the user message (three JSON context blobs + a
"User Message:" prefix) not matching the phase prompt's example format. This
script runs four message-structure variants against the Phase-1 smoke inputs
(A01/A02/A06/A07/A08) and prints per-variant pass rates.

Read-only: hits Ollama only, no DB, no agent flow.
"""
import json
import os
import sys
import time

import requests

# Make the project root importable when run as `python -m scripts.diagnose_planner`.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings

PHASE1_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "app", "agent", "planner_prompt_phase1.txt"
)

# (id, user_text, expected decision tag, expected action or None)
SMOKE = [
    ("A01", "I'm in the mood for Italian", "reply", None),
    ("A02", "Any Italian places in South?", "execute", "search_restaurants_by_filters"),
    ("A06", "Any recommendations?", "execute", "recommend_venues"),
    ("A07", "GoodFoods Grill", "execute", "get_seating_labels"),
    ("A08", "I want food", "reply", None),
]


def call_ollama(system, user, settings):
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": settings.ollama_think,
        "format": "json",
    }
    t0 = time.perf_counter()
    r = requests.post(settings.ollama_chat_url, json=payload, timeout=settings.ollama_timeout)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    data = r.json()
    raw = (data.get("message", {}).get("content") or data.get("response") or "").strip()
    return dt, raw


def parse_decision(raw):
    """Return ('reply', text) | ('execute', action) | ('unparsed', snippet)."""
    try:
        obj = json.loads(raw)
    except Exception:
        return ("unparsed", raw[:80])
    plan = obj.get("plan")
    if plan == "reply":
        return ("reply", (obj.get("reply") or "")[:60])
    if plan == "execute":
        return ("execute", obj.get("action") or "")
    return ("unparsed", raw[:80])


def is_pass(decision, expected):
    plan, payload = decision
    exp_plan, exp_action = expected
    if exp_plan == "reply":
        return plan == "reply"
    if exp_plan == "execute":
        return plan == "execute" and payload == exp_action
    return False


def v1_current(phase1, user_text):
    """Replicates the current integrated user message exactly."""
    system = phase1
    user = (
        "Conversation Memory (Persisted User Details):\n"
        '{\n  "phase": "discovery"\n}\n\n'
        "Recent Results (JSON):\n[]\n\n"
        "Customer Profile (JSON):\n{}\n\n"
        f"User Message:\n{user_text}\n\n"
        "Respond ONLY with one valid JSON object (no text outside JSON)."
    )
    return system, user


def v2_minimal(phase1, user_text):
    """Bare user text — matches the phase prompt's `User says X` examples."""
    return phase1, user_text


def v3_context_in_system(phase1, user_text):
    """Context lives under the rules in the system message; user stays clean."""
    system = (
        f"{phase1}\n\n"
        "---\n"
        "Conversation Memory (Persisted User Details):\n"
        '{\n  "phase": "discovery"\n}\n'
        "Recent Results (JSON): []\n"
        "Customer Profile (JSON): {}\n"
    )
    return system, user_text


def v4_text_first(phase1, user_text):
    """User text first, context appended afterwards, no `User Message:` label."""
    system = phase1
    user = (
        f"{user_text}\n\n"
        "--- Context (do not echo back) ---\n"
        'Memory: {"phase": "discovery"}\n'
        "Recent results: []\n"
        "Customer profile: {}\n"
        "Respond ONLY with one valid JSON object."
    )
    return system, user


VARIANTS = [
    ("V1 current (ctx in user)", v1_current),
    ("V2 minimal (raw user only)", v2_minimal),
    ("V3 context in system", v3_context_in_system),
    ("V4 text-first w/ trailing ctx", v4_text_first),
]


def main():
    settings = get_settings()
    with open(PHASE1_PROMPT_PATH, "r", encoding="utf-8") as f:
        phase1 = f.read()

    print(f"Model: {settings.ollama_model} | think={settings.ollama_think}\n")

    summary = {}
    for vname, builder in VARIANTS:
        print(f"===== {vname} =====")
        passes = 0
        for cid, user_text, exp_plan, exp_action in SMOKE:
            system, user = builder(phase1, user_text)
            try:
                dt, raw = call_ollama(system, user, settings)
            except Exception as e:
                print(f"  {cid}  ERROR  {e}")
                continue
            decision = parse_decision(raw)
            ok = is_pass(decision, (exp_plan, exp_action))
            passes += 1 if ok else 0
            mark = "PASS" if ok else "FAIL"
            exp_str = exp_plan if exp_plan == "reply" else f"{exp_plan}:{exp_action}"
            got_str = f"{decision[0]}:{decision[1]}" if decision[0] != "reply" else f"reply:{decision[1]!r}"
            print(f"  {cid}  {dt:5.2f}s  [{mark}]  exp={exp_str:40s}  got={got_str}")
        summary[vname] = passes
        print(f"  -> {passes}/{len(SMOKE)} pass\n")

    print("===== SUMMARY =====")
    for vname, passes in summary.items():
        print(f"  {vname:38s}  {passes}/{len(SMOKE)}")


if __name__ == "__main__":
    main()
