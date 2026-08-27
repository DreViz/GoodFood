#!/usr/bin/env python
"""Synthetic trajectory generator for QLoRA fine-tuning of the planner.

Produces a chat-format JSONL ({"messages": [system, user, assistant], "meta"})
teaching the 3-phase planner procedure. Runtime fidelity is non-negotiable: the
three parts must match what planner_agent.call_planner_llm sends Ollama at
inference time — the raw phase-prompt file as system, build_user_content() as
user, the decision as bare JSON with no <think> preamble — and the memory blob
is serialised through the real ConversationMemory so key order matches
production.

Integrity: every utterance comes from the template banks in this file. Nothing
is copied or adapted from tests/eval/conversations.yaml (or heldout) — those
stay an untouched benchmark.

    python -m scripts.generate_training_data --n 3000 \
        --out data/planner_train.jsonl --val-split 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ConversationMemory is stdlib-only (logging + typing), so importing it keeps
# the generator dependency-free while guaranteeing the memory JSON matches
# production byte-for-byte.
from app.agent.conversation_memory import ConversationMemory  # noqa: E402

# Runtime contract constants — mirrored from the agent (see module docstring).

AGENT_DIR = REPO_ROOT / "app" / "agent"
SEED_DATA_PATH = REPO_ROOT / "app" / "data" / "goodfoods_locations_unique_50.json"

# planner_agent.load_prompt_for_phase()
PHASE_PROMPT_FILES = {
    "discovery": "planner_prompt_phase1.txt",
    "availability": "planner_prompt_phase2.txt",
    "booking": "planner_prompt_phase3.txt",
}

# planner_agent.validate_planner_json() — the whitelist a decision must pass
# before any tool runs. Mirrored here (it is a function-local set upstream);
# tests/training assert this copy still matches the live validator.
ALLOWED_ACTIONS = frozenset({
    "search_restaurants_by_filters",
    "recommend_venues",
    "check_availability",
    "create_reservation",
    "get_seating_map",
    "get_amenities",
    "get_booking_details",
    "get_seating_labels",
    "cancel_reservation",
    "modify_reservation",
})

# validate_planner_json accepts 4..200 chars; the phase prompts ask for 5..180.
# Generate to the tighter window so every target satisfies both.
REPLY_MIN_CHARS = 5
REPLY_MAX_CHARS = 180

# Actions this generator is allowed to emit. cancel/modify are deliberately
# absent: at runtime the whole manage flow is decided by Python interceptors
# (planner_agent guards 1, 2, 4, 5) before the LLM is ever called, and both
# models already score 9/9 there. Teaching them here could only regress it.
GENERATED_ACTIONS = frozenset({
    "search_restaurants_by_filters",
    "recommend_venues",
    "get_seating_labels",
    "check_availability",
    "create_reservation",
})

# Balance targets (fraction of the dataset per group).
#
# Booking is deliberately smaller than the 30% originally planned, and
# availability correspondingly larger. Reason: most booking-phase turns are
# decided by pre-LLM interceptors (planner_agent guards 3 and 8) that return
# create_reservation WITHOUT calling the model at all, so budget spent there
# buys little — while the failure being fixed (booking without checking
# availability first) is decided in the availability phase, which the model
# does own. Spending the budget where the model actually decides also stops
# "emit create_reservation" from dominating the dataset, which would push the
# model toward the very over-eagerness this fine-tune exists to remove.
GROUP_WEIGHTS = {
    "availability": 0.50,
    "booking": 0.20,
    "discovery": 0.20,
    "edge": 0.10,
}


@lru_cache(maxsize=None)
def load_phase_prompt(phase: str) -> str:
    """Return the system prompt for `phase`, exactly as the planner loads it.

    Mirrors planner_agent.load_prompt_for_phase: unknown/empty phases fall back
    to the Phase-1 (discovery) prompt.
    """
    filename = PHASE_PROMPT_FILES.get(phase or "discovery", PHASE_PROMPT_FILES["discovery"])
    return (AGENT_DIR / filename).read_text(encoding="utf-8")


def memory_inline(memory_state: Dict[str, Any]) -> str:
    """Serialise a memory state the way call_planner_llm does.

    Routed through the real ConversationMemory so that (a) None values and the
    internal ``intent`` key are dropped and (b) key order is production order.
    """
    mem = ConversationMemory()
    mem.update_from_planner(dict(memory_state))
    merged = mem.merge_into_context({})
    return json.dumps(merged.get("memory", {}), ensure_ascii=False)


def build_user_content(
    user_text: str,
    memory_state: Dict[str, Any],
    recent_results: Optional[List[Dict[str, Any]]] = None,
    customer_profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Reproduce planner_agent.call_planner_llm's `user_content` exactly."""
    results_inline = json.dumps(recent_results or [], ensure_ascii=False)
    profile_inline = json.dumps(customer_profile or {}, ensure_ascii=False)
    return (
        f"{user_text}\n\n"
        "--- Context (do not echo back) ---\n"
        f"Memory: {memory_inline(memory_state)}\n"
        f"Recent results: {results_inline}\n"
        f"Customer profile: {profile_inline}\n"
        "Respond ONLY with one valid JSON object."
    )


def build_sample(case: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a rule-engine case into a chat-format training record."""
    phase = case["phase"]
    plan = case["plan"]
    record = {
        "messages": [
            {"role": "system", "content": load_phase_prompt(phase)},
            {
                "role": "user",
                "content": build_user_content(
                    case["user"], case["memory"], case.get("recent_results"),
                ),
            },
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
        ],
        "meta": {
            "case": case["case"],
            "group": case["group"],
            "phase": phase,
            "plan": plan.get("plan"),
            "action": plan.get("action"),
            "memory_keys": sorted(k for k, v in case["memory"].items() if v is not None),
        },
    }
    return record


# Real entities from the seed JSON so the model learns real names/zones/cuisines.
class SeedCorpus:
    """Restaurant names, zones, cuisines and tags pulled from the seed JSON."""

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
        self.names = [r["unit_name"] for r in rows]
        self.zones = sorted({r["zone"] for r in rows if r.get("zone")})
        self.cuisines = sorted({c for r in rows for c in (r.get("cuisines") or [])})
        self.tags = sorted({t for r in rows for t in (r.get("tags") or [])})

    def restaurant(self, rng: random.Random) -> Dict[str, Any]:
        return rng.choice(self.rows)

    def recent_results(self, rng: random.Random, k: int = 3) -> List[Dict[str, Any]]:
        """Build a `recent_results` block shaped exactly like agent.py's.

        NOTE the ``"id": None``: agent.process_user_query builds these entries
        with ``r.get("id")``, but tool_calls.search_restaurants returns
        ``location_id`` — so at runtime this key is always null. Reproduced
        faithfully rather than "fixed" here; the agent is not touched by Phase 8.
        """
        picks = rng.sample(self.rows, k=min(k, len(self.rows)))
        return [
            {
                "id": None,
                "unit_name": r["unit_name"],
                "zone": r.get("zone"),
                "avg_price_per_person": r.get("avg_price_per_person"),
                "tags": r.get("tags", []),
                "cuisines": r.get("cuisines", []),
            }
            for r in picks
        ]


@lru_cache(maxsize=1)
def load_seed_corpus() -> SeedCorpus:
    if not SEED_DATA_PATH.exists():
        raise SystemExit(
            f"Seed data not found at {SEED_DATA_PATH}. "
            "The generator needs it for real restaurant names/zones/cuisines."
        )
    rows = json.loads(SEED_DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"Seed data at {SEED_DATA_PATH} is not a non-empty JSON list.")
    return SeedCorpus(rows)


# Template banks: written from the phase-prompt rules, independent of
# tests/eval/*.yaml.

# Only date forms date_utils.normalize_date_to_iso
# understands. Relative forms are intentionally kept RAW in the emitted args:
# resolving "tomorrow" to an ISO date at generation time would bake the
# generation date into the weights. Python normalises downstream, exactly as
# it does for the 4B model today.
DATE_PHRASES: List[str] = [
    "tomorrow",
    "today",
    "day after tomorrow",
    "this friday",
    "this saturday",
    "next monday",
    "next tuesday",
    "next thursday",
    "next sunday",
    "coming wednesday",
    "12 Dec",
    "3 Jan",
    "21 Feb",
    "9 Mar",
    "27 Apr",
    "16 Jun",
    "8th September",
    "23rd October",
    "Dec 12",
    "Jan 3",
    "Feb 21",
    "Sept 8",
    "12/12",
    "3/1",
    "21/2",
    "16-6",
]

# ISO dates used for memory slots. A wide, generation-date-independent pool so
# the model learns "copy memory.date", never a specific calendar day.
ISO_DATE_POOL: List[str] = [
    f"{year}-{month:02d}-{day:02d}"
    for year in (2026, 2027)
    for month in (1, 3, 5, 6, 8, 9, 11, 12)
    for day in (4, 11, 17, 23, 28)
]

# (user phrasing, normalised HH:MM). Times ARE normalised in the emitted args:
# the mapping is calendar-independent, the Phase-2 prompt asks for HH:MM, and
# memory always stores HH:MM.
TIME_PHRASES: List[Tuple[str, str]] = [
    ("7pm", "19:00"),
    ("7:30pm", "19:30"),
    ("8 pm", "20:00"),
    ("8:15pm", "20:15"),
    ("9pm", "21:00"),
    ("12:30pm", "12:30"),
    ("1pm", "13:00"),
    ("2:15pm", "14:15"),
    ("19:00", "19:00"),
    ("19:45", "19:45"),
    ("20:30", "20:30"),
    ("21:15", "21:15"),
    ("13:15", "13:15"),
]

# Phrasings that both the model and the Python guards
# (planner_agent.extract_party_size_from_text) resolve identically.
# Standalone clauses — drop into a sentence as-is ("18 Nov, table for two").
PARTY_PHRASES: List[Tuple[str, int]] = [
    ("table for two", 2),
    ("table for four", 4),
    ("party of three", 3),
    ("party of six", 6),
    ("party of eight", 8),
    ("we are 5", 5),
    ("we are 7", 7),
    ("for 2 people", 2),
    ("for 4 people", 4),
    ("6 guests", 6),
    ("9 people", 9),
    ("10 guests", 10),
    ("for 3 people", 3),
    ("table for 12", 12),
]

# Bare counts — for carrier templates ("make it {count}") where a full clause
# would read as broken English ("it'll be we are 7").
PARTY_COUNT_PHRASES: List[Tuple[str, int]] = [
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("six", 6),
    ("eight", 8),
    ("4 people", 4),
    ("6 people", 6),
    ("7 guests", 7),
    ("9 guests", 9),
    ("10 people", 10),
]

EMAIL_LOCALS = [
    "nina.rao", "arjun.m", "devika.s", "farhan.k", "meera93", "rohit.b",
    "tanvi.p", "yusuf.a", "lakshmi.n", "kabir.d", "shreya.v", "imran.q",
    "anita.g", "vikram.t", "priyaa", "noor.h",
]
EMAIL_DOMAINS = ["example.com", "example.org", "example.net", "mailbox.example"]

# Malformed addresses used by the invalid-email edge case. Deliberately without
# any '@' so no regex anywhere in the stack can mistake them for an address.
INVALID_EMAILS = [
    "nina.rao at example dot com",
    "arjun.m(at)example.org",
    "devika.s.example.com",
    "just my usual address",
    "farhan dot k at mailbox",
]

SEATING_PHRASES: List[Tuple[str, str]] = [
    ("outdoor seating please", "outdoor"),
    ("we'd prefer the terrace", "outdoor"),
    ("somewhere outside if possible", "outdoor"),
    ("rooftop if you have it", "outdoor"),
    ("indoor is better for us", "indoor"),
    ("inside please", "indoor"),
    ("indoor seating", "indoor"),
]

# Index 0 of each bank is the exact wording from the phase prompt / the
# deterministic guard; the rest are variants that keep the same key substrings
# the eval scorer matches on ("date", "guests"/"how many", "email", "confirm",
# "area"). Canonical is emitted most of the time (see reply()).
REPLY_BANKS: Dict[str, List[str]] = {
    "discovery_more_filters": [
        "Would you like to add any other preferences such as area, budget, rating, or tags?",
        "Any other preferences — area, budget, rating or a particular vibe?",
        "Would you like to narrow it down by area, budget or rating?",
    ],
    "discovery_area": [
        "Which area would you like to dine in?",
        "Which area should I search in?",
        "Which part of town would you like to dine in?",
    ],
    "discovery_clarify": [
        "Could you clarify what you'd like to eat or where you'd like to dine?",
        "Could you tell me what you'd like to eat or which area you prefer?",
        "Happy to help with dining — what would you like to eat, or where?",
    ],
    "availability_date": [
        "What date should I check?",
        "Which date should I check availability for?",
        "What day would you like me to check?",
    ],
    "availability_party": [
        "For how many guests should I check availability?",
        "How many guests should I check availability for?",
        "For how many people should I check?",
    ],
    "availability_party_invalid": [
        "For how many guests should I check availability? Please enter a positive number.",
        "That party size isn't valid — for how many guests should I check availability?",
        "Please give a positive number of guests, and I'll check availability.",
    ],
    "availability_party_too_large": [
        "I can book up to 50 guests in one reservation. For how many guests should I check availability?",
        "That's above the 50-guest limit for a single booking — how many guests should I check for?",
        "For groups that large we book in parts. For how many guests should I check availability?",
    ],
    "availability_single_time": [
        "Which time would you prefer? Please pick a single time.",
        "Could you pick one time? Which time would you prefer?",
        "Which single time should I check for you?",
    ],
    "booking_email": [
        "Which email should I use for the reservation?",
        "Which email address should I use for the booking?",
        "What email should I put on the reservation?",
    ],
    "booking_email_invalid": [
        "That email doesn't look valid — which email should I use for the reservation?",
        "I couldn't read that email address. Which email should I use for the reservation?",
        "That doesn't look like a valid email. Which email should I use?",
    ],
    "booking_confirm": [
        "Shall I confirm your reservation now?",
        "Shall I go ahead and confirm this reservation?",
        "Everything looks set — shall I confirm the reservation now?",
    ],
    "booking_time": [
        "What time should I book the reservation for?",
        "Which time should I book the reservation for?",
        "What time would you like the reservation?",
    ],
    "booking_date": [
        "Which date should I use for the reservation?",
        "What date should I book the reservation for?",
        "Which day should I put on the reservation?",
    ],
    "booking_party": [
        "For how many guests should I make the reservation?",
        "How many guests should the reservation be for?",
        "For how many people should I make the reservation?",
    ],
    "booking_declined": [
        "No problem — I won't book anything yet. Just say the word when you're ready.",
        "Sure, nothing is booked. Let me know whenever you'd like to confirm.",
        "Understood — I'll hold off on the reservation until you're ready.",
    ],
}

CUISINE_ONLY_TEMPLATES = [
    "craving {cuisine} tonight",
    "I feel like {cuisine} today",
    "{cuisine} sounds good",
    "looking for {cuisine} food",
    "we want {cuisine}",
    "how about some {cuisine}?",
    "{cuisine} please",
]

CUISINE_ZONE_TEMPLATES = [
    "{cuisine} in the {zone} zone",
    "show me {cuisine} spots in {zone}",
    "{cuisine} somewhere in {zone}",
    "we're in {zone} and want {cuisine}",
    "any {cuisine} around {zone}?",
    "{cuisine} restaurants, {zone} side",
]

CUISINE_ZONE_PRICE_TEMPLATES = [
    "{cuisine} in {zone} under {price}",
    "{cuisine} around {zone}, budget {price} per head",
    "looking for {cuisine} in {zone} below {price}",
    "{cuisine} spots in {zone} that stay under {price} a head",
]

ZONE_ONLY_TEMPLATES = [
    "what's good in {zone}?",
    "somewhere to eat in {zone}",
    "options in the {zone} zone please",
    "dinner spots around {zone}",
]

ZONE_PRICE_TEMPLATES = [
    "somewhere in {zone} under {price}",
    "{zone} area, keep it below {price} per person",
    "cheap eats in {zone}, max {price}",
]

ZONE_RATING_TEMPLATES = [
    "highly rated places in {zone}, at least {rating} stars",
    "{zone} zone but only {rating}+ rated",
    "anything above {rating} stars in {zone}?",
]

TAG_TEMPLATES = [
    "somewhere {tag} in {zone}",
    "{tag} places around {zone}",
    "we want a {tag} spot in the {zone} zone",
    "{tag} dining in {zone} please",
]

RECOMMEND_PLAIN_TEMPLATES = [
    "anything you'd recommend?",
    "recommend something",
    "what do you suggest?",
    "suggest a place",
    "surprise me with a suggestion",
]

RECOMMEND_QUERY_TEMPLATES: List[Tuple[str, str]] = [
    ("suggest somewhere for a big family dinner", "big family dinner"),
    ("recommend a spot for a quiet catch-up", "quiet catch-up"),
    ("what do you suggest for a celebration meal?", "celebration meal"),
    ("recommend somewhere for a work lunch", "work lunch"),
    ("suggest a place for a first date", "first date"),
    ("any suggestion for a lazy sunday brunch?", "lazy sunday brunch"),
]

NAMED_RESTAURANT_TEMPLATES = [
    "let's go with {name}",
    "{name} please",
    "I'd like {name}",
    "book us into {name}",
    "we'll take {name}",
    "check tables at {name}",
    "what about {name}?",
    "{name} works for us",
]

VAGUE_ZONE_TEMPLATES = [
    "anything good near me?",
    "{cuisine} places near me",
    "{cuisine} somewhere around here",
    "somewhere nearby please",
    "what's around here?",
    "restaurants close by",
]

OUT_OF_SCOPE_TEMPLATES = [
    "what's the weather like this weekend?",
    "can you book me a cab?",
    "who won the match last night?",
    "do you sell gift cards for flights?",
    "hello there",
    "tell me a joke",
]

# Availability-phase user utterances. {date}/{party}/{time} are filled with
# phrases from the banks above.
AVAIL_DATE_PARTY_TIME_TEMPLATES = [
    "{date} at {time}, {party}",
    "{party} on {date} around {time}",
    "we'd like {date}, {time}, {party}",
    "{date} {time} — {party}",
    "{party}, {date} at {time} please",
]

AVAIL_DATE_PARTY_TEMPLATES = [
    "{date}, {party}",
    "{party} on {date}",
    "we're thinking {date} — {party}",
    "{date} please, {party}",
    "{party} coming {date}",
]

AVAIL_TIME_ONLY_TEMPLATES = [
    "can we do {time}?",
    "{time} would suit us",
    "let's try {time}",
    "how about {time}?",
    "is {time} free?",
    "{time} then",
]

# Standalone-clause carriers (take a PARTY_PHRASES entry verbatim). As with
# dates, no bare "{party}" — it would collide with common one-line turns.
AVAIL_PARTY_ONLY_TEMPLATES = [
    "{party} please",
    "it's {party}",
    "{party}, thanks",
]

# Bare-count carriers (take a PARTY_COUNT_PHRASES entry).
AVAIL_PARTY_COUNT_TEMPLATES = [
    "it'll be {count}",
    "make it {count}",
    "{count} in total",
    "we'll be {count}",
    "just {count}",
]

# Terse-but-distinct: a bare "{date}" would collide verbatim with common
# single-word turns, so every carrier adds at least one word.
AVAIL_DATE_ONLY_TEMPLATES = [
    "{date} works",
    "we're thinking {date}",
    "how about {date}?",
    "{date} if that works",
    "let's say {date}",
]

AVAIL_NO_INFO_TEMPLATES = [
    "we'd like to come in for dinner",
    "planning a meal there",
    "want to eat there soon",
    "thinking of visiting",
    "we'd like a table",
]

AVAIL_SEATING_REQUEST_TEMPLATES = [
    "what seating do they have?",
    "show me the seating options",
    "list the seating sections",
    "which seating areas are there?",
    "can you repeat the seating options?",
]

AVAIL_TWO_TIME_TEMPLATES = [
    "{time_a} or {time_b}?",
    "either {time_a} or {time_b}",
    "{time_a} or {time_b} — whichever is open",
]

AVAIL_BAD_PARTY_TEMPLATES = [
    "for 0 people",
    "party of 0",
    "-2 people",
    "- 3 guests",
    "0 guests please",
]

AVAIL_HUGE_PARTY_TEMPLATES = [
    "we are 80 people",
    "for 120 guests",
    "party of 200",
]

BOOKING_NUDGE_TEMPLATES = [
    "let's finish the booking",
    "how do we complete this?",
    "go ahead and set it up",
    "what else do you need?",
    "please get it booked",
]

BOOKING_EMAIL_TEMPLATES = [
    "{email}",
    "my email is {email}",
    "use {email}",
    "here is my email: {email}",
    "you can reach me at {email}",
]

BOOKING_EMAIL_SEATING_TEMPLATES = [
    "{email}, {seating}",
    "{email} — {seating}",
    "use {email} and {seating}",
]

BOOKING_CONFIRM_TEMPLATES = [
    "yes",
    "yep",
    "please confirm",
    "book it",
    "go ahead",
    "sounds good",
    "confirm it",
    "yes please",
    "do it",
]

BOOKING_DECLINE_TEMPLATES = [
    "hold off for now",
    "not right now",
    "I'll decide later",
    "let me check with the others first",
    "maybe another day",
    "we'll skip it for now",
]

BOOKING_SEATING_ONLY_TEMPLATES = [
    "{seating}",
    "also, {seating}",
    "one more thing — {seating}",
]


def reply(rng: random.Random, bank: str) -> Dict[str, Any]:
    """Build a `plan=reply` decision.

    The canonical prompt wording is emitted ~70% of the time; the remainder
    uses a variant so the model learns the decision, not one sentence.
    """
    options = REPLY_BANKS[bank]
    text = options[0] if rng.random() < 0.7 else rng.choice(options[1:])
    # A raise, not an assert: `python -O` strips asserts, and a reply outside
    # the validator's bounds is silently rejected at runtime.
    if not REPLY_MIN_CHARS <= len(text) <= REPLY_MAX_CHARS:
        raise ValueError(f"reply length {len(text)} out of bounds: {text!r}")
    return {"plan": "reply", "reply": text}


def execute(action: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if action not in GENERATED_ACTIONS:
        raise ValueError(f"action not permitted by the generator: {action}")
    return {"plan": "execute", "action": action, "args": args}


def memory_state(phase: str, **slots: Any) -> Dict[str, Any]:
    """Assemble a memory snapshot; None-valued slots are simply absent."""
    state: Dict[str, Any] = {"phase": phase}
    state.update({k: v for k, v in slots.items() if v is not None})
    return state


def party_utterance(rng: random.Random) -> Tuple[str, int]:
    """A user message that supplies only a party size, plus the parsed count.

    Draws from either the standalone-clause bank or the bare-count bank so the
    sentence always reads naturally.
    """
    if rng.random() < 0.5:
        phrase, size = rng.choice(PARTY_PHRASES)
        return rng.choice(AVAIL_PARTY_ONLY_TEMPLATES).format(party=phrase), size
    phrase, size = rng.choice(PARTY_COUNT_PHRASES)
    return rng.choice(AVAIL_PARTY_COUNT_TEMPLATES).format(count=phrase), size


def pick_email(rng: random.Random) -> str:
    return f"{rng.choice(EMAIL_LOCALS)}@{rng.choice(EMAIL_DOMAINS)}"


def pick_price(rng: random.Random) -> int:
    return rng.choice([400, 500, 600, 700, 800, 900, 1000, 1200])


def case(name: str, group: str, phase: str, memory: Dict[str, Any],
         user: str, plan: Dict[str, Any],
         recent_results: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "case": name,
        "group": group,
        "phase": phase,
        "memory": memory,
        "user": user,
        "plan": plan,
        "recent_results": recent_results,
    }


# RULE ENGINE — discovery phase. Encodes the Phase-1 prompt's decision table
# plus the deterministic guards planner_agent applies on top of it.

def d_cuisine_only(rng: random.Random) -> Dict[str, Any]:
    """cuisine and nothing else -> ask for more filters (never search)."""
    corpus = load_seed_corpus()
    cuisine = rng.choice(corpus.cuisines)
    user = rng.choice(CUISINE_ONLY_TEMPLATES).format(cuisine=cuisine)
    return case("d_cuisine_only", "discovery", "discovery",
                memory_state("discovery"), user,
                reply(rng, "discovery_more_filters"))


def d_cuisine_zone(rng: random.Random) -> Dict[str, Any]:
    """cuisine + at least one more filter -> search."""
    corpus = load_seed_corpus()
    cuisine = rng.choice(corpus.cuisines)
    zone = rng.choice(corpus.zones)
    args: Dict[str, Any] = {"cuisine": cuisine, "zone": zone}
    if rng.random() < 0.45:
        price = pick_price(rng)
        user = rng.choice(CUISINE_ZONE_PRICE_TEMPLATES).format(
            cuisine=cuisine, zone=zone, price=price)
        args["max_price"] = price
    else:
        user = rng.choice(CUISINE_ZONE_TEMPLATES).format(cuisine=cuisine, zone=zone)
    return case("d_cuisine_zone", "discovery", "discovery",
                memory_state("discovery"), user,
                execute("search_restaurants_by_filters", args))


def d_carry_over_zone(rng: random.Random) -> Dict[str, Any]:
    """Cuisine already in memory, user adds a zone -> search with BOTH.

    Mirrors the Phase-1 filter carry-over: a follow-up turn must not drop the
    filter the user gave earlier.
    """
    corpus = load_seed_corpus()
    cuisine = rng.choice(corpus.cuisines)
    zone = rng.choice(corpus.zones)
    user = rng.choice([
        "{zone} please", "make it {zone}", "in {zone}", "{zone} side",
        "somewhere in {zone}",
    ]).format(zone=zone)
    return case("d_carry_over_zone", "discovery", "discovery",
                memory_state("discovery", cuisine=cuisine), user,
                execute("search_restaurants_by_filters",
                        {"cuisine": cuisine, "zone": zone}))


def d_filters_no_cuisine(rng: random.Random) -> Dict[str, Any]:
    """Filters present but no cuisine -> search with what we have."""
    corpus = load_seed_corpus()
    zone = rng.choice(corpus.zones)
    roll = rng.random()
    if roll < 0.34:
        user = rng.choice(ZONE_ONLY_TEMPLATES).format(zone=zone)
        args: Dict[str, Any] = {"zone": zone}
    elif roll < 0.67:
        price = pick_price(rng)
        user = rng.choice(ZONE_PRICE_TEMPLATES).format(zone=zone, price=price)
        args = {"zone": zone, "max_price": price}
    else:
        rating = rng.choice([4, 4.2, 4.5])
        user = rng.choice(ZONE_RATING_TEMPLATES).format(zone=zone, rating=rating)
        args = {"zone": zone, "min_rating": rating}
    return case("d_filters_no_cuisine", "discovery", "discovery",
                memory_state("discovery"), user,
                execute("search_restaurants_by_filters", args))


def d_tag_search(rng: random.Random) -> Dict[str, Any]:
    """A vibe/tag plus a zone -> search carrying the tag."""
    corpus = load_seed_corpus()
    tag = rng.choice(corpus.tags)
    zone = rng.choice(corpus.zones)
    user = rng.choice(TAG_TEMPLATES).format(tag=tag.replace("-", " "), zone=zone)
    return case("d_tag_search", "discovery", "discovery",
                memory_state("discovery"), user,
                execute("search_restaurants_by_filters", {"zone": zone, "tag": tag}))


def d_recommend(rng: random.Random) -> Dict[str, Any]:
    """Suggestion request with no filters -> recommend_venues."""
    if rng.random() < 0.5:
        user = rng.choice(RECOMMEND_PLAIN_TEMPLATES)
        args: Dict[str, Any] = {}
    else:
        user, query = rng.choice(RECOMMEND_QUERY_TEMPLATES)
        args = {"query": query}
    return case("d_recommend", "discovery", "discovery",
                memory_state("discovery"), user,
                execute("recommend_venues", args))


def d_named_restaurant(rng: random.Random) -> Dict[str, Any]:
    """A specific restaurant is named -> get_seating_labels (phase 1 -> 2)."""
    corpus = load_seed_corpus()
    restaurant = corpus.restaurant(rng)
    name = restaurant["unit_name"]
    user = rng.choice(NAMED_RESTAURANT_TEMPLATES).format(name=name)
    # Half of these follow a search, so the model sees a populated
    # `Recent results` block as well as an empty one.
    results = corpus.recent_results(rng) if rng.random() < 0.5 else None
    return case("d_named_restaurant", "discovery", "discovery",
                memory_state("discovery"), user,
                execute("get_seating_labels", {"restaurant": name}),
                recent_results=results)


# RULE ENGINE — availability phase. CORE INVARIANT: the planner NEVER emits
# create_reservation here; a stated time means check_availability. This is the
# 1.7B's dominant measured failure (8 of 13 failed conversations) and the
# reason this dataset exists.

def _avail_restaurant(rng: random.Random) -> str:
    return load_seed_corpus().restaurant(rng)["unit_name"]


def a_date_party_time(rng: random.Random) -> Dict[str, Any]:
    """Date + party + time in one turn -> check_availability (NOT a booking)."""
    name = _avail_restaurant(rng)
    date_phrase = rng.choice(DATE_PHRASES)
    party_phrase, party = rng.choice(PARTY_PHRASES)
    time_phrase, time_hhmm = rng.choice(TIME_PHRASES)
    user = rng.choice(AVAIL_DATE_PARTY_TIME_TEMPLATES).format(
        date=date_phrase, party=party_phrase, time=time_phrase)
    return case("a_date_party_time", "availability", "availability",
                memory_state("availability", restaurant=name), user,
                execute("check_availability", {
                    "restaurant": name,
                    "date": date_phrase,
                    "party_size": party,
                    "time": time_hhmm,
                }))


def a_date_party_no_time(rng: random.Random) -> Dict[str, Any]:
    """Date + party, no time -> check_availability with time null."""
    name = _avail_restaurant(rng)
    date_phrase = rng.choice(DATE_PHRASES)
    party_phrase, party = rng.choice(PARTY_PHRASES)
    user = rng.choice(AVAIL_DATE_PARTY_TEMPLATES).format(
        date=date_phrase, party=party_phrase)
    return case("a_date_party_no_time", "availability", "availability",
                memory_state("availability", restaurant=name), user,
                execute("check_availability", {
                    "restaurant": name,
                    "date": date_phrase,
                    "party_size": party,
                    "time": None,
                }))


def a_time_completes(rng: random.Random) -> Dict[str, Any]:
    """Date + party already in memory, user adds a time -> check_availability.

    The single most important case in the dataset: the 1.7B currently jumps
    straight to create_reservation here.
    """
    name = _avail_restaurant(rng)
    iso_date = rng.choice(ISO_DATE_POOL)
    _, party = rng.choice(PARTY_PHRASES)
    time_phrase, time_hhmm = rng.choice(TIME_PHRASES)
    user = rng.choice(AVAIL_TIME_ONLY_TEMPLATES).format(time=time_phrase)
    return case("a_time_completes", "availability", "availability",
                memory_state("availability", restaurant=name, date=iso_date,
                             party_size=party),
                user,
                execute("check_availability", {
                    "restaurant": name,
                    "date": iso_date,
                    "party_size": party,
                    "time": time_hhmm,
                }))


def a_party_completes(rng: random.Random) -> Dict[str, Any]:
    """Date in memory, user supplies the party size -> check_availability."""
    name = _avail_restaurant(rng)
    iso_date = rng.choice(ISO_DATE_POOL)
    user, party = party_utterance(rng)
    return case("a_party_completes", "availability", "availability",
                memory_state("availability", restaurant=name, date=iso_date),
                user,
                execute("check_availability", {
                    "restaurant": name,
                    "date": iso_date,
                    "party_size": party,
                    "time": None,
                }))


def a_date_completes(rng: random.Random) -> Dict[str, Any]:
    """Party in memory, user supplies the date -> check_availability."""
    name = _avail_restaurant(rng)
    _, party = rng.choice(PARTY_PHRASES)
    date_phrase = rng.choice(DATE_PHRASES)
    user = rng.choice(AVAIL_DATE_ONLY_TEMPLATES).format(date=date_phrase)
    return case("a_date_completes", "availability", "availability",
                memory_state("availability", restaurant=name, party_size=party),
                user,
                execute("check_availability", {
                    "restaurant": name,
                    "date": date_phrase,
                    "party_size": party,
                    "time": None,
                }))


def a_missing_date(rng: random.Random) -> Dict[str, Any]:
    """No date anywhere -> ask for the date FIRST (fixed collection order)."""
    name = _avail_restaurant(rng)
    if rng.random() < 0.5:
        user, _ = party_utterance(rng)
        mem = memory_state("availability", restaurant=name)
    else:
        user = rng.choice(AVAIL_NO_INFO_TEMPLATES)
        _, party = rng.choice(PARTY_PHRASES)
        mem = memory_state("availability", restaurant=name, party_size=party)
    return case("a_missing_date", "availability", "availability", mem, user,
                reply(rng, "availability_date"))


def a_missing_party(rng: random.Random) -> Dict[str, Any]:
    """Date known, party unknown, nothing new supplied -> ask party size."""
    name = _avail_restaurant(rng)
    iso_date = rng.choice(ISO_DATE_POOL)
    user = rng.choice(AVAIL_NO_INFO_TEMPLATES)
    return case("a_missing_party", "availability", "availability",
                memory_state("availability", restaurant=name, date=iso_date),
                user, reply(rng, "availability_party"))


def a_seating_request(rng: random.Random) -> Dict[str, Any]:
    """"show/list/repeat the seating options" -> get_seating_labels."""
    name = _avail_restaurant(rng)
    iso_date = rng.choice(ISO_DATE_POOL) if rng.random() < 0.5 else None
    user = rng.choice(AVAIL_SEATING_REQUEST_TEMPLATES)
    return case("a_seating_request", "availability", "availability",
                memory_state("availability", restaurant=name, date=iso_date),
                user, execute("get_seating_labels", {"restaurant": name}))


# NOTE — no "slot accepted in the availability phase" case exists on purpose.
# The Phase-2 prompt has a rule for it (slot picked + email missing -> ask for
# the email), but the memory state it needs is unreachable: a memory holding a
# `time` while phase is still "availability" means check_availability came back
# UNAVAILABLE — update_phase_after_check_availability advances to "booking"
# whenever the slot was free. Training "the user accepts, so ask for their
# email" on that state would teach accept-without-verify in the one phase whose
# entire purpose is verification. The legitimate ask-for-email decision lives in
# the booking phase and is covered by b_missing_email.


# RULE ENGINE — booking phase.

def _booking_memory(rng: random.Random, *, email: Optional[str] = None,
                    seating: Optional[str] = None,
                    drop: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """A booking-phase memory with every slot filled, minus `drop`."""
    name = _avail_restaurant(rng)
    slots: Dict[str, Any] = {
        "restaurant": name,
        "date": rng.choice(ISO_DATE_POOL),
        "time": rng.choice(TIME_PHRASES)[1],
        "party_size": rng.choice(PARTY_PHRASES)[1],
        "customer_email": email,
        "seating_pref": seating,
    }
    if drop:
        slots[drop] = None
    return memory_state("booking", **slots), name


def b_missing_email(rng: random.Random) -> Dict[str, Any]:
    """No email on file -> ask for it before anything else."""
    mem, _ = _booking_memory(rng)
    user = rng.choice(BOOKING_NUDGE_TEMPLATES)
    return case("b_missing_email", "booking", "booking", mem, user,
                reply(rng, "booking_email"))


def b_email_supplied(rng: random.Random) -> Dict[str, Any]:
    """Email arrives and every other field is known -> ask to confirm."""
    mem, _ = _booking_memory(rng)
    email = pick_email(rng)
    user = rng.choice(BOOKING_EMAIL_TEMPLATES).format(email=email)
    return case("b_email_supplied", "booking", "booking", mem, user,
                reply(rng, "booking_confirm"))


def b_email_with_seating(rng: random.Random) -> Dict[str, Any]:
    """Email + a seating preference in one message -> still ask to confirm."""
    mem, _ = _booking_memory(rng)
    email = pick_email(rng)
    seating_phrase, _ = rng.choice(SEATING_PHRASES)
    user = rng.choice(BOOKING_EMAIL_SEATING_TEMPLATES).format(
        email=email, seating=seating_phrase)
    return case("b_email_with_seating", "booking", "booking", mem, user,
                reply(rng, "booking_confirm"))


def b_seating_supplied(rng: random.Random) -> Dict[str, Any]:
    """Seating preference stated once the email is known -> ask to confirm."""
    mem, _ = _booking_memory(rng, email=pick_email(rng))
    seating_phrase, _ = rng.choice(SEATING_PHRASES)
    user = rng.choice(BOOKING_SEATING_ONLY_TEMPLATES).format(seating=seating_phrase)
    return case("b_seating_supplied", "booking", "booking", mem, user,
                reply(rng, "booking_confirm"))


def b_confirm(rng: random.Random) -> Dict[str, Any]:
    """Every field present AND the user confirms -> create_reservation.

    This is the ONLY case in the generator that emits create_reservation, and
    it is unreachable without customer_email in memory.
    """
    seating = rng.choice([None, "outdoor", "indoor"])
    mem, name = _booking_memory(rng, email=pick_email(rng), seating=seating)
    user = rng.choice(BOOKING_CONFIRM_TEMPLATES)
    args: Dict[str, Any] = {
        "restaurant": mem["restaurant"],
        "date": mem["date"],
        "time": mem["time"],
        "party_size": mem["party_size"],
        "customer_email": mem["customer_email"],
    }
    # Only carry seating when the user actually expressed one; location_id is
    # never emitted — dispatch_tool resolves it from the restaurant name, and
    # memory.location_id is never populated at runtime.
    if mem.get("seating_pref"):
        args["seating_pref"] = mem["seating_pref"]
    return case("b_confirm", "booking", "booking", mem, user,
                execute("create_reservation", args))


def b_declined(rng: random.Random) -> Dict[str, Any]:
    """User backs out with everything on file -> reply, and fire NO tool."""
    mem, _ = _booking_memory(rng, email=pick_email(rng))
    user = rng.choice(BOOKING_DECLINE_TEMPLATES)
    return case("b_declined", "booking", "booking", mem, user,
                reply(rng, "booking_declined"))


def b_missing_time(rng: random.Random) -> Dict[str, Any]:
    mem, _ = _booking_memory(rng, email=pick_email(rng), drop="time")
    user = rng.choice(BOOKING_NUDGE_TEMPLATES)
    return case("b_missing_time", "booking", "booking", mem, user,
                reply(rng, "booking_time"))


def b_missing_date(rng: random.Random) -> Dict[str, Any]:
    mem, _ = _booking_memory(rng, email=pick_email(rng), drop="date")
    user = rng.choice(BOOKING_NUDGE_TEMPLATES)
    return case("b_missing_date", "booking", "booking", mem, user,
                reply(rng, "booking_date"))


def b_missing_party(rng: random.Random) -> Dict[str, Any]:
    mem, _ = _booking_memory(rng, email=pick_email(rng), drop="party_size")
    user = rng.choice(BOOKING_NUDGE_TEMPLATES)
    return case("b_missing_party", "booking", "booking", mem, user,
                reply(rng, "booking_party"))


# RULE ENGINE — edge guards. Each mirrors a deterministic guard in
# planner_agent, so the model agrees with the Python layer instead of fighting it.

def e_party_zero_or_negative(rng: random.Random) -> Dict[str, Any]:
    """party_size 0 or negative -> re-ask, never store, never execute."""
    name = _avail_restaurant(rng)
    iso_date = rng.choice(ISO_DATE_POOL) if rng.random() < 0.6 else None
    user = rng.choice(AVAIL_BAD_PARTY_TEMPLATES)
    return case("e_party_zero_or_negative", "edge", "availability",
                memory_state("availability", restaurant=name, date=iso_date),
                user, reply(rng, "availability_party_invalid"))


def e_party_too_large(rng: random.Random) -> Dict[str, Any]:
    """party_size above the accepted 1-50 range -> re-ask."""
    name = _avail_restaurant(rng)
    iso_date = rng.choice(ISO_DATE_POOL)
    user = rng.choice(AVAIL_HUGE_PARTY_TEMPLATES)
    return case("e_party_too_large", "edge", "availability",
                memory_state("availability", restaurant=name, date=iso_date),
                user, reply(rng, "availability_party_too_large"))


def e_ambiguous_time(rng: random.Random) -> Dict[str, Any]:
    """Two candidate times -> ask the user to pick one.

    The one deliberate exception to "a time in the availability phase means
    check_availability": an ambiguous pair cannot be checked. The unit tests
    exclude this case by name from that invariant.
    """
    name = _avail_restaurant(rng)
    iso_date = rng.choice(ISO_DATE_POOL)
    _, party = rng.choice(PARTY_PHRASES)
    (time_a, _), (time_b, _) = rng.sample(TIME_PHRASES, 2)
    user = rng.choice(AVAIL_TWO_TIME_TEMPLATES).format(time_a=time_a, time_b=time_b)
    return case("e_ambiguous_time", "edge", "availability",
                memory_state("availability", restaurant=name, date=iso_date,
                             party_size=party),
                user, reply(rng, "availability_single_time"))


def e_invalid_email(rng: random.Random) -> Dict[str, Any]:
    """Unparseable email -> re-ask; never book with it."""
    mem, _ = _booking_memory(rng)
    user = rng.choice(INVALID_EMAILS)
    return case("e_invalid_email", "edge", "booking", mem, user,
                reply(rng, "booking_email_invalid"))


def e_vague_zone(rng: random.Random) -> Dict[str, Any]:
    """"near me" is not a zone -> ask for a real area instead of searching."""
    corpus = load_seed_corpus()
    template = rng.choice(VAGUE_ZONE_TEMPLATES)
    user = template.format(cuisine=rng.choice(corpus.cuisines)) if "{cuisine}" in template else template
    return case("e_vague_zone", "edge", "discovery",
                memory_state("discovery"), user, reply(rng, "discovery_area"))


def e_out_of_scope(rng: random.Random) -> Dict[str, Any]:
    """Off-topic or contentless -> the Phase-1 failsafe clarification."""
    user = rng.choice(OUT_OF_SCOPE_TEMPLATES)
    return case("e_out_of_scope", "edge", "discovery",
                memory_state("discovery"), user, reply(rng, "discovery_clarify"))


# Case registry. Weights are within-group; group sizes come from GROUP_WEIGHTS
# and are allocated deterministically, so `--n` splits reproducibly.

CaseBuilder = Callable[[random.Random], Dict[str, Any]]

CASE_REGISTRY: Dict[str, List[Tuple[CaseBuilder, float]]] = {
    "discovery": [
        (d_cuisine_only, 0.15),
        (d_cuisine_zone, 0.27),
        (d_carry_over_zone, 0.10),
        (d_filters_no_cuisine, 0.14),
        (d_tag_search, 0.10),
        (d_recommend, 0.10),
        (d_named_restaurant, 0.14),
    ],
    "availability": [
        (a_date_party_time, 0.24),
        (a_date_party_no_time, 0.14),
        # +0.04 absorbed from the dropped slot-accepted case: it goes to the
        # anti-premature-booking core rather than to another reply case.
        (a_time_completes, 0.26),
        (a_party_completes, 0.11),
        (a_date_completes, 0.09),
        (a_missing_date, 0.06),
        (a_missing_party, 0.05),
        (a_seating_request, 0.05),
    ],
    # b_confirm is down from 0.55: it is the case the interceptors own most
    # completely (every phrase in BOOKING_CONFIRM_TEMPLATES is caught by guard 3
    # or 8 before the LLM is called). It stays represented — the guards do miss
    # some phrasings, e.g. "that all looks right, go ahead", where the model has
    # to produce the call itself — just no longer dominant.
    # The cases the model genuinely decides (missing field, decline, seating)
    # absorb the difference. Note b_email_supplied/-with_seating teach "an email
    # arrived, do NOT book yet, ask first", which reinforces the goal.
    "booking": [
        (b_confirm, 0.25),
        (b_email_supplied, 0.13),
        (b_email_with_seating, 0.07),
        (b_seating_supplied, 0.07),
        (b_missing_email, 0.15),
        (b_declined, 0.14),
        (b_missing_time, 0.07),
        (b_missing_date, 0.06),
        (b_missing_party, 0.06),
    ],
    "edge": [
        (e_party_zero_or_negative, 0.24),
        (e_party_too_large, 0.12),
        (e_ambiguous_time, 0.18),
        (e_invalid_email, 0.20),
        (e_vague_zone, 0.14),
        (e_out_of_scope, 0.12),
    ],
}


def allocate_group_counts(n: int) -> Dict[str, int]:
    """Split `n` across groups by GROUP_WEIGHTS, largest-remainder rounding."""
    raw = {g: n * w for g, w in GROUP_WEIGHTS.items()}
    counts = {g: int(v) for g, v in raw.items()}
    remainder = n - sum(counts.values())
    # Hand out the leftovers to the largest fractional parts (ties by name for
    # determinism).
    order = sorted(raw, key=lambda g: (-(raw[g] - counts[g]), g))
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def generate_cases(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Build `n` rule-engine cases, balanced across groups."""
    counts = allocate_group_counts(n)
    cases: List[Dict[str, Any]] = []
    for group in sorted(counts):
        builders = [b for b, _ in CASE_REGISTRY[group]]
        weights = [w for _, w in CASE_REGISTRY[group]]
        for _ in range(counts[group]):
            builder = rng.choices(builders, weights=weights, k=1)[0]
            cases.append(builder(rng))
    rng.shuffle(cases)
    return cases


# Validation — fail loudly rather than train on a broken target.

def validate_record(record: Dict[str, Any]) -> None:
    """Assert one record satisfies every contract the runtime enforces."""
    messages = record["messages"]
    roles = [m["role"] for m in messages]
    if roles != ["system", "user", "assistant"]:
        raise ValueError(f"unexpected role sequence: {roles}")

    plan = json.loads(messages[2]["content"])
    if plan.get("plan") not in ("reply", "execute"):
        raise ValueError(f"invalid plan: {plan!r}")

    if plan["plan"] == "reply":
        text = plan.get("reply")
        if not isinstance(text, str) or not (REPLY_MIN_CHARS <= len(text) <= REPLY_MAX_CHARS):
            raise ValueError(f"reply length out of bounds: {text!r}")
        return

    action = plan.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"action not in the planner whitelist: {action!r}")
    if action not in GENERATED_ACTIONS:
        raise ValueError(f"action outside the generator's scope: {action!r}")
    if not isinstance(plan.get("args"), dict):
        raise ValueError("args must be an object")

    if action == "create_reservation":
        # The invariant this dataset exists to teach.
        if record["meta"]["phase"] != "booking":
            raise ValueError("create_reservation outside the booking phase")
        if not plan["args"].get("customer_email"):
            raise ValueError("create_reservation without customer_email")


def split_train_val(
    records: List[Dict[str, Any]], val_split: float, rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split by unique user content so no user message spans both files.

    Template banks legitimately produce repeats; letting the same user turn sit
    in train and val would make the val loss meaningless.
    """
    if val_split <= 0:
        return records, []

    by_user: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        by_user.setdefault(rec["messages"][1]["content"], []).append(rec)

    keys = sorted(by_user)
    rng.shuffle(keys)
    n_val_keys = max(1, round(len(keys) * val_split)) if len(keys) > 1 else 0
    val_keys = set(keys[:n_val_keys])

    train = [r for r in records if r["messages"][1]["content"] not in val_keys]
    val = [r for r in records if r["messages"][1]["content"] in val_keys]
    return train, val


def build_manifest(train: List[Dict[str, Any]], val: List[Dict[str, Any]],
                   args: argparse.Namespace) -> Dict[str, Any]:
    every = train + val

    def tally(key: Callable[[Dict[str, Any]], Any]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for rec in every:
            k = key(rec)
            out[str(k)] = out.get(str(k), 0) + 1
        return dict(sorted(out.items()))

    replies = sum(1 for r in every if r["meta"]["plan"] == "reply")
    executes = len(every) - replies

    return {
        "generator": "scripts/generate_training_data.py",
        "seed": args.seed,
        "requested": args.n,
        "total": len(every),
        "train": len(train),
        "val": len(val),
        "val_split": args.val_split,
        "unique_user_messages": len({r["messages"][1]["content"] for r in every}),
        "by_phase": tally(lambda r: r["meta"]["phase"]),
        "by_group": tally(lambda r: r["meta"]["group"]),
        "by_case": tally(lambda r: r["meta"]["case"]),
        "by_action": tally(lambda r: r["meta"]["action"] or "(reply)"),
        "plan_counts": {"reply": replies, "execute": executes},
        "reply_ratio": round(replies / len(every), 4) if every else 0.0,
        "execute_ratio": round(executes / len(every), 4) if every else 0.0,
    }


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def val_path_for(out: Path) -> Path:
    return out.with_name(f"{out.stem}_val{out.suffix}")


def generate_dataset(n: int, seed: int, val_split: float) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Full pipeline as a function so tests can call it without touching disk."""
    rng = random.Random(seed)
    records = [build_sample(c) for c in generate_cases(n, rng)]
    for rec in records:
        validate_record(rec)
    return split_train_val(records, val_split, rng)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic planner trajectories for QLoRA fine-tuning.",
    )
    p.add_argument("--n", type=int, default=3000,
                   help="Total samples to generate (train + val). Default: 3000.")
    p.add_argument("--out", default="data/planner_train.jsonl",
                   help="Training JSONL path. The val file is <stem>_val.jsonl.")
    p.add_argument("--val-split", type=float, default=0.05,
                   help="Fraction of unique user messages held out. Default: 0.05.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed — same seed + same --n gives identical output.")
    p.add_argument("--manifest", default=None,
                   help="Manifest path. Default: manifest.json beside --out.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        print("--n must be >= 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.val_split < 1.0:
        print("--val-split must be in [0.0, 1.0)", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    val_path = val_path_for(out_path)
    manifest_path = Path(args.manifest) if args.manifest else out_path.parent / "manifest.json"
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    train, val = generate_dataset(args.n, args.seed, args.val_split)

    write_jsonl(out_path, train)
    write_jsonl(val_path, val)

    manifest = build_manifest(train, val, args)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    print(f"train   : {len(train):>6}  -> {out_path}")
    print(f"val     : {len(val):>6}  -> {val_path}")
    print(f"manifest: {manifest_path}")
    print(f"  phases : {manifest['by_phase']}")
    print(f"  actions: {manifest['by_action']}")
    print(f"  reply/execute: {manifest['plan_counts']['reply']}/"
          f"{manifest['plan_counts']['execute']} "
          f"({manifest['reply_ratio']:.0%} / {manifest['execute_ratio']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
