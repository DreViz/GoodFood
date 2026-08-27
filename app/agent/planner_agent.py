import json
import logging
import requests
import os
import re
from typing import Optional
from app.agent.mock_llm import mock_planner_output
from app.agent.conversation_memory import ConversationMemory
from app.agent.llm_utils import strip_model_reasoning
from app.config import get_settings


logger = logging.getLogger(__name__)


def load_prompt_for_phase(phase: str) -> str:
    base = os.path.dirname(__file__)

    if phase in (None, "", "discovery"):
        file = "planner_prompt_phase1.txt"
    elif phase == "availability":
        file = "planner_prompt_phase2.txt"
    elif phase == "booking":
        file = "planner_prompt_phase3.txt"
    else:
        file = "planner_prompt_phase1.txt"

    path = os.path.join(base, file)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Shared conversation state for the agent flow (the frontend resets it via
# POST /agent/memory/reset). Model/endpoint settings come from app.config.
memory = ConversationMemory()


def repair_malformed_json(candidate: str) -> str:
    if not candidate:
        return candidate

    first_open = candidate.find("{")
    last_close = candidate.rfind("}")
    if first_open != -1 and last_close != -1:
        candidate = candidate[first_open:last_close + 1]

    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    candidate = re.sub(r",\s*,+", ",", candidate)
    candidate = re.sub(r"\n{2,}", "\n", candidate).strip()
    return candidate


def validate_planner_json(candidate: str) -> dict | None:
    if not candidate:
        return None

    candidate = repair_malformed_json(candidate)

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    if not isinstance(parsed, dict) or "plan" not in parsed:
        return None

    plan = parsed.get("plan")

    if plan not in ("reply", "execute"):
        return None

    if plan == "reply":
        reply_text = parsed.get("reply", "")
        if isinstance(reply_text, str) and 4 <= len(reply_text) <= 200:
            return parsed
        return None

    if plan == "execute":
        action = parsed.get("action")
        args = parsed.get("args", {}) or {}
        if not isinstance(action, str) or not isinstance(args, dict):
            return None

        allowed_actions = {
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
        }

        if action not in allowed_actions:
            return None

        return parsed

    return None


# These field cleaners run before memory updates to drop invalid or fabricated
# values (e.g. party_size=0, date="") the LLM may emit.

def safe_extract_party_size(value):
    """Accept a plausible party size (1-50); reject anything else."""
    if value is None:
        return None

    try:
        v = int(value)
        if v >= 1 and v <= 50:
            return v
    except Exception:
        pass

    return None


def safe_extract_date(value):
    """Normalize a user/LLM-supplied date to "YYYY-MM-DD", or None.

    Memory dates must always be ISO: slot_manager.parse_opening_hours only
    accepts ISO, so letting "tomorrow" or "18 Nov" through crashes every
    downstream availability check. All supported forms (relative terms,
    this/next <weekday>, DD Mon, Mon DD, DD/MM, ISO) are handled by
    normalize_date_to_iso in date_utils.
    """
    if not value:
        return None
    from app.api.utils.date_utils import normalize_date_to_iso
    return normalize_date_to_iso(value)



def safe_extract_time(value):
    """Normalize a time value to 24-hour HH:MM, falling back to the raw text.

    The planner prompt asks for HH:MM but qwen3:4b often emits "7:30pm" /
    "8pm". normalize_time handles those; if it can't (truly unknown), we keep
    the stripped original so downstream code can decide rather than silently
    dropping a value the user clearly provided.
    """
    if not value:
        return None
    from app.api.utils.date_utils import normalize_time
    return normalize_time(value) or str(value).strip()


# Word -> digit map for natural party-size phrases ("table for two" -> 2).
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def extract_party_size_from_text(text: str):
    """Deterministically pull a party size (1-50) out of free-form user text.

    Used by the collection guards and the modify interceptor, where we must
    know what the user said without trusting the LLM's extraction. Handles:

      - digits: "for 4", "party of 8", "we are 5", "table for 4", "4 people"
      - words:  "table for two", "party of eight"

    Rejects 0, negatives, and out-of-range values (returns None) so it cannot
    store a hallucinated size.
    """
    if not text:
        return None
    t = text.lower()

    # Word-number forms, only in a party-size context to avoid matching "one"
    # inside "someone" / "anyone". Context words: for|of|table|party|guests...
    m = re.search(
        r"\b(?:for|of|table|party|guests?|people|persons?|pax|we\s+are)\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        t,
    )
    if m and _WORD_NUMBERS.get(m.group(1)) is not None:
        return _WORD_NUMBERS[m.group(1)]
    # "... are <word> people" / "we were eight"
    m = re.search(
        r"\b(?:are|were|for|of)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:people|guests?|persons?)\b",
        t,
    )
    if m and _WORD_NUMBERS.get(m.group(1)) is not None:
        return _WORD_NUMBERS[m.group(1)]

    # Digit forms. Reject an explicitly-negative token ("-3 people") outright
    # so the guard re-asks instead of storing the LLM's sign-stripped 3.
    if re.search(r"-\s*\d+\s*(?:people|guests?|persons?|pax)", t):
        return None
    m = re.search(
        r"(?:for|of|party\s+of|table\s+for|we\s+are|guests?|pax|people|persons?)\s*"
        r"(\d{1,2})\b",
        t,
    )
    if m:
        return safe_extract_party_size(int(m.group(1)))
    # bare "8 people" / "4 guests" with no preposition
    m = re.search(r"\b(\d{1,2})\s+(?:people|guests?|persons?|pax)\b", t)
    if m:
        return safe_extract_party_size(int(m.group(1)))

    return None


def _is_text_negative_party(text: str) -> bool:
    """True when the user explicitly wrote an invalid party count (negative or zero).

    The LLM normalizes "-3" to "3" before the cleaner ever sees it, so the
    raw text is the only reliable place to catch it. Zero is equally invalid,
    and the model tends to produce unparseable JSON on such input — so we
    intercept before the LLM round-trip.
    """
    if not text:
        return False
    t = text.lower()
    # Negative: "-3 people", "- 1 guest"
    if re.search(r"-\s*\d+\s*(?:people|guests?|persons?|pax)", t):
        return True
    # Zero: "0 people", "for 0 guests", "party of 0"
    if re.search(r"\b0\s*(?:people|guests?|persons?|pax)", t):
        return True
    return False


# Cuisines present in the seed data + common synonyms. Used by the
# discovery-phase filter accumulator so a cuisine mentioned on a REPLY turn
# (where the planner emits no args) still persists to memory for the next
# turn's carry-over.
_KNOWN_CUISINES = [
    "italian", "chinese", "american", "continental", "european", "tex-mex",
    "bbq", "pan-asian", "south indian", "andhra", "kerala",
    "coastal karnataka", "asian", "indian", "mexican", "thai", "japanese",
    "mediterranean", "middle eastern", "north indian", "mughlai", "pizza",
]


def _extract_cuisine_from_text(text: str) -> Optional[str]:
    """Pull a known cuisine name out of free-form user text, or None.

    Matches against the seed-data cuisine list so we never store a bogus
    value. Returns the canonical Title-cased form for memory storage.
    """
    if not text:
        return None
    t = text.lower()
    for c in _KNOWN_CUISINES:
        if re.search(r"\b" + re.escape(c) + r"\b", t):
            return c.title()
    return None


# Booking confirmation classification. Matches affirmative openers ("yes",
# "yep", "please confirm", "that's right, book it", "sounds good", "do it"),
# explicitly rejects decline phrases ("actually, let me think about it" must
# never match), and is bounded to short messages (<=6 words) so "yes, I want
# to change the time" cannot fire the create_reservation shortcut.
_CONFIRM_START = re.compile(
    r"^(?:yes|yep|yeah|sure|ok|okay|please|confirm|confirmed|go ahead|"
    r"book it|that'?s right|sounds good|sound(s)? good|do it|please do|"
    r"proceed|affirmative|correct|perfect|go for it|make it so)\b",
    re.IGNORECASE,
)
_DECLINE_WORDS = re.compile(
    r"\b(?:no|nope|actually|wait|stop|cancel|don'?t|do not|never mind|"
    r"let me think|change|modify|different|hold on|not yet|back|other)\b",
    re.IGNORECASE,
)


def _is_booking_confirmation(text: str) -> bool:
    """True when the user is clearly confirming a booking (not declining).

    Three conditions, all required:
      1. Text starts with a confirmation word/phrase.
      2. Text does NOT contain any decline keyword.
      3. Text is short (<=6 words) — a real confirmation is a brief "yes",
         not a sentence with additional instructions.
    """
    if not text:
        return False
    t = text.strip()
    if not _CONFIRM_START.search(t):
        return False
    if _DECLINE_WORDS.search(t.lower()):
        return False
    if len(t.split()) > 6:
        return False
    return True


def extract_seating_pref(value):
    """Normalize a seating preference to "outdoor" or "indoor", or None.

    Canonicalizes to the short form the booking tool expects. Accepts the raw
    user phrase ("outdoor seating", "patio", "indoor") or whatever the LLM
    emitted. Also used on free-text email turns where the user pairs their
    email with a seating note ("sky@..., outdoor").
    """
    if not value:
        return None
    t = str(value).lower()
    if re.search(r"\b(outdoor|outside|terrace|patio|open\s*air|rooftop|garden|alfresco)\b", t):
        return "outdoor"
    if re.search(r"\b(indoor|inside|interior)\b", t):
        return "indoor"
    return None



# Fix: Never infer missing values from memory
def strip_memory_if_unsupported(args):
    """
    Remove invalid or fabricated values before memory.update().
    Memory must represent *explicit user-provided fields only*.
    """
    cleaned = {}

    if "restaurant" in args and args["restaurant"]:
        cleaned["restaurant"] = args["restaurant"]

    if "date" in args and args["date"]:
        # Uninterpretable dates return None here, the field is dropped, and
        # the collection guard re-asks. Memory only ever stores ISO dates.
        d = safe_extract_date(args["date"])
        if d:
            cleaned["date"] = d


    if "time" in args:
        t = safe_extract_time(args["time"])
        if t:
            cleaned["time"] = t

    if "party_size" in args:
        ps = safe_extract_party_size(args["party_size"])
        if ps:
            cleaned["party_size"] = ps
    
    if "customer_email" in args:
         email_val = args["customer_email"]
         if isinstance(email_val, str) and email_val.strip():
             cleaned["customer_email"] = email_val.strip()

    if "seating_pref" in args:
        sp = extract_seating_pref(args["seating_pref"]) or (
            str(args["seating_pref"]).strip() if args["seating_pref"] else None
        )
        if sp:
            cleaned["seating_pref"] = sp


    return cleaned


def call_planner_llm(
    user_text: str,
    context: str = "",
    recent_results: list = None,
    customer_profile: dict = None,
) -> dict:

    recent_results = recent_results or []
    customer_profile = customer_profile or {}

    planner_context = {
        "user_message": user_text,
        "recent_results": recent_results,
        "customer_profile": customer_profile,
    }

    merged_context = memory.merge_into_context(planner_context)

    memory_json = json.dumps(merged_context.get("memory", {}), indent=2, ensure_ascii=False)
    recent_results_json = json.dumps(recent_results, indent=2, ensure_ascii=False)
    customer_profile_json = json.dumps(customer_profile, indent=2, ensure_ascii=False)

    current_phase = memory.state.get("phase") or "discovery"
    PHASE_PROMPT = load_prompt_for_phase(current_phase)

    full_prompt = (
        f"{PHASE_PROMPT}\n\n---\n"
        f"Conversation Memory (Persisted User Details):\n{memory_json}\n\n"
        f"Recent Results (JSON):\n{recent_results_json}\n\n"
        f"Customer Profile (JSON):\n{customer_profile_json}\n\n"
        f"User Message:\n{user_text}\n\n"
        "Respond ONLY with one valid JSON object (no text outside JSON)."
    )
    logger.info("\n\n===== PLANNER PHASE: %s =====", current_phase)
    logger.info("===== FULL PROMPT SENT TO LLM =====\n%s", full_prompt)

    # PRE-LLM INTERCEPTORS — deterministic guards that fire before the LLM
    # round-trip, covering decisions the model makes unreliably under
    # format=json + think=false (cancel/modify intent, manage-flow email
    # capture, booking confirmation, modify extraction). Each returns a
    # planner-shaped dict mirroring what the LLM would emit if it followed
    # the prompt reliably — and skips the round-trip entirely, so these
    # turns cost ~0 latency.
    _text_lower = user_text.strip().lower()
    _phase = memory.state.get("phase")

    # (0) Bare restaurant-name guard — discovery phase. The model sometimes
    # treats a bare brand name ("GoodFoods", "GoodFoods Grill") as ambiguous
    # and replies asking to clarify instead of emitting get_seating_labels.
    # Narrow by design: the whole message must be a short "GoodFoods[ <word>]
    # [<word>]" token with no filter / question / manage keywords, so it
    # cannot swallow real searches ("Any Italian places?") or booking phrases.
    if _phase in (None, "", "discovery"):
        _bare = user_text.strip().lower()
        _bare = re.sub(r"[^\w\s]", " ", _bare).strip()
        _bare = re.sub(r"\s+", " ", _bare)
        _is_bare_name = bool(re.match(r"^goodfoods(?:\s+\w+){0,2}$", _bare))
        _name_blockers = re.search(
            r"\b(any|some|recommend|suggest|find|search|book|reserve|check|"
            r"availability|cancel|modify|change|update|view|near|in|at|under|"
            r"italian|asian|indian|chinese|mughlai|bbq|mediterranean|"
            r"south|east|west|north|central|food|places?|restaurant|table|"
            r"weather|who|what|where|when|why|how)\b",
            _bare,
        )
        if _is_bare_name and not _name_blockers:
            _name = user_text.strip()
            memory.update_from_planner({"restaurant": _name, "phase": "availability"})
            logger.info("===== BARE RESTAURANT-NAME GUARD -> get_seating_labels =====")
            return {
                "plan": "execute",
                "action": "get_seating_labels",
                "args": {"restaurant": _name},
            }

    # (1) Cancel/modify intent interceptor — discovery -> booking manage.
    # The Phase-1 prompt has no rule for cancel/modify intent; without this
    # guard the model falls through to failsafe or hallucinates a restaurant.
    # Scan for an inline email FIRST: "cancel booking for x@example.com" has
    # the email right there, and asking "Which email?" wastes a turn (worse,
    # the next turn's text gets captured as the email). Only ask when the
    # email is genuinely missing.
    if _phase in (None, "", "discovery"):
        _manage_patterns = [
            r"\b(cancel|modify|change|update|view)\b.{0,30}\b(booking|reservation)\b",
            r"\b(booking|reservation)\b.{0,30}\b(cancel|modify|change|update|view)\b",
            r"\bmy\s+(booking|reservation)\b",
        ]
        if any(re.search(p, _text_lower) for p in _manage_patterns):
            _inline_email = re.search(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_text
            )
            if _inline_email:
                _email = _inline_email.group(0)
                memory.update_from_planner({
                    "phase": "booking",
                    "intent": "manage",
                    "customer_email": _email,
                })
                logger.info("===== MANAGE INTENT + inline email -> get_booking_details =====")
                return {
                    "plan": "execute",
                    "action": "get_booking_details",
                    "args": {"customer_email": _email},
                }
            logger.info("===== CANCEL/MODIFY INTENT INTERCEPTOR (Phase 1 -> manage) =====")
            memory.update_from_planner({"phase": "booking", "intent": "manage"})
            return {
                "plan": "reply",
                "reply": "Which email is the reservation under?",
            }

    # (2) Manage-flow email capture -> get_booking_details directly.
    # Guard (1) asked for the email; a bare email on the next turn is the
    # answer. Without this the Phase-3 prompt loops on "Which email?" because
    # the email never lands in memory.
    if (_phase == "booking"
            and memory.state.get("intent") == "manage"
            and not memory.state.get("customer_email")):
        _email_match_pre = re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_text
        )
        if _email_match_pre:
            _email = _email_match_pre.group(0)
            memory.update_from_planner({"customer_email": _email})
            logger.info("===== MANAGE FLOW: email captured -> get_booking_details =====")
            return {
                "plan": "execute",
                "action": "get_booking_details",
                "args": {"customer_email": _email},
            }

    # (3) Booking confirmation short-circuit — create flow only. The model
    # emits invalid JSON on a bare "yes"; when all required booking fields
    # are in memory and the user gives a strict affirmative, emit
    # create_reservation directly. Excluded for intent=manage, where "yes"
    # might mean "yes cancel" (guard 4).
    if _phase == "booking" and memory.state.get("intent") in (None, "create"):
        _mem = memory.state
        _has_required = all([
            _mem.get("restaurant"),
            _mem.get("date"),
            _mem.get("time"),
            _mem.get("party_size"),
            _mem.get("customer_email"),
        ])
        _strict_affirmatives = {
            "yes", "y", "yeah", "yep", "ok", "okay", "sure", "confirm",
            "confirmed", "please confirm", "go ahead", "do it", "book it",
            "yes please", "yes confirm", "confirm it", "book it please",
            "yes go ahead", "yes book it",
        }
        if _has_required and _text_lower in _strict_affirmatives:
            logger.info("===== BOOKING CONFIRMATION SHORT-CIRCUIT =====")
            return {
                "plan": "execute",
                "action": "create_reservation",
                "args": {
                    "restaurant": _mem.get("restaurant"),
                    "date": _mem.get("date"),
                    "time": _mem.get("time"),
                    "party_size": _mem.get("party_size"),
                    "customer_email": _mem.get("customer_email"),
                    "seating_pref": _mem.get("seating_pref"),
                },
            }

    # (4) Cancel confirmation — manage flow. "yes cancel it" / "cancel
    # please" fires cancel_reservation without depending on the model to
    # follow the Phase-3 cancel rule.
    if (_phase == "booking"
            and memory.state.get("intent") == "manage"
            and memory.state.get("customer_email")):
        if "cancel" in _text_lower and any(
            w in _text_lower for w in ("yes", "it", "please", "confirm", "go", "now")
        ):
            logger.info("===== CANCEL CONFIRMATION INTERCEPTOR =====")
            return {
                "plan": "execute",
                "action": "cancel_reservation",
                "args": {"customer_email": memory.state.get("customer_email")},
            }

    # (5) Manage-flow modify — booking + intent=manage + email known. After
    # get_booking_details returns the reservation, the next turn supplies the
    # modification ("change the time to 20:00", "make it 6 people"). The
    # Phase-3 prompt would ask one clarifying question first — a needless
    # round-trip the model also handles unreliably. Extract the requested
    # value(s) deterministically and emit modify_reservation with the new_*
    # arg names the tool expects. Guarded by a change-keyword plus an actual
    # value so it cannot fire on "actually leave it as is" or a bare "yes".
    if (_phase == "booking"
            and memory.state.get("intent") == "manage"
            and memory.state.get("customer_email")):
        _change_kw = re.search(
            r"\b(change|move|make it|update|shift|reschedule|to|different|new)\b",
            _text_lower,
        )
        if _change_kw and "cancel" not in _text_lower:
            from app.api.utils.date_utils import (
                extract_date_from_text, extract_time_from_text,
            )
            _mod_time = extract_time_from_text(user_text)
            _mod_date = extract_date_from_text(user_text)
            _mod_party = extract_party_size_from_text(user_text)
            if _mod_time or _mod_date or _mod_party is not None:
                _mod_args = {"customer_email": memory.state.get("customer_email")}
                if _mod_time:
                    _mod_args["new_time"] = _mod_time
                if _mod_date:
                    _mod_args["new_date"] = _mod_date
                if _mod_party is not None:
                    _mod_args["new_party_size"] = _mod_party
                logger.info("===== MANAGE MODIFY INTERCEPTOR -> modify_reservation =====")
                return {
                    "plan": "execute",
                    "action": "modify_reservation",
                    "args": _mod_args,
                }

    # (6) Negative/zero party guard — availability phase. The model
    # normalizes "-3" to "3" before the cleaner sees it; scan the raw text
    # and ask the user to re-enter the party size.
    if _phase == "availability" and _is_text_negative_party(user_text):
        logger.info("===== NEGATIVE PARTY SIZE GUARD -> re-ask =====")
        return {
            "plan": "reply",
            "reply": "For how many guests should I check availability? "
                     "Please enter a positive number.",
        }

    # (7) Ambiguous-time guard — availability phase. "tomorrow at 7 or 8pm"
    # offers two times; passing "7 or 8pm" to check_availability matches no
    # slot and the user sees a confusing error. The pattern requires at least
    # one am/pm anchor so "7 or 8 people" cannot match.
    if _phase == "availability":
        _two_times = (
            re.search(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?\s*(?:or|/)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)", _text_lower)
            or re.search(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*(?:or|/)\s*\d{1,2}", _text_lower)
        )
        if _two_times:
            logger.info("===== AMBIGUOUS TIME GUARD -> ask to pick one =====")
            return {
                "plan": "reply",
                "reply": "Which time would you prefer? Please pick a single time.",
            }

    # (8) Booking confirmation — booking phase. Confirmation phrases other
    # than a bare "yes" ("sounds good", "that's right, book it") sometimes
    # get a "Shall I confirm?" re-ask instead of create_reservation. When all
    # booking fields are present and the message matches the confirmation
    # classifier, fire the booking directly.
    if _phase == "booking":
        _has_all_fields = all([
            memory.state.get("restaurant"),
            memory.state.get("date"),
            memory.state.get("time"),
            memory.state.get("party_size"),
            memory.state.get("customer_email"),
        ])
        if _has_all_fields and _is_booking_confirmation(user_text):
            logger.info("===== BOOKING CONFIRMATION INTERCEPTOR -> create_reservation =====")
            return {
                "plan": "execute",
                "action": "create_reservation",
                "args": {
                    "restaurant": memory.state.get("restaurant"),
                    "location_id": memory.state.get("location_id"),
                    "date": memory.state.get("date"),
                    "time": memory.state.get("time"),
                    "party_size": memory.state.get("party_size"),
                    "customer_email": memory.state.get("customer_email"),
                    "seating_pref": memory.state.get("seating_pref"),
                },
            }
    # END PRE-LLM INTERCEPTORS

    try:
        settings = get_settings()
        # /api/chat (not /api/generate) so the model's chat template is
        # applied — that is what makes `format: "json"` work. The JSON
        # constraint also suppresses qwen3's reasoning preamble (it cannot
        # emit a think paragraph if every token must keep the output valid
        # JSON). Measured: ~75s/turn via the raw generate path -> ~3s here.
        #
        # Known limitation: cuisine-only discovery inputs (rule: "ask for
        # more filters") sometimes still execute a search — the model can't
        # reason through the AND-condition without thinking tokens. The
        # deterministic cuisine-only guard below covers it.
        system_content = PHASE_PROMPT
        # Keep the user message text-first and the context JSON inline
        # (single-line). Pretty-printed JSON in the prompt gets treated as a
        # template to copy — the model echoes `{"phase": "discovery"}` verbatim
        # instead of producing a decision. Inline JSON after the user text
        # breaks the copy pattern while still exposing the values needed for
        # `memory.X` substitution.
        memory_inline = json.dumps(merged_context.get("memory", {}), ensure_ascii=False)
        results_inline = json.dumps(recent_results, ensure_ascii=False)
        profile_inline = json.dumps(customer_profile, ensure_ascii=False)
        user_content = (
            f"{user_text}\n\n"
            "--- Context (do not echo back) ---\n"
            f"Memory: {memory_inline}\n"
            f"Recent results: {results_inline}\n"
            f"Customer profile: {profile_inline}\n"
            "Respond ONLY with one valid JSON object."
        )
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "think": settings.ollama_think,
            "format": "json",
        }

        r = requests.post(settings.ollama_chat_url, json=payload, timeout=settings.ollama_timeout)
        r.raise_for_status()
        data = r.json()

        raw_response = (
            data.get("message", {}).get("content")
            or data.get("response")
            or data.get("text")
            or ""
        ).strip()

        # qwen3 with /api/chat + think=false + format=json emits clean JSON
        # directly. strip_model_reasoning stays as a defensive fallback in
        # case a model swap or future prompt change reintroduces a preamble.
        raw_response = strip_model_reasoning(raw_response)

        json_start = raw_response.find("{")

        if json_start == -1:
            candidate = raw_response
        else:
            depth = 0
            json_end = None

            # bracket-match the outermost JSON object
            for i in range(json_start, len(raw_response)):
                if raw_response[i] == "{":
                    depth += 1
                elif raw_response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_end = i
                        break
            if json_end is not None:
                candidate = raw_response[json_start:json_end + 1]
            else:
                candidate = raw_response[json_start:]  

        logger.info("===== JSON CANDIDATE EXTRACTED =====\n%s", candidate)

        if "memory.restaurant" in candidate:
            candidate = candidate.replace("memory.restaurant", json.dumps(memory.state.get("restaurant")))

        if "memory.date" in candidate:
            candidate = candidate.replace("memory.date", json.dumps(memory.state.get("date")))

        if "memory.party_size" in candidate:
            candidate = candidate.replace("memory.party_size", json.dumps(memory.state.get("party_size")))

        if "memory.customer_email" in candidate:
            candidate = candidate.replace("memory.customer_email", json.dumps(memory.state.get("customer_email")))

        candidate = re.sub(r"<([^>]+)>", r"\1", candidate)

        # EARLY EMAIL CAPTURE (create flow only — the manage flow is handled
        # by pre-LLM guard 2). A matched email counts as "intended for
        # capture" when (a) we're in booking phase waiting for one, or (b) an
        # explicit intent keyword is present ("use X", "my email is X") —
        # which covers email changes and out-of-phase provision.
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_text)
        lower_text = user_text.lower()
        email_intent_regex = re.compile(r"\b(use|email is|my email|here is my email|here is|contact email|to create|to book|use email)\b")
        _in_booking_email_needed = (
            memory.state.get("phase") == "booking"
            and memory.state.get("customer_email") is None
        )
        pure_email_intent = bool(
            email_match
            and (_in_booking_email_needed or email_intent_regex.search(lower_text))
        )

        if memory.state.get("phase") == "booking" and pure_email_intent:
            email = email_match.group(0)
            memory.update_from_planner({"customer_email": email})

            # A seating preference often rides along with the email
            # ("sky@example.com, outdoor seating please") — capture it now or
            # it never reaches create_reservation.
            _sp = extract_seating_pref(user_text)
            if _sp:
                memory.update_from_planner({"seating_pref": _sp})

            mem = memory.state
            has_required = all([
                mem.get("restaurant"),
                mem.get("date"),
                mem.get("time"),
                mem.get("party_size"),
            ])

            if has_required:
                return {"plan": "reply", "reply": "Shall I confirm your reservation now?"}

            # time is the next missing field in the booking flow
            return {"plan": "reply", "reply": "What time should I book it for?"}

        validated = validate_planner_json(candidate)
        logger.info("===== VALIDATED PLANNER JSON =====\n%s", validated)

        # The modify tool's contract uses new_date / new_time / new_party_size /
        # new_seating_pref, but the model often emits the bare field names.
        # Rename bare -> new_* so dispatch always sees consistent keys.
        if validated and validated.get("plan") == "execute" \
                and validated.get("action") == "modify_reservation":
            _margs = validated.get("args", {}) or {}
            for _bare, _new in (
                ("time", "new_time"),
                ("date", "new_date"),
                ("party_size", "new_party_size"),
                ("seating_pref", "new_seating_pref"),
            ):
                if _bare in _margs and _new not in _margs:
                    _margs[_new] = _margs.pop(_bare)
            validated["args"] = _margs

        # Once a restaurant is selected (availability/booking phase), an
        # availability-style turn ("18 Nov for 4 people") sometimes still
        # emits search_restaurants_by_filters with date/party_size in the args
        # — the wrong tool, and dispatch would reject the unknown kwargs.
        # Re-route to check_availability with the selected restaurant plus
        # whatever date/party/time we can pull from the args or user text.
        if (validated
                and validated.get("plan") == "execute"
                and validated.get("action") == "search_restaurants_by_filters"
                and memory.state.get("phase") in ("availability", "booking")
                and memory.state.get("restaurant")):
            from app.api.utils.date_utils import (
                extract_date_from_text, extract_time_from_text,
            )
            _sargs = validated.get("args", {}) or {}
            _rdate = (
                _sargs.get("date")
                or extract_date_from_text(user_text)
                or memory.state.get("date")
            )
            _rdate_iso = safe_extract_date(_rdate) if _rdate else None
            _rparty = (
                _sargs.get("party_size")
                or _sargs.get("group_size")
                or extract_party_size_from_text(user_text)
                or memory.state.get("party_size")
            )
            _rparty_n = safe_extract_party_size(_rparty)
            _rtime = extract_time_from_text(user_text) or memory.state.get("time")
            _new_args = {"restaurant": memory.state.get("restaurant")}
            if _rdate_iso:
                _new_args["date"] = _rdate_iso
            if _rparty_n:
                _new_args["party_size"] = _rparty_n
            if _rtime:
                _new_args["time"] = _rtime
            logger.info("===== RE-ROUTE search -> check_availability: %s =====", _new_args)
            validated = {
                "plan": "execute",
                "action": "check_availability",
                "args": _new_args,
            }

        # SAFETY CHECKS — all of these run BEFORE any memory update.
        def _is_missing(value):
            if value is None:
                return True
            if isinstance(value, str):
                s = value.strip().lower()
                return s in ("", "null", "none", "undefined")
            return False

        # If LLM returned an execute action in booking phase to create_reservation,
        # block it here if required booking fields are missing in memory.
        if validated and validated.get("plan") == "execute":
            action_candidate = validated.get("action")
            if memory.state.get("phase") == "booking" and action_candidate == "create_reservation":
                mem = memory.state
                missing = []
                if _is_missing(mem.get("time")):
                    missing.append("time")
                if _is_missing(mem.get("date")):
                    missing.append("date")
                if _is_missing(mem.get("party_size")):
                    missing.append("party_size")
                if _is_missing(mem.get("customer_email")):
                    missing.append("customer_email")
                if _is_missing(mem.get("restaurant")):
                    missing.append("restaurant")

                if missing:
                    # Ask for the first missing field.
                    field = missing[0]
                    questions = {
                        "customer_email": "Which email should I use for the reservation?",
                        "time": "What time should I book the reservation for?",
                        "date": "Which date should I use for the reservation?",
                        "party_size": "For how many guests should I make the reservation?",
                        "restaurant": "Which restaurant should I book?"
                    }
                    return {"plan": "reply", "reply": questions.get(field, "Could you confirm the missing details?")}

        # PHASE-2 COLLECTION-ORDER GUARD (reply path). In availability the
        # model loses track of what memory already holds — it asks for a field
        # it has, or asks for the wrong one. On a REPLY, extract any fields
        # the user just stated, store them, then ask for the first missing
        # field in the fixed order [date, party_size]. Seating-option requests
        # and slot picks are excluded so this cannot smother those flows.
        if (validated
                and validated.get("plan") == "reply"
                and memory.state.get("phase") == "availability"):
            _tl = user_text.lower()
            _is_meta_request = re.search(
                r"\b(seating|seat|show|list|options|works|book|available|slots?)\b",
                _tl,
            )
            if not _is_meta_request:
                from app.api.utils.date_utils import extract_date_from_text
                _d = extract_date_from_text(user_text)
                _p = extract_party_size_from_text(user_text)
                if _d and not memory.state.get("date"):
                    memory.update_from_planner({"date": _d})
                if _p and not memory.state.get("party_size"):
                    memory.update_from_planner({"party_size": _p})
                if not memory.state.get("date"):
                    logger.info("===== PHASE-2 COLLECTION GUARD: ask date =====")
                    return {"plan": "reply", "reply": "What date should I check?"}
                if not memory.state.get("party_size"):
                    logger.info("===== PHASE-2 COLLECTION GUARD: ask party =====")
                    return {"plan": "reply",
                            "reply": "For how many guests should I check availability?"}

        # PHASE-1 FILTER ACCUMULATOR. On a cuisine-only turn the prompt makes
        # the planner REPLY asking for more preferences — and replies carry no
        # args, so the cuisine the user just stated is never stored. The next
        # turn ("In South") then finds memory.cuisine empty. Extract and
        # persist the cuisine now, regardless of what the planner decided.
        if (validated
                and validated.get("plan") == "reply"
                and memory.state.get("phase") in (None, "", "discovery")
                and not memory.state.get("cuisine")):
            _cu = _extract_cuisine_from_text(user_text)
            if _cu:
                memory.update_from_planner({"cuisine": _cu})
                logger.info(f"===== DISCOVERY CUISINE ACCUMULATOR -> memory.cuisine={_cu} =====")

        # The execute-path carry-over below already merges memory filters when
        # the model emits a search. The residual gap is the model REPLYING on a
        # follow-up ("In South") instead of searching — it re-asks for
        # preferences even after the user adds a second filter. This guard is
        # deliberately pinched: it fires ONLY when memory already holds a
        # cuisine, the model returned a reply, and the user's message is a
        # short zone phrase.
        if (validated
                and validated.get("plan") == "reply"
                and memory.state.get("phase") in (None, "", "discovery")
                and memory.state.get("cuisine")):
            _tl = user_text.lower()
            _zone_m = re.search(
                r"\b(south|east|west|north|central|north-east|north-west|"
                r"south-east|far south)\b", _tl,
            )
            _is_short_zone_phrase = (
                _zone_m is not None and len(user_text.split()) <= 4
            )
            if _is_short_zone_phrase:
                _fargs = {
                    "cuisine": memory.state.get("cuisine"),
                    "zone": _zone_m.group(1),
                }
                memory.update_from_planner({"zone": _zone_m.group(1)})
                logger.info("===== PHASE-1 REPLY-PATH CARRY-OVER -> search =====")
                return {
                    "plan": "execute",
                    "action": "search_restaurants_by_filters",
                    "args": _fargs,
                }

        # PHASE-1 NEAR-ME / VAGUE-ZONE GUARD. The prompt maps "near me /
        # around here" to a clarifying question, but the model ignores it and
        # emits a search with zone="near me", which matches nothing. If a
        # discovery search carries a vague zone, ask for a real area instead
        # of running the empty search. Runs before the cuisine-only guard so
        # "Italian near me" still clarifies.
        if (validated
                and validated.get("plan") == "execute"
                and validated.get("action") == "search_restaurants_by_filters"
                and memory.state.get("phase") in (None, "", "discovery")):
            _args = validated.get("args", {}) or {}
            _zone_val = str(_args.get("zone") or "").strip().lower()
            if _zone_val in ("near me", "nearme", "around here", "aroundhere",
                             "nearby", "near", "here", "close by", "closeby"):
                logger.info("===== PHASE-1 NEAR-ME GUARD -> ask area =====")
                return {
                    "plan": "reply",
                    "reply": "Which area would you like to dine in?",
                }

        # PHASE-1 CUISINE-ONLY GUARD. The model cannot reliably apply the
        # "cuisine exists AND no other filters -> ask for more preferences"
        # AND-condition — without thinking tokens it sees a cuisine and
        # immediately fires a one-filter search. Enforce the rule here.
        if (validated
                and validated.get("plan") == "execute"
                and validated.get("action") == "search_restaurants_by_filters"
                and memory.state.get("phase") in (None, "", "discovery")):
            _args = validated.get("args", {}) or {}
            _filter_keys = {"cuisine", "zone", "max_price", "min_rating", "tag"}
            _present = {
                k for k in _filter_keys
                if _args.get(k) not in (None, "", [], 0)
            }
            if _present == {"cuisine"}:
                logger.info("===== PHASE-1 CUISINE-ONLY GUARD FIRED =====")
                # Persist the cuisine before returning so the next turn can
                # build on it ("I want Italian" -> "In South" must carry the
                # Italian forward).
                if _args.get("cuisine"):
                    memory.update_from_planner({"cuisine": _args["cuisine"]})
                return {
                    "plan": "reply",
                    "reply": "Would you like to add any other preferences such as area, budget, rating, or tags?",
                }

        # PHASE-1 FILTER CARRY-OVER. When the user progressively reveals
        # filters across turns, the model often emits only the newly-mentioned
        # one, dropping what memory already holds. Merge memory filters into
        # the args; args already present take precedence (the user may be
        # overriding a prior value).
        if (validated
                and validated.get("plan") == "execute"
                and validated.get("action") == "search_restaurants_by_filters"
                and memory.state.get("phase") in (None, "", "discovery")):
            _args = validated.get("args", {}) or {}
            for _filt in ("cuisine", "zone", "max_price", "min_rating", "tag"):
                if not _args.get(_filt) and memory.state.get(_filt):
                    _args[_filt] = memory.state[_filt]
            validated["args"] = _args
            logger.info("===== PHASE-1 FILTER CARRY-OVER: %s =====", validated["args"])

        # Memory updates happen only past this point — every guard above
        # runs before anything is persisted.
        phase = memory.state.get("phase")

        if validated and validated.get("plan") == "execute":
            action = validated.get("action")
            args = validated.get("args", {}) or {}
            cleaned = strip_memory_if_unsupported(args)
            # A create_reservation attempt in availability means the model
            # skipped the email question — persist what we have and ask.
            if phase == "availability":
                if action == "create_reservation" and cleaned.get("customer_email"):

                    
                    memory.update_from_planner({
                        "restaurant": cleaned.get("restaurant"),
                        "date": cleaned.get("date"),
                        "time": cleaned.get("time"),
                        "party_size": cleaned.get("party_size"),
                        "phase": "booking",
                    })

                    return {
                        "plan": "reply",
                        "reply": "Which email should I use for the reservation?"
                    }


            logger.info("===== EXECUTE ACTION DETECTED: %s =====", action)
            logger.info("===== EXECUTE ARGS ===== %s", args)

            # some models emit restaurant_name instead of restaurant
            if "restaurant_name" in args and "restaurant" not in args:
                args["restaurant"] = args.pop("restaurant_name")
            validated["args"] = args

            logger.info("===== CLEANED ARGS ===== %s", cleaned)
            logger.info("===== MEMORY BEFORE UPDATE ===== %s", memory.state)

            # If all booking fields are known, capture the email and ask for
            # confirmation instead of firing get_booking_details.
            if action == "get_booking_details" and cleaned.get("customer_email"):
                mem = memory.state
                has_required = all([
                    mem.get("restaurant"),
                    mem.get("date"),
                    mem.get("time"),
                    mem.get("party_size")
                ])
                if mem.get("phase") == "booking" and has_required:
                    memory.update_from_planner({"customer_email": cleaned["customer_email"]})
                    # Ask for confirmation — do NOT call get_booking_details
                    return {"plan": "reply", "reply": "Shall I confirm your reservation now?"}

            if action == "get_seating_labels":
                memory.update_from_planner({
                    "restaurant": cleaned.get("restaurant"),
                    "phase": "availability",
                })

            elif action == "check_availability":
                memory.update_from_planner({
                    "restaurant": cleaned.get("restaurant"),
                    "date": cleaned.get("date"),
                    "time": cleaned.get("time"),
                    "party_size": cleaned.get("party_size"),
                    "phase": "availability",
                })
                # NOTE: do NOT auto-transition to booking here. Whether the
                # slot is actually available is only known after the tool
                # runs. The orchestrator calls update_phase_after_check_availability()
                # post-dispatch to advance phase=booking only when
                # is_available=true. Transitioning blindly on `time` present
                # would send unavailable-slot turns into the booking phase.

                # If check_availability fired but a required field is missing
                # after cleaning (e.g. party_size=0 was rejected, or the
                # action fired prematurely), re-ask for the first missing
                # field instead of dispatching a broken availability check.
                if not memory.state.get("date"):
                    logger.info("===== PHASE-2 EXECUTE GUARD: ask date =====")
                    return {"plan": "reply", "reply": "What date should I check?"}
                if not memory.state.get("party_size"):
                    logger.info("===== PHASE-2 EXECUTE GUARD: ask party =====")
                    return {"plan": "reply",
                            "reply": "For how many guests should I check availability?"}

            elif action == "create_reservation":
                payload = {"phase": "booking"}

                if cleaned.get("restaurant") is not None:
                    payload["restaurant"] = cleaned["restaurant"]

                if cleaned.get("date") is not None:
                    payload["date"] = cleaned["date"]

                if cleaned.get("time") is not None:
                    payload["time"] = cleaned["time"]

                if cleaned.get("party_size") is not None:
                    payload["party_size"] = cleaned["party_size"]

                if cleaned.get("customer_email") is not None:
                    payload["customer_email"] = cleaned["customer_email"]

                if cleaned.get("seating_pref") is not None:
                    payload["seating_pref"] = cleaned["seating_pref"]

                memory.update_from_planner(payload)

            elif action == "get_booking_details":
                update_payload = {}
                if cleaned.get("customer_email"):
                    update_payload["customer_email"] = cleaned["customer_email"]
                memory.update_from_planner(update_payload)

            logger.info("===== MEMORY AFTER UPDATE ===== %s", memory.state)

        # Replies carry no user fields — only an explicit phase change is
        # honored, never slot values.
        if validated and validated.get("plan") == "reply":
            args = validated.get("args", {}) or {}
            if args.get("phase"):
                memory.update_from_planner({"phase": args["phase"]})

        if validated:
            return validated

        # second attempt repair
        repaired = repair_malformed_json(raw_response)
        validated2 = validate_planner_json(repaired)
        if validated2:
            return validated2

        # Third attempt: retry the LLM with strict format enforcement. The
        # JSON constraint still lets unparseable output through occasionally
        # (grammar-sampler edge cases). Rather than falling back to the
        # generic reply, give the model one more shot with an explicit
        # "your last response was invalid" warning — this recovers roughly
        # half of the would-be failures.
        try:
            _retry_user = user_content + (
                "\n\n=== FORMAT ERROR — READ CAREFULLY ===\n"
                "Your previous response was NOT valid JSON. "
                "Output ONLY a single JSON object. "
                "Start with { and end with }. "
                "No reasoning, no commentary, no markdown, no text outside the JSON."
            )
            _retry_payload = {
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": _retry_user},
                ],
                "stream": False,
                "think": settings.ollama_think,
                "format": "json",
            }
            _rr = requests.post(settings.ollama_chat_url, json=_retry_payload, timeout=settings.ollama_timeout)
            _rr.raise_for_status()
            _rdata = _rr.json()
            _retry_raw = (
                _rdata.get("message", {}).get("content")
                or _rdata.get("response")
                or ""
            ).strip()
            _retry_raw = strip_model_reasoning(_retry_raw)

            # Bracket-match the retry output, same as the first attempt.
            _js = _retry_raw.find("{")
            if _js != -1:
                _depth = 0
                _je = None
                for _i in range(_js, len(_retry_raw)):
                    if _retry_raw[_i] == "{":
                        _depth += 1
                    elif _retry_raw[_i] == "}":
                        _depth -= 1
                        if _depth == 0:
                            _je = _i
                            break
                _retry_candidate = _retry_raw[_js:_je + 1] if _je is not None else _retry_raw[_js:]
            else:
                _retry_candidate = _retry_raw

            validated3 = validate_planner_json(_retry_candidate)
            if validated3:
                logger.info("===== PLANNER RETRY SUCCEEDED (attempt 3) =====")
                return validated3
            # Last resort: repair the retry output too
            repaired_retry = repair_malformed_json(_retry_candidate)
            validated4 = validate_planner_json(repaired_retry)
            if validated4:
                logger.info("===== PLANNER RETRY+REPAIR SUCCEEDED (attempt 4) =====")
                return validated4
        except Exception as _retry_err:
            logger.warning(f"Planner retry call failed: {_retry_err}")

        return {"plan": "reply", "reply": "Could you clarify — would you like to search or book a specific restaurant?"}

    except Exception as e:
        logger.warning(f"Planner failed: {e}")
        return mock_planner_output(user_text)


def update_phase_after_check_availability(tool_result: dict) -> None:
    """Advance memory.phase to "booking" iff check_availability confirmed the
    requested slot is available.

    Called by the orchestrator (agent.process_user_query) AFTER the
    check_availability tool runs. The planner cannot do this itself because
    it sees only the planner decision, not the tool result.

    Behavior:
      mode="single" + is_available=True  -> phase = "booking"
      mode="single" + is_available=False -> phase stays "availability"
      mode="list"                        -> phase stays "availability"
      ok=False                           -> phase stays "availability"
    """
    if not isinstance(tool_result, dict):
        return
    if tool_result.get("mode") == "single" and tool_result.get("is_available") is True:
        memory.update_from_planner({"phase": "booking"})
        logger.info("===== POST-DISPATCH: check_availability available -> phase=booking =====")
    else:
        cur = memory.state.get("phase")
        if cur == "booking":
            # Defensive: if a previous turn set booking but the latest
            # availability check came back unavailable/list/error, revert
            # so the user is not asked to confirm an unbookable slot.
            memory.update_from_planner({"phase": "availability"})
            logger.info("===== POST-DISPATCH: check_availability not available -> phase=availability =====")