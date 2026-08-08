# app/agent/planner_agent.py
import json
import logging
import requests
import os
import re
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


# Model, endpoint, timeout and think-mode all come from app.config.
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
    if not value:
        return None
    return str(value).strip()



def safe_extract_time(value):
    if not value:
        return None
    return str(value).strip()



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
        cleaned["date"] = str(args["date"]).strip()


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


    return cleaned

# Main Planner Function
def call_planner_llm(
    user_text: str,
    context: str = "",
    recent_results: list = None,
    customer_profile: dict = None,
) -> dict:

    recent_results = recent_results or []
    customer_profile = customer_profile or {}

    # merge memory into planner context (without mutating)
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

    # =====================================================================
    # PRE-LLM INTERCEPTORS
    # Deterministic guards that fire before the LLM round-trip, covering
    # decisions qwen3:4b + format=json + think=false makes unreliably:
    #   1. Cancel/modify intent detection in Phase 1 (no Phase-1 prompt rule)
    #   2. Manage-flow email capture -> get_booking_details
    #   3. Booking confirmation short-circuit (LLM emits invalid JSON on "yes")
    #   4. Cancel confirmation in manage flow
    # Each returns a planner-shaped dict mirroring what the LLM would emit
    # if it followed the prompt reliably. Skipping the round-trip here
    # also keeps per-turn latency at ~0s for these covered cases.
    # =====================================================================
    _text_lower = user_text.strip().lower()
    _phase = memory.state.get("phase")

    # (1) Cancel/modify intent interceptor — Phase 1 -> Phase 3 manage.
    # Phase-1 prompt has no rule for cancel/modify intent; without this
    # guard the model falls through to failsafe or hallucinates a restaurant.
    if _phase in (None, "", "discovery"):
        _manage_patterns = [
            r"\b(cancel|modify|change|update|view)\b.{0,30}\b(booking|reservation)\b",
            r"\b(booking|reservation)\b.{0,30}\b(cancel|modify|change|update|view)\b",
            r"\bmy\s+(booking|reservation)\b",
        ]
        if any(re.search(p, _text_lower) for p in _manage_patterns):
            logger.info("===== CANCEL/MODIFY INTENT INTERCEPTOR (Phase 1 -> manage) =====")
            memory.update_from_planner({"phase": "booking", "intent": "manage"})
            return {
                "plan": "reply",
                "reply": "Which email is the reservation under?",
            }

    # (2) Manage-flow email capture -> emit get_booking_details directly.
    # The cancel/modify interceptor above set intent=manage and asked for
    # email; on the next turn a bare email is the answer. The Phase-3
    # prompt would otherwise loop on "Which email?" because it does not
    # see the email in memory yet.
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

    # (3) Booking confirmation short-circuit — create flow only.
    # C01 T4 "yes" regressed to invalid JSON / fallback. When all required
    # booking fields are in memory and the user gives a strict affirmative,
    # emit create_reservation directly. Excluded for intent=manage (where
    # "yes" might mean "yes cancel" — handled by guard 4).
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

    # (4) Cancel confirmation interceptor — manage flow.
    # "yes cancel it" / "cancel please" should fire cancel_reservation
    # without depending on the LLM to follow the Phase-3 cancel rule.
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
    # END PRE-LLM INTERCEPTORS

    try:
        settings = get_settings()
        # Use /api/chat (not /api/generate) so qwen3's chat template is applied
        # — this is what makes `format: "json"` actually work. With the raw
        # /api/generate path the chat template is skipped, qwen3 still reasons
        # in plain text (~1800 tokens of preface before the JSON), and the
        # planner takes ~75s/turn instead of ~3s.
        #
        # `format: "json"` constrains decoding to valid-JSON tokens only,
        # which is what suppresses the reasoning preamble (qwen3 cannot emit
        # a reasoning paragraph if every token must keep the output valid
        # JSON). Measured: 75s -> 3s, decisions stay correct across all three
        # phase prompts (verified for execute/reply across P1/P2/P3).
        #
        # Known limitation: cuisine-only Phase-1 inputs (rule: "ask for more
        # filters") sometimes still execute a search, because the model can't
        # reason through the AND-condition without thinking tokens. Documented
        # as an eval finding rather than patched here.
        system_content = PHASE_PROMPT
        # CRITICAL: keep the user message text-first and the context JSON
        # inline (single-line). With format=json + think=false, qwen3:4b
        # treats any pretty-printed JSON in the prompt as a template to
        # copy — earlier turns ended up echoing `{"phase": "discovery"}`
        # verbatim instead of producing a planner decision. Inline JSON
        # after the user text (with an explicit do-not-echo guard) breaks
        # the copy pattern while still giving the model the values it
        # needs for `memory.X` substitution.
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

        #taking response from ollama
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

            #logic to extract json from response bracket wise
            for i in range(json_start, len(raw_response)):
                if raw_response[i] == "{":
                    depth += 1
                elif raw_response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_end = i
                        break
            #slicing extra output from llm before and after json
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

        # EARLY EMAIL CAPTURE (restricted to clear email-provision intents)
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_text)
        lower_text = user_text.lower()
        # Treat a matched email as "intended for capture" when:
        #   (a) booking phase AND customer_email currently missing in memory
        #       — the user is answering the standard "which email?" question,
        #       so a bare email qualifies (no keyword needed); OR
        #   (b) an explicit email-intent keyword is present
        #       ("use X", "my email is X", etc.) — covers email changes /
        #       out-of-phase provision.
        # Pre-LLM guard (2) already handles the manage flow, so this branch
        # only fires for the create flow.
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

            # Save email
            memory.update_from_planner({"customer_email": email})

            # Check if all booking fields already exist
            mem = memory.state
            has_required = all([
                mem.get("restaurant"),
                mem.get("date"),
                mem.get("time"),
                mem.get("party_size"),
            ])

            # If everything present → ask for confirmation directly
            if has_required:
                return {"plan": "reply", "reply": "Shall I confirm your reservation now?"}

            # Otherwise ask next missing field (time is highest priority missing in booking flow)
            return {"plan": "reply", "reply": "What time should I book it for?"}
        # END EARLY EMAIL CAPTURE

        validated = validate_planner_json(candidate)
        logger.info("===== VALIDATED PLANNER JSON =====\n%s", validated)

        # SAFETY CHECKS (run BEFORE any memory updates)
        # Helper to detect missing/invalid values robustly
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
                    # Ask for the FIRST missing field (per your prompt rules)
                    field = missing[0]
                    questions = {
                        "customer_email": "Which email should I use for the reservation?",
                        "time": "What time should I book the reservation for?",
                        "date": "Which date should I use for the reservation?",
                        "party_size": "For how many guests should I make the reservation?",
                        "restaurant": "Which restaurant should I book?"
                    }
                    return {"plan": "reply", "reply": questions.get(field, "Could you confirm the missing details?")}

        # PHASE-1 CUISINE-ONLY GUARD
        # qwen3:4b + format=json + think=false cannot reliably apply the
        # "cuisine exists AND no other filters -> ask for more preferences"
        # AND-condition (Phase-1 prompt rule, lines 82-83). Without thinking
        # tokens the model sees cuisine and immediately fires a search. Enforce
        # the rule deterministically here so A01/A04-turn-1 style inputs ask
        # for additional filters instead of executing a one-filter search.
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
                # Persist cuisine before returning so the next turn can build
                # on it (A04 "I want Italian" -> "In South" must carry Italian
                # forward; without this, T2 sees an empty memory).
                if _args.get("cuisine"):
                    memory.update_from_planner({"cuisine": _args["cuisine"]})
                return {
                    "plan": "reply",
                    "reply": "Would you like to add any other preferences such as area, budget, rating, or tags?",
                }
        # END PHASE-1 CUISINE-ONLY GUARD

        # PHASE-1 FILTER CARRY-OVER
        # When the user progressively reveals filters across turns, qwen3:4b
        # + think=false often emits only the newly-mentioned filter, dropping
        # ones already in memory. Merge memory filters into args so the
        # search sees the full intent. Args already-present take precedence
        # (user may be overriding a prior value).
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
        # END PHASE-1 FILTER CARRY-OVER
        # END SAFETY CHECKS

        # MEMORY UPDATES ONLY HERE
        phase = memory.state.get("phase")

        if validated and validated.get("plan") == "execute":
            action = validated.get("action")
            args = validated.get("args", {}) or {}
            cleaned = strip_memory_if_unsupported(args)
            #  If Phase-2 and create_reservation has no email → ask for email instead of executing tool
            if phase == "availability":
                if action == "create_reservation" and cleaned.get("customer_email"):

                    
                    memory.update_from_planner({
                        "restaurant": cleaned.get("restaurant"),
                        "date": cleaned.get("date"),
                        "time": cleaned.get("time"),
                        "party_size": cleaned.get("party_size"),
                        "phase": "booking",
                    })

                    # 2THEN ask for email
                    return {
                        "plan": "reply",
                        "reply": "Which email should I use for the reservation?"
                    }


            # continue with your normal execution logic

            logger.info("===== EXECUTE ACTION DETECTED: %s =====", action)
            logger.info("===== EXECUTE ARGS ===== %s", args)

            # normalize naming
            if "restaurant_name" in args and "restaurant" not in args:
                args["restaurant"] = args.pop("restaurant_name")
            validated["args"] = args

            # FIELD-SAFETY CLEANING **(cleaned is created here!)**

            logger.info("===== CLEANED ARGS ===== %s", cleaned)
            logger.info("===== MEMORY BEFORE UPDATE ===== %s", memory.state)

            # Once all fields are known, capture the email and ask for
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
                    # Persist the provided email (we already cleaned it)
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
                # sends unavailable-slot turns (D01) into the booking phase.

            elif action == "create_reservation":
                # Persist the booking fields carried over from the availability step.
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

        # SAFETY FIX: never update memory on reply in phase 2
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

        return {"plan": "reply", "reply": "Could you clarify — would you like to search or book a specific restaurant?"}

    except Exception as e:
        logger.warning(f"Planner failed: {e}")
        return mock_planner_output(user_text)


# POST-DISPATCH PHASE HOOK
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