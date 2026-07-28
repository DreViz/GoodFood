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

    try:
        settings = get_settings()
        payload = {
            "model": settings.ollama_model,
            "prompt": full_prompt,
            "stream": False,
            # qwen3: disable reasoning for latency + clean JSON. Any residual
            # preamble is removed by strip_model_reasoning below.
            "think": settings.ollama_think,
        }

        r = requests.post(settings.ollama_generate_url, json=payload, timeout=settings.ollama_timeout)
        r.raise_for_status()
        data = r.json()

        #taking response from ollama
        raw_response = (
            data.get("message", {}).get("content")
            or data.get("response")
            or data.get("text")
            or ""
        ).strip()

        # qwen3 emits a reasoning preamble ending in </think> even with
        # think disabled — strip it before any JSON extraction.
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
        # only treat as "provide email" when there's clear intent keywords
        email_intent_regex = re.compile(r"\b(use|email is|my email|here is my email|here is|contact email|to create|to book|use email)\b")
        pure_email_intent = bool(email_match and email_intent_regex.search(lower_text))

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
                # If time is present → auto-transition to booking
                if cleaned.get("time"):
                    memory.update_from_planner({"phase": "booking"})

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