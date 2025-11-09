# app/agent/planner_agent.py
import json
import logging
import requests
import os
import re
from app.agent.mock_llm import mock_planner_output
from app.agent.conversation_memory import ConversationMemory

# ---------------- Setup ----------------
logger = logging.getLogger(__name__)

# Load the planner prompt
PLANNER_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "planner_prompt.txt")
with open(PLANNER_PROMPT_PATH, "r", encoding="utf-8") as f:
    PLANNER_PROMPT = f.read()

# Ollama configuration
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"  # Chat endpoint for llama3
MODEL = "llama3.2:3b"
TIMEOUT = 120

# Global memory instance
memory = ConversationMemory()


# ---------------- Helper: JSON Validator & Repair ----------------
def repair_malformed_json(candidate: str) -> str:
    """
    Try basic repairs on slightly-broken JSON produced by LLMs:
    - remove trailing commas before } or ]
    - collapse multiple newlines
    - strip leading/trailing junk outside the outermost {...}
    Returns the best-effort repaired JSON string.
    """
    if not candidate:
        return candidate

    # Keep only the first {...} block if multiple braces
    first_open = candidate.find("{")
    last_close = candidate.rfind("}")
    if first_open != -1 and last_close != -1 and last_close > first_open:
        candidate = candidate[first_open:last_close + 1]

    # remove trailing commas like: "date": "2025-11-09",}
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    # remove repeated commas: ", ,"
    candidate = re.sub(r",\s*,+", ",", candidate)

    # collapse multiple newlines and whitespace
    candidate = re.sub(r"\n{2,}", "\n", candidate).strip()

    return candidate


def validate_planner_json(candidate: str) -> dict | None:
    """
    Validates planner JSON structure before parsing. Returns dict if valid, else None.
    This mirrors the earlier validator but accepts repaired JSON too.
    """
    if not candidate:
        return None

    # quick repair attempt
    candidate = repair_malformed_json(candidate)

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    # Ensure it's a dict with required top-level keys
    if not isinstance(parsed, dict) or "plan" not in parsed:
        return None

    plan = parsed.get("plan")

    # Validate plan type
    if plan not in ("reply", "execute"):
        return None

    # Schema: reply
    if plan == "reply":
        reply = parsed.get("reply", "")
        if isinstance(reply, str) and 4 <= len(reply) <= 200:
            return parsed
        return None

    # Schema: execute
    if plan == "execute":
        if "action" not in parsed or "args" not in parsed:
            return None
        action = parsed["action"]
        args = parsed["args"]
        if not isinstance(args, dict) or not isinstance(action, str):
            return None

        # Disallow obvious placeholders
        invalid_tokens = ["<", ">", "your_action", "optional", "ask", "question", "what"]
        if any(tok in json.dumps(parsed).lower() for tok in invalid_tokens):
            return None

        # Ensure action is known
        allowed_actions = {
            "search_restaurants_by_filters",
            "recommend_venues",
            "get_restaurant_info",
            "check_availability",
            "create_reservation",
            "get_seating_map",
        }
        if action not in allowed_actions:
            return None
        return parsed

    return None


# ---------------- Planner Function ----------------
def call_planner_llm(
    user_text: str,
    context: str = "",
    recent_results: list = None,
    customer_profile: dict = None,
) -> dict:
    """
    Sends user query to the planner LLM to decide the next system action.
    Integrates persistent memory for tracking user intent (cuisine, restaurant, date, etc.).
    """

    recent_results = recent_results or []
    customer_profile = customer_profile or {}

    # Step 1: Update conversation memory from user message
    memory.update_from_user(user_text)

    # Step 2: Merge context
    planner_context = {
        "user_message": user_text,
        "recent_results": recent_results,
        "customer_profile": customer_profile,
    }
    merged_context = memory.merge_into_context(planner_context)

    # Step 3: Prepare structured prompt
    memory_json = json.dumps(merged_context.get("memory", {}), indent=2, ensure_ascii=False)
    recent_results_json = json.dumps(recent_results, indent=2, ensure_ascii=False)
    customer_profile_json = json.dumps(customer_profile, indent=2, ensure_ascii=False)

    full_prompt = (
        f"{PLANNER_PROMPT}\n\n---\n"
        f"Conversation Memory (Persisted User Details):\n{memory_json}\n\n"
        f"Recent Results (JSON):\n{recent_results_json}\n\n"
        f"Customer Profile (JSON):\n{customer_profile_json}\n\n"
        f"User Message:\n{user_text}\n\n"
        "Respond ONLY with one valid JSON object (no text outside JSON)."
    )

    # Step 4: Call Ollama Chat API
    try:
        logger.info(f" [Planner Input Context]: {json.dumps(merged_context, indent=2, ensure_ascii=False)}")
        payload = {
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "stop": ["}\n", "}\r", "}\r\n", "}\n\n"]
        }

        # Debug print (optional)
        print(payload)

        r = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        # Extract chat message
        raw_response = (
            str(
                data.get("message", {}).get("content")
                or data.get("response")
                or data.get("text")
                or ""
            ).strip()
        )
        logger.debug(f"Raw planner response: {raw_response}")

        # Extract JSON substring (best effort)
        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}")
        candidate = raw_response[json_start:json_end + 1] if json_start != -1 and json_end != -1 else raw_response

        # Try to repair and validate
        validated = validate_planner_json(candidate)
        if validated:
            # Before returning, ensure we normalize keys so downstream dispatch sees expected fields.
            # e.g. if memory uses 'restaurant' make sure planner's args include 'restaurant' (not restaurant_name)
            if validated.get("plan") == "execute":
                args = validated.get("args", {})
                # normalize keys (planner might output restaurant_name)
                if "restaurant_name" in args and "restaurant" not in args:
                    args["restaurant"] = args.pop("restaurant_name")
                validated["args"] = args
            return validated

        # If not valid, attempt a second pass by repairing entire raw_response
        repaired = repair_malformed_json(raw_response)
        validated2 = validate_planner_json(repaired)
        if validated2:
            if validated2.get("plan") == "execute":
                args = validated2.get("args", {})
                if "restaurant_name" in args and "restaurant" not in args:
                    args["restaurant"] = args.pop("restaurant_name")
                validated2["args"] = args
            return validated2

        logger.warning(f" Invalid or placeholder planner JSON: {candidate}")
        # fallback: ask the user a clarifying question
        return {"plan": "reply", "reply": "Could you clarify — would you like to search or book a specific restaurant?"}

    except Exception as e:
        logger.warning(f" Planner LLM failed, using fallback mock: {e}")
        return mock_planner_output(user_text)
