import json
import logging
import requests
import os

from app.agent.llm_utils import strip_model_reasoning
from app.config import get_settings

logger = logging.getLogger(__name__)

RESPONDER_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "responder_prompt.txt")
with open(RESPONDER_PROMPT_PATH, "r", encoding="utf-8") as f:
    RESPONDER_PROMPT = f.read()


def normalize_reservation_error(error_text: str) -> str:
    """
    Converts raw SQL/technical errors into user-friendly messages.
    """

    if not error_text:
        return "Something went wrong."

    txt = error_text.lower()

    # Duplicate booking constraint
    if "unique_customer_booking_per_slot" in txt or "duplicate key" in txt:
        return (
            "You already have a reservation for this restaurant at that date and time."
        )

    # Invalid restaurant
    if "restaurant" in txt and "not found" in txt:
        return "I couldn’t find that restaurant."

    # Invalid email
    if "invalid customer_email" in txt:
        return "The email address doesn’t seem to be valid."

    # Invalid date or time
    if "missing required date" in txt:
        return "The reservation date is missing."
    if "missing required time" in txt:
        return "The reservation time is missing."

    # DB or system-level error fallback
    if "psycopg2" in txt or "sqlalchemy" in txt:
        return "There was a system issue while creating the reservation."

    # Final fallback
    return error_text

def format_reservation_message(result: dict) -> str:
    """Format a user-friendly reservation confirmation or error message."""

    if result.get("ok"):
        # reservation stands even when the email fails — say so explicitly
        if result.get("email_sent") is False:
            return (
                f"Your reservation at {result['restaurant']} is confirmed for "
                f"{result.get('date')} at {result.get('time')}. "
                "However, we couldn’t send a confirmation email to that address. "
                "Would you like to update the email?"
            )

        return (
            f"Your reservation is confirmed at {result['restaurant']} on "
            f"{result.get('date')} at {result.get('time')}. "
            f"The confirmation email has been sent to {result.get('customer_email')}. "
            "Would you like anything else?"
        )

    friendly_error = normalize_reservation_error(result.get("error"))
    return (
        f"{friendly_error} "
        "Would you like to try a different time?"
    )




def call_responder_llm(user_text: str, tool_output: dict) -> str:
    """
    Converts structured tool_output into a natural, conversational reply.
    No cleaning or post-processing.
    """

    logger.info(f"[RESPONDER] Received structured payload: {json.dumps(tool_output, indent=2, ensure_ascii=False)}")

    # reservation confirmations are formatted deterministically — no LLM
    # round-trip for the one message that must never be wrong
    if tool_output.get("action") == "create_reservation":
        return format_reservation_message(tool_output.get("result", {}))

    try:
        settings = get_settings()
        # /api/chat + a strict JSON schema so the responder cannot emit a
        # reasoning preamble: forcing `{"reply": "<string>"}` via format=
        # schema keeps decoding constrained to valid JSON tokens, which
        # suppresses the reasoning phase entirely (same trick as the planner).
        # The reply text is then extracted from the parsed object.
        r = requests.post(
            settings.ollama_chat_url,
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": RESPONDER_PROMPT.strip()},
                    {"role": "user", "content": (
                        f"User said: {user_text.strip()}\n"
                        f"Tool result (structured):\n{json.dumps(tool_output, indent=2, ensure_ascii=False)}\n\n"
                        "Write the final user-facing reply as JSON: "
                        '{"reply": "<your reply text>"}. '
                        "Do not include any other keys or commentary."
                    )},
                ],
                "stream": False,
                "think": settings.ollama_think,
                "format": {
                    "type": "object",
                    "properties": {
                        "reply": {"type": "string"},
                    },
                    "required": ["reply"],
                    "additionalProperties": False,
                },
            },
            timeout=settings.ollama_timeout,
        )
        r.raise_for_status()

        data = r.json()
        raw_content = (
            data.get("message", {}).get("content")
            or data.get("response")
            or ""
        ).strip()

        # Two-layer extraction: try JSON first, fall back to plain text.
        cleaned = ""
        try:
            obj = json.loads(raw_content)
            if isinstance(obj, dict) and isinstance(obj.get("reply"), str):
                cleaned = obj["reply"].strip()
        except Exception:
            # fall back to stripping a reasoning preamble
            cleaned = strip_model_reasoning(raw_content)

        if not cleaned or len(cleaned) < 2:
            cleaned = "I'm here to help — could you repeat that?"

        return cleaned

    except Exception as e:
        logger.warning(f"Responder LLM failed: {e}")
        return "I'm sorry, something went wrong while generating my response."
