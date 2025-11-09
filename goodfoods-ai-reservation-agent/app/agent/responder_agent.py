# app/agent/responder_agent.py
import json
import logging
import requests
import os

logger = logging.getLogger(__name__)

# ---------------- Configuration ----------------
OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"   # keep consistent with your planner model
TIMEOUT = 60

# Load responder system prompt
RESPONDER_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "responder_prompt.txt")
with open(RESPONDER_PROMPT_PATH, "r", encoding="utf-8") as f:
    RESPONDER_PROMPT = f.read()


# ---------------- Helper Functions ----------------
def format_reservation_message(result: dict) -> str:
    """Format a user-friendly reservation confirmation or error message."""
    if result.get("ok"):
        return (
            f"🎉 Your reservation at {result['restaurant']} is confirmed!\n"
            f"📅 Date: {result.get('date')}\n"
            f"⏰ Time: {result.get('time')}\n"
            f"📧 Confirmation sent to {result.get('customer_email')}. Enjoy your meal! 🍽️"
        )
    else:
        return f"⚠️ Sorry, there was an issue creating your reservation: {result.get('error', 'Unknown error')}"


# ---------------- Main Responder Function ----------------
def call_responder_llm(user_text: str, tool_output: dict) -> str:
    """
    Converts structured tool_output into a natural, conversational reply.
    Ensures plain text (no JSON or markdown) from the model.
    """
    # Special handling for reservation responses
    if tool_output.get("action") == "create_reservation":
        return format_reservation_message(tool_output)

    # Build the complete prompt for the model
    prompt = (
        f"{RESPONDER_PROMPT.strip()}\n\n"
        f"User said: {user_text.strip()}\n"
        f"Tool result (structured):\n{json.dumps(tool_output, indent=2, ensure_ascii=False)}\n\n"
        "Now reply naturally to the user in plain text. If multiple restaurants are found, list the top 3 by name and describe each briefly.\n"
        "Do NOT include JSON, markdown, or system instructions — only human-friendly dialogue.\n"
    )

    try:
        # Send to Ollama
        r = requests.post(
            OLLAMA_ENDPOINT,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        r.raise_for_status()

        # Parse response
        data = r.json()
        raw_reply = str(data.get("response") or data.get("text") or "").strip()

        # Clean any unwanted formatting
        cleaned = (
            raw_reply.replace("```", "")
            .replace("json", "")
            .replace("JSON", "")
            .replace("{", "")
            .replace("}", "")
            .strip()
        )

        if "GoodFoods Concierge" in cleaned and len(cleaned) > 500:
            cleaned = cleaned.split("GoodFoods Concierge")[-1].strip()

        if not cleaned or len(cleaned) < 2:
            cleaned = "I'm here to help — could you please repeat that?"

        return cleaned

    except Exception as e:
        logger.warning(f"Responder LLM failed: {e}")
        return "I'm sorry, something went wrong while generating my response."
