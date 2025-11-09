# app/agent/mock_llm.py
from typing import Dict, Any
import re

"""
A minimal mock LLM to simulate reasoning + tool-calling.
It lets you test your FastAPI /agent route without Ollama running.
"""
# app/agent/mock_llm.py

def mock_planner_output(user_text: str):
    """
    Fallback response when Ollama is not running or planner fails.
    """
    return {
        "plan": "reply",
        "reply": "Hi there! I'm GoodFoods Concierge. How can I help you with your dining plans today?"
    }

def mock_llm_infer(prompt: str) -> Dict[str, Any]:
    prompt_lower = prompt.lower()

    # --- Intent recognition ---
    if "slot" in prompt_lower or "availability" in prompt_lower:
        return {
            "tool": "check_availability",
            "params": {
                "location_id": 1,
                "date": "2025-11-09",
                "party_size": 2
            }
        }

    elif "book" in prompt_lower or "reserve" in prompt_lower:
        # simple extraction of party size
        party_match = re.search(r"for (\d+)", prompt_lower)
        party_size = int(party_match.group(1)) if party_match else 2

        return {
            "action": "create_reservation",
            "params": {
                "customer_email": "test@example.com",
                "location_id": 1,
                "date": "2025-11-09",
                "time": "19:30",
                "party_size": party_size,
                "seating_preference": "window-side"
            }
        }

    elif "cuisine" in prompt_lower or "recommend" in prompt_lower:
        return {
            "action": "recommend_restaurants",
            "params": {"cuisine": "italian", "budget": 800}
        }

    # --- Default fallback ---
    return {"action": "chat", "response": "I didn’t understand. Try asking about booking or availability."}
