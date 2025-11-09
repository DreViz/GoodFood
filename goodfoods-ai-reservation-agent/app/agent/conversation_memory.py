"""
Persistent conversational memory for GoodFoods Concierge.
Tracks user intent, preferences, and context across multiple turns in a session.
A new conversation (fresh context) starts only when the app or page reloads.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any


class ConversationMemory:
    def __init__(self):
        # Persistent memory during session lifetime
        # NOTE: use 'restaurant' (not restaurant_name) to match planner prompt expectations
        self.state: Dict[str, Any] = {
            "cuisine": None,
            "restaurant": None,
            "location": None,
            "date": None,
            "time": None,
            "party_size": None,
            "customer_email": None,
            "intent": None,  # e.g., "search" or "book"
        }

    # ------------------------------------------------------
    # 🧠 Main update function
    # ------------------------------------------------------
    def update_from_user(self, user_message: str):
        """Parse and update memory from a user's natural language message."""
        if not user_message:
            return

        text = user_message.lower().strip()
        if self.state.get("intent") == "book" and not self.state.get("date"):
            self.state["date"] = datetime.now().strftime("%Y-%m-%d")
        # Temp dict to avoid overwriting existing fields with None
        detected_values: Dict[str, Any] = {}

        # --- Intent detection ---
        if re.search(r"\b(book|reserve|booking|booked|make a reservation|i'd like to book|i want to book)\b", text):
            detected_values["intent"] = "book"
        elif re.search(r"\b(search|find|show|looking for|recommend)\b", text):
            if self.state.get("intent") != "book":
                detected_values["intent"] = "search"

        # --- Detect cuisine ---
        cuisines = ["italian", "indian", "mexican", "chinese", "thai", "continental", "vegan", "asian", "mughlai"]
        for c in cuisines:
            if re.search(rf"\b{re.escape(c)}\b", text):
                detected_values["cuisine"] = c.capitalize()

        # --- Detect date ---
        if re.search(r"\btoday\b", text):
            detected_values["date"] = datetime.now().strftime("%Y-%m-%d")
        elif re.search(r"\btomorrow\b", text):
            tomorrow = datetime.now() + timedelta(days=1)
            detected_values["date"] = tomorrow.strftime("%Y-%m-%d")
        else:
            # detect "11th November", "11 Nov", "9/11", "2025-11-09"
            date_match = re.search(
                r"(\d{1,2})(st|nd|rd|th)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*",
                text
            )
            if date_match:
                day = int(date_match.group(1))
                month_str = date_match.group(3)[:3].title()
                try:
                    date_obj = datetime.strptime(f"{day} {month_str} {datetime.now().year}", "%d %b %Y")
                    if date_obj.date() < datetime.now().date():
                        date_obj = datetime.strptime(f"{day} {month_str} {datetime.now().year + 1}", "%d %b %Y")
                    detected_values["date"] = date_obj.strftime("%Y-%m-%d")
                except Exception:
                    pass
            else:
                iso_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
                if iso_match:
                    detected_values["date"] = iso_match.group(1)

        # --- Detect time ---
        time_match = re.search(r"(\b\d{1,2})([:\.](\d{2}))?\s?(am|pm)?\b", text)
        if time_match:
            hour = int(time_match.group(1))
            minute = time_match.group(3)
            minute = int(minute) if minute else 0
            suffix = time_match.group(4)
            if suffix:
                suffix = suffix.lower()
            if suffix == "pm" and hour < 12:
                hour += 12
            if suffix == "am" and hour == 12:
                hour = 0
            if 0 <= hour < 24 and 0 <= minute < 60:
                detected_values["time"] = f"{hour:02d}:{minute:02d}"

        # --- Detect party size ---
        party_match = re.search(r"\bfor\s+(\d{1,2})\b", text)
        if party_match:
            try:
                detected_values["party_size"] = int(party_match.group(1))
            except Exception:
                pass
        else:
            party_match2 = re.search(r"\bparty of\s+(\d{1,2})\b", text)
            if party_match2:
                try:
                    detected_values["party_size"] = int(party_match2.group(1))
                except Exception:
                    pass

        # --- Detect restaurant ---
        rest_match = re.search(r"\b(?:book|reserve|i want to book|i'd like to book)?\s*(?:at\s+)?(goodfoods(?:\s+[A-Za-z0-9&\-\']+)?)\b", text)
        if rest_match:
            name = rest_match.group(1).strip()
            name = " ".join([w.capitalize() for w in name.split()])
            detected_values["restaurant"] = name

        short_rest_match = re.search(r"\b(book|reserve)\s+(grill|bistro|cantina|garden|atrium|commons|cafe|hub|club)\b", text)
        if short_rest_match and not self.state.get("restaurant"):
            candidate = short_rest_match.group(2).capitalize()
            detected_values["restaurant"] = f"GoodFoods {candidate}"

        # --- Detect location ---
        locations = ["koramangala", "hebbal", "whitefield", "indiranagar", "marathahalli", "south", "east", "north"]
        for loc in locations:
            if re.search(rf"\b{re.escape(loc)}\b", text):
                detected_values["location"] = loc.title()
                break

        # --- Detect and validate email ---
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.[a-z]{2,}", text)
        if email_match:
            email = email_match.group(0)
            # Ensure domain part is valid like .com, .net, .org etc.
            domain_part = email.split("@")[-1]
            if "." in domain_part and len(domain_part.split(".")[-1]) >= 2:
                detected_values["customer_email"] = email

        # --- Merge detected values into memory (non-destructive) ---
        for key, value in detected_values.items():
            if value:
                self.state[key] = value

    # ------------------------------------------------------
    # 🧩 Utilities
    # ------------------------------------------------------
    def merge_into_context(self, planner_context: dict) -> dict:
        """Inject stored fields into planner context before LLM call."""
        merged = planner_context.copy()

        merged_memory = {
            k: v for k, v in self.state.items()
            if v is not None or k == "customer_email"
        }

        merged["memory"] = merged_memory
        return merged


    def reset(self):
        """Reset memory completely (e.g., when user refreshes the page)."""
        for key in self.state.keys():
            self.state[key] = None

    def dump(self) -> Dict[str, Any]:
        """Return current memory for debugging/logging."""
        return {k: v for k, v in self.state.items() if v is not None}
    
