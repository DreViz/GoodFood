# app/agent/conversation_memory.py

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


"""
Conversation Memory for GoodFoods Concierge
Pure storage only — NO NLP, NO parsing.
Planner controls everything.
"""


class ConversationMemory:
    def __init__(self):
        # full unified state (everything the planner may choose to store)
        self.state: Dict[str, Any] = {
            "phase": "discovery",
            "cuisine": None,
            "restaurant": None,
            "location_id": None,
            "location": None,
            "date": None,
            "time": None,
            "party_size": None,
            "customer_email": None,
            "seating_pref": None,
        }

    # --------------------------------------------------------------
    # ONLY PLANNER MAY UPDATE MEMORY (NO AI GUESSING)
    # --------------------------------------------------------------
    def update_from_planner(self, data: Dict[str, Any] = None, **kwargs):
        """
        Unified update interface.
        Supports:
            memory.update_from_planner({"restaurant": "GoodFoods Grill"})
            memory.update_from_planner(restaurant="GoodFoods Grill", phase="availability")
        """

        if data is None:
            data = {}

        # merge kwargs
        data.update(kwargs)

        # apply updates
        for key, value in data.items():
            if key not in self.state:
                continue
            if value is None:
                continue  # never overwrite with empty values
            self.state[key] = value

        logger.info(f"[MEMORY] Updated: {self.state}")

    # --------------------------------------------------------------
    def merge_into_context(self, planner_context: dict) -> dict:
        """
        Add memory into planner context so prompts can use it.
        """
        merged = planner_context.copy()
        merged["memory"] = {k: v for k, v in self.state.items() if v is not None}
        return merged

    # --------------------------------------------------------------
    def dump(self):
        """
        For debugging — returns non-None memory keys.
        """
        return {k: v for k, v in self.state.items() if v is not None}

    # --------------------------------------------------------------
    def reset(self):
        """
        Reset entire conversation but return phase to 'discovery'.
        """
        for k in self.state:
            self.state[k] = None
        self.state["phase"] = "discovery"

        logger.info("[MEMORY] Reset to discovery")
