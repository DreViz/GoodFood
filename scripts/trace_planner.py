"""
Reproduce the A02 failure path: invoke call_planner_llm with the same arguments
process_user_query uses, with INFO logging enabled, to see what the LLM
actually returns and why validate_planner_json rejects it.
"""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

from app.agent.planner_agent import call_planner_llm, memory

# Mirror what evaluate.py does before each conversation.
memory.reset()
print(f"\n[probe] memory.state after reset = {memory.state}\n")

# Mirror process_user_query call signature (no customer_profile passed).
result = call_planner_llm("Any Italian places in South?", "", recent_results=[])
print(f"\n[probe] FINAL RESULT: {result}")
print(f"[probe] memory.state after call: {memory.state}")
