# app/agent/agent.py
import json
import logging
import time
from typing import Any, Dict, Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.agent.planner_agent import call_planner_llm
from app.agent.responder_agent import call_responder_llm
from app.agent.tool_manager import ToolManager
from app.agent.tool_calls import dispatch_tool, TOOL_SPEC

logger = logging.getLogger(__name__)
router = APIRouter()

# -----------------------------------------
# TOOL REGISTRY (AUTO)
# -----------------------------------------
tool_manager = ToolManager()

# Automatically register all tools from TOOL_SPEC
for name, spec in TOOL_SPEC.items():
    schema = spec.get("input_schema", {}).get("properties", {})

    def make_tool_fn(tool_name):
        # bind tool_name in closure
        return lambda **args: dispatch_tool(tool_name, args)

    tool_manager.register(name, make_tool_fn(name), schema)

TOOL_ALIASES = {
    "book_table": "create_reservation",
    "reserve": "create_reservation",
    "find_slots": "check_availability",
    "availability": "check_availability",
    "recommend": "recommend_venues",
    "search": "search_restaurants_by_filters",
    "search_restaurants": "search_restaurants_by_filters",
}

# ----------------- In-memory user context -----------------
# Stores per-user recent_results and optionally customer_profile.
# In production, replace with per-session store (Redis, DB, etc.)
user_context: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------
# MAIN ORCHESTRATION LOGIC
# -----------------------------------------
def process_user_query(user_text: str, user_id: Optional[str] = None, context: Optional[str] = "") -> Dict[str, Any]:
    """
    High-level orchestrator:
    1. Sends user input to the planner model (to get plan JSON), including recent_results
    2. Executes tool calls if needed
    3. Passes results to the responder model for a natural reply
    """
    if not user_id:
        user_id = "anonymous"

    logger.info(f"Processing query for user {user_id}: {user_text}")

    # Ensure we have a context bucket for the user
    if user_id not in user_context:
        user_context[user_id] = {"recent_results": [], "customer_profile": None}

    recent_results = user_context[user_id].get("recent_results", [])
    customer_profile = user_context[user_id].get("customer_profile", None)

    # Step 1: Planner decides plan JSON
    plan_obj = call_planner_llm(user_text, context or "", recent_results=recent_results, customer_profile=customer_profile)
    plan = plan_obj.get("plan")

    logger.info(f"Planner returned: {plan_obj}")

    # Step 2: Handle simple replies
    if plan == "reply":
        reply_text = plan_obj.get("reply", "").strip()
        return {"reply": reply_text}

    # Step 3: Handle tool execution
    if plan == "execute":
        action = plan_obj.get("action")
        args = plan_obj.get("args", {}) or {}

        # Normalize aliases
        if action not in tool_manager.registry:
            action = TOOL_ALIASES.get(action, action)

        tool_entry = tool_manager.get_tool(action)
        if not tool_entry:
            logger.warning(f"Unknown action: {action}")
            return {"reply": f"Sorry, I didn’t recognize that action '{action}'."}

        tool_fn = tool_entry["fn"]

        try:
            logger.info(f"Executing tool: {action} with args {args}")
            tool_output = tool_fn(**args)
        except Exception as e:
            logger.exception(f"Tool execution failed for {action}")
            return {"reply": f"An error occurred while performing '{action}': {e}"}

        # If it was a restaurant search, store the results for future turns
        if action in ("search_restaurants_by_filters", "search_restaurants"):
            results = tool_output.get("results") if isinstance(tool_output, dict) else None
            if isinstance(results, list):
                user_context[user_id]["recent_results"] = [
                    {
                        "id": r.get("id"),
                        "unit_name": r.get("unit_name"),
                        "zone": r.get("zone"),
                        "avg_price_per_person": r.get("avg_price_per_person"),
                        "tags": r.get("tags", []),
                        "cuisines": r.get("cuisines", []),
                    }
                    for r in results
                ]
            else:
                user_context[user_id]["recent_results"] = []
        else:
            pass

        reply_text = call_responder_llm(user_text, tool_output)
        return {"reply": reply_text, "tool_output": tool_output}

    logger.warning("No valid plan received from planner.")
    return {"reply": "Sorry, I couldn’t determine what to do with your request."}


# -----------------------------------------
# STREAMING ENDPOINT (Typing Effect)
# -----------------------------------------
@router.post("/chat/stream")
async def chat_stream(request: Request):
    """
    Streams the assistant's reply word-by-word for a natural typing effect.
    """
    data = await request.json()
    user_query = data.get("query", "").strip()
    context = data.get("context", "")

    # Run the full query once (non-streaming)
    output = process_user_query(user_query, user_id="anonymous", context=context)
    full_reply = output.get("reply", "I'm sorry, I couldn't generate a response.")
    
    # Typing effect generator
    def stream_fake_typing():
        time.sleep(1.0)
        for word in full_reply.split():
            yield f"data: {json.dumps({'message': word + ' '})}\n\n"
            time.sleep(0.05)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_fake_typing(), media_type="text/event-stream")
