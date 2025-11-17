# app/agent/agent.py
import json
import logging
from typing import Any, Dict, Optional

from app.agent.planner_agent import call_planner_llm, memory
from app.agent.responder_agent import call_responder_llm
from app.agent.tool_manager import ToolManager
from app.agent.tool_calls import dispatch_tool, TOOL_SPEC

logger = logging.getLogger(__name__)

# -----------------------------------------
# TOOL REGISTRY
# -----------------------------------------
tool_manager = ToolManager()

# Register tools defined in TOOL_SPEC
for name, spec in TOOL_SPEC.items():
    schema = {arg: Any for arg in spec.get("args", [])}

    def make_tool_fn(tool_name):
        return lambda **args: dispatch_tool(tool_name, args)

    tool_manager.register(name, make_tool_fn(name), schema)

# when llm responds with alias
TOOL_ALIASES = {
    "book_table": "create_reservation",
    "reserve": "create_reservation",
    "find_slots": "check_availability",
    "availability": "check_availability",
    "recommend": "recommend_venues",
    "search": "search_restaurants_by_filters",
    "search_restaurants": "search_restaurants_by_filters",
}


# -----------------------------------------
# PER-USER CONTEXT (recent results only)
# -----------------------------------------
user_context: Dict[str, Dict[str, Any]] = {}


# -----------------------------------------
# PARSE PLANNER OUTPUT
# -----------------------------------------
def parse_planner_output(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            pass
    raise ValueError("Planner output not valid JSON")

# -----------------------------------------
# MAIN PROCESSOR
# -----------------------------------------
def process_user_query(
    user_text: str,
    user_id: Optional[str] = None,
    context: Optional[str] = "",
) -> Dict[str, Any]:

    if not user_id:
        user_id = "anonymous"

    logger.info(f"Incoming user message (user={user_id}): {user_text}")

    # Initialize user bucket
    if user_id not in user_context:
        user_context[user_id] = {
            "recent_results": [],
            "customer_profile": None,
        }

    recent_results = user_context[user_id]["recent_results"]
    customer_profile = user_context[user_id]["customer_profile"]

    logger.info(f"[AGENT] Incoming user text: {user_text}")
    logger.info(f"[AGET] Memory BEFORE planner: {memory.dump()}")


    # 1 — CALL PLANNER

    raw_plan = call_planner_llm(
        user_text,
        context or "",
        recent_results=recent_results,
        customer_profile=customer_profile,
    )
    logger.info(f"Raw planner output: {raw_plan}")


    # 2 — VALIDATE json from planner_agent

    try:
        plan_obj = parse_planner_output(raw_plan)
        logger.info(f"[AGENT] Parsed planner JSON: {plan_obj}")
    except Exception:
        logger.warning("Planner returned invalid JSON — fallback reply.")
        return {"reply": "Could you clarify what you'd like me to do?"}
    
    #get plan from the planner_agent
    plan = plan_obj.get("plan")
    if not plan:
        return {"reply": "Could you clarify what you'd like me to do?"}


    # 3 — PLAN → reply

    if plan == "reply":
        reply_txt = plan_obj.get("reply", "").strip()
        logger.info(f"[AGENT] reply text when plan: {reply_txt}")
        return {"reply": reply_txt}


    # 4 — PLAN → execute

    if plan == "execute":

        action = plan_obj.get("action")
        args = plan_obj.get("args", {}) or {}

        # Makes planner flexible in naming 
        if action not in tool_manager.registry:
            action = TOOL_ALIASES.get(action, action)


        #fetch tool function
        tool_entry = tool_manager.get_tool(action)
        if not tool_entry:
            logger.warning("Planner requested unknown tool: %s", action)
            return {"reply": f"Sorry, I didn’t recognize the action '{action}'."}

        tool_fn = tool_entry["fn"]

        # EXECUTE TOOL from planner_agent

        try:
            #this dispacthces the tool call to tool_manager.execute
            logger.info(f"Running tool: {action} args={args}")
            raw_tool_output = tool_fn(**args)
            logger.info(f"[AGENT] Tool '{action}' output: {raw_tool_output}")

            tool_result = (
                raw_tool_output["result"]
                if isinstance(raw_tool_output, dict) and "result" in raw_tool_output
                else raw_tool_output
            )

        except Exception as e:
            logger.exception("Tool execution failed for %s", action)
            return {"reply": f"An error occurred while performing '{action}': {e}"}

        # ---------------------------------------------------------
        # UPDATE RECENT RESULTS FOR SEARCHES
        # ---------------------------------------------------------
        if action in ("search_restaurants_by_filters", "search_restaurants"):
            results = tool_result.get("results") if isinstance(tool_result, dict) else None

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


        # BACKEND MEMORY UPDATES

        logger.info(f"[AGENT] Memory update triggered by action '{action}' with args: {args}")

        if action == "get_seating_labels":
            memory.update_from_planner({
                "restaurant": args.get("restaurant"),
                "phase": "availability",
            })

        elif action == "check_availability":
            mem_updates = {
                "restaurant": args.get("restaurant"),
                "date": args.get("date"),
                "party_size": args.get("party_size"),
                "phase": "availability",
            }
            if args.get("time") is not None:
                mem_updates["time"] = args.get("time")

            memory.update_from_planner(mem_updates)

        elif action == "create_reservation":
            memory.update_from_planner({"phase": "booking"})

        logger.info(f"[AGENT] Memory AFTER update:  {memory.dump()}")


        # 5 — SEND STRUCTURED PAYLOAD TO RESPONDER

        responder_payload = {
            "action": action,
            "args": args,
            "result": tool_result
        }

        reply_text = call_responder_llm(user_text, responder_payload)

        logger.warning(f"\n\n\n[DEBUG FINAL REPLY]: {reply_text}\n\n")

        return {
            "reply": reply_text,
            "tool_output": responder_payload
        }

    return {"reply": "Sorry, I couldn't understand that."}
