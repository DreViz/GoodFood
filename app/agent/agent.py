# app/agent/agent.py
import json
import logging
from typing import Any, Dict, Optional

from app.agent.planner_agent import call_planner_llm
from app.agent.responder_agent import call_responder_llm
from app.agent.tool_calls import dispatch_tool

logger = logging.getLogger(__name__)


# -----------------------------------------
# PER-USER CONTEXT (recent search results)
# -----------------------------------------
recent_results: list = []


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

    logger.info(f"Incoming user message: {user_text}")
    logger.info(f"[AGENT] Memory BEFORE planner: {call_planner_llm.__module__}")

    # 1 — CALL PLANNER
    raw_plan = call_planner_llm(
        user_text,
        context or "",
        recent_results=recent_results,
    )
    logger.info(f"Raw planner output: {raw_plan}")

    # 2 — VALIDATE JSON
    try:
        plan_obj = parse_planner_output(raw_plan)
        logger.info(f"[AGENT] Parsed planner JSON: {plan_obj}")
    except Exception:
        logger.warning("Planner returned invalid JSON — fallback reply.")
        return {"reply": "Could you clarify what you'd like me to do?"}

    plan = plan_obj.get("plan")
    if not plan:
        return {"reply": "Could you clarify what you'd like me to do?"}

    # 3 — PLAN → reply
    if plan == "reply":
        reply_txt = plan_obj.get("reply", "").strip()
        logger.info(f"[AGENT] reply text: {reply_txt}")
        return {"reply": reply_txt}

    # 4 — PLAN → execute
    if plan == "execute":
        action = plan_obj.get("action")
        args = plan_obj.get("args", {}) or {}

        # EXECUTE TOOL via dispatch_tool
        try:
            logger.info(f"Running tool: {action} args={args}")
            raw_tool_output = dispatch_tool(action, args)
            logger.info(f"[AGENT] Tool '{action}' output: {raw_tool_output}")

            tool_result = (
                raw_tool_output.get("result")
                if isinstance(raw_tool_output, dict) and "result" in raw_tool_output
                else raw_tool_output
            )

        except Exception as e:
            logger.exception("Tool execution failed for %s", action)
            return {"reply": f"An error occurred while performing '{action}': {e}"}

        # Track recent search results for planner context
        if action in ("search_restaurants_by_filters", "search_restaurants"):
            results = tool_result.get("results") if isinstance(tool_result, dict) else None
            if isinstance(results, list):
                recent_results.clear()
                recent_results.extend([
                    {
                        "id": r.get("id"),
                        "unit_name": r.get("unit_name"),
                        "zone": r.get("zone"),
                        "avg_price_per_person": r.get("avg_price_per_person"),
                        "tags": r.get("tags", []),
                        "cuisines": r.get("cuisines", []),
                    }
                    for r in results
                ])

        # 5 — SEND STRUCTURED PAYLOAD TO RESPONDER
        responder_payload = {
            "action": action,
            "args": args,
            "result": tool_result,
        }

        reply_text = call_responder_llm(user_text, responder_payload)

        return {
            "reply": reply_text,
            "tool_output": responder_payload,
        }

    return {"reply": "Sorry, I couldn't understand that."}
