# app/agent/llm_interface.py
import requests, json, os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


from app.agent.tools_registry import dispatch_tool
from app.agent.planner_agent import call_planner_llm
from app.agent.responder_agent import call_responder_llm

def chat_with_tools(user_text: str, recent_results=None, customer_profile=None):
    """Main orchestrator between Planner → Tool → Responder."""
    plan = call_planner_llm(user_text, recent_results=recent_results, customer_profile=customer_profile)
    print("\n🧭 PLAN:", plan)

    if plan.get("plan") == "execute":
        tool_name = plan.get("action")
        args = plan.get("args", {})
        tool_output = dispatch_tool(tool_name, args)
        return call_responder_llm(user_text, tool_output)

    elif plan.get("plan") == "reply":
        return plan.get("reply", "I'm here to help — could you please clarify?")
    
    else:
        return "Sorry, I couldn't determine what to do next."