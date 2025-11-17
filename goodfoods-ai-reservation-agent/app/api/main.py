# app/api/main.py
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from app.agent.planner_agent import memory
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from app.api.routes import (
    restaurants,
    reservations,
    customers,
    notifications,
    analytics,
    agent,
    availability,
)
from app.agent.conversation_memory import ConversationMemory
from app.data.db_connection import engine, Base

load_dotenv()


app = FastAPI(
    title="GoodFoods Reservation API",
    version="1.0.0",
    description="AI-powered restaurant reservation and dining management system",
)


conversation_memory = ConversationMemory()


app.include_router(restaurants.router, prefix="/restaurants", tags=["Restaurants"])
app.include_router(reservations.router, prefix="/reservations", tags=["Reservations"])
app.include_router(customers.router, prefix="/customers", tags=["Customers"])
app.include_router(notifications.router, prefix="/notifications", tags=["Notifications"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(agent.router, prefix="/agent", tags=["Agent"])
app.include_router(availability.router, prefix="/availability", tags=["Availability"])



#  attach memory reset endpoint (called from frontend once per page refresh)
@app.post("/agent/memory/reset", tags=["Agent Memory"])
def reset_conversation_memory():
    """
    Reset the in-memory conversation context.
    Called once by the Streamlit frontend when the page is refreshed.
    """
    try:
        memory.reset()
        return {"ok": True, "message": " Conversation memory cleared successfully."}
    except Exception as e:
        return {"ok": False, "error": str(e)}


#update memory endpoint (called before each chat message)
@app.post("/agent/memory/update", tags=["Agent Memory"])
async def update_conversation_memory(request: Request):
    """
    Update the persistent conversation memory with the latest user message.
    This allows the planner to retain context (like cuisine, date, etc.) across turns.
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        if not text:
            return {"ok": False, "error": "Missing 'text' in request body."}

        conversation_memory.update_from_user(text)
        return {"ok": True, "memory": conversation_memory.state}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/", tags=["Root"])
def root():
    return {"message": "Welcome to GoodFoods Reservation API "}
