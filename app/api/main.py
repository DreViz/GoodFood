# app/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    shortcuts,
)
from app.config import get_settings

load_dotenv()

settings = get_settings()

app = FastAPI(
    title="GoodFoods Reservation API",
    version="1.0.0",
    description="AI-powered restaurant reservation and dining management system",
)

# CORS — allow the Next.js frontend (localhost:3000) and legacy Streamlit (8501).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(restaurants.router, prefix="/restaurants", tags=["Restaurants"])
app.include_router(reservations.router, prefix="/reservations", tags=["Reservations"])
app.include_router(customers.router, prefix="/customers", tags=["Customers"])
app.include_router(notifications.router, prefix="/notifications", tags=["Notifications"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(agent.router, prefix="/agent", tags=["Agent"])
app.include_router(availability.router, prefix="/availability", tags=["Availability"])
app.include_router(shortcuts.router, tags=["Shortcuts"])



# Called by the frontend on every page refresh so stale conversation state
# doesn't leak into a new session.
@app.post("/agent/memory/reset", tags=["Agent Memory"])
def reset_conversation_memory():
    try:
        memory.reset()
        return {"ok": True, "message": " Conversation memory cleared successfully."}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/", tags=["Root"])
def root():
    return {"message": "Welcome to GoodFoods Reservation API "}
