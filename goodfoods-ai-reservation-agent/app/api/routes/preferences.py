from fastapi import APIRouter
from pydantic import BaseModel
from app.data.db_connection import SessionLocal
from app.data.db_models import Customer, CustomerPreferences
import datetime

router = APIRouter()

class PrefRequest(BaseModel):
    email: str
    cuisine: str | None = None
    vibe_tags: str | None = None
    max_price: float | None = None
    guests: int | None = None
    date: str | None = None

@router.post("/save")
def save_preferences(req: PrefRequest):
    session = SessionLocal()
    try:
        cust = session.query(Customer).filter_by(email=req.email).first()
        if not cust:
            return {"ok": False, "msg": "customer not found"}

        prefs = session.query(CustomerPreferences).filter_by(customer_id=cust.id).first()
        if not prefs:
            prefs = CustomerPreferences(customer_id=cust.id)
            session.add(prefs)

        prefs.cuisine = req.cuisine
        prefs.vibe_tags = req.vibe_tags
        prefs.max_price = req.max_price
        prefs.guests = req.guests
        prefs.date = datetime.date.fromisoformat(req.date) if req.date else None

        session.commit()
        return {"ok": True, "msg": "Preferences saved"}
    finally:
        session.close()
