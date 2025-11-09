# app/api/routes/restaurants.py
from fastapi import APIRouter
from sqlalchemy import func
from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant
from app.api.utils.slot_manager import get_available_slots
from app.data.db_models import Reservation

router = APIRouter()

@router.get("/", summary="Get all restaurants")
def get_restaurants(limit: int = 10):
    session = SessionLocal()
    restaurants = session.query(Restaurant).limit(limit).all()
    data = [{"id": r.id, "name": r.unit_name, "zone": r.zone, "rating": r.rating} for r in restaurants]
    session.close()
    return data

@router.get("/{restaurant_id}/menu", summary="Get restaurant menu")
def get_menu(restaurant_id: int):
    session = SessionLocal()
    menu = session.query(MenuItem).filter_by(restaurant_id=restaurant_id).all()
    data = [{"item": m.item, "price": m.price, "veg": m.veg} for m in menu]
    session.close()
    return data

@router.get("/{restaurant_id}/slots", summary="Check available slots for a restaurant")
def check_availability(restaurant_id: int, date: str, party_size: int):
    result = get_available_slots(restaurant_id, date, party_size)
    if "error" in result:
        return {"error": result["error"]}
    return result

@router.get("/{restaurant_id}/analytics", summary="Get restaurant analytics")
def restaurant_analytics(restaurant_id: int):
    session = SessionLocal()

    total_reservations = session.query(Reservation).filter_by(restaurant_id=restaurant_id).count()
    avg_party_size = session.query(func.avg(Reservation.party_size)).filter_by(restaurant_id=restaurant_id).scalar()
    popular_slots = (
        session.query(Reservation.time, func.count(Reservation.id))
        .filter_by(restaurant_id=restaurant_id)
        .group_by(Reservation.time)
        .order_by(func.count(Reservation.id).desc())
        .limit(3)
        .all()
    )

    session.close()
    return {
        "restaurant_id": restaurant_id,
        "total_reservations": total_reservations,
        "average_party_size": round(avg_party_size or 0, 2),
        "top_slots": [{"time": t, "count": c} for t, c in popular_slots]
    }
