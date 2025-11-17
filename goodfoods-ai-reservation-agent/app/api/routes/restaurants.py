# app/api/routes/restaurants.py
from fastapi import APIRouter
from sqlalchemy import func
import logging
from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant, Reservation
from app.api.utils.slot_manager import get_available_slots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/restaurants", tags=["Restaurants"])

@router.get("/", summary="Get all restaurants")
def get_restaurants(limit: int = 10):
    logger.info("[API] /restaurants → Fetching all restaurants")
    session = SessionLocal()
    restaurants = session.query(Restaurant).limit(limit).all()
    data = [{"id": r.id, "name": r.unit_name, "zone": r.zone, "rating": r.rating} for r in restaurants]
    session.close()
    logger.info(f"[API] Returned {len(data)} restaurants")
    return data


@router.get("/{restaurant_id}/menu", summary="Get restaurant menu")
def get_menu(restaurant_id: int):
    logger.info(f"[API] /restaurants/{restaurant_id}/menu → Fetching menu")
    session = SessionLocal()
    menu = session.query(MenuItem).filter_by(restaurant_id=restaurant_id).all()
    data = [{"item": m.item, "price": m.price, "veg": m.veg} for m in menu]
    session.close()
    logger.info(f"[API] Returned {len(data)} menu items")
    return data


@router.get("/{restaurant_id}/slots", summary="Check available slots for a restaurant")
def check_availability(restaurant_id: int, date: str, party_size: int):
    logger.info(f" [API] /restaurants/{restaurant_id}/slots → Checking availability for {date}, party_size={party_size}")
    result = get_available_slots(restaurant_id, date, party_size)
    if "error" in result:
        logger.warning(f" [API] Slot check failed: {result['error']}")
        return {"error": result["error"]}
    logger.info("[API] Slot check successful")
    return result


@router.get("/{restaurant_id}/analytics", summary="Get restaurant analytics")
def restaurant_analytics(restaurant_id: int):
    logger.info(f"[API] /restaurants/{restaurant_id}/analytics → Calculating analytics")
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
    logger.info(f"✅ [API] Analytics calculated for restaurant_id={restaurant_id}")
    return {
        "restaurant_id": restaurant_id,
        "total_reservations": total_reservations,
        "average_party_size": round(avg_party_size or 0, 2),
        "top_slots": [{"time": t, "count": c} for t, c in popular_slots]
    }
