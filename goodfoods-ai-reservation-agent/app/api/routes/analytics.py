# app/api/routes/analytics.py
from fastapi import APIRouter
from sqlalchemy import func
from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant, Customer

router = APIRouter()


@router.get("/summary", summary="Get analytics summary")
def analytics_summary():
    session = SessionLocal()
    try:
        total_restaurants = session.query(Restaurant).count()
        total_customers = session.query(Customer).count()

        avg_rating = session.query(func.avg(Restaurant.rating)).scalar() or 0.0

        # Just for demo: zones distribution
        zones = session.query(Restaurant.zone, func.count(Restaurant.zone)).group_by(Restaurant.zone).all()
        top_zones = sorted(zones, key=lambda z: z[1], reverse=True)[:3]
        zone_summary = [z[0] for z in top_zones]

        return {
            "total_customers": total_customers,
            "total_restaurants": total_restaurants,
            "average_rating": round(avg_rating, 2),
            "top_zones": zone_summary
        }
    finally:
        session.close()


@router.get("/popular-cuisines", summary="Top cuisines across restaurants")
def analytics_popular_cuisines():
    session = SessionLocal()
    try:
        # Each restaurant has cuisines in JSON (list)
        restaurants = session.query(Restaurant).all()
        cuisine_count = {}

        for r in restaurants:
            # Skip if null
            if hasattr(r, "menu_items"):
                for item in r.menu_items:
                    cuisine_count[item.item] = cuisine_count.get(item.item, 0) + 1

        # Top 5 cuisines
        top_cuisines = sorted(cuisine_count.items(), key=lambda x: x[1], reverse=True)[:5]
        return {"top_cuisines": [{"name": c, "count": n} for c, n in top_cuisines]}
    finally:
        session.close()


@router.get("/seating-trends", summary="Customer seating preferences trend")
def analytics_seating_trends():
    session = SessionLocal()
    try:
        # Fetch from customers table
        preferences = session.query(Customer.seating_preference, func.count(Customer.id)).group_by(Customer.seating_preference).all()
        results = [
            {"type": pref[0] or "Not specified", "count": pref[1]}
            for pref in preferences
        ]
        return {"seating_preferences": results}
    finally:
        session.close()
