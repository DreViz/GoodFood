# scripts/add_opening_hours.py

from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant
from sqlalchemy import text

# Define a standard opening hours template
DEFAULT_HOURS = {
    "mon_thu": "12:00-15:30,19:00-23:00",
    "fri": "12:00-16:00,19:00-23:30",
    "sat": "12:00-16:00,19:00-23:30",
    "sun": "12:00-16:00,19:00-22:30"
}

def add_opening_hours():
    session = SessionLocal()
    restaurants = session.query(Restaurant).all()

    if not restaurants:
        print("No restaurants found. Did you run load_restaurants.py first?")
        return

    for r in restaurants:
        r.opening_hours = DEFAULT_HOURS

    session.commit()
    session.close()
    print(f"Added opening_hours to {len(restaurants)} restaurants successfully!")

if __name__ == "__main__":
    add_opening_hours()
