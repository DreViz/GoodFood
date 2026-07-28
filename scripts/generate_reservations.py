# scripts/generate_reservations.py
import random
from datetime import datetime, timedelta
from app.data.db_connection import SessionLocal
from app.data.db_models import Reservation, Customer, Restaurant

def generate_random_datetime():
    """Generate a random date & time within the next 10 days."""
    start_date = datetime.now()
    random_days = random.randint(1, 10)
    random_hour = random.choice([12, 13, 19, 20, 21])  # realistic restaurant hours
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d"), f"{random_hour}:00"

def generate_reservations(n=15):
    session = SessionLocal()

    customers = session.query(Customer).all()
    restaurants = session.query(Restaurant).all()

    if not customers or not restaurants:
        print("No customers or restaurants found. Please populate them first.")
        session.close()
        return

    for _ in range(n):
        customer = random.choice(customers)
        restaurant = random.choice(restaurants)
        date, time = generate_random_datetime()
        party_size = random.randint(2, 6)
        seating_preference = random.choice([
            "Window-side", "Near Fountain", "Indoor dining", 
            "Private dining room", "Bar high-tops"
        ])

        reservation = Reservation(
            customer_id=customer.id,
            restaurant_id=restaurant.id,
            date=date,
            time=time,
            party_size=party_size,
            seating_preference=seating_preference,
            status=random.choice(["confirmed", "completed", "cancelled"])
        )

        session.add(reservation)

    session.commit()
    print(f"Successfully inserted {n} random reservations.")
    session.close()

if __name__ == "__main__":
    generate_reservations(20)
