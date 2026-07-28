# scripts/generate_customers.py
import random
from faker import Faker
from app.data.db_connection import SessionLocal
from app.data.db_models import Customer

fake = Faker()

def generate_random_customer():
    cuisines = ["Italian", "Chinese", "Indian", "Mexican", "Thai", "Japanese"]
    seating_preferences = [
        "Near Fountain", "Window-side", "Bar high-tops", 
        "Private dining", "Indoor dining", "Near live kitchen"
    ]
    allergies = ["nuts", "gluten", "dairy", "soy", "seafood", None]

    return Customer(
        name=fake.name(),
        email=fake.email(),
        allergies=random.sample([a for a in allergies if a], k=random.randint(0, 2)),
        preferred_cuisines=random.sample(cuisines, k=random.randint(1, 2)),
        avoid_music=random.choice([True, False]),
        seating_preference=random.choice(seating_preferences)
    )

def populate_customers(n=10):
    session = SessionLocal()
    customers = [generate_random_customer() for _ in range(n)]
    session.add_all(customers)
    session.commit()
    print(f"Successfully inserted {n} random customers.")
    session.close()

if __name__ == "__main__":
    populate_customers(10)
