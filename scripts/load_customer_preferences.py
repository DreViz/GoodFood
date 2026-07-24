import json
from sqlalchemy.exc import IntegrityError
from datetime import date
from app.data.db_connection import SessionLocal, engine
from app.data.db_models import Base, Customer, CustomerPreferences

# --- STEP 1: Ensure table exists ---
print("📦 Creating missing tables if any...")
Base.metadata.create_all(bind=engine)

# --- STEP 2: Open session ---
session = SessionLocal()

# --- STEP 3: Optional: Insert demo customers if not present ---
demo_customers = [
    {"name": "John Doe", "email": "john@example.com"},
    {"name": "Alice Smith", "email": "alice@example.com"},
]

for cust_data in demo_customers:
    existing = session.query(Customer).filter_by(email=cust_data["email"]).first()
    if not existing:
        cust = Customer(**cust_data)
        session.add(cust)
        session.commit()
        print(f"✅ Added demo customer: {cust.email}")
    else:
        print(f"⚠️ Customer already exists: {existing.email}")

# --- STEP 4: Insert customer preferences ---
preferences_data = [
    {
        "email": "john@example.com",
        "cuisine": "Italian",
        "vibe_tags": "date-night,outdoor",
        "max_price": 1200,
        "guests": 2,
        "date": date(2025, 11, 10),
    },
    {
        "email": "alice@example.com",
        "cuisine": "North Indian",
        "vibe_tags": "family-friendly",
        "max_price": 800,
        "guests": 4,
        "date": date(2025, 11, 11),
    },
]

for pref_data in preferences_data:
    try:
        # Get customer ID by email
        cust = session.query(Customer).filter_by(email=pref_data["email"]).first()
        if not cust:
            print(f"❌ Customer not found for {pref_data['email']} — skipping.")
            continue

        # Check if preferences already exist
        existing_pref = (
            session.query(CustomerPreferences)
            .filter_by(customer_id=cust.id)
            .first()
        )
        if existing_pref:
            print(f"⚠️ Preferences already exist for {cust.email}")
            continue

        pref = CustomerPreferences(
            customer_id=cust.id,
            cuisine=pref_data["cuisine"],
            vibe_tags=pref_data["vibe_tags"],
            max_price=pref_data["max_price"],
            guests=pref_data["guests"],
            date=pref_data["date"],
        )

        session.add(pref)
        session.commit()
        print(f"✅ Added preferences for {cust.email}")

    except IntegrityError as e:
        session.rollback()
        print(f"⚠️ Skipped duplicate: {pref_data['email']} - {e}")

session.close()
print("🎉 Customer preferences loaded successfully!")
