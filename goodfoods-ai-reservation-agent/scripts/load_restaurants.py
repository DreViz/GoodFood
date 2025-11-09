# scripts/load_restaurants.py
import json
from sqlalchemy.exc import IntegrityError
from app.data.db_connection import SessionLocal, init_db
from app.data.db_models import Restaurant

init_db()
session = SessionLocal()

print("Creating tables (if missing)...")

with open("app/data/goodfoods_locations_unique_50.json", "r", encoding="utf-8") as f:
    restaurants_data = json.load(f)

inserted, skipped, failed = 0, 0, 0

for rest in restaurants_data:
    try:
        restaurant = Restaurant(
            location_id=rest["location_id"],
            brand=rest.get("brand"),
            unit_name=rest.get("unit_name"),
            zone=rest.get("zone"),
            address=rest.get("address"),
            pincode=rest.get("pincode"),
            latitude=rest.get("latitude"),
            longitude=rest.get("longitude"),
            rating=rest.get("rating"),
            avg_price_per_person=rest.get("avg_price_per_person"),
            capacity=rest.get("capacity"),
            description=rest.get("description"),
            contact_phone=rest.get("contact", {}).get("phone"),
            contact_email=rest.get("contact", {}).get("email"),
            cuisines=rest.get("cuisines", []),
            amenities=rest.get("amenities", []),
            seating_sections=rest.get("seating_sections", []),
            menu=rest.get("menu", []),
            policies=rest.get("policies", {}),
            opening_hours=rest.get("opening_hours", {}),
            ops_metrics=rest.get("ops_metrics", {}),
            contact=rest.get("contact", {}),
            tags=rest.get("tags", []),
        )

        session.add(restaurant)
        session.commit()
        inserted += 1
        print(f"✅ Inserted: {restaurant.unit_name}")

    except IntegrityError:
        session.rollback()
        skipped += 1
        print(f"⚠️ Skipped duplicate: {rest['unit_name']}")
    except Exception as e:
        session.rollback()
        failed += 1
        print(f"❌ Failed to insert {rest['unit_name']}: {e}")

session.close()
print(f"\n🎉 Done! Inserted: {inserted}, Skipped: {skipped}, Failed: {failed}")


#python -m scripts.load_restaurants