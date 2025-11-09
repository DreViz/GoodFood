# app/api/utils/slot_manager.py

from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant, Reservation
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_available_slots(location_id: int, date: str, party_size: int):
    """
    Get available reservation slots for a restaurant by location_id on a given date.
    Checks total capacity and excludes already booked times.
    """
    session = SessionLocal()
    try:
        # --- Step 1: Validate & find restaurant ---
        location_id = int(location_id)
        restaurant = session.query(Restaurant).filter_by(location_id=location_id).first()

        if not restaurant:
            msg = f"❌ Restaurant not found for location_id={location_id}"
            logger.warning(msg)
            return {"error": msg}

        # --- Step 2: Parse opening hours for the given date ---
        slots = parse_opening_hours(restaurant.opening_hours, date)
        if not slots:
            msg = f"No slots found for {restaurant.unit_name} on {date} (possibly closed)"
            logger.info(msg)
            return {"error": msg}

        available_slots = []

        # --- Step 3: Check booked capacity for each slot ---
        for slot in slots:
            reservations = (
                session.query(Reservation)
                .filter_by(restaurant_id=restaurant.id, date=date, time=slot)
                .all()
            )
            reserved_capacity = sum(r.party_size for r in reservations)
            remaining_capacity = (restaurant.capacity or 50) - reserved_capacity

            if remaining_capacity >= party_size:
                available_slots.append({
                    "time": slot,
                    "remaining_capacity": remaining_capacity
                })

        logger.info(
            f"✅ Available slots for '{restaurant.unit_name}' (loc_id={location_id}) on {date}: "
            f"{[s['time'] for s in available_slots]}"
        )

        return {
            "ok": True,
            "restaurant": restaurant.unit_name,
            "location_id": location_id,
            "date": date,
            "party_size": party_size,
            "available_slots": available_slots
        }

    except Exception as e:
        logger.exception(f"[get_available_slots] Unexpected error: {e}")
        return {"error": str(e)}

    finally:
        session.close()


# --- Utility: Parse opening hours JSON into slot list ---
def parse_opening_hours(opening_hours: dict, date_str: str):
    """
    Parses the opening_hours JSON and returns a list of time slots (e.g. ['12:00', '12:30', ...]).
    Handles weekday matching and fallback gracefully.
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        weekday = date_obj.strftime("%a").lower()

        # Match to key like 'mon_thu', 'fri', 'sat', 'sun'
        matched_hours = None
        for key, hours in (opening_hours or {}).items():
            if key.startswith(weekday[:3]) or weekday in key:
                matched_hours = hours
                break

        if not matched_hours:
            # Fallback to mon_thu if not found
            matched_hours = opening_hours.get("mon_thu", "")

        if not matched_hours:
            logger.warning(f"No matching opening hours for weekday '{weekday}'")
            return []

        # Example: "12:00-15:30,19:00-23:00"
        time_ranges = [r.strip() for r in matched_hours.split(",") if r.strip()]
        all_slots = []

        for time_range in time_ranges:
            start_str, end_str = time_range.split("-")
            start_time = datetime.strptime(start_str.strip(), "%H:%M")
            end_time = datetime.strptime(end_str.strip(), "%H:%M")

            # Generate 30-min slots
            while start_time < end_time:
                all_slots.append(start_time.strftime("%H:%M"))
                start_time += timedelta(minutes=30)

        return all_slots

    except Exception as e:
        logger.warning(f"[parse_opening_hours] Failed to parse: {e}")
        return []
