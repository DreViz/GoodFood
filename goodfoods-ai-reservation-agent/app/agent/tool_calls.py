import logging
import re
from typing import Any, Dict, Optional
from sqlalchemy import cast, String, func
from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant, Customer, Reservation
from app.api.utils.slot_manager import get_available_slots
from app.api.utils.email_service import send_email

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# 🔧 TOOL SPECIFICATION (LLM Reference)
# -----------------------------------------------------
TOOL_SPEC = {
    "search_restaurants_by_filters": {
        "description": "Find restaurants by cuisine, zone, price, rating, or tag filters.",
        "args": ["cuisine", "zone", "max_price", "min_rating", "tag"],
    },
    "recommend_venues": {
        "description": "Suggest restaurants based on user preferences (cuisine, tags, budget).",
        "args": ["cuisine", "max_price", "tags"],
    },
    "check_availability": {
        "description": "Check available slots for a restaurant by name or location ID.",
        "args": ["restaurant", "date", "time", "party_size"],
    },
    "create_reservation": {
        "description": "Create and confirm a reservation for a restaurant.",
        "args": ["restaurant", "date", "time", "party_size", "customer_email", "seating_pref"],
    },
    "get_seating_map": {
        "description": "Get available seating sections for a restaurant.",
        "args": ["restaurant"],
    },
    "get_amenities": {
        "description": "Fetch amenities and facilities offered by a restaurant.",
        "args": ["restaurant"],
    },
    "get_booking_details": {
        "description": "Fetch existing booking details for a given customer email.",
        "args": ["customer_email"],
    }
}

# -----------------------------------------------------
# 🧠 CORE TOOL IMPLEMENTATIONS
# -----------------------------------------------------

def search_restaurants(cuisine=None, zone=None, max_price=None, min_rating=None, tag=None, limit=5):
    """Search restaurants dynamically."""
    session = SessionLocal()
    try:
        logger.info(f"🔍 Searching restaurants with filters: cuisine={cuisine}, zone={zone}, max_price={max_price}, min_rating={min_rating}, tag={tag}")
        q = session.query(Restaurant)
        if cuisine:
            q = q.filter(func.lower(cast(Restaurant.cuisines, String)).ilike(f"%{cuisine.lower()}%"))
        if zone:
            q = q.filter(Restaurant.zone.ilike(f"%{zone}%"))
        if max_price:
            q = q.filter(Restaurant.avg_price_per_person <= max_price)
        if min_rating:
            q = q.filter(Restaurant.rating >= min_rating)
        if tag:
            q = q.filter(cast(Restaurant.tags, String).ilike(f"%{tag}%"))

        results = q.limit(limit).all()
        logger.info(f" Found {len(results)} restaurants matching filters.")
        return [
            {
                "location_id": r.location_id,
                "unit_name": r.unit_name,
                "zone": r.zone,
                "avg_price_per_person": r.avg_price_per_person,
                "rating": r.rating,
                "cuisines": r.cuisines,
                "tags": r.tags,
            }
            for r in results
        ]
    except Exception as e:
        logger.exception(" Error searching restaurants")
        return []
    finally:
        session.close()


def recommend_venues(cuisine=None, max_price=None, tags=None):
    """Recommend restaurants based on preferences."""
    session = SessionLocal()
    try:
        logger.info(f" Recommending venues for cuisine={cuisine}, max_price={max_price}, tags={tags}")
        q = session.query(Restaurant)
        if cuisine:
            q = q.filter(cast(Restaurant.cuisines, String).ilike(f"%{cuisine}%"))
        if max_price:
            q = q.filter(Restaurant.avg_price_per_person <= max_price)
        results = q.limit(3).all()
        logger.info(f"Found {len(results)} recommended venues.")
        return [
            {
                "location_id": r.location_id,
                "unit_name": r.unit_name,
                "zone": r.zone,
                "avg_price_per_person": r.avg_price_per_person,
                "rating": r.rating,
            }
            for r in results
        ]
    except Exception as e:
        logger.exception(" Error recommending venues")
        return []
    finally:
        session.close()


def get_seating_map(location_id: int):
    """Retrieve seating options for a restaurant."""
    session = SessionLocal()
    try:
        logger.info(f"🪑 Fetching seating map for location_id={location_id}")
        r = session.query(Restaurant).filter_by(location_id=location_id).first()
        if not r:
            return {"error": f"No restaurant found with location_id={location_id}"}
        return {"unit_name": r.unit_name, "sections": r.seating_sections or []}
    except Exception as e:
        logger.exception(" Error getting seating map")
        return {"error": str(e)}
    finally:
        session.close()


def get_amenities(restaurant: Optional[str] = None, location_id: Optional[int] = None):
    """Fetch amenities and facilities of a restaurant."""
    session = SessionLocal()
    try:
        if not restaurant and not location_id:
            return {"error": "Please specify a restaurant name or location_id."}

        if not location_id and restaurant:
            rest = (
                session.query(Restaurant)
                .filter(func.lower(Restaurant.unit_name).like(f"%{restaurant.lower()}%"))
                .first()
            )
        else:
            rest = session.query(Restaurant).filter_by(location_id=location_id).first()

        if not rest:
            return {"error": f"No restaurant found for '{restaurant or location_id}'"}

        amenities = []
        if getattr(rest, "amenities", None):
            if isinstance(rest.amenities, str):
                amenities = [a.strip() for a in rest.amenities.replace("•", ",").split(",") if a.strip()]
            else:
                amenities = rest.amenities
        else:
            amenities = [
                "Air Conditioning",
                "Valet Parking",
                "Kids High-Chair",
                "Live Music (Weekends)",
                "Pet Friendly Outdoor Seating",
            ]

        description = (
            f"{rest.unit_name} offers a premium dining experience with the following amenities:\n• "
            + " • ".join(amenities)
            + "."
        )

        if getattr(rest, "has_rooftop", False):
            description += " It also features a beautiful rooftop area."
        if getattr(rest, "has_private_lounge", False):
            description += " It includes a private lounge for small gatherings."

        return {
            "ok": True,
            "restaurant": rest.unit_name,
            "location_id": rest.location_id,
            "amenities": amenities,
            "summary": description.strip(),
        }

    except Exception as e:
        logger.exception("Error fetching amenities")
        return {"error": str(e)}
    finally:
        session.close()


def get_booking_details(customer_email: str):
    """Fetch booking details for a given customer email."""
    session = SessionLocal()
    try:
        if not customer_email or "@" not in customer_email:
            return {"error": "Invalid or missing customer_email"}

        cust = session.query(Customer).filter_by(email=customer_email).first()
        if not cust:
            return {"error": f"No customer found with email '{customer_email}'"}

        reservations = (
            session.query(Reservation, Restaurant)
            .join(Restaurant, Reservation.restaurant_id == Restaurant.id)
            .filter(Reservation.customer_id == cust.id)
            .order_by(Reservation.date.desc(), Reservation.time.desc())
            .all()
        )

        if not reservations:
            return {"message": "No active bookings found."}

        details = [
            {
                "reservation_id": res.id,
                "restaurant": rest.unit_name,
                "date": res.date,
                "time": res.time,
                "party_size": res.party_size,
                "status": res.status,
                "seating_preference": res.seating_preference,
            }
            for res, rest in reservations
        ]

        latest = details[0]
        summary = (
            f"Your latest booking is at {latest['restaurant']} on {latest['date']} at {latest['time']} "
            f"for {latest['party_size']} guests. Status: {latest['status'].capitalize()}."
        )

        return {"ok": True, "customer_email": customer_email, "bookings": details, "summary": summary}

    except Exception as e:
        logger.exception(" Error fetching booking details")
        return {"error": str(e)}
    finally:
        session.close()


def check_availability(
    location_id: Optional[int] = None,
    restaurant: Optional[str] = None,
    date: Optional[str] = None,
    time: Optional[str] = None,
    party_size: Optional[int] = None,
    **kwargs,
):
    """Check availability for a restaurant by name or ID."""
    if "email" in kwargs:
        logger.warning("⚠️ Ignoring unexpected 'email' parameter in check_availability().")

    session = SessionLocal()
    try:
        logger.info("🔍 [check_availability] START")
        logger.info(f"Incoming params → location_id={location_id}, restaurant='{restaurant}', date={date}, time={time}, party_size={party_size}")

        rest = None
        if not location_id:
            if not restaurant:
                return {"error": "Please specify a restaurant name or location_id."}

            cleaned_name = re.sub(r"\b(in|at|near|around|the)\b", "", restaurant.strip().lower(), flags=re.IGNORECASE)
            rest = (
                session.query(Restaurant)
                .filter(func.lower(Restaurant.unit_name).like(f"%{cleaned_name}%"))
                .first()
            )

            if not rest:
                return {"error": f"No restaurant found matching '{restaurant}'"}

            location_id = rest.location_id
        else:
            rest = session.query(Restaurant).filter_by(location_id=location_id).first()
            if not rest:
                return {"error": "Invalid restaurant location_id."}

        if not date:
            return {"error": "Please provide a date for checking availability."}

        slots_data = get_available_slots(location_id, date, party_size)
        if not slots_data or "error" in slots_data:
            return {"error": slots_data.get("error", "No available slots.")}

        if time:
            all_slots = [s["time"] for s in slots_data.get("available_slots", [])]
            is_available = time in all_slots
            return {
                "ok": True,
                "mode": "single",
                "restaurant": rest.unit_name,
                "location_id": location_id,
                "date": date,
                "time": time,
                "party_size": party_size,
                "is_available": is_available,
                "available_slots": None if is_available else all_slots,
            }

        return {"ok": True, "mode": "list", "restaurant": rest.unit_name, "availability": slots_data}

    except Exception as e:
        logger.exception(f"[check_availability] ERROR: {e}")
        return {"error": str(e)}
    finally:
        session.close()
        logger.info("🔚 [check_availability] Session closed.")


def create_reservation(location_id, date, time, party_size, customer_email, seating_pref=None):
    """Create and confirm a reservation using location_id."""
    session = SessionLocal()
    try:
        if not customer_email:
            return {"ok": False, "error": "Missing required customer_email"}

        logger.info(f"🪑 Creating reservation: loc_id={location_id}, date={date}, time={time}, size={party_size}, email={customer_email}")
        cust = session.query(Customer).filter_by(email=customer_email).first()
        if not cust:
            cust = Customer(name="Guest", email=customer_email)
            session.add(cust)
            session.commit()
            session.refresh(cust)

        rest = session.query(Restaurant).filter_by(location_id=location_id).first()
        if not rest:
            return {"error": f"Invalid location_id: {location_id}"}

        reservation = Reservation(
            customer_id=cust.id,
            restaurant_id=rest.id,
            date=date,
            time=time,
            party_size=party_size,
            seating_preference=seating_pref,
            status="confirmed",
        )
        session.add(reservation)
        session.commit()

        try:
            send_email(
                to_email=cust.email,
                subject="Your GoodFoods reservation is confirmed!",
                body_html=f"""
                    <b>Reservation Confirmed!</b><br>
                    Restaurant: {rest.unit_name}<br>
                    Date: {date}<br>
                    Time: {time}<br>
                    Guests: {party_size}<br><br>
                    We look forward to hosting you! 🍽️
                """,
            )
        except Exception as e:
            logger.warning(f"Email delivery failed: {e}")

        return {
            "ok": True,
            "restaurant": rest.unit_name,
            "reservation_id": reservation.id,
            "customer_email": cust.email,
        }

    except Exception as e:
        logger.exception("Error creating reservation")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()


# -----------------------------------------------------
#  TOOL DISPATCHER (Unified)
# -----------------------------------------------------
TOOL_FUNCTIONS = {
    "search_restaurants_by_filters": search_restaurants,
    "recommend_venues": recommend_venues,
    "get_seating_map": get_seating_map,
    "check_availability": check_availability,
    "create_reservation": create_reservation,
    "get_amenities": get_amenities,
    "get_booking_details": get_booking_details,
}


def dispatch_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch planner tool calls safely and consistently."""
    session = None
    try:
        if not name:
            return {"error": "Missing tool name in dispatch."}

        rename_map = {"restaurant_name": "restaurant", "zone": None}
        normalized_args = {
            rename_map.get(k, k): v for k, v in args.items() if rename_map.get(k, k) is not None
        }

        func = TOOL_FUNCTIONS.get(name)
        if not func:
            return {"error": f"Unknown tool '{name}'"}

        if name in {"create_reservation", "check_availability"}:
            if "restaurant" in normalized_args and "location_id" not in normalized_args:
                restaurant_name = normalized_args.pop("restaurant")
                session = SessionLocal()
                rest = (
                    session.query(Restaurant)
                    .filter(Restaurant.unit_name.ilike(f"%{restaurant_name}%"))
                    .first()
                )
                if rest:
                    normalized_args["location_id"] = rest.location_id
                    logger.info(f" Mapped restaurant '{restaurant_name}' → location_id={rest.location_id}")
                else:
                    return {"error": f"Restaurant '{restaurant_name}' not found."}

        if name == "create_reservation":
            required = ["location_id", "date", "time", "party_size", "customer_email"]
            missing = [r for r in required if not normalized_args.get(r)]
            if missing:
                return {"error": f"Missing required parameters for reservation: {', '.join(missing)}"}

        logger.info(f"Dispatching → {name} | args={normalized_args}")

        result = func(**normalized_args)

        logger.info(f" Tool executed successfully: {name}")
        return {"result": result}

    except TypeError as te:
        logger.exception(f" TypeError in tool '{name}': {te}")
        return {"error": f"Invalid parameters for tool '{name}': {te}"}
    except Exception as e:
        logger.exception(f" Tool dispatch failed for {name}: {e}")
        return {"error": str(e)}
    finally:
        if session:
            session.close()
            logger.info(" Session closed after dispatch.")
