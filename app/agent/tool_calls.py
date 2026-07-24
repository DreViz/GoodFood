# app/agent/tool_calls.py
import json
import logging
import re
import unicodedata
from typing import Any, Dict, Optional
from sqlalchemy import cast, String, func

from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant, Customer, Reservation
from app.api.utils.slot_manager import get_available_slots
from app.api.utils.email_service import send_email
from datetime import datetime

logger = logging.getLogger(__name__)

# -----------------------------------------------------
# TOOL SPEC (for LLM & registration)
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
        "args": ["restaurant", "location_id"],
    },
    "get_amenities": {
        "description": "Fetch amenities and facilities offered by a restaurant.",
        "args": ["restaurant", "location_id"],
    },
    "get_booking_details": {
        "description": "Fetch existing booking details for a given customer email.",
        "args": ["customer_email"],
    },
    "get_seating_labels": {
        "description": "Returns only seating section labels for a restaurant.",
        "args": ["restaurant"],
    },
}

# -----------------------------------------------------
# Helper: normalize restaurant name for fuzzy matching
# -----------------------------------------------------
def normalize_name(name: str) -> str:
    """Normalize restaurant name for fuzzy matching."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name

# -----------------------------------------------------
# CORE TOOL IMPLEMENTATIONS
# -----------------------------------------------------

def search_restaurants(cuisine=None, zone=None, max_price=None, min_rating=None, tag=None, limit=5):
    """Search restaurants dynamically. Returns {'ok': True, 'results': [...] } on success."""
    session = SessionLocal()
    try:
        logger.info("Searching restaurants with filters: %s", {"cuisine": cuisine, "zone": zone, "max_price": max_price, "min_rating": min_rating, "tag": tag})
        q = session.query(Restaurant)

        if cuisine:
            q = q.filter(func.lower(cast(Restaurant.cuisines, String)).ilike(f"%{cuisine.lower()}%"))
        if zone:
            q = q.filter(Restaurant.zone.ilike(f"%{zone}%"))
        if max_price:
            try:
                q = q.filter(Restaurant.avg_price_per_person <= float(max_price))
            except Exception:
                logger.warning("max_price could not be parsed to number: %s", max_price)
        if min_rating:
            try:
                q = q.filter(Restaurant.rating >= float(min_rating))
            except Exception:
                logger.warning("min_rating could not be parsed to number: %s", min_rating)
        if tag:
            q = q.filter(cast(Restaurant.tags, String).ilike(f"%{tag}%"))

        results = q.limit(limit).all()
        formatted = [
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

        return {"ok": True, "results": formatted}
    except Exception as e:
        logger.exception("Error searching restaurants")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

def recommend_venues(cuisine=None, max_price=None, tags=None):
    """Recommend restaurants based on preferences. Returns {'ok': True, 'results': [...] }."""
    session = SessionLocal()
    try:
        logger.info("Recommending venues: cuisine=%s max_price=%s tags=%s", cuisine, max_price, tags)
        q = session.query(Restaurant)
        if cuisine:
            q = q.filter(cast(Restaurant.cuisines, String).ilike(f"%{cuisine}%"))
        if max_price:
            try:
                q = q.filter(Restaurant.avg_price_per_person <= float(max_price))
            except Exception:
                logger.warning("max_price could not be parsed to number: %s", max_price)
        results = q.limit(3).all()
        formatted = [
            {
                "location_id": r.location_id,
                "unit_name": r.unit_name,
                "zone": r.zone,
                "avg_price_per_person": r.avg_price_per_person,
                "rating": r.rating,
            }
            for r in results
        ]
        return {"ok": True, "results": formatted}
    except Exception as e:
        logger.exception("Error recommending venues")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

def _resolve_restaurant_to_location(session, restaurant: Optional[str] = None, location_id: Optional[int] = None):
    """Resolve either location_id (preferred) or restaurant name to a Restaurant record."""
    if location_id:
        rest = session.query(Restaurant).filter_by(location_id=location_id).first()
        if rest:
            return rest
        return None
    if not restaurant:
        return None
    normalized = normalize_name(restaurant)
    if not normalized:
        return None
    cleaned = re.sub(r"\b(in|at|near|around|the)\b", "", normalized).strip()
    rest = session.query(Restaurant).filter(func.lower(Restaurant.unit_name).like(f"%{cleaned}%")).first()
    return rest

def get_seating_map(restaurant: Optional[str] = None, location_id: Optional[int] = None):
    """Retrieve seating sections for a restaurant. Returns {'ok': True, 'unit_name': ..., 'sections': [...] }."""
    session = SessionLocal()
    try:
        rest = _resolve_restaurant_to_location(session, restaurant=restaurant, location_id=location_id)
        if not rest:
            return {"ok": False, "error": f"No restaurant found for '{restaurant or location_id}'"}
        return {"ok": True, "unit_name": rest.unit_name, "sections": rest.seating_sections or []}
    except Exception as e:
        logger.exception("Error getting seating map")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

def get_amenities(restaurant: Optional[str] = None, location_id: Optional[int] = None):
    """Fetch amenities and a short summary for the restaurant."""
    session = SessionLocal()
    try:
        rest = _resolve_restaurant_to_location(session, restaurant=restaurant, location_id=location_id)
        if not rest:
            return {"ok": False, "error": f"No restaurant found for '{restaurant or location_id}'"}

        raw = getattr(rest, "amenities", None)
        if isinstance(raw, list):
            amenities = raw
        elif isinstance(raw, str):
            amenities = [a.strip() for a in raw.replace("•", ",").split(",") if a.strip()]
        else:
            amenities = []

        description = f"{rest.unit_name} offers: " + ", ".join(amenities) if amenities else f"{rest.unit_name} — amenities not listed."
        return {"ok": True, "restaurant": rest.unit_name, "location_id": rest.location_id, "amenities": amenities, "summary": description}
    except Exception as e:
        logger.exception("Error getting amenities")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

def get_booking_details(customer_email: str):
    """Fetch booking details for a given customer email."""
    session = SessionLocal()
    try:
        if not customer_email or "@" not in customer_email:
            return {"ok": False, "error": "Invalid or missing customer_email"}

        cust = session.query(Customer).filter_by(email=customer_email).first()
        if not cust:
            return {"ok": False, "error": f"No customer found with email '{customer_email}'"}

        reservations = (
            session.query(Reservation, Restaurant)
            .join(Restaurant, Reservation.restaurant_id == Restaurant.id)
            .filter(Reservation.customer_id == cust.id)
            .order_by(Reservation.date.desc(), Reservation.time.desc())
            .all()
        )

        if not reservations:
            return {"ok": True, "bookings": [], "message": "No active bookings found."}

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
        logger.exception("Error fetching booking details")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

def _normalize_date(d: str) -> str:
    """Convert common user formats to YYYY-MM-DD safely."""
    if not d:
        return d

    # If already ISO, accept
    try:
        datetime.strptime(d, "%Y-%m-%d")
        return d
    except:
        pass

    # Try common user formats
    formats = [
        "%d-%m-%y",
        "%d-%m-%Y",
        "%d/%m/%y",
        "%d/%m/%Y",
        "%d.%m.%Y",
        "%d.%m.%y"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except:
            continue

    return d  # fallback (unchanged)


def check_availability(location_id: Optional[int] = None, restaurant: Optional[str] = None, date: Optional[str] = None, time: Optional[str] = None, party_size: Optional[int] = None, **kwargs):
    """Check availability for a restaurant by name or ID."""
    session = SessionLocal()
    try:
        rest = _resolve_restaurant_to_location(session, restaurant=restaurant, location_id=location_id)
        if not rest:
            return {"ok": False, "error": "Please specify a valid restaurant name or location_id."}
        location_id = rest.location_id

        if not date:
            return {"ok": False, "error": "Please provide a date for checking availability."}
        
        date = _normalize_date(date)

        slots_data = get_available_slots(location_id, date, party_size)
        if not slots_data or "error" in slots_data:
            return {"ok": False, "error": slots_data.get("error", "No available slots.")}

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

        # list mode
        return {"ok": True, "mode": "list", "restaurant": rest.unit_name, "availability": slots_data}
    except Exception as e:
        logger.exception("Error in check_availability")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

def create_reservation(
    restaurant: str = None,
    location_id: Optional[int] = None,
    date: str = None,
    time: str = None,
    party_size: int = None,
    customer_email: str = None,
    seating_pref: str = None
):
    """
    Create and confirm a reservation by restaurant name (resolved to location_id).
    Validates required fields, ensures customer exists (created if necessary), writes reservation.
    Returns `email_sent: False` if email delivery fails BUT reservation is still created.
    """
    session = SessionLocal()
    try:
        # Validate inputs
        if not restaurant:
            return {"ok": False, "error": "Missing required restaurant name."}
        if not customer_email or "@" not in customer_email:
            return {"ok": False, "error": "Missing or invalid customer_email."}
        if not date:
            return {"ok": False, "error": "Missing required date."}
        if not time:
            return {"ok": False, "error": "Missing required time."}
        if not party_size:
            return {"ok": False, "error": "Missing required party_size."}

        # Resolve restaurant -> record
        rest = _resolve_restaurant_to_location(session, restaurant=restaurant, location_id=None)
        if not rest:
            return {"ok": False, "error": f"Restaurant '{restaurant}' not found."}
        location_id = rest.location_id

        # Ensure customer exists
        cust = session.query(Customer).filter_by(email=customer_email).first()
        if not cust:
            cust = Customer(name="Guest", email=customer_email)
            session.add(cust)
            session.commit()
            session.refresh(cust)

        # Create reservation
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
        session.refresh(reservation)

        # ----------------------------
        # Email sending (best effort)
        # ----------------------------
        email_sent = True
        try:
            send_email(
                to_email=cust.email,    # keep your existing call unchanged
                subject="Your GoodFoods reservation is confirmed!",
                body_html=f"""
                    <b>Reservation Confirmed!</b><br>
                    Restaurant: {rest.unit_name}<br>
                    Date: {date}<br>
                    Time: {time}<br>
                    Guests: {party_size}<br><br>
                    We look forward to hosting you!
                """,
            )
        except Exception as e:
            email_sent = False
            logger.error(f"Email delivery failed (ignored): {e}")

        # Return reservation result INCLUDING email_sent flag
        return {
            "ok": True,
            "restaurant": rest.unit_name,
            "location_id": location_id,
            "reservation_id": reservation.id,
            "customer_email": cust.email,
            "date": date,
            "time": time,
            "party_size": party_size,
            "seating_pref": seating_pref,
            "email_sent": email_sent,
        }

    except Exception as e:
        logger.exception("Error creating reservation")
        return {"ok": False, "error": str(e)}

    finally:
        session.close()


def get_seating_labels(restaurant: Optional[str] = None, location_id: Optional[int] = None):
    """Return only seating section labels for a restaurant."""
    session = SessionLocal()
    try:
        rest = _resolve_restaurant_to_location(session, restaurant=restaurant, location_id=location_id)
        if not rest:
            return {"ok": False, "error": f"No restaurant found for '{restaurant or location_id}'"}
        sections = rest.seating_sections or []
        labels = [sec.get("label") for sec in sections if isinstance(sec, dict) and sec.get("label")]
        return {"ok": True, "restaurant": rest.unit_name, "location_id": rest.location_id, "seating_labels": labels}
    except Exception as e:
        logger.exception("Error in get_seating_labels")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

# -----------------------------------------------------
# TOOL FUNCTIONS map and dispatcher
# -----------------------------------------------------
TOOL_FUNCTIONS = {
    "search_restaurants_by_filters": search_restaurants,
    "recommend_venues": recommend_venues,
    "get_seating_map": get_seating_map,
    "check_availability": check_availability,
    "create_reservation": create_reservation,
    "get_amenities": get_amenities,
    "get_booking_details": get_booking_details,
    "get_seating_labels": get_seating_labels,
}

def dispatch_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch planner tool calls in a safe, consistent way.
    - Normalizes certain argument keys (restaurant_name -> restaurant).
    - Maps restaurant -> location_id for tools that need it.
    - Executes the underlying function and returns {"result": <function_return>} on success,
      or {"error": <message>} on failure.
    """
    session = None
    try:
        if not name:
            return {"error": "Missing tool name in dispatch."}

        logger.info("[Dispatch] Tool requested: %s", name)
        logger.debug("[Dispatch Args] %s", json.dumps(args, indent=2, default=str))

        normalized_args = dict(args)

        # Handle occasional LLM output of "restaurant_name" instead of "restaurant"
        if "restaurant_name" in normalized_args and "restaurant" not in normalized_args:
            normalized_args["restaurant"] = normalized_args.pop("restaurant_name")

        func = TOOL_FUNCTIONS.get(name)
        if not func:
            logger.warning("Unknown tool requested: %s", name)
            return {"error": f"Unknown tool '{name}'"}

        # Map restaurant -> location_id for tools that need both
        needs_mapping = name in {"check_availability", "create_reservation", "get_seating_labels", "get_seating_map", "get_amenities"}
        if needs_mapping and "restaurant" in normalized_args and "location_id" not in normalized_args:
            restaurant_name = normalized_args.get("restaurant")
            session = SessionLocal()
            rest = session.query(Restaurant).filter(Restaurant.unit_name.ilike(f"%{restaurant_name}%")).first()
            if rest:
                normalized_args["location_id"] = rest.location_id
                logger.info("Mapped restaurant '%s' → location_id=%s", restaurant_name, rest.location_id)
            else:
                return {"error": f"Restaurant '{restaurant_name}' not found."}

        # Execute tool
        logger.info("Executing tool function: %s", func.__name__)
        result = func(**normalized_args)

        logger.info("Dispatch result (truncated): %s", str(result)[:1000])
        return {"result": result}
    except Exception as e:
        logger.exception("Tool dispatch failed for %s: %s", name, e)
        return {"error": str(e)}
    finally:
        if session:
            session.close()
            logger.info("[Dispatch] DB session closed after dispatch.")
