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

# Tool metadata exposed to the planner LLM.
TOOL_SPEC = {
    "search_restaurants_by_filters": {
        "description": "Find restaurants by cuisine, zone, price, rating, or tag filters. Results are semantically ranked by cuisine + the optional query vibe when available.",
        "args": ["cuisine", "zone", "max_price", "min_rating", "tag", "query"],
    },
    "recommend_venues": {
        "description": "Suggest restaurants based on user preferences (cuisine, tags, budget) or a free-text vibe.",
        "args": ["cuisine", "max_price", "tags", "query"],
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
    "cancel_reservation": {
        "description": "Cancel the customer's most recent confirmed reservation (by email).",
        "args": ["customer_email"],
    },
    "modify_reservation": {
        "description": "Modify the customer's most recent confirmed reservation (date/time/party_size/seating).",
        "args": ["customer_email", "new_date", "new_time", "new_party_size", "new_seating_pref"],
    },
}

def normalize_name(name: str) -> str:
    """Normalize a restaurant name for fuzzy matching."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def search_restaurants(cuisine=None, zone=None, max_price=None, min_rating=None, tag=None, limit=5, query=None):
    """Search restaurants dynamically. Returns {'ok': True, 'results': [...] } on success.

    Hybrid retrieval (see SEMANTIC_SEARCH_PLAN.md): hard filters (zone, price,
    rating, tag) apply as exact SQL predicates; soft semantics (cuisine +
    optional ``query`` free-text vibe) rank the SQL candidate set via
    sentence-embedding cosine similarity. If the embedding model is
    unavailable, falls back to the legacy ``ilike`` order without failing.
    """
    session = SessionLocal()
    try:
        logger.info("Searching restaurants with filters: %s", {"cuisine": cuisine, "zone": zone, "max_price": max_price, "min_rating": min_rating, "tag": tag, "query": query})
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

        # Fetch the FULL candidate set first (not pre-limited), so semantic
        # rank picks the best top-K across all matches rather than re-ordering
        # an already-truncated 5 rows.
        results = q.order_by(Restaurant.rating.desc()).all()

        # Build the soft-semantics query string from cuisine + free-text vibe.
        # Cuisine is both a SQL filter (above) and a semantic signal — including
        # it here keeps it on the ranker's radar for tie-breaking.
        semantic_query = " ".join(p for p in [cuisine, query] if p).strip()

        ranked = None
        if semantic_query and results:
            from app.retrieval.embeddings import semantic_rank
            candidate_ids = [r.location_id for r in results]
            ranked = semantic_rank(semantic_query, candidate_ids, top_k=limit)

        if ranked:
            # Re-order the loaded rows by the ranker's order.
            id_to_row = {r.location_id: r for r in results}
            ordered = [id_to_row[lid] for lid, _ in ranked if lid in id_to_row]
        else:
            # Fallback: SQL order (rating desc). Slice to limit here since the
            # ranker didn't run / didn't return usable scores.
            ordered = results[:limit]

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
            for r in ordered
        ]

        return {"ok": True, "results": formatted}
    except Exception as e:
        logger.exception("Error searching restaurants")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()

def recommend_venues(cuisine=None, max_price=None, tags=None, query=None, limit=3):
    """Recommend restaurants based on preferences. Returns {'ok': True, 'results': [...]}.

    Two paths:
      - Hard filters present (cuisine/max_price) -> SQL pre-filter, then
        semantic-rank the candidate set.
      - No hard filters -> semantic-rank ALL restaurants by the vibe string
        (cuisine + tags + free-text query). This is the genuine "recommend me
        something" path.

    Tags are soft semantics, not exact predicates — they fold into the
    semantic query string ("romantic" matches even when the DB stores
    "date-night"). Falls back to ilike/rating order if the embedding model
    is unavailable.
    """
    session = SessionLocal()
    try:
        logger.info("Recommending venues: cuisine=%s max_price=%s tags=%s query=%s", cuisine, max_price, tags, query)
        q = session.query(Restaurant)
        if cuisine:
            q = q.filter(cast(Restaurant.cuisines, String).ilike(f"%{cuisine}%"))
        if max_price:
            try:
                q = q.filter(Restaurant.avg_price_per_person <= float(max_price))
            except Exception:
                logger.warning("max_price could not be parsed to number: %s", max_price)

        results = q.order_by(Restaurant.rating.desc()).all()

        # Build the semantic vibe string from every soft signal available.
        soft_parts = []
        if cuisine:
            soft_parts.append(cuisine)
        if tags:
            # tags may arrive as a list (planner) or comma-string.
            if isinstance(tags, (list, tuple)):
                soft_parts.extend(str(t) for t in tags if t)
            else:
                soft_parts.append(str(tags))
        if query:
            soft_parts.append(query)
        semantic_query = " ".join(soft_parts).strip()

        ranked = None
        if semantic_query and results:
            from app.retrieval.embeddings import semantic_rank
            candidate_ids = [r.location_id for r in results]
            ranked = semantic_rank(semantic_query, candidate_ids, top_k=limit)

        if ranked:
            id_to_row = {r.location_id: r for r in results}
            ordered = [id_to_row[lid] for lid, _ in ranked if lid in id_to_row]
        else:
            ordered = results[:limit]

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
            for r in ordered
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
    """Convert any user date form to YYYY-MM-DD.

    Delegates to date_utils.normalize_date_to_iso so this layer shares one
    implementation with the planner (safe_extract_date) and slot_manager.
    Handles relative terms (today/tomorrow), this/next <weekday>, DD Mon,
    Mon DD, DD/MM (day-first), and ISO. Returns the original string if it
    can't be interpreted (callers treat that as a parse error downstream).
    """
    if not d:
        return d
    from app.api.utils.date_utils import normalize_date_to_iso
    iso = normalize_date_to_iso(d)
    return iso if iso else d


def _normalize_time(t: str) -> str:
    """Convert any user time form to 24-hour HH:MM.

    The slot manager emits slots as "19:30", but the planner often passes
    "7:30pm" / "8 pm" verbatim — and an unnormalized 12-hour input never
    matches a slot, so is_available is always False and the flow never
    advances to booking. normalize_time maps "7:30pm"->"19:30", "8pm"->
    "20:00". Returns the original string if it can't be interpreted.
    """
    if not t:
        return t
    from app.api.utils.date_utils import normalize_time
    nt = normalize_time(t)
    return nt if nt else t


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
            # Normalize before the comparison so "7:30pm" matches slot "19:30".
            time = _normalize_time(time)
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

        rest = _resolve_restaurant_to_location(session, restaurant=restaurant, location_id=None)
        if not rest:
            return {"ok": False, "error": f"Restaurant '{restaurant}' not found."}
        location_id = rest.location_id

        cust = session.query(Customer).filter_by(email=customer_email).first()
        if not cust:
            cust = Customer(name="Guest", email=customer_email)
            session.add(cust)
            session.commit()
            session.refresh(cust)

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

        # best-effort confirmation email — the reservation stands regardless
        email_sent = True
        try:
            send_email(
                cust.email,
                "Your GoodFoods reservation is confirmed!",
                f"""
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


def _latest_confirmed_reservation(session, customer_email: str):
    """
    Return (reservation, restaurant, customer) for the customer's most recent
    CONFIRMED reservation, or None.

    "One active reservation per customer email" is defined as the most recent
    status='confirmed' row (ordered by date, then time). This is the single
    reservation that cancel/modify operate on.
    """
    if not customer_email or "@" not in customer_email:
        return None
    cust = session.query(Customer).filter_by(email=customer_email).first()
    if not cust:
        return None
    row = (
        session.query(Reservation, Restaurant)
        .join(Restaurant, Reservation.restaurant_id == Restaurant.id)
        .filter(Reservation.customer_id == cust.id, Reservation.status == "confirmed")
        .order_by(Reservation.date.desc(), Reservation.time.desc())
        .first()
    )
    if not row:
        return None
    reservation, restaurant = row
    return reservation, restaurant, cust


def cancel_reservation(customer_email: str = None, **kwargs):
    """
    Cancel the customer's most recent confirmed reservation.

    Soft-delete only: status -> 'cancelled' (never hard-deleted, preserving
    history). Sends a best-effort cancellation email. Returns {ok: False, ...}
    when there is no active reservation for the email.
    """
    session = SessionLocal()
    try:
        if not customer_email or "@" not in customer_email:
            return {"ok": False, "error": "Missing or invalid customer_email."}

        found = _latest_confirmed_reservation(session, customer_email)
        if not found:
            return {"ok": False, "error": f"No active reservation for {customer_email}."}

        reservation, restaurant, cust = found
        reservation.status = "cancelled"
        session.commit()
        session.refresh(reservation)

        email_sent = True
        try:
            send_email(
                cust.email,
                "Your GoodFoods reservation has been cancelled",
                f"""
                    <b>Reservation Cancelled</b><br>
                    Restaurant: {restaurant.unit_name}<br>
                    Date: {reservation.date}<br>
                    Time: {reservation.time}<br><br>
                    We hope to host you another time.
                """,
            )
        except Exception as e:
            email_sent = False
            logger.error(f"Cancellation email failed (ignored): {e}")

        return {
            "ok": True,
            "reservation_id": reservation.id,
            "restaurant": restaurant.unit_name,
            "date": reservation.date,
            "time": reservation.time,
            "status": reservation.status,
            "customer_email": cust.email,
            "email_sent": email_sent,
        }
    except Exception as e:
        session.rollback()
        logger.exception("Error cancelling reservation")
        return {"ok": False, "error": str(e)}
    finally:
        session.close()


def modify_reservation(
    customer_email: str = None,
    new_date: str = None,
    new_time: str = None,
    new_party_size: int = None,
    new_seating_pref: str = None,
    **kwargs,
):
    """
    Modify the customer's most recent confirmed reservation.

    Only supplied fields change. Whenever date/time/party_size change, the new
    slot is re-validated via get_available_slots. Returns {ok: False, ...} for:
    no active reservation, invalid new party_size, or an unavailable requested
    slot (with `available_slots` alternatives).
    """
    session = SessionLocal()
    try:
        if not customer_email or "@" not in customer_email:
            return {"ok": False, "error": "Missing or invalid customer_email."}

        # Accept both new_* and bare field names from the planner.
        new_date = new_date or kwargs.get("date")
        new_time = new_time or kwargs.get("time")
        if new_party_size is None:
            new_party_size = kwargs.get("party_size")
        new_seating_pref = new_seating_pref or kwargs.get("seating_pref")

        found = _latest_confirmed_reservation(session, customer_email)
        if not found:
            return {"ok": False, "error": f"No active reservation for {customer_email}."}

        reservation, restaurant, cust = found

        if new_party_size is not None:
            try:
                new_party_size = int(new_party_size)
            except (TypeError, ValueError):
                return {"ok": False, "error": "Invalid new party_size."}
            if new_party_size < 1:
                return {"ok": False, "error": "Invalid new party_size."}

        # Effective values after applying requested changes.
        eff_date = _normalize_date(new_date) if new_date else reservation.date
        eff_time = new_time or reservation.time
        eff_party = new_party_size if new_party_size is not None else reservation.party_size

        # Re-validate the slot whenever anything affecting availability changes.
        if new_date or new_time or new_party_size is not None:
            slots_data = get_available_slots(restaurant.location_id, eff_date, eff_party)
            if not slots_data or "error" in slots_data:
                return {"ok": False, "error": slots_data.get("error", "No available slots for the requested change.")}
            available = [s["time"] for s in slots_data.get("available_slots", [])]
            if eff_time not in available:
                return {
                    "ok": False,
                    "error": f"{eff_time} is not available on {eff_date}.",
                    "available_slots": available,
                }

        reservation.date = eff_date
        reservation.time = eff_time
        reservation.party_size = eff_party
        if new_seating_pref:
            reservation.seating_preference = new_seating_pref
        session.commit()
        session.refresh(reservation)

        email_sent = True
        try:
            send_email(
                cust.email,
                "Your GoodFoods reservation has been updated",
                f"""
                    <b>Reservation Updated</b><br>
                    Restaurant: {restaurant.unit_name}<br>
                    Date: {reservation.date}<br>
                    Time: {reservation.time}<br>
                    Guests: {reservation.party_size}<br><br>
                    See you soon!
                """,
            )
        except Exception as e:
            email_sent = False
            logger.error(f"Modification email failed (ignored): {e}")

        return {
            "ok": True,
            "reservation_id": reservation.id,
            "restaurant": restaurant.unit_name,
            "date": reservation.date,
            "time": reservation.time,
            "party_size": reservation.party_size,
            "seating_pref": reservation.seating_preference,
            "status": reservation.status,
            "customer_email": cust.email,
            "email_sent": email_sent,
        }
    except Exception as e:
        session.rollback()
        logger.exception("Error modifying reservation")
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

TOOL_FUNCTIONS = {
    "search_restaurants_by_filters": search_restaurants,
    "recommend_venues": recommend_venues,
    "get_seating_map": get_seating_map,
    "check_availability": check_availability,
    "create_reservation": create_reservation,
    "get_amenities": get_amenities,
    "get_booking_details": get_booking_details,
    "get_seating_labels": get_seating_labels,
    "cancel_reservation": cancel_reservation,
    "modify_reservation": modify_reservation,
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

        # Drop kwargs the target function cannot accept. The model sometimes
        # routes an availability-style turn to search_restaurants_by_filters
        # with date/party_size in the args; search_restaurants() has no such
        # params (and no **kwargs), so the call would raise TypeError. Filter
        # to the function's declared parameters — functions that accept
        # **kwargs (check_availability, modify_reservation) keep everything.
        import inspect as _inspect
        _sig = _inspect.signature(func)
        _accepts_var_kw = any(
            p.kind == _inspect.Parameter.VAR_KEYWORD
            for p in _sig.parameters.values()
        )
        if not _accepts_var_kw:
            _valid = set(_sig.parameters.keys())
            _dropped = {k: v for k, v in normalized_args.items() if k not in _valid}
            if _dropped:
                logger.warning("[Dispatch] Dropped unsupported kwargs for %s: %s", name, list(_dropped))
            normalized_args = {k: v for k, v in normalized_args.items() if k in _valid}

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
