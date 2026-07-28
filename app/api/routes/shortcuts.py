# app/api/routes/shortcuts.py
"""
Direct (non-agent) shortcut endpoints over the tool layer (Phase 2 / WS2).

These expose the same underlying tools the planner uses, but as clean,
schema-validated HTTP endpoints for the frontend and the Phase 3 eval harness.
"""
from fastapi import APIRouter, HTTPException

from app.agent.tool_calls import search_restaurants, create_reservation
from app.api.schemas import SearchRequest, BookingRequest

router = APIRouter()


@router.post("/search", summary="Search restaurants by filters")
def search(data: SearchRequest):
    result = search_restaurants(
        cuisine=data.cuisine,
        zone=data.zone,
        max_price=data.max_price,
        min_rating=data.min_rating,
        tag=data.tag,
        limit=data.limit,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Search failed."))
    return result


@router.post("/book", summary="Create a reservation (full validation)")
def book(data: BookingRequest):
    result = create_reservation(
        restaurant=data.restaurant,
        location_id=data.location_id,
        date=data.date,
        time=data.time,
        party_size=data.party_size,
        customer_email=data.customer_email,
        seating_pref=data.seating_pref,
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Booking failed."))
    return result
