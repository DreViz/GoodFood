# app/api/routes/availability.py

from fastapi import APIRouter, HTTPException
from app.agent.tool_calls import check_availability
from app.api.schemas import AvailabilityRequest

# NOTE: no prefix here — main.py mounts this router with prefix="/availability".
router = APIRouter(tags=["Availability"])


@router.post("/", summary="Check restaurant availability by name or location ID")
def check_available_slots(data: AvailabilityRequest):
    try:
        result = check_availability(
            location_id=data.location_id,
            restaurant=data.restaurant,
            date=data.date,
            time=data.time,
            party_size=data.party_size,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
