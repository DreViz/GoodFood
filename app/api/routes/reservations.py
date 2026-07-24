from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from app.data.db_connection import SessionLocal
from app.data.db_models import Reservation, Customer, Restaurant
from app.api.utils.slot_manager import get_available_slots
from app.api.utils.email_service import send_email
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

router = APIRouter(prefix="/reservations", tags=["Reservations"])

# ---------------------------- FASTAPI DB DEPENDENCY ----------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------- MODELS ----------------------------
class ReservationRequest(BaseModel):
    customer_email: EmailStr
    restaurant_id: int
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    party_size: int
    seating_preference: Optional[str] = None


class ReservationResponse(BaseModel):
    id: int
    customer_id: int
    restaurant_id: int
    date: str
    time: str
    party_size: int
    seating_preference: Optional[str]
    status: str

    class Config:
        from_attributes = True  # ✅ Pydantic v2 compatible


# ---------------------------- EMAIL ----------------------------
def send_confirmation_email(customer_email, restaurant_name, data, customer_name):
    """Helper function for background email sending"""
    subject = f"GoodFoods Reservation Confirmed — {restaurant_name}"
    body = f"""
    <html>
      <body>
        <h2>Your reservation is confirmed ✅</h2>
        <p><strong>Restaurant:</strong> {restaurant_name}</p>
        <p><strong>Date:</strong> {data.date}</p>
        <p><strong>Time:</strong> {data.time}</p>
        <p><strong>Party size:</strong> {data.party_size}</p>
        <p><strong>Seating preference:</strong> {data.seating_preference or 'Not specified'}</p>
        <hr>
        <p>We look forward to serving you — GoodFoods 🍽️</p>
      </body>
    </html>
    """
    send_email(customer_email, subject, body)


# ---------------------------- CREATE RESERVATION ----------------------------
@router.post("/", summary="Create a new reservation (async email confirmation)")
def create_reservation(data: ReservationRequest, background_tasks: BackgroundTasks):
    session = SessionLocal()
    try:
        # Verify customer
        customer = session.query(Customer).filter_by(email=data.customer_email).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found. Please save preferences first.")

        # Verify restaurant
        restaurant = session.query(Restaurant).filter_by(id=data.restaurant_id).first()
        if not restaurant:
            raise HTTPException(status_code=404, detail="Restaurant not found.")

        # Check slot availability
        availability = get_available_slots(restaurant.id, data.date, data.party_size)
        if "error" in availability:
            raise HTTPException(status_code=400, detail=availability["error"])

        available_times = [s["time"] for s in availability["available_slots"]]
        if data.time not in available_times:
            raise HTTPException(status_code=400, detail=f"Slot {data.time} is not available for this restaurant.")

        # Create reservation
        reservation = Reservation(
            customer_id=customer.id,
            restaurant_id=restaurant.id,
            date=data.date,
            time=data.time,
            party_size=data.party_size,
            seating_preference=data.seating_preference,
            status="confirmed"
        )
        session.add(reservation)
        session.commit()
        session.refresh(reservation)

        # Cache values before closing session
        restaurant_name = restaurant.unit_name
        customer_name = customer.name
        customer_email = customer.email

    except HTTPException:
        session.rollback()
        raise
    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

    # Send confirmation email asynchronously
    background_tasks.add_task(send_confirmation_email, customer_email, restaurant_name, data, customer_name)

    return {
        "message": "✅ Reservation created successfully",
        "reservation_id": reservation.id,
        "restaurant": restaurant_name,
        "customer": customer_name,
        "date": data.date,
        "time": data.time,
        "email_status": "Email is being sent in the background"
    }


# ---------------------------- FETCH ALL RESERVATIONS ----------------------------
@router.get("/", response_model=List[ReservationResponse], summary="Fetch all reservations")
def get_all_reservations(db: Session = Depends(get_db)):
    try:
        reservations = db.query(Reservation).all()
        if not reservations:
            raise HTTPException(status_code=404, detail="No reservations found.")
        return reservations
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ---------------------------- FETCH BY CUSTOMER EMAIL ----------------------------
@router.get("/{email}", response_model=List[ReservationResponse], summary="Fetch reservations by customer email")
def get_reservations_by_email(email: str, db: Session = Depends(get_db)):
    try:
        customer = db.query(Customer).filter_by(email=email).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found.")

        reservations = db.query(Reservation).filter_by(customer_id=customer.id).all()
        if not reservations:
            raise HTTPException(status_code=404, detail=f"No reservations found for {email}.")
        return reservations
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
