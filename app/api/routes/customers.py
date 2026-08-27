# app/api/routes/customers.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.data.db_connection import SessionLocal
from app.data.db_models import Customer
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter()


class CustomerProfile(BaseModel):
    name: str
    email: str
    allergies: list[str] = []
    preferred_cuisines: list[str] = []
    avoid_music: bool = False
    seating_preference: str | None = None


@router.post("/profile", summary="Save or update customer profile")
def save_customer_profile(data: CustomerProfile):
    session = SessionLocal()
    try:
        customer = session.query(Customer).filter(Customer.email == data.email).first()

        if customer:
            customer.name = data.name
            customer.allergies = data.allergies
            customer.preferred_cuisines = data.preferred_cuisines
            customer.avoid_music = data.avoid_music
            customer.seating_preference = data.seating_preference
        else:
            customer = Customer(
                name=data.name,
                email=data.email,
                allergies=data.allergies,
                preferred_cuisines=data.preferred_cuisines,
                avoid_music=data.avoid_music,
                seating_preference=data.seating_preference,
            )
            session.add(customer)

        session.commit()
        session.refresh(customer)

        return {
            "message": "Profile saved successfully!",
            "data": {
                "id": customer.id,
                "name": customer.name,
                "email": customer.email,
                "allergies": customer.allergies,
                "preferred_cuisines": customer.preferred_cuisines,
                "avoid_music": customer.avoid_music,
                "seating_preference": customer.seating_preference,
            },
        }

    except SQLAlchemyError as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        session.close()


@router.get("/{email}", summary="Get customer profile by email")
def get_customer_by_email(email: str):
    session = SessionLocal()
    try:
        customer = session.query(Customer).filter(Customer.email == email).first()
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        return {
            "id": customer.id,
            "name": customer.name,
            "email": customer.email,
            "allergies": customer.allergies,
            "preferred_cuisines": customer.preferred_cuisines,
            "avoid_music": customer.avoid_music,
            "seating_preference": customer.seating_preference,
        }

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        session.close()
