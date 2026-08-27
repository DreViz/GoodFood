# app/data/db_models.py
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, ForeignKey, UniqueConstraint, Date
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from app.data.db_connection import Base


class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    location_id = Column(Integer, unique=True, nullable=False)
    brand = Column(String, nullable=False)
    unit_name = Column(String, nullable=False)
    zone = Column(String)
    address = Column(String)
    pincode = Column(String(10))
    latitude = Column(Float)
    longitude = Column(Float)
    rating = Column(Float)
    avg_price_per_person = Column(Integer)
    capacity = Column(Integer)
    description = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    cuisines = Column(JSONB, default=[])         
    amenities = Column(JSONB, default=[])           
    seating_sections = Column(JSONB, default=[])    
    menu = Column(JSONB, default=[])                
    policies = Column(JSONB, default={})
    opening_hours = Column(JSONB, default={})
    ops_metrics = Column(JSONB, default={})
    contact = Column(JSONB, default={})
    tags = Column(JSONB, default=[])

    reservations = relationship("Reservation", back_populates="restaurant", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("unit_name", "address", name="unique_restaurant_location"),)


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    allergies = Column(JSONB, nullable=True)
    preferred_cuisines = Column(JSONB, nullable=True)
    avoid_music = Column(Boolean, default=False)
    seating_preference = Column(String, nullable=True)

    preferences = relationship("CustomerPreferences", back_populates="customer", uselist=False)
    reservations = relationship("Reservation", back_populates="customer", cascade="all, delete-orphan")


class Reservation(Base):
    __tablename__ = "reservations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id", ondelete="CASCADE"), nullable=False)

    date = Column(String, nullable=False)
    time = Column(String, nullable=False)
    party_size = Column(Integer, nullable=False)
    seating_preference = Column(String, nullable=True)
    status = Column(String, default="confirmed")
    created_at = Column(String, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    customer = relationship("Customer", back_populates="reservations")
    restaurant = relationship("Restaurant", back_populates="reservations")

class CustomerPreferences(Base):
    __tablename__ = "customer_preferences"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"))
    cuisine = Column(String, nullable=True)
    vibe_tags = Column(String, nullable=True)
    max_price = Column(Float, nullable=True)
    guests = Column(Integer, nullable=True)
    date = Column(Date, nullable=True)

    customer = relationship("Customer", back_populates="preferences")
