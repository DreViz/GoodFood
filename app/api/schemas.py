# app/api/schemas.py
"""
Pydantic request/response schemas for the HTTP surface (Phase 2 / WS2).

Centralising these keeps validation consistent across the agent chat endpoint
and the shortcut endpoints (/search, /availability, /book) consumed by the
frontend and the Phase 3 eval harness.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# Agent chat

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's chat message.")
    session_id: Optional[str] = Field(None, description="Opaque client session id.")
    context: Optional[str] = Field("", description="Optional prior-conversation context.")


class ChatResponse(BaseModel):
    reply: str
    tool_output: Optional[Dict[str, Any]] = None
    phase: str


# Shortcut endpoints

class SearchRequest(BaseModel):
    cuisine: Optional[str] = None
    zone: Optional[str] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    tag: Optional[str] = None
    limit: int = 5


class AvailabilityRequest(BaseModel):
    location_id: Optional[int] = None
    restaurant: Optional[str] = None
    date: str
    time: Optional[str] = None
    party_size: int


class BookingRequest(BaseModel):
    restaurant: Optional[str] = None
    location_id: Optional[int] = None
    date: str
    time: str
    party_size: int
    customer_email: str
    seating_pref: Optional[str] = None
