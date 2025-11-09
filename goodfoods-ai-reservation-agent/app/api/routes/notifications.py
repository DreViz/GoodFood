# app/api/routes/notifications.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from app.api.utils.email_service import send_email

router = APIRouter()

class EmailRequest(BaseModel):
    to: EmailStr
    subject: str
    message: str


@router.post("/send", summary="Send an email notification")
def send_email_notification(data: EmailRequest):
    """
    Sends a test or custom email to a recipient.
    """
    try:
        response = send_email(data.to, data.subject, data.message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
