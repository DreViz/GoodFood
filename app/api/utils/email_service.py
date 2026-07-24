import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import HTTPException
import os


# ----- Configuration -----
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("GOODFOODS_EMAIL", "your_email@gmail.com")
SENDER_PASSWORD = os.getenv("GOODFOODS_EMAIL_PASSWORD", "your_app_password")


def send_email(recipient: str, subject: str, body: str):
    """
    Sends an email using SMTP with proper error handling.
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "html"))

        # Connect and send
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        return {"message": f"✅ Email sent successfully to {recipient}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email send failed: {str(e)}")
