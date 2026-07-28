# scripts/reset_db.py
from app.data.db_connection import engine, Base

print("Dropping all tables...")
Base.metadata.drop_all(bind=engine)

print("Recreating schema...")
Base.metadata.create_all(bind=engine)

print("Database schema reset successfully.")
