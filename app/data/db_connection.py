# app/data/db_connection.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuration: point to your local Postgres 'goodfoods' DB
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/goodfoods"
)

# Defined here once; models import this Base (not the other way around).
Base = declarative_base()

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Import models so they register with Base (here, not at top, to avoid
    circular imports), then create tables."""
    import app.data.db_models  # noqa: F401
    print("Creating tables (if missing)...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
