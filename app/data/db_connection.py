# app/data/db_connection.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuration: point to your local Postgres 'goodfoods' DB
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/goodfoods"
)

# Define Base here once. Models should import this Base (not the other way around).
Base = declarative_base()

# Create engine & session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Ensure models are imported so they register with Base, then create tables.
    Importing models here (after Base is created) avoids circular imports.
    """
    # Import models to ensure SQLAlchemy metadata is populated
    # (module import order matters: import here, not at top)
    import app.data.db_models  # noqa: F401  (models register themselves on import)
    print("Creating tables (if missing)...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
