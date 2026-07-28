"""
Delete all restaurants from the database safely.
Useful before reloading fresh data from JSON.
"""

from app.data.db_connection import SessionLocal
from app.data.db_models import Restaurant

def clear_restaurants():
    session = SessionLocal()
    try:
        count = session.query(Restaurant).delete()
        session.commit()
        print(f"Deleted {count} restaurants from the database!")
    except Exception as e:
        session.rollback()
        print(f"Error while deleting restaurants: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    clear_restaurants()
