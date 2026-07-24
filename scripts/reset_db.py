# scripts/reset_db.py
from app.data.db_connection import engine, Base

print("⚠️ Dropping all tables...")
Base.metadata.drop_all(bind=engine)

print("✅ Recreating schema...")
Base.metadata.create_all(bind=engine)

print("🎉 Database schema reset successfully.")


#python -m scripts.reset_db

# to delete data
# from app.data.db_connection import SessionLocal
# from app.data.db_models import Restaurant

# session = SessionLocal()
# session.query(Restaurant).delete()
# session.commit()
# session.close()
# print("✅ All restaurants deleted!")
