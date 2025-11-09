# scripts/update_restaurant_schema.py
from sqlalchemy import text
from app.data.db_connection import engine  # ✅ uses same connection as your app

alter_statements = [
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS cuisines JSONB DEFAULT '[]';",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS amenities JSONB DEFAULT '[]';",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS seating_sections JSONB DEFAULT '[]';",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS menu JSONB DEFAULT '[]';",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS policies JSONB DEFAULT '{}'::jsonb;",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS opening_hours JSONB DEFAULT '{}'::jsonb;",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS ops_metrics JSONB DEFAULT '{}'::jsonb;",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS contact JSONB DEFAULT '{}'::jsonb;",
    "ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]';",
]

with engine.connect() as conn:
    for stmt in alter_statements:
        print(f"Executing: {stmt}")
        conn.execute(text(stmt))
    conn.commit()

print("✅ Restaurant schema successfully updated.")
