import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Читај users од SQLite
sqlite_conn = sqlite3.connect("users.db")
users_df = pd.read_sql_query("SELECT username, password_hash FROM users", sqlite_conn)
sqlite_conn.close()

print(f"{len(users_df)} users од SQLite")

# 2. Конекција кон Azure PostgreSQL (исто како migrate.py)
pg_uri = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}?sslmode=require"
)
pg_engine = create_engine(pg_uri)

# 3. Запиши во табела users (ќе ја направи ако ја нема)
users_df.to_sql("users", pg_engine, if_exists="append", index=False)

print("✅ Users префрлени во Azure PostgreSQL!")
