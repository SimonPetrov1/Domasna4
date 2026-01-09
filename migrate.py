import sqlite3
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
load_dotenv()

# 1. Читај од SQLite3
sqlite_conn = sqlite3.connect("users.db")
df = pd.read_sql_query("SELECT * FROM coins", sqlite_conn)
print(f"📊 {len(df)} coins од SQLite3")
sqlite_conn.close()

# 2. Пиши во Azure PostgreSQL
pg_uri = f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}?sslmode=require"
pg_engine = create_engine(pg_uri)

df.to_sql("coins", pg_engine, if_exists="replace", index=False)
print("✅ 514 coins ПРЕФРЛЕНИ во Azure PostgreSQL!")

# 3. Провери
check_df = pd.read_sql_query("SELECT count(*) as count FROM coins", pg_engine)
print(f"✅ Во Azure: {check_df['count'].iloc[0]} записи")
