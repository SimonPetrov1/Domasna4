import sqlite3
import pandas as pd

conn = sqlite3.connect("users.db")
df = pd.read_sql_query("SELECT * FROM coins", conn)
conn.close()


print("Coins imported.")
