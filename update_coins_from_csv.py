import pandas as pd
import sqlite3

CSV_PATH = ("data/processed/all_coins.csv")
DB_PATH = "users.db"

df = pd.read_csv(CSV_PATH)

# осигурај се дека time е integer
df["time"] = df["time"].astype(int)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# избриши само coins, НЕ users
cur.execute("DELETE FROM coins")

# внесе нови податоци
df.to_sql("coins", conn, if_exists="append", index=False)

conn.commit()
conn.close()

print("Coins table successfully updated from CSV")
