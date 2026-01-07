import sqlite3

conn = sqlite3.connect("users.db")  # ќе се креира users.db во овој фолдер
with open("schema.sql", "r") as f:
    conn.executescript(f.read())
conn.close()

print("Database initialized.")