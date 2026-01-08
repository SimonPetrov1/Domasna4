from __future__ import annotations

import os
import sqlite3
from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

APP_PORT = int(os.getenv("PORT", "5001"))
DB_PATH = os.getenv("AUTH_DB_PATH", "users.db")

app = Flask(__name__)

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@app.get("/health")
def health():
    return {"ok": True, "service": "auth-service"}

@app.post("/api/register")
def register():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"ok": False, "error": "username and password required"}), 400

    pw_hash = generate_password_hash(password)

    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pw_hash))
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
    except sqlite3.IntegrityError:
        return jsonify({"ok": False, "error": "username already exists"}), 409

    return jsonify({"ok": True, "user": {"id": user_id, "username": username}}), 201

@app.post("/api/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username or not password:
        return jsonify({"ok": False, "error": "username and password required"}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"ok": False, "error": "invalid credentials"}), 401

    return jsonify({"ok": True, "user": {"id": int(row["id"]), "username": row["username"]}})

@app.get("/api/users/<int:user_id>")
def get_user(user_id: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return jsonify({"ok": False, "error": "not found"}), 404
    return jsonify({"ok": True, "user": {"id": int(row["id"]), "username": row["username"]}})

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
