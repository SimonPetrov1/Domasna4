from __future__ import annotations

import os
from flask import Flask, jsonify, request, session
import requests

APP_PORT = int(os.getenv("PORT", "5000"))
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:5001")
TECH_SERVICE_URL = os.getenv("TECH_SERVICE_URL", "http://localhost:5002")
PRED_SERVICE_URL = os.getenv("PRED_SERVICE_URL", "http://localhost:5003")

app = Flask(__name__)
app.secret_key = os.getenv("UI_SECRET_KEY", "dev-secret-change-me")

@app.get("/health")
def health():
    return {"ok": True, "service": "ui-service"}

# ---------- Auth proxy (UI -> auth-service) ----------
@app.post("/register")
def register():
    data = request.get_json(force=True, silent=True) or {}
    r = requests.post(f"{AUTH_SERVICE_URL}/api/register", json=data, timeout=15)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})

@app.post("/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    r = requests.post(f"{AUTH_SERVICE_URL}/api/login", json=data, timeout=15)
    if r.ok:
        payload = r.json()
        if payload.get("ok") and payload.get("user"):
            session["user"] = payload["user"]
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})

@app.post("/logout")
def logout():
    session.pop("user", None)
    return jsonify({"ok": True})

@app.get("/me")
def me():
    return jsonify({"ok": True, "user": session.get("user")})

# ---------- Market / TA proxy (UI -> technical-service) ----------
@app.get("/coins")
def coins():
    limit = request.args.get("limit", "200")
    r = requests.get(f"{TECH_SERVICE_URL}/api/coins", params={"limit": limit}, timeout=30)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})

@app.get("/analysis/<symbol>")
def analysis(symbol: str):
    n = request.args.get("n", "120")
    r = requests.get(f"{TECH_SERVICE_URL}/api/analysis/{symbol}", params={"n": n}, timeout=60)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})

# ---------- Prediction proxy (UI -> prediction-service) ----------
@app.get("/predict/<symbol>")
def predict(symbol: str):
    params = {
        "horizon": request.args.get("horizon", "7"),
        "lookback": request.args.get("lookback", "60"),
        "n": request.args.get("n", "900"),
    }
    r = requests.get(f"{PRED_SERVICE_URL}/api/predict/{symbol}", params=params, timeout=120)
    return (r.text, r.status_code, {"Content-Type": r.headers.get("Content-Type", "application/json")})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
