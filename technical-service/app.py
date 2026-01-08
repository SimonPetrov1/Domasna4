from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, request

from analysis.technical_analysis import add_indicators, generate_signals

APP_PORT = int(os.getenv("PORT", "5002"))
DB_PATH = os.getenv("MARKET_DB_PATH", "market.db")
CSV_PATH = os.getenv("COINS_CSV_PATH", os.path.join("data", "processed", "all_coins.csv"))

app = Flask(__name__)

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_coins_loaded() -> None:
    """
    Creates/refreshes the 'coins' table from CSV if it doesn't exist.
    This keeps the service self-contained and easy to run for grading.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='coins'")
    exists = cur.fetchone() is not None
    conn.close()
    if exists:
        return

    # Create table by importing CSV
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"COINS_CSV_PATH not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    # normalize time column if present
    if "time" in df.columns:
        df["time"] = df["time"].astype(int, errors="ignore")

    conn = get_db()
    df.to_sql("coins", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def load_symbol_df(symbol: str) -> pd.DataFrame:
    conn = get_db()
    q = "SELECT * FROM coins WHERE symbol = ? ORDER BY time ASC"
    df = pd.read_sql_query(q, conn, params=(symbol.upper(),))
    conn.close()
    if df.empty:
        raise KeyError(f"symbol not found: {symbol}")
    # try to standardize a 'date' column for frontends
    if "time" in df.columns and "date" not in df.columns:
        # keep as int epoch seconds/millis - do not guess units; client can format
        df["date"] = df["time"]
    return df

@app.get("/health")
def health():
    return {"ok": True, "service": "technical-service"}

@app.get("/api/coins")
def coins():
    """
    Returns a list of symbols with simple last-known stats.
    Query params: limit (default 200)
    """
    limit = int(request.args.get("limit", "200"))
    conn = get_db()
    # Get the latest row per symbol (SQLite trick using MAX(time))
    df = pd.read_sql_query("""
        SELECT c.*
        FROM coins c
        JOIN (
            SELECT symbol, MAX(time) AS max_time
            FROM coins
            GROUP BY symbol
        ) latest
        ON c.symbol = latest.symbol AND c.time = latest.max_time
        ORDER BY c.symbol ASC
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()

    out = []
    for _, r in df.iterrows():
        out.append({
            "symbol": r.get("symbol"),
            "close": float(r.get("close")) if "close" in r and pd.notna(r.get("close")) else None,
            "volume": float(r.get("volume")) if "volume" in r and pd.notna(r.get("volume")) else None,
            "time": int(r.get("time")) if "time" in r and pd.notna(r.get("time")) else None,
        })
    return jsonify({"ok": True, "coins": out})

@app.get("/api/coin/<symbol>/history")
def coin_history(symbol: str):
    """
    Returns OHLCV history for a symbol.
    Query params: n (default 300)
    """
    n = int(request.args.get("n", "300"))
    df = load_symbol_df(symbol).tail(n)
    cols = [c for c in ["date", "time", "open", "high", "low", "close", "volume"] if c in df.columns]
    return jsonify({"ok": True, "symbol": symbol.upper(), "history": df[cols].to_dict(orient="records")})

@app.get("/api/analysis/<symbol>")
def analysis(symbol: str):
    """
    Adds indicators + generates signals. Returns last N rows with key columns.
    Query params: n (default 120)
    """
    n = int(request.args.get("n", "120"))
    df = load_symbol_df(symbol).copy()

    # Your indicator code expects columns like close/high/low etc.
    df = add_indicators(df)
    df = generate_signals(df)

    keep_cols = [c for c in [
        "date", "time", "open", "high", "low", "close", "volume",
        "RSI", "Stoch_K", "Stoch_D",
        "MACD", "MACD_signal",
        "ADX", "CCI",
        "SMA_20", "EMA_20", "WMA_20",
        "BB_high", "BB_low",
        "signal"
    ] if c in df.columns]

    view = df.tail(n)[keep_cols]
    return jsonify({"ok": True, "symbol": symbol.upper(), "analysis": view.to_dict(orient="records")})

if __name__ == "__main__":
    ensure_coins_loaded()
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
