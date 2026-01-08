from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

APP_PORT = int(os.getenv("PORT", "5003"))
TECH_SERVICE_URL = os.getenv("TECH_SERVICE_URL", "http://localhost:5002")

app = Flask(__name__)

def fetch_history(symbol: str, n: int = 900) -> pd.DataFrame:
    url = f"{TECH_SERVICE_URL}/api/coin/{symbol.upper()}/history"
    r = requests.get(url, params={"n": n}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if not payload.get("ok"):
        raise RuntimeError(payload)
    rows = payload["history"]
    df = pd.DataFrame(rows)
    if "close" not in df.columns:
        raise ValueError("history does not contain 'close'")
    # ensure order oldest->newest
    if "time" in df.columns:
        df = df.sort_values("time")
    return df.reset_index(drop=True)

def run_lstm_prediction_df(df: pd.DataFrame, symbol: str, lookback: int = 60, horizon: int = 7) -> Dict[str, Any]:
    """
    Minimal LSTM prediction on closing prices.
    Returns metrics on a hold-out test split + future horizon forecast.
    """
    series = df["close"].astype(float).values.reshape(-1, 1)
    if len(series) < lookback + 30:
        raise ValueError(f"Not enough data for LSTM (need >= {lookback+30}, got {len(series)})")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es],
        verbose=0
    )

    preds = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test)
    preds_inv = scaler.inverse_transform(preds)

    rmse = float(np.sqrt(mean_squared_error(y_test_inv, preds_inv)))
    mape = float(mean_absolute_percentage_error(y_test_inv, preds_inv))
    r2 = float(r2_score(y_test_inv, preds_inv))

    # forecast future horizon
    last_window = scaled[-lookback:].copy()  # shape (lookback, 1)
    future_scaled = []
    for _ in range(horizon):
        pred = model.predict(last_window.reshape(1, lookback, 1), verbose=0)
        future_scaled.append(pred[0][0])
        last_window = np.vstack([last_window[1:], pred.reshape(1, 1)])

    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()

    history = df[["close"]].tail(50).copy()
    if "date" in df.columns:
        history = df[["date", "close"]].tail(50).copy()
    elif "time" in df.columns:
        history = df[["time", "close"]].tail(50).copy()

    return {
        "symbol": symbol.upper(),
        "metrics": {"rmse": rmse, "mape": mape, "r2": r2},
        "forecast": [{"day": i + 1, "predicted_close": float(future[i])} for i in range(horizon)],
        "history": history.to_dict(orient="records"),
    }

@app.get("/health")
def health():
    return {"ok": True, "service": "prediction-service"}

@app.get("/api/predict/<symbol>")
def predict(symbol: str):
    horizon = int(request.args.get("horizon", "7"))
    lookback = int(request.args.get("lookback", "60"))
    n = int(request.args.get("n", "900"))

    try:
        df = fetch_history(symbol, n=n)
        result = run_lstm_prediction_df(df, symbol, lookback=lookback, horizon=horizon)
        return jsonify({"ok": True, **result})
    except requests.RequestException as e:
        return jsonify({"ok": False, "error": f"Failed to fetch history from technical-service: {e}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    # TensorFlow can be noisy; keep defaults, but ensure it imports
    _ = tf.__version__
    app.run(host="0.0.0.0", port=APP_PORT, debug=True)
