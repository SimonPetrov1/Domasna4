import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import (
    MACD,
    ADXIndicator,
    CCIIndicator,
    SMAIndicator,
    EMAIndicator,
    WMAIndicator,
)
from ta.volatility import BollingerBands

# -------- LOAD AND PREPARE DATA -------- #

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("date")


# -------- ADD TECHNICAL INDICATORS -------- #

def add_indicators(df):
    # ===== OSCILLATORS (5) =====
    df["RSI"] = RSIIndicator(df["close"]).rsi()

    macd = MACD(close=df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    stoch = StochasticOscillator(
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )
    df["Stochastic"] = stoch.stoch()

    df["ADX"] = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ).adx()

    df["CCI"] = CCIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ).cci()

    # ===== MOVING AVERAGES (5) =====
    df["SMA_20"] = SMAIndicator(df["close"], window=20).sma_indicator()
    df["EMA_20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["WMA_20"] = WMAIndicator(df["close"], window=20).wma()

    bb = BollingerBands(df["close"])
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    # Volume Moving Average
    df["Volume_SMA_20"] = df["volume"].rolling(window=20).mean()

    return df


# -------- BUY / SELL / HOLD SIGNALS -------- #

def generate_signals(df):
    df["signal"] = "HOLD"

    buy_condition = (
        (df["RSI"] < 30) &
        (df["close"] > df["EMA_20"]) &
        (df["MACD"] > df["MACD_signal"])
    )

    sell_condition = (
        (df["RSI"] > 70) &
        (df["close"] < df["EMA_20"]) &
        (df["MACD"] < df["MACD_signal"])
    )

    df.loc[buy_condition, "signal"] = "BUY"
    df.loc[sell_condition, "signal"] = "SELL"

    return df


# -------- RESAMPLE TIMEFRAMES -------- #

def analyze_timeframes(df):
    df = df.set_index("date")

    frames = {
        "1D": df.resample("1D").last(),
        "1W": df.resample("1W").last(),
        "1M": df.resample("1M").last(),
    }

    return frames


# -------- FULL PIPELINE -------- #

def run_technical_analysis(csv_path):
    df = load_data(csv_path)
    df = add_indicators(df)
    df = generate_signals(df)
    timeframes = analyze_timeframes(df)

    return df, timeframes


# -------- RUN STANDALONE (OPTIONAL) -------- #

if __name__ == "__main__":
    df, timeframes = run_technical_analysis("../data/processed/all_coins.csv")

    print("\n=== Latest Signals ===")
    print(df[["date", "close", "RSI", "MACD", "CCI", "signal"]].tail())

    print("\n=== Weekly Analysis Sample ===")
    print(timeframes["1W"].tail())
