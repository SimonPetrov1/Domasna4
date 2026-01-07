import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, ADXIndicator, CCIIndicator, SMAIndicator, EMAIndicator, WMAIndicator
from ta.volatility import BollingerBands

class TechnicalAnalysis:
    """
    Technical Analysis Strategy
    """

    def __init__(self, coin_symbol="BTC"):
        self.coin_symbol = coin_symbol

    def analyze(self, csv_path, return_results=False):
        # ---------- Load Data ----------
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("date")
        coin_df = df[df["symbol"] == self.coin_symbol].copy()
        if coin_df.empty:
            raise ValueError(f"No data found for symbol {self.coin_symbol}")

        # ---------- Indicators ----------
        coin_df["RSI"] = RSIIndicator(coin_df["close"]).rsi()

        macd = MACD(close=coin_df["close"])
        coin_df["MACD"] = macd.macd()
        coin_df["MACD_signal"] = macd.macd_signal()

        stoch = StochasticOscillator(high=coin_df["high"], low=coin_df["low"], close=coin_df["close"])
        coin_df["Stochastic"] = stoch.stoch()

        coin_df["ADX"] = ADXIndicator(high=coin_df["high"], low=coin_df["low"], close=coin_df["close"]).adx()
        coin_df["CCI"] = CCIIndicator(high=coin_df["high"], low=coin_df["low"], close=coin_df["close"]).cci()

        coin_df["SMA_20"] = SMAIndicator(coin_df["close"], window=20).sma_indicator()
        coin_df["EMA_20"] = EMAIndicator(coin_df["close"], window=20).ema_indicator()
        coin_df["WMA_20"] = WMAIndicator(coin_df["close"], window=20).wma()

        bb = BollingerBands(coin_df["close"])
        coin_df["BB_upper"] = bb.bollinger_hband()
        coin_df["BB_lower"] = bb.bollinger_lband()

        coin_df["Volume_SMA_20"] = coin_df["volume"].rolling(window=20).mean()

        # ---------- Signals ----------
        coin_df["signal"] = "HOLD"
        buy_condition = (coin_df["RSI"] < 30) & (coin_df["close"] > coin_df["EMA_20"]) & (coin_df["MACD"] > coin_df["MACD_signal"])
        sell_condition = (coin_df["RSI"] > 70) & (coin_df["close"] < coin_df["EMA_20"]) & (coin_df["MACD"] < coin_df["MACD_signal"])
        coin_df.loc[buy_condition, "signal"] = "BUY"
        coin_df.loc[sell_condition, "signal"] = "SELL"

        # ---------- Timeframes ----------
        coin_df = coin_df.set_index("date")
        timeframes = {
            "1D": coin_df.resample("1D").last(),
            "1W": coin_df.resample("1W").last(),
            "1M": coin_df.resample("1M").last(),
        }

        if return_results:
            return coin_df, timeframes
