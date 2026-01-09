import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from .analysis_strategy import AnalysisStrategy

class LSTMAnalysis(AnalysisStrategy):
    """
    Concrete Strategy: LSTM Analysis (DB-based)
    """

    def __init__(
        self,
        coin_symbol="BTC",
        lookback=30,
        train_ratio=0.7,
        epochs=20,
        batch_size=32
    ):
        self.coin_symbol = coin_symbol
        self.lookback = lookback
        self.train_ratio = train_ratio
        self.epochs = epochs
        self.batch_size = batch_size

    def analyze(self, df: pd.DataFrame, return_results=False):
        # ----------------- FILTER COIN -----------------
        coin_df = df[df["symbol"] == self.coin_symbol].sort_values("time").copy()
        if coin_df.empty:
            raise ValueError(f"No rows found for symbol {self.coin_symbol}")

        prices = coin_df["close"].values.reshape(-1, 1)

        # ----------------- SCALE DATA -----------------
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_scaled = scaler.fit_transform(prices)

        # ----------------- CREATE SEQUENCES -----------------
        X, y = self._create_sequences(prices_scaled)

        # ----------------- TRAIN / TEST SPLIT -----------------
        train_size = int(len(X) * self.train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        if len(X_test) == 0:
            raise ValueError("Not enough data for training/testing split.")

        # ----------------- BUILD MODEL -----------------
        model = models.Sequential([
            layers.LSTM(
                units=50,
                activation="tanh",
                input_shape=(self.lookback, 1)
            ),
            layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        # ----------------- TRAIN -----------------
        model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # ----------------- PREDICTION -----------------
        y_pred_scaled = model.predict(X_test)
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred_scaled)

        # ----------------- METRICS -----------------
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)

        print(f"\n========== LSTM RESULTS for {self.coin_symbol} ==========")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R²: {r2:.4f}")
        print("==================================\n")

        if return_results:
            return y_test_inv, y_pred_inv

        plt.figure(figsize=(10,5))
        plt.plot(y_test_inv, label="Real price")
        plt.plot(y_pred_inv, label="Predicted price")
        plt.title(f"LSTM price prediction for {self.coin_symbol}")
        plt.legend()
        plt.show()

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)
