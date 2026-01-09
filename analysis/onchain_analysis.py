import sqlite3
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# APIs
BLOCKCHAIN_API_BTC = "https://api.blockchain.info/stats?format=json"
COINGECKO_API = "https://api.coingecko.com/api/v3"

ETHERSCAN_API_KEY = "6GJ56G62UE3M4SX2AFTFQHRFB6P5S6PEEZ"  # бесплатен API за ETH
ETH_HASH_RATE = 120_000_000  # proxy hash rate

# Coins со поддршка за реална on-chain анализа
REAL_ONCHAIN_COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum"
}

# Сите coinsi што сакаме да поддржиме
ONCHAIN_SUPPORTED_COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "DOGE": "dogecoin",
    "SOL": "solana",
    "ADA": "cardano",
    "BNB": "binancecoin"
}


class OnChainAnalysis:
    def __init__(self, coin_symbol="BTC"):
        if coin_symbol not in ONCHAIN_SUPPORTED_COINS:
            raise ValueError(f"On-chain analysis not supported for {coin_symbol}")

        self.coin_symbol = coin_symbol
        self.coingecko_id = ONCHAIN_SUPPORTED_COINS[coin_symbol]
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, price_df=None, return_results=False):
        # Ако е даден price_df, користи го, инаку повикај база/requests
        if price_df is not None:
            if not price_df.empty:
                price = float(price_df["close"].iloc[-1])
                volume = float(price_df["volume"].iloc[-1])
            else:
                price, volume = 0, 0
        else:
            # постоечка логика од база
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute(
                "SELECT close, volume FROM coins WHERE symbol=? ORDER BY time DESC LIMIT 1",
                (self.coin_symbol,)
            )
            row = cur.fetchone()
            conn.close()
            price, volume = row if row else (0, 0)

        # ---------- Real or proxy on-chain metrics ----------
        if self.coin_symbol == "BTC":
            metrics = self._btc_onchain()
        elif self.coin_symbol == "ETH":
            metrics = self._eth_onchain()
        else:
            if self.coin_symbol not in REAL_ONCHAIN_COINS:
                transactions = self._transactions_proxy()
                exchange_flows = self._exchange_flows() or transactions * 50
                active_addresses = self._active_addresses_proxy() or transactions * 0.01
                hash_rate = 0
                metrics = {
                    "hash_rate": hash_rate,
                    "transactions": transactions,
                    "exchange_flows": exchange_flows,
                    "active_addresses": active_addresses
                }

        market_cap = self._market_cap()
        whale_movements = self._whale_movements()
        tvl = self._tvl()
        nvt = market_cap / metrics.get("transactions", 0) if metrics.get("transactions") else 0
        mvrv = market_cap / (market_cap * 0.7) if market_cap else 0
        sentiment = self._sentiment_score()

        results = {
            "symbol": self.coin_symbol,
            "price": round(price, 4),
            "volume": round(volume, 4),
            "market_cap": round(market_cap, 2),
            "active_addresses": round(metrics.get("active_addresses", 0), 2),
            "transactions": round(metrics.get("transactions", 0), 2),
            "exchange_flows": round(metrics.get("exchange_flows", 0), 2),
            "whale_movements": whale_movements,
            "hash_rate": metrics.get("hash_rate", 0),
            "tvl": round(tvl, 2),
            "nvt": round(nvt, 4),
            "mvrv": round(mvrv, 4),
            "sentiment": round(sentiment, 4),
        }

        if return_results:
            return results

    # ---------- Helpers ----------

    def _btc_onchain(self):
        try:
            r = requests.get(BLOCKCHAIN_API_BTC, timeout=10).json()
            return {
                "hash_rate": r.get("hash_rate", 0),
                "transactions": r.get("n_tx", 0),
                "exchange_flows": r.get("total_btc_sent", 0),
                "active_addresses": r.get("estimated_btc_sent", 0)
            }
        except:
            return {}

    def _eth_onchain(self):
        try:
            # Transactions proxy
            r = requests.get(f"{COINGECKO_API}/coins/markets", params={"vs_currency": "usd", "ids": "ethereum"},
                             timeout=10).json()
            transactions = r[0].get("total_volume", 0) if r else 0

            # Exchange Flows proxy
            exchange_flows = transactions * 50  # груба проценка

            # Active addresses proxy
            r2 = requests.get(f"{COINGECKO_API}/coins/ethereum", timeout=10).json()
            active_addresses = r2.get("market_data", {}).get("circulating_supply", 0)

            return {
                "hash_rate": ETH_HASH_RATE,
                "transactions": transactions,
                "exchange_flows": exchange_flows,
                "active_addresses": active_addresses
            }
        except:
            return {}

    def _market_cap(self):
        try:
            r = requests.get(f"{COINGECKO_API}/coins/markets", params={"vs_currency": "usd", "ids": self.coingecko_id},
                             timeout=10).json()
            return r[0].get("market_cap", 0) if r else 0
        except:
            return 0

    def _whale_movements(self, threshold=1_000_000):
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM coins WHERE symbol=? AND volume>=?",
            (self.coin_symbol, threshold)
        )
        count = cur.fetchone()[0]
        conn.close()
        return count

    def _exchange_flows(self):
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute(
            "SELECT SUM(volume) FROM coins WHERE symbol=?",
            (self.coin_symbol,)
        )
        val = cur.fetchone()[0]
        conn.close()
        return val or 0

    def _active_addresses_proxy(self):
        try:
            r = requests.get(
                f"{COINGECKO_API}/coins/{self.coingecko_id}", timeout=10
            ).json()
            val = r.get("market_data", {}).get("price_change_percentage_24h", 0)
            return abs(val)
        except:
            return 0

    def _transactions_proxy(self):
        try:
            r = requests.get(
                f"{COINGECKO_API}/coins/markets",
                params={"vs_currency": "usd", "ids": self.coingecko_id},
                timeout=10
            ).json()
            return r[0].get("total_volume", 0) if r else 0
        except:
            return 0

    def _tvl(self):
        try:
            r = requests.get(f"{COINGECKO_API}/defi/tvl", timeout=10).json()
            for d in r:
                if d["id"] == self.coingecko_id:
                    return d.get("tvl", 0)
        except:
            pass
        return 0

    def _sentiment_score(self):
        text = f"{self.coin_symbol} crypto market outlook investors"
        return self.analyzer.polarity_scores(text)["compound"]