from flask import Flask, render_template, abort, request, redirect, url_for, session
import pandas as pd
import json
import os
import sqlite3

from matplotlib import pyplot as plt
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session, redirect, url_for, render_template, request
from werkzeug.security import check_password_hash, generate_password_hash
import io
import base64
from analysis import lstm_price_prediction
from analysis.lstm_price_prediction import LSTMAnalysis
from analysis.technical_analysis import TechnicalAnalysis
from analysis.onchain_analysis import OnChainAnalysis
from analysis.onchain_analysis import ONCHAIN_SUPPORTED_COINS, REAL_ONCHAIN_COINS


app = Flask(__name__)
app.secret_key = "secret123"


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    message = None
    username = session["user"]

    if request.method == "POST":
        old_pw = request.form.get("old_password")
        new_pw = request.form.get("new_password")
        confirm = request.form.get("confirm_password")

        conn = get_db()
        cur = conn.cursor()

        # земи го корисникот
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()

        if not user or not check_password_hash(user["password_hash"], old_pw):
            message = "Wrong current password."
        elif new_pw != confirm:
            message = "Passwords do not match."
        else:
            new_hash = generate_password_hash(new_pw)
            cur.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (new_hash, username),
            )
            conn.commit()
            message = "Password successfully changed."

        conn.close()

    return render_template("profile.html", username=username, message=message)


def get_db():
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            return render_template("register.html", error="Please fill all fields.")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            conn.close()
            return render_template("register.html", error="Username already exists.")

        pwd_hash = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, pwd_hash),
        )
        conn.commit()
        conn.close()

        session["user"] = username
        return redirect(url_for("markets"))

    return render_template("register.html")


def fmt_number(x):
    x = float(x)
    abs_x = abs(x)
    if abs_x >= 1_000_000_000_000:
        return f"{x/1_000_000_000_000:.2f}T"
    elif abs_x >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    elif abs_x >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    elif abs_x >= 1_000:
        return f"{x/1_000:.2f}K"
    else:
        return f"{x:.2f}"


conn = sqlite3.connect("users.db")
df = pd.read_sql_query("SELECT * FROM coins", conn)
conn.close()

df["date"] = pd.to_datetime(df["time"], unit="s").dt.date


@app.route("/")
def index():
    # земи последни 2 реда по coin за цена + 24h change
    tmp = df.copy().sort_values("time")
    latest_two = tmp.groupby("symbol").tail(2)

    stats = {}
    for symbol, g in latest_two.groupby("symbol"):
        g = g.sort_values("time")
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None

        price = float(last["close"])
        volume = float(last["volume"])
        market_cap = price * volume

        change_24h = 0.0
        if prev is not None and prev["close"] != 0:
            change_24h = (price - float(prev["close"])) / float(prev["close"]) * 100.0

        stats[symbol] = {
            "price": price,
            "volume": volume,
            "market_cap": market_cap,
            "price_fmt": f"${fmt_number(price)}",
            "volume_fmt": fmt_number(volume),
            "market_cap_fmt": fmt_number(market_cap),
            "change_24h": change_24h,
        }

    # топ 10 по market cap за табелата долу
    sorted_by_mcap = sorted(
        stats.keys(), key=lambda s: stats[s]["market_cap"], reverse=True
    )[:10]
    top10 = {s: stats[s] for s in sorted_by_mcap}

    # топ 3 по цена за Market Overview box
    sorted_by_price = sorted(
        stats.keys(), key=lambda s: stats[s]["price"], reverse=True
    )[:3]
    top3_price = [(s, stats[s]) for s in sorted_by_price]

    return render_template("index.html", top10=top10, top3_price=top3_price)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row["password_hash"], password):
            session["user"] = username
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))


@app.route("/help")
def help_page():
    return render_template("help.html")

@app.route("/markets")
def markets():
    if "user" not in session:
        return redirect(url_for("login"))

    query = request.args.get("q", "").upper()
    sort = request.args.get("sort", "price")
    direction = request.args.get("dir", "desc")
    min_vol = request.args.get("min_vol", 0, type=float)
    min_price = request.args.get("min_price", 0, type=float)
    max_price = request.args.get("max_price", 0, type=float)

    page = request.args.get("page", 1, type=int)
    per_page = 15

    tmp = df.copy().sort_values("time")
    latest_two = tmp.groupby("symbol").tail(2)

    stats = {}
    for symbol, g in latest_two.groupby("symbol"):
        g = g.sort_values("time")
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None

        price = float(last["close"])
        volume = float(last["volume"])
        market_cap = price * volume

        change_24h = 0.0
        if prev is not None and prev["close"] != 0:
            change_24h = (price - float(prev["close"])) / float(prev["close"]) * 100.0

        stats[symbol] = {
            "last_close": price,
            "last_volume": volume,
            "market_cap": market_cap,
            "price_fmt": f"${fmt_number(price)}",
            "volume_fmt": fmt_number(volume),
            "market_cap_fmt": fmt_number(market_cap),
            "change_24h": change_24h,
        }

    # филтри
    stats = {s: v for s, v in stats.items() if v["last_volume"] >= min_vol}

    if min_price > 0:
        stats = {s: v for s, v in stats.items() if v["last_close"] >= min_price}

    if max_price > 0:
        stats = {s: v for s, v in stats.items() if v["last_close"] <= max_price}

    if query:
        stats = {s: v for s, v in stats.items() if query in s}

    # тотален market cap и 24h volume
    total_market_cap = sum(v["market_cap"] for v in stats.values())
    total_volume = sum(v["last_volume"] for v in stats.values())
    total_market_cap_fmt = f"${fmt_number(total_market_cap)}"
    total_volume_fmt = fmt_number(total_volume)

    # trending цени за BTC, ETH, SOL
    btc_price = stats.get("BTC", {}).get("price_fmt")
    eth_price = stats.get("ETH", {}).get("price_fmt")
    sol_price = stats.get("SOL", {}).get("price_fmt")

    # сортирање
    reverse = (direction == "desc")
    if sort == "price":
        all_symbols = sorted(
            stats.keys(), key=lambda s: stats[s]["last_close"], reverse=reverse
        )
    else:
        all_symbols = sorted(
            stats.keys(), key=lambda s: stats[s]["last_volume"], reverse=reverse
        )

    # pagination
    total = len(all_symbols)
    start = (page - 1) * per_page
    end = start + per_page
    symbols = all_symbols[start:end]
    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "coins.html",
        symbols=symbols,
        stats=stats,
        query=query,
        sort=sort,
        direction=direction,
        min_vol=min_vol,
        min_price=min_price,
        max_price=max_price,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        total_market_cap=total_market_cap_fmt,
        total_volume=total_volume_fmt,
        btc_price=btc_price,
        eth_price=eth_price,
        sol_price=sol_price,
    )

@app.route("/coin/<symbol>")
def coin_detail(symbol):
    coin_df = df[df["symbol"] == symbol].sort_values("time")
    if coin_df.empty:
        abort(404)

    rows = coin_df.to_dict(orient="records")
    last_row = rows[-1]

    def pct_change(days):
        if len(rows) <= days:
            return 0.0
        prev = rows[-(days + 1)]["close"]
        curr = last_row["close"]
        if prev == 0:
            return 0.0
        return (curr - prev) / prev * 100.0

    change_1h = pct_change(1)
    change_7d = pct_change(7)
    change_30d = pct_change(30)

    chart_data = {
        "dates": [str(r["date"]) for r in rows],
        "prices": [r["close"] for r in rows],
    }

    def fmt_number(x):
        x = float(x)
        abs_x = abs(x)
        if abs_x >= 1_000_000_000_000:
            return f"{x / 1_000_000_000_000:.2f}T"
        elif abs_x >= 1_000_000_000:
            return f"{x / 1_000_000_000:.2f}B"
        elif abs_x >= 1_000_000:
            return f"{x / 1_000_000:.2f}M"
        elif abs_x >= 1_000:
            return f"{x / 1_000:.2f}K"
        else:
            return f"{x:.2f}"

    trading_vol_fmt = fmt_number(last_row["volume"])
    market_cap_val = last_row["close"] * last_row["volume"]
    market_cap_fmt = fmt_number(market_cap_val)
    close_fmt = fmt_number(last_row["close"])

    # HIGH–LOW за целиот период
    lowest_low = min(r["low"] for r in rows)
    highest_high = max(r["high"] for r in rows)
    range_low_fmt = f"{lowest_low:,.2f}"
    range_high_fmt = f"{highest_high:,.2f}"

    # RANGE = high - low за последниот ден
    price_range = float(last_row["high"]) - float(last_row["low"])
    range_value = f"{price_range:,.2f}"


    return render_template(
        "coin_detail.html",
        symbol=symbol,
        last=last_row,  # ако немаш друго 'last'
        rows=rows,
        change_1h=change_1h,
        change_7d=change_7d,
        change_30d=change_30d,
        trading_vol_fmt=trading_vol_fmt,
        market_cap_fmt=market_cap_fmt,
        close_fmt=close_fmt,
        range_low_fmt=range_low_fmt,
        range_high_fmt=range_high_fmt,
        range_value=range_value,
        chart_data=chart_data,
    )


@app.route("/technical/<symbol>")
def technical_analysis_page(symbol):
    if "user" not in session:
        return redirect(url_for("login"))

    analysis = TechnicalAnalysis(coin_symbol=symbol)
    df, timeframes = analysis.analyze("data/processed/all_coins.csv", return_results=True)

    # Последни 20 реда за табела
    latest_signals = df.tail(20)

    # Генерирање графици за сите временски рамки
    plots = {}
    for tf_name, tf_df in timeframes.items():
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # ---------- Цена ----------
        ax1.plot(tf_df["close"], color="blue", label="Close")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # ---------- RSI на secondary y-axis ----------
        if "RSI" in tf_df:
            ax2 = ax1.twinx()
            ax2.plot(tf_df["RSI"], color="red", label="RSI")
            ax2.set_ylabel("RSI", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

        ax1.set_title(f"{symbol} - {tf_name} timeframe")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        plots[tf_name] = img_base64
        plt.close(fig)

    # ---------- Дополнителен график: MACD + signal за 1D ----------
    macd_img = None
    if "1D" in timeframes:
        tf_df = timeframes["1D"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tf_df["MACD"], label="MACD", color="blue")
        ax.plot(tf_df["MACD_signal"], label="Signal", color="orange")
        ax.set_title(f"{symbol} - MACD (1D)")
        ax.set_xlabel("Date")
        ax.set_ylabel("MACD")
        ax.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        macd_img = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

    return render_template(
        "technical.html",
        symbol=symbol,
        latest_signals=latest_signals,
        plots=plots,
        macd_img=macd_img
    )


ONCHAIN_COINS = ["BTC", "ETH", "SOL", "BNB", "ADA"]

@app.route("/onchain/<symbol>")
def onchain_page(symbol):
    if "user" not in session:
        return redirect(url_for("login"))

    if symbol not in ONCHAIN_COINS:
        # ако coin не е поддржан
        return render_template(
            "onchain.html",
            symbol=symbol,
            supported=False,
            data=None
        )

    analysis = OnChainAnalysis(symbol)
    data = analysis.analyze(return_results=True)

    return render_template(
        "onchain.html",
        symbol=symbol,
        supported=True,
        data=data
    )


@app.route("/onchain")
def onchain_list():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template(
        "onchain_list.html",
        ONCHAIN_SUPPORTED_COINS=ONCHAIN_SUPPORTED_COINS,
        REAL_ONCHAIN_COINS=REAL_ONCHAIN_COINS
    )

if __name__ == "__main__":
    app.run(debug=True)
