import base64

from flask import Flask, render_template, abort, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from sqlalchemy import text
import pandas as pd
import os
from flask import jsonify
import io
import base64
from analysis.technical_analysis import TechnicalAnalysis
from analysis.onchain_analysis import OnChainAnalysis
from analysis.onchain_analysis import ONCHAIN_SUPPORTED_COINS, REAL_ONCHAIN_COINS
from matplotlib import pyplot as plt

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "secret123")

# =========================
# DATABASE CONFIG (AZURE)
# =========================

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://dbadmin:Dbdasdomasno!@crypto-app.postgres.database.azure.com:5432/example_db?sslmode=require",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================
# MODELS
# =========================
class User(db.Model):
    __tablename__ = "users"

    username = db.Column(db.String(50), primary_key=True)
    password_hash = db.Column(db.String(255), nullable=False)


class Coin(db.Model):
    __tablename__ = 'coins'

    time = db.Column(db.BigInteger, primary_key=True)  # timestamp
    open_price = db.Column(db.Float, name='open')  # open
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)  # користи close како price
    volume = db.Column(db.Float)
    symbol = db.Column(db.String(10))


# coins табелата ја користиме директно преку pandas + SQL,
# не ни треба ORM модел ако не менуваме шема.


# =========================
# HELPERS
# =========================
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


def load_coins_df():
    # користиме SQLAlchemy engine кон Azure PostgreSQL
    with db.engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM coins"), conn)
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.date
    return df


# =========================
# PROFILE
# =========================
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect(url_for("login"))

    message = None
    username = session["user"]

    user = User.query.filter_by(username=username).first()
    if not user:
        # ако не постои во базата, одјави го
        return redirect(url_for("logout"))

    if request.method == "POST":
        old_pw = request.form.get("old_password")
        new_pw = request.form.get("new_password")
        confirm = request.form.get("confirm_password")

        if not check_password_hash(user.password_hash, old_pw or ""):
            message = "Wrong current password."
        elif new_pw != confirm:
            message = "Passwords do not match."
        else:
            user.password_hash = generate_password_hash(new_pw)
            db.session.commit()
            message = "Password successfully changed."

    return render_template("profile.html", username=username, message=message)


# =========================
# INDEX
# =========================
@app.route("/")
def index():
    df = load_coins_df()

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

    sorted_by_mcap = sorted(
        stats.keys(), key=lambda s: stats[s]["market_cap"], reverse=True
    )[:10]
    top10 = {s: stats[s] for s in sorted_by_mcap}

    sorted_by_price = sorted(
        stats.keys(), key=lambda s: stats[s]["price"], reverse=True
    )[:3]
    top3_price = [(s, stats[s]) for s in sorted_by_price]

    return render_template("index.html", top10=top10, top3_price=top3_price)


# =========================
# AUTH
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session["user"] = username
            return redirect(url_for("index"))

        return render_template("login.html", error="Invalid username or password.")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            return render_template("register.html", error="Please fill all fields.")

        if User.query.filter_by(username=username).first():
            return render_template("register.html", error="Username already exists.")

        user = User(
            username=username,
            password_hash=generate_password_hash(password),
        )
        db.session.add(user)
        db.session.commit()

        session["user"] = username
        return redirect(url_for("markets"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("index"))


@app.route("/help")
def help_page():
    return render_template("help.html")


# =========================
# MARKETS
# =========================
@app.route("/markets")
def markets():
    if "user" not in session:
        return redirect(url_for("login"))

    df = load_coins_df()

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

    total_market_cap = sum(v["market_cap"] for v in stats.values())
    total_volume = sum(v["last_volume"] for v in stats.values())
    total_market_cap_fmt = f"${fmt_number(total_market_cap)}"
    total_volume_fmt = fmt_number(total_volume)

    btc_price = stats.get("BTC", {}).get("price_fmt")
    eth_price = stats.get("ETH", {}).get("price_fmt")
    sol_price = stats.get("SOL", {}).get("price_fmt")

    reverse = (direction == "desc")
    if sort == "price":
        all_symbols = sorted(
            stats.keys(),
            key=lambda s: stats[s]["last_close"],
            reverse=reverse,
        )
    else:
        all_symbols = sorted(
            stats.keys(),
            key=lambda s: stats[s]["last_volume"],
            reverse=reverse,
        )

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


# =========================
# COIN DETAIL
# =========================
@app.route("/coin/<symbol>")
def coin_detail(symbol):
    df = load_coins_df()
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

    trading_vol_fmt = fmt_number(last_row["volume"])
    market_cap_val = last_row["close"] * last_row["volume"]
    market_cap_fmt = fmt_number(market_cap_val)
    close_fmt = fmt_number(last_row["close"])

    lowest_low = min(r["low"] for r in rows)
    highest_high = max(r["high"] for r in rows)
    range_low_fmt = f"{lowest_low:,.2f}"
    range_high_fmt = f"{highest_high:,.2f}"
    price_range = float(last_row["high"]) - float(last_row["low"])
    range_value = f"{price_range:,.2f}"

    # ATH / ATL за template ако ги користиш
    max_close = max(r["close"] for r in rows)
    min_close = min(r["close"] for r in rows)

    return render_template(
        "coin_detail.html",
        symbol=symbol,
        last=last_row,
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
        max_close=max_close,
        min_close=min_close,
    )


@app.route('/api/coins')
def get_coins():
    # Најди последниот запис за секој symbol
    latest_coins = db.session.query(
        Coin.symbol.label('symbol'),
        db.func.max(Coin.time).label('max_time')
    ).group_by(Coin.symbol).subquery()

    # Join за да земеш ги целите записи
    coins = db.session.query(Coin).join(
        latest_coins,
        (Coin.symbol == latest_coins.c.symbol) &
        (Coin.time == latest_coins.c.max_time)
    ).order_by(Coin.symbol).all()

    return jsonify([{
        'symbol': coin.symbol,
        'name': coin.symbol,  # или земи од API ако имаш
        'price': coin.close,
        'high': coin.high,
        'low': coin.low,
        'volume': coin.volume,
        'change_24h': 0.0  # пресметај ако сакаш
    } for coin in coins])


@app.route('/api/coins/<symbol>')
def get_coin(symbol):
    latest = Coin.query.filter_by(symbol=symbol.upper()).\
        order_by(Coin.time.desc()).first()
    if latest:
        return jsonify({
            'symbol': latest.symbol,
            'name': latest.symbol,
            'price': latest.close,
            'high': latest.high,
            'low': latest.low,
            'volume': latest.volume
        })
    return jsonify({'error': f'{symbol} not found'}), 404

@app.route('/api/coins/all')  # Нова рута
def get_all_coins():
    coins = Coin.query.order_by(Coin.time.desc()).all()
    return jsonify([{
        'symbol': coin.symbol,
        'price': coin.close,
        'time': coin.time,
        'high': coin.high,
        'low': coin.low,
        'volume': coin.volume
    } for coin in coins])

def load_coin_df(symbol):
    with db.engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM coins WHERE symbol = :symbol ORDER BY time"),
            conn,
            params={"symbol": symbol}
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["time"], unit="s")
    return df


@app.route("/technical/<symbol>")
def technical_analysis_page(symbol):
    if "user" not in session:
        return redirect(url_for("login"))

    df = load_coin_df(symbol)
    if df.empty:
        abort(404)

    analysis = TechnicalAnalysis(coin_symbol=symbol)
    df, timeframes = analysis.analyze(df=df, return_results=True)

    # Последни 20 реда за табела
    latest_signals = df.tail(20)

    plots = {}
    for tf_name, tf_df in timeframes.items():
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # -------- PRICE --------
        ax1.plot(tf_df["close"], color="blue", label="Close")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # -------- RSI --------
        if "RSI" in tf_df:
            ax2 = ax1.twinx()
            ax2.plot(tf_df["RSI"], color="red", label="RSI")
            ax2.set_ylabel("RSI", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

        ax1.set_title(f"{symbol} - {tf_name}")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plots[tf_name] = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)

    # -------- MACD (1D) --------
    macd_img = None
    if "1D" in timeframes:
        tf_df = timeframes["1D"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(tf_df["MACD"], label="MACD")
        ax.plot(tf_df["MACD_signal"], label="Signal")
        ax.legend()
        ax.set_title(f"{symbol} - MACD (1D)")

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
        return render_template(
            "onchain.html",
            symbol=symbol,
            supported=False,
            data=None
        )

    price_df = load_coin_df(symbol)

    # Испрати само coin_symbol при иницијализација
    analysis = OnChainAnalysis(symbol)

    # Потоа прати df во analyze
    data = analysis.analyze(price_df=price_df, return_results=True)

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

lstm_analysis_cache = {}

@app.route("/lstm/<symbol>")
def lstm_analysis_page(symbol):
    if "user" not in session:
        return redirect(url_for("login"))

    df = load_coin_df(symbol)
    if df.empty:
        abort(404)

    # LAZY IMPORT: само кога ќе се повика страницата
    try:
        from analysis.lstm_price_prediction import LSTMAnalysis
    except Exception as e:
        return f"LSTM module not available: {e}", 500

    # LAZY MODEL LOAD
    if symbol not in lstm_analysis_cache:
        print(f"🔄 Loading LSTM model for {symbol}...")
        lstm_analysis_cache[symbol] = LSTMAnalysis(
            coin_symbol=symbol,
            lookback=30,
            train_ratio=0.7,
            epochs=10,  # за побрзо тестирање
            batch_size=32
        )

    analysis = lstm_analysis_cache[symbol]
    y_test_inv, y_pred_inv = analysis.analyze(df=df, return_results=True)

    # Плотирање
    import matplotlib.pyplot as plt
    import io, base64

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test_inv, label="Real price")
    ax.plot(y_pred_inv, label="Predicted price")
    ax.set_title(f"LSTM price prediction for {symbol}")
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return render_template(
        "lstm.html",
        symbol=symbol,
        plot_data=img_base64
    )

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
