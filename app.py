from flask import Flask, render_template, abort, request, redirect, url_for, session
import pandas as pd
import json
import os

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


app = Flask(__name__)
app.secret_key = "secret123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "all_coins.csv")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["time"], unit="s").dt.date


@app.route("/")
def index():
    latest_two = df.sort_values("time").groupby("symbol").tail(2)
    grouped = latest_two.groupby("symbol")

    stats = {}
    for symbol, g in grouped:
        g = g.sort_values("time")
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        change_24h = 0.0
        if prev is not None and prev["close"] != 0:
            change_24h = (last["close"] - prev["close"]) / prev["close"] * 100

        price = float(last["close"])
        volume = float(last["volume"])
        market_cap = price * volume

        stats[symbol] = {
            "price": price,
            "volume": volume,
            "market_cap": market_cap,
            "price_fmt": f"${fmt_number(price)}",
            "volume_fmt": fmt_number(volume),
            "market_cap_fmt": fmt_number(market_cap),
            "change_24h": change_24h,
        }

    sorted_symbols = sorted(
        stats.keys(),
        key=lambda s: stats[s]["price"],
        reverse=True
    )[:10]

    top10 = {s: stats[s] for s in sorted_symbols}
    return render_template("index.html", top10=top10)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin":
            session["user"] = username
            return redirect(url_for("coins"))
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

    tmp = df.copy()
    tmp = tmp[tmp["volume"] >= min_vol]
    if min_price > 0:
        tmp = tmp[tmp["close"] >= min_price]
    if max_price > 0:
        tmp = tmp[tmp["close"] <= max_price]

    latest_two = tmp.sort_values("time").groupby("symbol").tail(2)
    grouped = latest_two.groupby("symbol")

    stats = {}
    for symbol, g in grouped:
        g = g.sort_values("time")
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        change_24h = 0.0
        if prev is not None and prev["close"] != 0:
            change_24h = (last["close"] - prev["close"]) / prev["close"] * 100

        price = float(last["close"])
        volume = float(last["volume"])
        market_cap = price * volume

        stats[symbol] = {
            "last_close": price,
            "last_volume": volume,
            "market_cap": market_cap,
            "price_fmt": f"${fmt_number(price)}",
            "volume_fmt": fmt_number(volume),
            "market_cap_fmt": fmt_number(market_cap),
            "change_24h": change_24h,
        }

    if query:
        stats = {s: v for s, v in stats.items() if query in s}

    reverse = (direction == "desc")
    if sort == "price":
        symbols = sorted(stats.keys(), key=lambda s: stats[s]["last_close"], reverse=reverse)
    else:  # volume
        symbols = sorted(stats.keys(), key=lambda s: stats[s]["last_volume"], reverse=reverse)

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
    range_low_fmt = fmt_number(last_row["low"])
    range_high_fmt = fmt_number(last_row["high"])
    close_fmt = fmt_number(last_row["close"])

    return render_template(
        "coin_detail.html",
        symbol=symbol,
        rows=rows,
        last=last_row,
        chart_data=json.dumps(chart_data),
        change_1h=change_1h,
        change_7d=change_7d,
        change_30d=change_30d,
        trading_vol_fmt=trading_vol_fmt,
        market_cap_fmt=market_cap_fmt,
        range_low_fmt=range_low_fmt,
        range_high_fmt=range_high_fmt,
        close_fmt=close_fmt,
    )


if __name__ == "__main__":
    app.run(debug=True)
