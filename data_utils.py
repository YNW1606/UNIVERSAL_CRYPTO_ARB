"""
Universal market data fetchers.
Bybit first (no geo-restrictions from any server), Binance fallback.
All functions return Binance-compatible formats for drop-in replacement.
"""
import requests
import time
import os
import json

BYBIT_BASE   = "https://api.bybit.com"
BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"

BYBIT_INTERVAL = {
    "1m": "1",  "3m": "3",  "5m": "5",  "15m": "15",
    "30m": "30", "1h": "60", "2h": "120", "4h": "240",
    "6h": "360", "12h": "720", "1d": "D", "1w": "W", "1M": "M",
}

BYBIT_OI_PERIOD = {
    "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


# ── KLINES ────────────────────────────────────────────────────────────────────

def fetch_klines(symbol, interval="1h", limit=200, cache_dir=None, cache_secs=3300):
    """
    Fetch OHLCV klines. Returns Binance-compatible list of rows, oldest first:
    [[open_time_ms, open, high, low, close, volume, close_time, ...], ...]
    Optionally caches valid results to cache_dir.
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = os.path.join(cache_dir, f"{symbol}_{interval}_{limit}.json")
        if os.path.exists(cache_key) and time.time() - os.path.getmtime(cache_key) < cache_secs:
            try:
                with open(cache_key) as f:
                    d = json.load(f)
                if isinstance(d, list) and len(d) > 0 and isinstance(d[0], list):
                    return d
            except Exception:
                pass
    else:
        cache_key = None

    data = _fetch_klines_bybit(symbol, interval, limit)
    if data is None:
        data = _fetch_klines_binance(symbol, interval, limit)
    if data is None:
        raise RuntimeError(f"All klines sources failed for {symbol} {interval}")

    if cache_key:
        try:
            with open(cache_key, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    return data


def _fetch_klines_bybit(symbol, interval, limit):
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/kline",
            params={"category": "spot", "symbol": symbol,
                    "interval": BYBIT_INTERVAL.get(interval, "60"), "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") != 0:
            return None
        rows = resp["result"]["list"]
        if not rows:
            return None
        rows = list(rows)
        rows.reverse()  # Bybit is newest-first → flip to oldest-first (Binance order)
        # Pad each row to 12 columns matching Binance format:
        # [open_time, open, high, low, close, volume, close_time, qav, num_trades, tbbav, tbqav, ignore]
        padded = []
        for row in rows:
            while len(row) < 7:
                row = list(row) + ["0"]
            padded.append([
                row[0], row[1], row[2], row[3], row[4], row[5],
                row[0],                # close_time placeholder
                row[6] if len(row) > 6 else "0",  # qav = turnover
                "0", "0", "0", "0",   # num_trades, tbbav, tbqav, ignore
            ])
        return padded
    except Exception:
        return None


def _fetch_klines_binance(symbol, interval, limit):
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        if isinstance(d, list) and len(d) > 0:
            return d
    except Exception:
        pass
    return None


# ── ORDERBOOK ─────────────────────────────────────────────────────────────────

def fetch_orderbook(symbol, depth=20):
    """Returns {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}"""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": depth},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            result = resp["result"]
            return {"bids": result.get("b", []), "asks": result.get("a", [])}
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/depth",
            params={"symbol": symbol, "limit": depth},
            timeout=5,
        )
        d = r.json()
        if isinstance(d, dict) and "bids" in d:
            return d
    except Exception:
        pass
    return {"bids": [], "asks": []}


# ── TRADES ────────────────────────────────────────────────────────────────────

def fetch_agg_trades(symbol, limit=1000):
    """Returns list of {price, qty, is_buyer_maker, time_ms}"""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/recent-trade",
            params={"category": "spot", "symbol": symbol, "limit": min(limit, 1000)},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            trades = []
            for t in resp["result"]["list"]:
                trades.append({
                    "price": float(t.get("price", 0)),
                    "qty": float(t.get("qty", t.get("size", 0))),
                    "is_buyer_maker": t.get("side", "Buy") == "Sell",
                    "time_ms": int(t.get("time", 0)),
                })
            return trades
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/aggTrades",
            params={"symbol": symbol, "limit": limit},
            timeout=5,
        )
        trades = []
        for t in r.json():
            if isinstance(t, dict):
                trades.append({
                    "price": float(t.get("p", 0)),
                    "qty": float(t.get("q", 0)),
                    "is_buyer_maker": bool(t.get("m", False)),
                    "time_ms": int(t.get("T", 0)),
                })
        return trades
    except Exception:
        pass
    return []


# ── TICKER PRICE ──────────────────────────────────────────────────────────────

def fetch_ticker_price(symbol):
    """Returns current price as float."""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/tickers",
            params={"category": "spot", "symbol": symbol},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            items = resp["result"]["list"]
            if items:
                return float(items[0].get("lastPrice", 0))
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        d = r.json()
        if isinstance(d, dict) and "price" in d:
            return float(d["price"])
    except Exception:
        pass
    return 0.0


def fetch_book_ticker(symbol):
    """Returns {"bidPrice": ..., "askPrice": ..., "bidQty": ..., "askQty": ...}"""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": 1},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            result = resp["result"]
            bids = result.get("b", [])
            asks = result.get("a", [])
            return {
                "bidPrice": bids[0][0] if bids else "0",
                "askPrice": asks[0][0] if asks else "0",
                "bidQty":   bids[0][1] if bids else "0",
                "askQty":   asks[0][1] if asks else "0",
            }
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
            timeout=5,
        )
        d = r.json()
        if isinstance(d, dict) and "bidPrice" in d:
            return d
    except Exception:
        pass
    return {"bidPrice": "0", "askPrice": "0", "bidQty": "0", "askQty": "0"}


# ── PERP / FUNDING ────────────────────────────────────────────────────────────

def fetch_funding_rate(symbol):
    """Returns {lastFundingRate, nextFundingTime, markPrice, indexPrice}"""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            items = resp["result"]["list"]
            if items:
                t = items[0]
                return {
                    "lastFundingRate": float(t.get("fundingRate", 0)),
                    "nextFundingTime": int(t.get("nextFundingTime", 0)),
                    "markPrice":       float(t.get("markPrice", 0)),
                    "indexPrice":      float(t.get("indexPrice", 0)),
                }
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=5,
        )
        d = r.json()
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {}


def fetch_funding_history(symbol, limit=20):
    """Returns list of dicts with fundingRate key, oldest first."""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/funding/history",
            params={"category": "linear", "symbol": symbol, "limit": limit},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            rows = list(resp["result"]["list"])
            rows.reverse()  # Bybit newest-first → oldest-first
            return [{"fundingRate": float(row.get("fundingRate", 0)),
                     "fundingTime": int(row.get("fundingRateTimestamp", 0))} for row in rows]
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": limit},
            timeout=5,
        )
        d = r.json()
        if isinstance(d, list):
            return d
    except Exception:
        pass
    return []


def fetch_oi_history(symbol, period="1h", limit=48):
    """Returns list of dicts with sumOpenInterest key, oldest first."""
    try:
        bybit_period = BYBIT_OI_PERIOD.get(period, "1h")
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/open-interest",
            params={"category": "linear", "symbol": symbol,
                    "intervalTime": bybit_period, "limit": limit},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            rows = list(resp["result"]["list"])
            rows.reverse()  # Bybit newest-first → oldest-first
            return [{"sumOpenInterest": row.get("openInterest", "0"),
                     "timestamp": row.get("timestamp", 0)} for row in rows]
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": period, "limit": limit},
            timeout=5,
        )
        d = r.json()
        if isinstance(d, list):
            return d
    except Exception:
        pass
    return []


def fetch_perp_price(symbol):
    """Returns perpetual futures mark price as float."""
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            items = resp["result"]["list"]
            if items:
                return float(items[0].get("markPrice", 0))
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        d = r.json()
        if isinstance(d, dict):
            return float(d.get("price", 0))
    except Exception:
        pass
    return 0.0
