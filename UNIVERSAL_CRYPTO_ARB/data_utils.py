"""
Universal market data fetchers.
Priority: KuCoin (works from any IP) → Bybit → Binance.
All functions return Binance-compatible formats for drop-in replacement.

Why KuCoin first: GitHub Actions runs on Azure US servers.
  - Binance returns HTTP 451 (US restricted)
  - Bybit returns HTTP 403 (CloudFront geo-block)
  - KuCoin returns HTTP 200 from anywhere
"""
import requests
import time
import os
import json

KUCOIN_BASE  = "https://api.kucoin.com"
BYBIT_BASE   = "https://api.bybit.com"
BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoQuantBot/1.0)"}

KUCOIN_INTERVAL = {
    "1m": "1min",  "3m": "3min",  "5m": "5min",  "15m": "15min",
    "30m": "30min", "1h": "1hour", "2h": "2hour", "4h": "4hour",
    "6h": "6hour", "8h": "8hour", "12h": "12hour",
    "1d": "1day", "1w": "1week",
}

BYBIT_INTERVAL = {
    "1m": "1",  "3m": "3",  "5m": "5",  "15m": "15",
    "30m": "30", "1h": "60", "2h": "120", "4h": "240",
    "6h": "360", "12h": "720", "1d": "D", "1w": "W", "1M": "M",
}

BYBIT_OI_PERIOD = {
    "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


def _binance_symbol_to_kucoin(symbol):
    """Convert 'BTCUSDT' → 'BTC-USDT'. Falls back to adding a dash before USDT."""
    if symbol.endswith("USDT"):
        return symbol[:-4] + "-USDT"
    if symbol.endswith("BTC"):
        return symbol[:-3] + "-BTC"
    return symbol


# ── KLINES ────────────────────────────────────────────────────────────────────

def fetch_klines(symbol, interval="1h", limit=200, cache_dir=None, cache_secs=3300):
    """
    Fetch OHLCV klines. Returns Binance-compatible list of rows, oldest first:
    [[open_time_ms, open, high, low, close, volume, close_time, ...], ...]
    Tries KuCoin first, then Bybit, then Binance.
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

    data = _fetch_klines_kucoin(symbol, interval, limit)
    if data is None:
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


def _fetch_klines_kucoin(symbol, interval, limit):
    """
    KuCoin spot klines. Works from GitHub Actions US servers.
    Returns rows in Binance format: [[ts_ms, o, h, l, c, vol, ...], ...]

    KuCoin row format (newest first): [ts_sec, open, close, high, low, volume, turnover]
    Note: KuCoin index order is different — close at [2], high at [3], low at [4].
    """
    try:
        kucoin_sym = _binance_symbol_to_kucoin(symbol)
        kucoin_int = KUCOIN_INTERVAL.get(interval, "1hour")

        # KuCoin returns latest ~1500 candles without startAt/endAt
        # Requesting limit via time range if needed
        params = {"type": kucoin_int, "symbol": kucoin_sym}
        if limit and limit < 1500:
            # Set startAt to get enough history
            seconds_per_bar = {
                "1min": 60, "3min": 180, "5min": 300, "15min": 900,
                "30min": 1800, "1hour": 3600, "2hour": 7200, "4hour": 14400,
                "6hour": 21600, "8hour": 28800, "12hour": 43200,
                "1day": 86400, "1week": 604800,
            }.get(kucoin_int, 3600)
            params["startAt"] = int(time.time()) - (limit + 2) * seconds_per_bar
            params["endAt"] = int(time.time())

        r = requests.get(
            f"{KUCOIN_BASE}/api/v1/market/candles",
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != "200000":
            return None
        rows = resp.get("data", [])
        if not rows:
            return None

        rows = list(rows)
        rows.reverse()  # KuCoin newest-first → oldest-first
        # Convert KuCoin format to Binance format:
        # KuCoin: [ts_sec, open, close, high, low, volume, turnover]
        # Binance: [ts_ms, open, high, low, close, volume, close_time, qav, ...]
        padded = []
        for row in rows:
            ts_ms = str(int(row[0]) * 1000)
            padded.append([
                ts_ms,    # open_time_ms
                row[1],   # open
                row[3],   # high  (KuCoin index 3)
                row[4],   # low   (KuCoin index 4)
                row[2],   # close (KuCoin index 2)
                row[5],   # volume
                ts_ms,    # close_time placeholder
                row[6] if len(row) > 6 else "0",  # qav = turnover
                "0", "0", "0", "0",
            ])
        return padded if len(padded) > 0 else None
    except Exception:
        return None


def _fetch_klines_bybit(symbol, interval, limit):
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/kline",
            params={"category": "spot", "symbol": symbol,
                    "interval": BYBIT_INTERVAL.get(interval, "60"), "limit": limit},
            headers=HEADERS,
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
        rows.reverse()  # Bybit newest-first → oldest-first
        padded = []
        for row in rows:
            while len(row) < 7:
                row = list(row) + ["0"]
            padded.append([
                row[0], row[1], row[2], row[3], row[4], row[5],
                row[0], row[6] if len(row) > 6 else "0",
                "0", "0", "0", "0",
            ])
        return padded
    except Exception:
        return None


def _fetch_klines_binance(symbol, interval, limit):
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            headers=HEADERS,
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
    # Try KuCoin first
    try:
        kucoin_sym = _binance_symbol_to_kucoin(symbol)
        r = requests.get(
            f"{KUCOIN_BASE}/api/v1/market/orderbook/level2_20",
            params={"symbol": kucoin_sym},
            headers=HEADERS,
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") == "200000":
            d = resp.get("data", {})
            return {
                "bids": [[item[0], item[1]] for item in d.get("bids", [])],
                "asks": [[item[0], item[1]] for item in d.get("asks", [])],
            }
    except Exception:
        pass
    # Try Bybit
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": depth},
            headers=HEADERS,
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            result = resp["result"]
            return {"bids": result.get("b", []), "asks": result.get("a", [])}
    except Exception:
        pass
    # Fall back to Binance
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/depth",
            params={"symbol": symbol, "limit": depth},
            headers=HEADERS,
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
    # Try KuCoin first
    try:
        kucoin_sym = _binance_symbol_to_kucoin(symbol)
        r = requests.get(
            f"{KUCOIN_BASE}/api/v1/market/histories",
            params={"symbol": kucoin_sym},
            headers=HEADERS,
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") == "200000":
            trades = []
            for t in resp.get("data", [])[:limit]:
                # KuCoin trade: {sequence, time, price, size, side}
                # side: "buy" = buyer aggressor (is_buyer_maker=False), "sell" = seller aggressor (is_buyer_maker=True)
                trades.append({
                    "price": float(t.get("price", 0)),
                    "qty": float(t.get("size", 0)),
                    "is_buyer_maker": t.get("side", "buy") == "sell",
                    "time_ms": int(t.get("time", 0)) // 1_000_000,  # KuCoin uses nanoseconds
                })
            if trades:
                return trades
    except Exception:
        pass
    # Try Bybit
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/recent-trade",
            params={"category": "spot", "symbol": symbol, "limit": min(limit, 1000)},
            headers=HEADERS,
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
    # Fall back to Binance
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/aggTrades",
            params={"symbol": symbol, "limit": limit},
            headers=HEADERS,
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
    # Try KuCoin first
    try:
        kucoin_sym = _binance_symbol_to_kucoin(symbol)
        r = requests.get(
            f"{KUCOIN_BASE}/api/v1/market/orderbook/level1",
            params={"symbol": kucoin_sym},
            headers=HEADERS,
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") == "200000":
            px = resp.get("data", {}).get("price", 0)
            if px:
                return float(px)
    except Exception:
        pass
    # Try Bybit
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/tickers",
            params={"category": "spot", "symbol": symbol},
            headers=HEADERS,
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
    # Fall back to Binance
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/ticker/price",
            params={"symbol": symbol},
            headers=HEADERS,
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
    # Try KuCoin first
    try:
        kucoin_sym = _binance_symbol_to_kucoin(symbol)
        r = requests.get(
            f"{KUCOIN_BASE}/api/v1/market/orderbook/level1",
            params={"symbol": kucoin_sym},
            headers=HEADERS,
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") == "200000":
            d = resp.get("data", {})
            return {
                "bidPrice": d.get("bestBid", "0"),
                "askPrice": d.get("bestAsk", "0"),
                "bidQty":   d.get("bestBidSize", "0"),
                "askQty":   d.get("bestAskSize", "0"),
            }
    except Exception:
        pass
    # Try Bybit
    try:
        r = requests.get(
            f"{BYBIT_BASE}/v5/market/orderbook",
            params={"category": "spot", "symbol": symbol, "limit": 1},
            headers=HEADERS,
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
    # Fall back to Binance
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
            headers=HEADERS,
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
            headers=HEADERS,
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
            headers=HEADERS,
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
            headers=HEADERS,
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            rows = list(resp["result"]["list"])
            rows.reverse()
            return [{"fundingRate": float(row.get("fundingRate", 0)),
                     "fundingTime": int(row.get("fundingRateTimestamp", 0))} for row in rows]
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": limit},
            headers=HEADERS,
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
            headers=HEADERS,
            timeout=5,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("retCode") == 0:
            rows = list(resp["result"]["list"])
            rows.reverse()
            return [{"sumOpenInterest": row.get("openInterest", "0"),
                     "timestamp": row.get("timestamp", 0)} for row in rows]
    except Exception:
        pass
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": period, "limit": limit},
            headers=HEADERS,
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
            headers=HEADERS,
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
            headers=HEADERS,
            timeout=5,
        )
        d = r.json()
        if isinstance(d, dict):
            return float(d.get("price", 0))
    except Exception:
        pass
    return 0.0
