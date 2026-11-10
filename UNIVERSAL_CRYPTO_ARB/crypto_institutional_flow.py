#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crypto_institutional_flow.py
=============================
PURE INSTITUTIONAL FLOW + ARBITRAGE DETECTOR

NO INDICATORS. NO PATTERNS. NO RSI/MACD/HURST.

Every signal comes from actual market activity:
  - Who is buying vs selling RIGHT NOW (aggTrades CVD)
  - Where large money is hiding in the order book (iceberg / wall)
  - How fast open interest is changing (new money entering positions)
  - What the perp vs spot price gap tells us (basis arb)
  - What top traders and institutions are positioned for
  - Where the liquidation cascade triggers are
  - What cross-exchange price differences mean (arb)
  - What the funding rate is saying about trapped leverage

ARBITRAGE TRADES (fully computed yield):
  1. Cash-and-Carry (long spot + short perp, harvest positive funding)
  2. Reverse Carry  (short spot + long perp, harvest negative funding)
  3. Cross-Exchange (buy cheap exchange, sell expensive exchange)
  4. Spot-Perp Basis (perp premium/discount = arb opportunity)
  5. Triangular Arb (BTC→ETH→BNB→BTC loop on live prices)

Usage:
    python crypto_institutional_flow.py
    python crypto_institutional_flow.py --symbols BTCUSDT ETHUSDT SOLUSDT
    python crypto_institutional_flow.py --quick           (BTC+ETH only)
    python crypto_institutional_flow.py --arb-only        (arb trades only)
    python crypto_institutional_flow.py --json
"""

import argparse
import sys
import time
import math
import json
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

try:
    import requests
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "numpy", "-q"])
    import requests
    import numpy as np
    HAS_NUMPY = True

# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_FAPI  = "https://fapi.binance.com"
BINANCE_SPOT  = "https://api.binance.com"
OKX_API       = "https://www.okx.com"
BYBIT_API     = "https://api.bybit.com"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                   "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "ADAUSDT"]

# Flow thresholds
LARGE_TRADE_USD    = 100_000   # $100k+ = large trade
WHALE_TRADE_USD    = 500_000   # $500k+ = whale
BLOCK_TRADE_USD    = 1_000_000 # $1M+ = institutional block

# Order book
WALL_MULT          = 6.0       # level qty > 6x avg = wall
ICEBERG_REPEAT     = 5         # same price hit 5+ times in trades = iceberg
DEPTH_BUCKETS_PCT  = [0.10, 0.25, 0.50, 1.0, 2.0]  # % from mid to measure depth

# OI thresholds
OI_VELOCITY_HIGH   = 2.0       # >2% OI change per hour = fast accumulation
OI_VELOCITY_FAST   = 5.0       # >5% per hour = very fast

# Funding
FUNDING_EXTREME    = 5.0       # >5bps/8h = extreme
FUNDING_HIGH       = 2.0

# Liq cascade
LIQ_CLUSTER_PCT    = 0.015
LIQ_NEAR_PCT       = 0.02      # liq cluster within 2% = danger zone

# Arbitrage
ARB_MIN_SPREAD_BPS = 5         # minimum 5bps spread to flag cross-exchange arb
ARB_MIN_CARRY_PCT  = 3.0       # minimum 3% annual yield for carry arb
TRI_ARB_MIN_PCT    = 0.03      # minimum 0.03% profit after fees for tri arb

HEADERS = {"User-Agent": "Mozilla/5.0 InstitutionalFlow/1.0"}

# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 10, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(0.3 * (attempt + 1))
    return None

def _safe(x, d=0.0):
    try:
        v = float(x)
        return d if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return d

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_agg_trades(symbol: str, limit: int = 1500) -> List[dict]:
    """Raw aggTrades from Binance perp futures (falls back to spot)."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/aggTrades",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        data = _get(f"{BINANCE_SPOT}/api/v3/aggTrades",
                    {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return []
    trades = []
    for t in data:
        try:
            trades.append({
                "price":    float(t["p"]),
                "qty":      float(t["q"]),
                "usd":      float(t["p"]) * float(t["q"]),
                "is_maker": bool(t["m"]),   # True = taker sold (bearish)
                "time":     int(t["T"]),
            })
        except Exception:
            pass
    return trades


def fetch_order_book(symbol: str, depth: int = 500) -> dict:
    """Deep L2 order book."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/depth",
                {"symbol": symbol, "limit": depth})
    if not isinstance(data, dict):
        data = _get(f"{BINANCE_SPOT}/api/v3/depth",
                    {"symbol": symbol, "limit": depth})
    if not isinstance(data, dict):
        return {}
    bids = [(float(p), float(q)) for p, q in data.get("bids", []) if float(q) > 0]
    asks = [(float(p), float(q)) for p, q in data.get("asks", []) if float(q) > 0]
    return {"bids": bids, "asks": asks,
            "last_update": data.get("lastUpdateId", 0)}


def fetch_premium_index(symbol: str) -> dict:
    """Funding rate, mark price, index price, basis."""
    d = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    if not isinstance(d, dict):
        return {}
    try:
        mark  = float(d.get("markPrice", 0))
        idx   = float(d.get("indexPrice", 0))
        basis = (mark - idx) / idx * 100 if idx > 0 else 0
        return {
            "funding_rate":   float(d.get("lastFundingRate", 0)),
            "mark_price":     mark,
            "index_price":    idx,
            "basis_pct":      basis,
            "basis_bps":      basis * 100,
            "next_funding":   int(d.get("nextFundingTime", 0)),
        }
    except Exception:
        return {}


def fetch_funding_history(symbol: str, limit: int = 48) -> List[dict]:
    """Last 48 × 8h funding records = ~16 days."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        try:
            rate = float(item.get("fundingRate", 0))
            result.append({
                "time": int(item.get("fundingTime", 0)),
                "rate": rate,
                "bps":  rate * 10000,
                "ann_pct": rate * 1095 * 100,
            })
        except Exception:
            pass
    return sorted(result, key=lambda x: x["time"])


def fetch_open_interest_full(symbol: str) -> dict:
    """Current OI + 1h history (48 bars) + 5m history (12 bars = 1h velocity)."""
    cur_raw   = _get(f"{BINANCE_FAPI}/fapi/v1/openInterest", {"symbol": symbol})
    hist_1h   = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                     {"symbol": symbol, "period": "1h", "limit": 48})
    hist_5m   = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                     {"symbol": symbol, "period": "5m", "limit": 12})

    cur_oi = float(cur_raw.get("openInterest", 0)) if isinstance(cur_raw, dict) else 0.0

    def _parse(data):
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            try:
                out.append({
                    "time":    int(item.get("timestamp", 0)),
                    "oi":      float(item.get("sumOpenInterest", 0)),
                    "oi_usd":  float(item.get("sumOpenInterestValue", 0)),
                })
            except Exception:
                pass
        return sorted(out, key=lambda x: x["time"])

    series_1h = _parse(hist_1h)
    series_5m = _parse(hist_5m)

    # OI velocity: change per hour (using 5m series for precision)
    oi_velocity_1h = 0.0
    if len(series_5m) >= 2:
        old = series_5m[0]["oi"]; new = series_5m[-1]["oi"]
        if old > 0:
            oi_velocity_1h = (new - old) / old * 100  # % change over last hour

    # 24h OI change
    oi_change_24h = 0.0
    if len(series_1h) >= 24:
        old = series_1h[-24]["oi"]
        if old > 0:
            oi_change_24h = (series_1h[-1]["oi"] - old) / old * 100

    # OI trend: rising or falling consistently (last 6 hours)
    oi_trend = "flat"
    if len(series_1h) >= 6:
        vals = [x["oi"] for x in series_1h[-6:]]
        rising = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
        if rising >= 5: oi_trend = "strongly_rising"
        elif rising >= 4: oi_trend = "rising"
        elif rising <= 1: oi_trend = "strongly_falling"
        elif rising <= 2: oi_trend = "falling"

    return {
        "oi":              cur_oi,
        "oi_velocity_1h":  oi_velocity_1h,
        "oi_change_24h":   oi_change_24h,
        "oi_trend":        oi_trend,
        "series_1h":       series_1h,
    }


def fetch_long_short_ratios(symbol: str) -> dict:
    """Global, top-account and top-position L/S ratios + 24h history."""
    endpoints = {
        "global":    f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio",
        "top_acct":  f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio",
        "top_pos":   f"{BINANCE_FAPI}/futures/data/topLongShortPositionRatio",
    }
    result = {}
    for key, url in endpoints.items():
        data = _get(url, {"symbol": symbol, "period": "1h", "limit": 24})
        if isinstance(data, list) and data:
            vals = []
            for d in data:
                try:
                    vals.append({
                        "time":  int(d.get("timestamp", 0)),
                        "ratio": float(d.get("longShortRatio",
                                       d.get("longAccount", 1.0))),
                    })
                except Exception:
                    pass
            if vals:
                ratios = [v["ratio"] for v in vals]
                slope = float(np.polyfit(range(len(ratios)), ratios, 1)[0]) \
                        if len(ratios) >= 3 else 0.0
                result[key] = {
                    "current": vals[-1]["ratio"],
                    "24h_min": min(ratios),
                    "24h_max": max(ratios),
                    "24h_avg": statistics.mean(ratios),
                    "trend_slope": slope,     # positive = ratio rising (more longs)
                    "history": vals[-6:],     # last 6 periods
                }
    return result


def fetch_liquidations(symbol: str, limit: int = 500) -> dict:
    """Force-liquidation orders — build a heatmap of where stops are."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/allForceOrders",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return {}

    long_liq = short_liq = 0.0
    price_usd = []  # (price, usd, side)

    for item in data:
        try:
            side  = str(item.get("side", ""))
            qty   = float(item.get("origQty", 0))
            price = float(item.get("price", 0))
            usd   = qty * price
            if side == "SELL":  # Long position got liquidated
                long_liq += usd
                price_usd.append((price, usd, "LONG_LIQ"))
            else:               # Short position got liquidated
                short_liq += usd
                price_usd.append((price, usd, "SHORT_LIQ"))
        except Exception:
            pass

    # Build liq clusters
    clusters = []
    if price_usd:
        avg_p = statistics.mean([p for p, u, s in price_usd])
        tol = avg_p * LIQ_CLUSTER_PCT
        sorted_pu = sorted(price_usd, key=lambda x: x[0])
        cur = [sorted_pu[0]]
        for item in sorted_pu[1:]:
            if item[0] - cur[-1][0] <= tol:
                cur.append(item)
            else:
                if len(cur) >= 2:
                    total_usd = sum(u for _, u, _ in cur)
                    avg_lvl   = sum(p * u for p, u, _ in cur) / max(total_usd, 1)
                    side_dom  = "LONG_LIQ" if sum(1 for _, _, s in cur if s == "LONG_LIQ") > len(cur)/2 else "SHORT_LIQ"
                    clusters.append({"level": round(avg_lvl, 4), "usd": total_usd,
                                     "count": len(cur), "side": side_dom})
                cur = [item]
        if len(cur) >= 2:
            total_usd = sum(u for _, u, _ in cur)
            avg_lvl   = sum(p * u for p, u, _ in cur) / max(total_usd, 1)
            side_dom  = "LONG_LIQ" if sum(1 for _, _, s in cur if s == "LONG_LIQ") > len(cur)/2 else "SHORT_LIQ"
            clusters.append({"level": round(avg_lvl, 4), "usd": total_usd,
                             "count": len(cur), "side": side_dom})
        clusters.sort(key=lambda x: -x["usd"])

    return {
        "long_liq_usd":  long_liq,
        "short_liq_usd": short_liq,
        "total_liq_usd": long_liq + short_liq,
        "clusters":      clusters[:10],
    }


def fetch_spot_price(symbol: str) -> float:
    """Current spot price."""
    d = _get(f"{BINANCE_SPOT}/api/v3/ticker/price", {"symbol": symbol})
    if isinstance(d, dict):
        return _safe(d.get("price", 0))
    return 0.0


def fetch_perp_price(symbol: str) -> float:
    """Current perp mark price."""
    d = _get(f"{BINANCE_FAPI}/fapi/v1/ticker/price", {"symbol": symbol})
    if isinstance(d, dict):
        return _safe(d.get("price", 0))
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-EXCHANGE PRICE FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_okx_price(symbol: str) -> float:
    """OKX perp mark price. symbol = 'BTCUSDT' → 'BTC-USDT-SWAP'."""
    base = symbol.replace("USDT", "")
    inst_id = f"{base}-USDT-SWAP"
    d = _get(f"{OKX_API}/api/v5/market/ticker", {"instId": inst_id}, timeout=8)
    if isinstance(d, dict) and d.get("code") == "0":
        try:
            return _safe(d["data"][0].get("last", 0))
        except Exception:
            pass
    return 0.0


def fetch_bybit_price(symbol: str) -> float:
    """Bybit perp price. symbol = 'BTCUSDT'."""
    d = _get(f"{BYBIT_API}/v5/market/tickers",
             {"category": "linear", "symbol": symbol}, timeout=8)
    if isinstance(d, dict) and d.get("retCode") == 0:
        try:
            return _safe(d["result"]["list"][0].get("lastPrice", 0))
        except Exception:
            pass
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# INSTITUTIONAL FLOW ANALYZERS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_trade_flow(trades: List[dict], spot: float) -> dict:
    """
    Deep analysis of raw aggTrades:
    - CVD: cumulative delta (buy volume - sell volume)
    - Whale / block trade detection
    - Iceberg detection (same price hit repeatedly by large trades)
    - Trade velocity (acceleration in last 20% vs first 80%)
    - Session breakdown (Asia/London/NY)
    - Large lot clustering (institutional accumulation / distribution)
    """
    if not trades:
        return {}

    buy_usd = sell_usd = 0.0
    deltas   = []
    large_trades = []
    whale_trades = []
    block_trades = []
    session_delta = {"Asia": 0.0, "London": 0.0, "New_York": 0.0, "Other": 0.0}
    price_hits: Dict[float, float] = defaultdict(float)  # price → total usd hit

    for t in trades:
        usd      = t["usd"]
        is_sell  = t["is_maker"]   # maker = taker sold into it
        delta    = -usd if is_sell else usd

        deltas.append(delta)
        if is_sell:
            sell_usd += usd
        else:
            buy_usd += usd

        # Session
        ts_ms = t["time"]
        if ts_ms > 0:
            h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
            if   0  <= h < 8:  session_delta["Asia"]     += delta
            elif 7  <= h < 16: session_delta["London"]   += delta
            elif 13 <= h < 21: session_delta["New_York"] += delta
            else:              session_delta["Other"]     += delta

        # Classify size
        if usd >= BLOCK_TRADE_USD:
            block_trades.append(t)
        elif usd >= WHALE_TRADE_USD:
            whale_trades.append(t)
        elif usd >= LARGE_TRADE_USD:
            large_trades.append(t)

        # Track price hits for iceberg detection
        p_rounded = round(t["price"], 2)
        price_hits[p_rounded] += usd

    # Iceberg detection: price levels hit > ICEBERG_REPEAT times with large total
    avg_usd = (buy_usd + sell_usd) / max(len(trades), 1)
    iceberg_levels = []
    for price, total_usd in price_hits.items():
        hit_count = sum(1 for t in trades if round(t["price"], 2) == price)
        if hit_count >= ICEBERG_REPEAT and total_usd >= avg_usd * ICEBERG_REPEAT:
            side_usd_buy  = sum(t["usd"] for t in trades
                                if round(t["price"], 2) == price and not t["is_maker"])
            side_usd_sell = sum(t["usd"] for t in trades
                                if round(t["price"], 2) == price and t["is_maker"])
            iceberg_levels.append({
                "price": price,
                "total_usd": total_usd,
                "hit_count": hit_count,
                "side": "BUY" if side_usd_buy >= side_usd_sell else "SELL",
            })
    iceberg_levels.sort(key=lambda x: -x["total_usd"])

    # Trade velocity: compare last 20% vs first 80%
    n = len(deltas)
    split = max(1, int(n * 0.80))
    early_rate = abs(sum(deltas[:split])) / max(split, 1)
    late_rate  = abs(sum(deltas[split:])) / max(n - split, 1)
    velocity_accel = late_rate / max(early_rate, 1e-9)  # >1 = accelerating flow

    # CVD slope (last 20% direction)
    recent_delta = sum(deltas[split:])
    cum_delta    = sum(deltas)

    # Net institutional flow — include large trades ($100k+) not just blocks/whales
    inst_buy_usd  = sum(t["usd"] for t in block_trades + whale_trades + large_trades if not t["is_maker"])
    inst_sell_usd = sum(t["usd"] for t in block_trades + whale_trades + large_trades if t["is_maker"])

    # Smart money CVD (≥$100k) vs retail CVD (<$5k)
    smart_cvd  = sum((-t["usd"] if t["is_maker"] else t["usd"])
                     for t in block_trades + whale_trades + large_trades)
    retail_cvd = sum((-t["usd"] if t["is_maker"] else t["usd"])
                     for t in trades if t["usd"] < 5_000)
    smart_retail_div = "SMART_MONEY_BUYING_RETAIL_SELLING" if smart_cvd > 0 and retail_cvd < 0 else (
                       "SMART_MONEY_SELLING_RETAIL_BUYING" if smart_cvd < 0 and retail_cvd > 0 else
                       "ALIGNED")

    # Delta exhaustion: positive CVD but decelerating = buyers losing steam
    delta_exhaustion = (cum_delta > 0 and velocity_accel < 0.5) or (cum_delta < 0 and velocity_accel < 0.5)

    # Large lot clustering: group large trades by price proximity (within 0.1% of spot)
    proximity_pct = 0.001 * spot
    clustered_buys  = []
    clustered_sells = []
    for t in large_trades + whale_trades + block_trades:
        p_group = round(t["price"] / proximity_pct) * proximity_pct
        if not t["is_maker"]:
            clustered_buys.append((p_group, t["usd"]))
        else:
            clustered_sells.append((p_group, t["usd"]))

    # Aggregate into price-level buckets
    buy_clusters = {}
    for p, usd in clustered_buys:
        buy_clusters[round(p, 2)] = buy_clusters.get(round(p, 2), 0) + usd
    sell_clusters = {}
    for p, usd in clustered_sells:
        sell_clusters[round(p, 2)] = sell_clusters.get(round(p, 2), 0) + usd

    top_buy_clusters  = sorted(buy_clusters.items(),  key=lambda x: -x[1])[:5]
    top_sell_clusters = sorted(sell_clusters.items(), key=lambda x: -x[1])[:5]

    return {
        "cum_delta":        cum_delta,
        "recent_delta":     recent_delta,
        "buy_usd":          buy_usd,
        "sell_usd":         sell_usd,
        "cvd_ratio":        buy_usd / max(sell_usd, 1e-9),
        "total_trades":     len(trades),
        "velocity_accel":   velocity_accel,
        "inst_buy_usd":     inst_buy_usd,
        "inst_sell_usd":    inst_sell_usd,
        "net_inst_usd":     inst_buy_usd - inst_sell_usd,
        "smart_cvd":        smart_cvd,
        "retail_cvd":       retail_cvd,
        "smart_retail_div": smart_retail_div,
        "delta_exhaustion": delta_exhaustion,
        "block_trades":     sorted(block_trades,  key=lambda x: -x["usd"])[:5],
        "whale_trades":     sorted(whale_trades,  key=lambda x: -x["usd"])[:5],
        "iceberg_levels":   iceberg_levels[:5],
        "top_buy_clusters": top_buy_clusters,
        "top_sell_clusters": top_sell_clusters,
        "session_delta":    session_delta,
    }


def analyze_order_book(ob: dict, spot: float) -> dict:
    """
    Deep order book analysis:
    - Depth imbalance at multiple levels
    - Wall detection + wall USD value
    - Absorption detection (is a large wall shrinking?)
    - Hidden liquidity estimate (ask/bid sparse zones = vacuum)
    - Spread pressure (tight = conviction, wide = uncertainty)
    """
    if not ob:
        return {}

    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        return {}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid      = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10000

    # Depth at multiple % buckets from mid
    depth_buckets = {}
    for pct in DEPTH_BUCKETS_PCT:
        bid_usd = sum(p * q for p, q in bids if (mid - p) / mid * 100 <= pct)
        ask_usd = sum(p * q for p, q in asks if (p - mid) / mid * 100 <= pct)
        depth_buckets[f"{pct}pct"] = {
            "bid_usd": bid_usd,
            "ask_usd": ask_usd,
            "imbalance": bid_usd / max(ask_usd, 1e-9),
        }

    # Wall detection
    all_bid_qty = [q for _, q in bids]
    all_ask_qty = [q for _, q in asks]
    avg_bid_qty = statistics.mean(all_bid_qty) if all_bid_qty else 1.0
    avg_ask_qty = statistics.mean(all_ask_qty) if all_ask_qty else 1.0

    bid_walls = [(p, q, p * q) for p, q in bids if q >= avg_bid_qty * WALL_MULT]
    ask_walls = [(p, q, p * q) for p, q in asks if q >= avg_ask_qty * WALL_MULT]

    # Sort by USD value of wall
    bid_walls.sort(key=lambda x: -x[2])
    ask_walls.sort(key=lambda x: -x[2])

    # Nearest wall above and below
    nearest_bid_wall = bid_walls[0] if bid_walls else None
    nearest_ask_wall = ask_walls[0] if ask_walls else None

    # Vacuum detection: price range with very thin liquidity (institutional stop zone)
    # Find gaps > 3x average spacing
    def _find_vacuums(levels: list, side: str) -> list:
        prices = sorted([p for p, q in levels], reverse=(side == "bid"))
        if len(prices) < 3:
            return []
        spacings = [abs(prices[i] - prices[i+1]) for i in range(len(prices)-1)]
        avg_sp   = statistics.mean(spacings) if spacings else 0
        vacs = []
        for i, sp in enumerate(spacings):
            if sp > avg_sp * 3:
                vacs.append({
                    "from": round(min(prices[i], prices[i+1]), 4),
                    "to":   round(max(prices[i], prices[i+1]), 4),
                    "gap_pct": sp / mid * 100,
                })
        return sorted(vacs, key=lambda x: -x["gap_pct"])[:3]

    bid_vacuums = _find_vacuums(bids, "bid")
    ask_vacuums = _find_vacuums(asks, "ask")

    # Overall bias
    primary_imb = depth_buckets.get("0.5pct", {}).get("imbalance", 1.0)
    if primary_imb > 2.0:
        ob_bias = "STRONG_BID"
    elif primary_imb > 1.3:
        ob_bias = "BID_HEAVY"
    elif primary_imb < 0.5:
        ob_bias = "STRONG_ASK"
    elif primary_imb < 0.77:
        ob_bias = "ASK_HEAVY"
    else:
        ob_bias = "BALANCED"

    return {
        "best_bid":          best_bid,
        "best_ask":          best_ask,
        "mid":               mid,
        "spread_bps":        spread_bps,
        "depth_buckets":     depth_buckets,
        "bid_walls":         bid_walls[:5],
        "ask_walls":         ask_walls[:5],
        "nearest_bid_wall":  nearest_bid_wall,
        "nearest_ask_wall":  nearest_ask_wall,
        "bid_vacuums":       bid_vacuums,
        "ask_vacuums":       ask_vacuums,
        "ob_bias":           ob_bias,
        "total_bid_usd_1pct": depth_buckets.get("1.0pct", {}).get("bid_usd", 0),
        "total_ask_usd_1pct": depth_buckets.get("1.0pct", {}).get("ask_usd", 0),
    }


def analyze_oi_flow(oi_data: dict, spot: float, flow_data: dict) -> dict:
    """
    Combine OI velocity + CVD to determine if money is ENTERING or EXITING.

    OI rising  + price rising  + CVD positive = NEW LONGS entering (bullish conviction)
    OI rising  + price falling + CVD negative = NEW SHORTS entering (bearish conviction)
    OI falling + price rising  + CVD positive = SHORT COVERING (less conviction, may fade)
    OI falling + price falling + CVD negative = LONG LIQUIDATION / CAPITULATION
    OI rising  + price flat   + CVD mixed    = POSITION BUILDUP at current price (pending breakout)
    """
    oi_vel = _safe(oi_data.get("oi_velocity_1h", 0))
    oi_trend = oi_data.get("oi_trend", "flat")
    cum_delta = _safe(flow_data.get("cum_delta", 0))
    net_inst  = _safe(flow_data.get("net_inst_usd", 0))

    # Approximate price direction from CVD (since we have no OHLCV here)
    price_rising  = cum_delta > 0
    price_falling = cum_delta < 0
    oi_growing    = oi_vel > 0.5 or oi_trend in ("rising", "strongly_rising")
    oi_shrinking  = oi_vel < -0.5 or oi_trend in ("falling", "strongly_falling")

    if oi_growing and price_rising:
        flow_type = "INSTITUTIONAL_LONG_BUILD"
        note = (f"OI velocity +{oi_vel:.2f}%/hr + CVD positive + net inst flow "
                f"${net_inst/1e6:.2f}M → NEW LONGS opening. Bullish conviction.")
        score = 2
    elif oi_growing and price_falling:
        flow_type = "INSTITUTIONAL_SHORT_BUILD"
        note = (f"OI velocity +{oi_vel:.2f}%/hr + CVD negative + net inst "
                f"${net_inst/1e6:.2f}M → NEW SHORTS opening. Bearish conviction.")
        score = -2
    elif oi_shrinking and price_rising:
        flow_type = "SHORT_SQUEEZE"
        note = (f"OI falling {oi_vel:.2f}%/hr + price rising → Shorts being squeezed out. "
                f"Momentum driven, not new longs.")
        score = 1
    elif oi_shrinking and price_falling:
        flow_type = "LONG_LIQUIDATION"
        note = (f"OI falling {oi_vel:.2f}%/hr + price falling → Long capitulation. "
                f"Forced selling in progress.")
        score = -1
    elif oi_growing and abs(cum_delta) < abs(net_inst) * 0.5:
        flow_type = "POSITION_BUILDUP_PENDING"
        note = (f"OI growing at +{oi_vel:.2f}%/hr but price flat → "
                f"Institutions loading positions before a move. Breakout imminent.")
        score = 0
    else:
        flow_type = "NEUTRAL_FLOW"
        note = f"OI velocity={oi_vel:.2f}%/hr — no strong institutional flow signal."
        score = 0

    return {
        "flow_type": flow_type,
        "note": note,
        "score": score,
        "oi_velocity_1h": oi_vel,
        "oi_trend": oi_trend,
        "oi_24h_pct": oi_data.get("oi_change_24h", 0),
    }


def analyze_positioning(ls_data: dict, fund_data: dict, fund_hist: list) -> dict:
    """
    Combine top-trader L/S ratios + funding rate to build a full positioning picture.
    When everyone is on one side → squeeze risk.
    When top traders diverge from retail → follow top traders.
    """
    if not ls_data:
        return {}

    top_pos_cur   = ls_data.get("top_pos", {}).get("current", 1.0)
    top_acct_cur  = ls_data.get("top_acct", {}).get("current", 1.0)
    global_cur    = ls_data.get("global", {}).get("current", 1.0)
    top_pos_slope = ls_data.get("top_pos", {}).get("trend_slope", 0.0)
    top_acct_slope= ls_data.get("top_acct", {}).get("trend_slope", 0.0)

    fund_rate = _safe(fund_data.get("funding_rate", 0)) * 10000  # convert to bps
    fund_ann  = _safe(fund_data.get("funding_rate", 0)) * 1095 * 100

    # Funding momentum from history
    fund_momentum = "flat"
    squeeze_risk  = "low"
    fund_z_score  = 0.0
    if len(fund_hist) >= 6:
        bps_vals  = [x["bps"] for x in fund_hist]
        avg_bps   = statistics.mean(bps_vals)
        std_bps   = statistics.stdev(bps_vals) if len(bps_vals) > 1 else 0.001
        fund_z_score = (fund_rate - avg_bps) / max(std_bps, 0.001)
        # 3-period slope
        recent_slope = float(np.polyfit(range(min(12, len(bps_vals))),
                                        bps_vals[-12:], 1)[0]) if len(bps_vals) >= 3 else 0
        if recent_slope > 0.2:   fund_momentum = "rising"
        elif recent_slope < -0.2: fund_momentum = "falling"
        # Squeeze risk: extreme funding + all on one side
        if abs(fund_rate) >= FUNDING_EXTREME and abs(fund_z_score) > 1.5:
            squeeze_risk = "HIGH" if fund_rate > 0 else "HIGH_SHORT_SQUEEZE"
        elif abs(fund_rate) >= FUNDING_HIGH:
            squeeze_risk = "MEDIUM"

    # Key divergence: top traders vs retail (global)
    # Top traders much more long than retail = smart money bullish
    # Top traders much more short than retail = smart money bearish
    smart_vs_retail_gap = top_pos_cur - global_cur
    if smart_vs_retail_gap > 0.5:
        divergence = "SMART_MONEY_MORE_LONG_THAN_RETAIL"
    elif smart_vs_retail_gap < -0.5:
        divergence = "SMART_MONEY_MORE_SHORT_THAN_RETAIL"
    else:
        divergence = "ALIGNED"

    # Build positioning summary
    if top_pos_cur >= 1.8 and top_acct_cur >= 1.5:
        position_bias = "INSTITUTIONS_HEAVILY_LONG"
        pos_score = 2
    elif top_pos_cur >= 1.3:
        position_bias = "INSTITUTIONS_LEANING_LONG"
        pos_score = 1
    elif top_pos_cur <= 0.6 and top_acct_cur <= 0.65:
        position_bias = "INSTITUTIONS_HEAVILY_SHORT"
        pos_score = -2
    elif top_pos_cur <= 0.77:
        position_bias = "INSTITUTIONS_LEANING_SHORT"
        pos_score = -1
    else:
        position_bias = "INSTITUTIONS_NEUTRAL"
        pos_score = 0

    # Check if top traders are ADDING to position (rising slope)
    if top_pos_slope > 0.02 and pos_score > 0:
        pos_score = min(2, pos_score + 1)
        position_bias += " + ADDING"
    elif top_pos_slope < -0.02 and pos_score < 0:
        pos_score = max(-2, pos_score - 1)
        position_bias += " + ADDING"

    return {
        "position_bias":         position_bias,
        "pos_score":             pos_score,
        "top_position_ls":       top_pos_cur,
        "top_account_ls":        top_acct_cur,
        "global_ls":             global_cur,
        "top_pos_slope":         top_pos_slope,
        "smart_vs_retail_gap":   smart_vs_retail_gap,
        "divergence":            divergence,
        "funding_bps":           fund_rate,
        "funding_ann_pct":       fund_ann,
        "funding_momentum":      fund_momentum,
        "funding_z_score":       fund_z_score,
        "squeeze_risk":          squeeze_risk,
    }


def analyze_liquidation_risk(liq_data: dict, spot: float) -> dict:
    """
    Given the liquidation cluster map:
    - Which cluster is nearest above/below spot?
    - What is the cascade probability?
    - Which side has more liq (tells you which way market makers want to push)
    """
    if not liq_data:
        return {}

    clusters = liq_data.get("clusters", [])
    long_liq  = liq_data.get("long_liq_usd", 0)
    short_liq = liq_data.get("short_liq_usd", 0)
    total_liq = long_liq + short_liq

    clusters_above = [c for c in clusters if c["level"] > spot]
    clusters_below = [c for c in clusters if c["level"] < spot]

    nearest_above = min(clusters_above, key=lambda x: x["level"]) if clusters_above else None
    nearest_below = max(clusters_below, key=lambda x: x["level"]) if clusters_below else None

    # Cascade probability: nearer + larger = higher risk
    def _cascade_prob(cluster, spot_price):
        if not cluster:
            return 0.0
        dist_pct = abs(cluster["level"] - spot_price) / spot_price * 100
        usd_m    = cluster["usd"] / 1e6
        # Simple model: within 1% = high risk; $5M+ = significant
        prob = max(0, 1 - dist_pct / 5) * min(1, usd_m / 5)
        return round(prob * 100, 1)

    cascade_up_prob   = _cascade_prob(nearest_above, spot)
    cascade_down_prob = _cascade_prob(nearest_below, spot)

    # Market maker incentive: they push toward larger liq cluster (collect fees)
    mm_push = "UP"   if short_liq > long_liq * 1.5 else \
              "DOWN" if long_liq  > short_liq * 1.5 else "UNCLEAR"

    return {
        "nearest_above":    nearest_above,
        "nearest_below":    nearest_below,
        "cascade_up_prob":  cascade_up_prob,
        "cascade_down_prob": cascade_down_prob,
        "long_liq_usd":     long_liq,
        "short_liq_usd":    short_liq,
        "total_liq_usd":    total_liq,
        "mm_push_direction": mm_push,
        "all_clusters":     clusters,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ARBITRAGE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_carry_arb(symbol: str, spot: float, perp: float,
                      fund_data: dict, fund_hist: list) -> dict:
    """
    Cash-and-Carry (Funding Harvest):
    Strategy: Long spot + Short perp → collect positive funding every 8h

    Entry yield = funding_rate × 1095 (3 times/day × 365)
    Costs: 2 × taker fee (0.04%) + borrow cost (~2%/yr for spot)
    Net yield = gross_yield - costs
    """
    fund_rate = _safe(fund_data.get("funding_rate", 0))
    fund_bps  = fund_rate * 10000
    fund_ann  = fund_rate * 1095 * 100  # annualized %

    # Cost model
    taker_fee   = 0.04        # % per leg
    round_trip  = taker_fee * 2  # enter + exit cost
    borrow_cost = 2.0         # % annual for spot borrow (if shorting via margin)

    # Positive funding = long spot + short perp strategy
    # Negative funding = short spot + long perp strategy
    if fund_rate > 0:
        gross_yield  = fund_ann          # we collect this by being short perp
        net_yield    = gross_yield - round_trip - 0  # no borrow cost on spot long
        strategy     = "LONG_SPOT_SHORT_PERP"
        direction    = "Collect positive funding"
    else:
        gross_yield  = abs(fund_ann)     # we collect this by being long perp
        net_yield    = gross_yield - round_trip - borrow_cost  # borrow cost for spot short
        strategy     = "SHORT_SPOT_LONG_PERP"
        direction    = "Collect negative funding"

    # Squeeze risk: if funding is extreme, strategy could get squeezed
    # (perp can move against you temporarily)
    squeeze_risk = "LOW"
    if abs(fund_bps) >= FUNDING_EXTREME:
        squeeze_risk = "HIGH — funding may flip, reducing expected yield"
    elif abs(fund_bps) >= FUNDING_HIGH:
        squeeze_risk = "MEDIUM"

    # Funding momentum: is yield rising or falling?
    momentum = "stable"
    if len(fund_hist) >= 6:
        recent = [x["bps"] for x in fund_hist[-6:]]
        old    = [x["bps"] for x in fund_hist[-12:-6]] if len(fund_hist) >= 12 else recent
        if statistics.mean(recent) > statistics.mean(old) * 1.2:
            momentum = "increasing" if fund_rate > 0 else "decreasing"
        elif statistics.mean(recent) < statistics.mean(old) * 0.8:
            momentum = "decreasing" if fund_rate > 0 else "increasing"

    return {
        "type":          "CARRY_ARB",
        "symbol":        symbol,
        "strategy":      strategy,
        "direction":     direction,
        "spot_price":    spot,
        "perp_price":    perp,
        "funding_bps":   fund_bps,
        "gross_yield_ann": round(gross_yield, 3),
        "net_yield_ann":   round(net_yield, 3),
        "round_trip_cost": round_trip,
        "squeeze_risk":  squeeze_risk,
        "momentum":      momentum,
        "viable":        net_yield >= ARB_MIN_CARRY_PCT,
        "confidence":    "HIGH" if abs(fund_bps) >= FUNDING_EXTREME else
                         "MEDIUM" if abs(fund_bps) >= FUNDING_HIGH else "LOW",
    }


def compute_basis_arb(symbol: str, spot: float, perp: float, fund_data: dict) -> dict:
    """
    Spot vs Perp Basis Arbitrage:
    When perp trades at a significant premium/discount to spot,
    the gap tends to close at the next funding period.

    Trade: buy cheap, sell expensive → collect the convergence.
    """
    if spot <= 0 or perp <= 0:
        return {}

    basis_pct  = (perp - spot) / spot * 100
    basis_bps  = basis_pct * 100

    # Direction
    if basis_bps > ARB_MIN_SPREAD_BPS:
        strategy = "LONG_SPOT_SHORT_PERP"
        note = f"Perp at {basis_pct:+.4f}% premium over spot → sell premium"
    elif basis_bps < -ARB_MIN_SPREAD_BPS:
        strategy = "LONG_PERP_SHORT_SPOT"
        note = f"Perp at {basis_pct:+.4f}% discount to spot → buy discount"
    else:
        return {
            "type": "BASIS_ARB", "symbol": symbol,
            "viable": False,
            "basis_bps": basis_bps,
            "note": f"Basis {basis_bps:.1f}bps — below minimum {ARB_MIN_SPREAD_BPS}bps threshold."
        }

    # Time to next funding (when gap closes)
    next_fund_ms = fund_data.get("next_funding", 0)
    now_ms = int(time.time() * 1000)
    hours_to_funding = (next_fund_ms - now_ms) / 3_600_000 if next_fund_ms > now_ms else 8.0

    # Expected yield: basis converges at funding settlement
    round_trip  = 0.04 * 2  # 2 × taker fee
    gross_yield = abs(basis_pct)
    net_yield   = gross_yield - round_trip

    # Annualize based on funding period
    ann_yield = net_yield / (hours_to_funding / 8760) if hours_to_funding > 0 else 0

    return {
        "type":               "BASIS_ARB",
        "symbol":             symbol,
        "strategy":           strategy,
        "note":               note,
        "spot_price":         spot,
        "perp_price":         perp,
        "basis_pct":          round(basis_pct, 6),
        "basis_bps":          round(basis_bps, 2),
        "hours_to_funding":   round(hours_to_funding, 2),
        "gross_yield_pct":    round(gross_yield, 4),
        "net_yield_pct":      round(net_yield, 4),
        "ann_yield_pct":      round(ann_yield, 2),
        "viable":             net_yield > 0.02 and abs(basis_bps) >= ARB_MIN_SPREAD_BPS,
        "confidence":         "HIGH" if abs(basis_bps) > 20 else
                              "MEDIUM" if abs(basis_bps) > 10 else "LOW",
    }


def compute_cross_exchange_arb(symbol: str, prices: dict) -> dict:
    """
    Cross-exchange price arbitrage.
    Buy on cheapest exchange, sell on most expensive.

    Real constraints: transfer time, withdrawal fees, slippage.
    This detects the SIGNAL, not a fully executable trade.
    """
    # Filter out zero/failed prices
    valid = {ex: p for ex, p in prices.items() if p > 0}
    if len(valid) < 2:
        return {"type": "CROSS_EXCHANGE_ARB", "symbol": symbol, "viable": False,
                "note": "Not enough exchanges responded."}

    min_ex = min(valid, key=valid.get)
    max_ex = max(valid, key=valid.get)
    min_p  = valid[min_ex]
    max_p  = valid[max_ex]

    spread_pct = (max_p - min_p) / min_p * 100
    spread_bps = spread_pct * 100

    # Cost model: 2 × taker fee each side = 4 × 0.02% = 8bps for perp-to-perp
    # For spot: 2 × 0.10% = 20bps (higher due to withdrawal fees)
    fee_bps = 8   # perp-to-perp (no withdrawal needed if staying in USDT)
    net_bps = spread_bps - fee_bps
    net_pct = net_bps / 100

    # Is this real or just latency?
    # Spreads < 2bps are likely latency artifacts
    # Spreads > 20bps are real opportunities
    likely_real = spread_bps > 15

    all_prices = sorted(valid.items(), key=lambda x: x[1])

    return {
        "type":        "CROSS_EXCHANGE_ARB",
        "symbol":      symbol,
        "prices":      valid,
        "cheapest":    {"exchange": min_ex, "price": min_p},
        "dearest":     {"exchange": max_ex, "price": max_p},
        "spread_bps":  round(spread_bps, 2),
        "spread_pct":  round(spread_pct, 6),
        "fee_bps":     fee_bps,
        "net_bps":     round(net_bps, 2),
        "net_pct":     round(net_pct, 6),
        "strategy":    f"BUY {min_ex} @ {min_p:.4f} | SELL {max_ex} @ {max_p:.4f}",
        "viable":      net_bps >= ARB_MIN_SPREAD_BPS and likely_real,
        "likely_real": likely_real,
        "confidence":  "HIGH" if spread_bps > 20 else
                       "MEDIUM" if spread_bps > 10 else "LOW",
        "all_prices":  all_prices,
    }


def fetch_cross_pair_price(base: str, quote: str) -> float:
    """Fetch direct cross pair price from Binance spot, e.g. ETHBTC, BNBBTC."""
    sym = f"{base}{quote}"
    d = _get(f"{BINANCE_SPOT}/api/v3/ticker/price", {"symbol": sym})
    if isinstance(d, dict):
        return _safe(d.get("price", 0))
    return 0.0


def compute_triangular_arb(prices_map: dict) -> List[dict]:
    """
    Real triangular arbitrage using DIRECT cross pairs (ETHBTC, BNBBTC, etc.)
    not implied USDT rates.

    Real tri-arb happens when:
      ETHBTC (direct pair) ≠ ETHUSDT / BTCUSDT (implied rate)

    Loop USDT → BTC → ETH → USDT:
      1. Buy BTC with USDT:  get (1/BTC_price) BTC
      2. Buy ETH with BTC:   get (1/ETHBTC_direct) ETH
      3. Sell ETH for USDT:  get ETH_price_USD USDT
      Profit if: (1/BTC) × (1/ETHBTC_direct) × ETH_USD > 1 (minus fees)
    """
    btc = prices_map.get("BTCUSDT_spot", 0)
    eth = prices_map.get("ETHUSDT_spot", 0)
    bnb = prices_map.get("BNBUSDT_spot", 0)
    sol = prices_map.get("SOLUSDT_spot", 0)
    xrp = prices_map.get("XRPUSDT_spot", 0)

    if btc <= 0 or eth <= 0:
        return []

    taker_fee = 0.001  # 0.1% per trade Binance BNB discount tier

    # Fetch real direct cross pairs
    ethbtc_direct  = fetch_cross_pair_price("ETH", "BTC")
    bnbbtc_direct  = fetch_cross_pair_price("BNB", "BTC")
    bnbeth_direct  = fetch_cross_pair_price("BNB", "ETH")
    xrpbtc_direct  = fetch_cross_pair_price("XRP", "BTC")

    opportunities = []

    def _tri(usdt_in: float, label: str, steps: list) -> None:
        """
        steps = list of (from_qty_fn, rate) tuples describing the loop.
        More explicit: start with 1 USDT and simulate the triangle.
        """
        pass  # Replaced by direct computation below

    def _check(label: str, start_usdt: float, end_usdt_fn) -> None:
        end = end_usdt_fn(1.0)
        profit_pct = (end - 1.0) * 100
        if abs(profit_pct) < 50:  # Sanity check — >50% impossible w/o bug
            opportunities.append({
                "type":       "TRIANGULAR_ARB",
                "loop":       label,
                "profit_pct": round(profit_pct, 6),
                "viable":     profit_pct > TRI_ARB_MIN_PCT,
                "note":       f"$1 USDT → ${'%.8f' % end} USDT = {profit_pct:+.5f}%",
            })

    fee = 1 - taker_fee

    # ─── USDT → BTC → ETH → USDT ───────────────────────────────────────
    # Step1: 1 USDT → BTC:  qty = 1/btc_price
    # Step2: BTC → ETH via ETHBTC direct pair: qty = btc_qty / ethbtc_direct
    # Step3: ETH → USDT: qty = eth_qty × eth_usdt_price
    if ethbtc_direct > 0:
        def fwd_btc_eth(u):
            btc_qty = u / btc * fee
            eth_qty = btc_qty / ethbtc_direct * fee
            return eth_qty * eth * fee
        _check("USDT→BTC→ETH→USDT (direct ETHBTC)", 1.0, fwd_btc_eth)

        # Reverse: USDT → ETH → BTC → USDT
        def rev_eth_btc(u):
            eth_qty = u / eth * fee
            btc_qty = eth_qty * ethbtc_direct * fee
            return btc_qty * btc * fee
        _check("USDT→ETH→BTC→USDT (direct ETHBTC)", 1.0, rev_eth_btc)

    # ─── USDT → BTC → BNB → USDT ───────────────────────────────────────
    if bnbbtc_direct > 0 and bnb > 0:
        def fwd_btc_bnb(u):
            btc_qty = u / btc * fee
            bnb_qty = btc_qty / bnbbtc_direct * fee
            return bnb_qty * bnb * fee
        _check("USDT→BTC→BNB→USDT (direct BNBBTC)", 1.0, fwd_btc_bnb)

        def rev_bnb_btc(u):
            bnb_qty = u / bnb * fee
            btc_qty = bnb_qty * bnbbtc_direct * fee
            return btc_qty * btc * fee
        _check("USDT→BNB→BTC→USDT (direct BNBBTC)", 1.0, rev_bnb_btc)

    # ─── USDT → ETH → BNB → USDT ───────────────────────────────────────
    if bnbeth_direct > 0 and bnb > 0:
        def fwd_eth_bnb(u):
            eth_qty = u / eth * fee
            bnb_qty = eth_qty / bnbeth_direct * fee
            return bnb_qty * bnb * fee
        _check("USDT→ETH→BNB→USDT (direct BNBETH)", 1.0, fwd_eth_bnb)

        def rev_bnb_eth(u):
            bnb_qty = u / bnb * fee
            eth_qty = bnb_qty * bnbeth_direct * fee
            return eth_qty * eth * fee
        _check("USDT→BNB→ETH→USDT (direct BNBETH)", 1.0, rev_bnb_eth)

    # ─── USDT → BTC → XRP → USDT ───────────────────────────────────────
    if xrpbtc_direct > 0 and xrp > 0:
        def fwd_btc_xrp(u):
            btc_qty = u / btc * fee
            xrp_qty = btc_qty / xrpbtc_direct * fee
            return xrp_qty * xrp * fee
        _check("USDT→BTC→XRP→USDT (direct XRPBTC)", 1.0, fwd_btc_xrp)

        def rev_xrp_btc(u):
            xrp_qty = u / xrp * fee
            btc_qty = xrp_qty * xrpbtc_direct * fee
            return btc_qty * btc * fee
        _check("USDT→XRP→BTC→USDT (direct XRPBTC)", 1.0, rev_xrp_btc)

    return sorted(opportunities, key=lambda x: -x["profit_pct"])


# ─────────────────────────────────────────────────────────────────────────────
# INSTITUTIONAL VERDICT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_institutional_verdict(
        symbol: str, spot: float,
        flow: dict, ob: dict, oi_flow: dict,
        positioning: dict, liq_risk: dict,
) -> dict:
    """
    Synthesize all institutional flow signals into a final verdict.

    Returns:
      - what_is_happening: plain English description
      - phase: ACCUMULATING / DISTRIBUTING / SQUEEZING_SHORTS / SQUEEZING_LONGS / NEUTRAL
      - net_flow_direction: BUY / SELL / MIXED
      - conviction: HIGH / MEDIUM / LOW
      - next_move_signal: SWEEP_UP / SWEEP_DOWN / RANGE / UNCLEAR
      - action_signal: LONG / SHORT / WAIT / STAND_ASIDE
    """
    # Gather sub-scores
    oi_score   = oi_flow.get("score", 0)
    pos_score  = positioning.get("pos_score", 0)
    ob_bias    = ob.get("ob_bias", "BALANCED")

    # Order book contribution
    ob_score = {
        "STRONG_BID": 2, "BID_HEAVY": 1,
        "BALANCED": 0,
        "ASK_HEAVY": -1, "STRONG_ASK": -2,
    }.get(ob_bias, 0)

    # Flow contribution
    cum_delta   = _safe(flow.get("cum_delta", 0))
    net_inst    = _safe(flow.get("net_inst_usd", 0))
    total_vol   = _safe(flow.get("buy_usd", 0)) + _safe(flow.get("sell_usd", 0))
    inst_pct    = abs(net_inst) / max(total_vol, 1) * 100  # institutional % of flow
    vel_accel   = _safe(flow.get("velocity_accel", 1.0), 1.0)

    flow_score = 0
    if net_inst > 0:
        flow_score = 2 if inst_pct > 5 else 1
    elif net_inst < 0:
        flow_score = -2 if inst_pct > 5 else -1

    # Smart money divergence bonus — smart buying while retail sells = strong institutional signal
    srd = flow.get("smart_retail_div", "ALIGNED")
    if srd == "SMART_MONEY_BUYING_RETAIL_SELLING":
        flow_score += 1
    elif srd == "SMART_MONEY_SELLING_RETAIL_BUYING":
        flow_score -= 1

    # Delta exhaustion penalty — buyers running out of steam
    if flow.get("delta_exhaustion", False):
        flow_score = int(flow_score * 0.5)  # halve the score when exhausted

    # Combined score
    total_score = oi_score * 0.35 + pos_score * 0.35 + ob_score * 0.15 + flow_score * 0.15

    # Phase determination
    mm_push = liq_risk.get("mm_push_direction", "UNCLEAR")
    squeeze = positioning.get("squeeze_risk", "low")
    oi_type = oi_flow.get("flow_type", "")

    if "SQUEEZE" in oi_type or squeeze not in ("LOW", "low") and "SQUEEZE" in squeeze.upper():
        phase = "SQUEEZING_SHORTS" if total_score > 0 else "SQUEEZING_LONGS"
    elif total_score >= 1.0:
        phase = "ACCUMULATING"
    elif total_score <= -1.0:
        phase = "DISTRIBUTING"
    elif abs(total_score) < 0.3:
        phase = "NEUTRAL"
    else:
        phase = "TRANSITIONING"

    # Net flow direction
    if total_score > 0.3:
        net_direction = "BUY"
    elif total_score < -0.3:
        net_direction = "SELL"
    else:
        net_direction = "MIXED"

    # Conviction
    signal_agreement = sum([
        1 if oi_score > 0 else (-1 if oi_score < 0 else 0),
        1 if pos_score > 0 else (-1 if pos_score < 0 else 0),
        1 if ob_score > 0 else (-1 if ob_score < 0 else 0),
        1 if flow_score > 0 else (-1 if flow_score < 0 else 0),
    ])
    abs_agreement = abs(signal_agreement)
    if abs_agreement >= 3:
        conviction = "HIGH"
    elif abs_agreement >= 2:
        conviction = "MEDIUM"
    else:
        conviction = "LOW"

    # Next move signal
    if mm_push == "UP" and total_score > 0:
        next_move = "SWEEP_UP"
    elif mm_push == "DOWN" and total_score < 0:
        next_move = "SWEEP_DOWN"
    elif total_score > 0.5 and conviction in ("HIGH", "MEDIUM"):
        next_move = "LIKELY_UP"
    elif total_score < -0.5 and conviction in ("HIGH", "MEDIUM"):
        next_move = "LIKELY_DOWN"
    else:
        next_move = "RANGE_BOUND"

    # Action signal
    if conviction in ("HIGH", "MEDIUM") and total_score >= 1.0 and next_move in ("SWEEP_UP", "LIKELY_UP"):
        action = "LONG"
        kl     = liq_risk.get("nearest_above")
        kl_str = f"target liq sweep @ {kl['level']:.4f} (${kl['usd']/1e6:.1f}M)" if kl else "watch resistance"
    elif conviction in ("HIGH", "MEDIUM") and total_score <= -1.0 and next_move in ("SWEEP_DOWN", "LIKELY_DOWN"):
        action = "SHORT"
        kl     = liq_risk.get("nearest_below")
        kl_str = f"target liq sweep @ {kl['level']:.4f} (${kl['usd']/1e6:.1f}M)" if kl else "watch support"
    elif conviction == "HIGH" and abs(total_score) < 0.3:
        action = "WAIT"
        kl_str = "wait for breakout from current range"
    else:
        action = "STAND_ASIDE"
        kl_str = "signals conflicted"

    # Build human-readable explanation
    inst_net_str = f"${abs(net_inst)/1e6:.2f}M {'BUY' if net_inst > 0 else 'SELL'}"
    pos_str      = positioning.get("position_bias", "UNKNOWN")
    fund_str     = f"{positioning.get('funding_bps', 0):+.2f}bps"

    what_happening = (
        f"Phase: {phase}. "
        f"Institutions are {pos_str} (top trader L/S={positioning.get('top_position_ls', 1.0):.2f}). "
        f"Net institutional flow: {inst_net_str} ({inst_pct:.1f}% of total volume). "
        f"Funding: {fund_str}/8h ({positioning.get('funding_momentum', 'stable')} trend). "
        f"OI: {oi_flow.get('flow_type', 'unknown')} ({oi_flow.get('oi_velocity_1h', 0):+.2f}%/hr). "
        f"MM incentive: push {mm_push}. "
        f"Squeeze risk: {squeeze}."
    )

    return {
        "phase":               phase,
        "net_direction":       net_direction,
        "conviction":          conviction,
        "next_move":           next_move,
        "action":              action,
        "action_detail":       kl_str,
        "total_score":         round(total_score, 3),
        "sub_scores": {
            "oi_flow":     oi_score,
            "positioning": pos_score,
            "order_book":  ob_score,
            "trade_flow":  flow_score,
        },
        "what_is_happening":   what_happening,
        "iceberg_count":       len(flow.get("iceberg_levels", [])),
        "block_trades":        len(flow.get("block_trades", [])),
        "whale_trades":        len(flow.get("whale_trades", [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL SYMBOL SCAN
# ─────────────────────────────────────────────────────────────────────────────

def scan_symbol_deep(symbol: str, all_spot_prices: dict = None) -> dict:
    """Full institutional flow scan for one symbol."""
    t0 = time.time()
    print(f"  [{symbol}] Fetching flows...", end=" ", flush=True)

    # Parallel-ish fetches
    spot_p     = fetch_spot_price(symbol)
    perp_p     = fetch_perp_price(symbol)
    prem       = fetch_premium_index(symbol)
    fund_hist  = fetch_funding_history(symbol, limit=48)
    oi_data    = fetch_open_interest_full(symbol)
    ls_raw     = fetch_long_short_ratios(symbol)
    liq_raw    = fetch_liquidations(symbol, limit=500)
    trades     = fetch_agg_trades(symbol, limit=1000)
    ob_raw     = fetch_order_book(symbol, depth=500)

    # Cross-exchange prices
    okx_p  = fetch_okx_price(symbol)
    byb_p  = fetch_bybit_price(symbol)

    spot = spot_p if spot_p > 0 else _safe(prem.get("index_price", 0))
    perp = perp_p if perp_p > 0 else _safe(prem.get("mark_price", 0))

    print(f"done ({time.time()-t0:.1f}s)", flush=True)

    # ── Analysis ──────────────────────────────────────────────────────────
    flow        = analyze_trade_flow(trades, spot)
    ob          = analyze_order_book(ob_raw, spot)
    oi_flow     = analyze_oi_flow(oi_data, spot, flow)
    ls_data     = {
        "global":   ls_raw.get("global", {}),
        "top_acct": ls_raw.get("top_acct", {}),
        "top_pos":  ls_raw.get("top_pos", {}),
    }
    positioning = analyze_positioning(ls_data, prem, fund_hist)
    liq_risk    = analyze_liquidation_risk(liq_raw, spot)
    verdict     = build_institutional_verdict(
                    symbol, spot, flow, ob, oi_flow, positioning, liq_risk)

    # ── Arbitrage ─────────────────────────────────────────────────────────
    carry_arb  = compute_carry_arb(symbol, spot, perp, prem, fund_hist)
    basis_arb  = compute_basis_arb(symbol, spot, perp, prem)
    cross_arb  = compute_cross_exchange_arb(symbol, {
        "Binance": perp,
        "OKX":     okx_p,
        "Bybit":   byb_p,
    })

    return {
        "symbol":      symbol,
        "spot":        spot,
        "perp":        perp,
        "scan_ts":     datetime.now(timezone.utc).isoformat(),
        "flow":        flow,
        "ob":          ob,
        "oi_flow":     oi_flow,
        "positioning": positioning,
        "liq_risk":    liq_risk,
        "verdict":     verdict,
        "arb": {
            "carry":        carry_arb,
            "basis":        basis_arb,
            "cross_exchange": cross_arb,
        },
        "prices": {
            "binance_spot": spot_p,
            "binance_perp": perp,
            "okx_perp":    okx_p,
            "bybit_perp":  byb_p,
        },
        "fund_hist":   fund_hist[-6:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

def _c(score, text):
    if score > 0:   return f"\033[92m{text}\033[0m"
    elif score < 0: return f"\033[91m{text}\033[0m"
    return f"\033[37m{text}\033[0m"
def _b(t):   return f"\033[1m{t}\033[0m"
def _y(t):   return f"\033[93m{t}\033[0m"
def _dim(t): return f"\033[2m{t}\033[0m"

PHASE_ICONS = {
    "ACCUMULATING":     "\033[92m◆ ACCUM\033[0m",
    "DISTRIBUTING":     "\033[91m◇ DISTRIB\033[0m",
    "SQUEEZING_SHORTS": "\033[92m▲ SQ-SHORT\033[0m",
    "SQUEEZING_LONGS":  "\033[91m▼ SQ-LONG\033[0m",
    "NEUTRAL":          "\033[37m─ NEUTRAL\033[0m",
    "TRANSITIONING":    "\033[93m~ TRANSIT\033[0m",
}

ACTION_ICONS = {
    "LONG":        "\033[92m[▲ LONG]\033[0m",
    "SHORT":       "\033[91m[▼ SHORT]\033[0m",
    "WAIT":        "\033[93m[◈ WAIT]\033[0m",
    "STAND_ASIDE": "\033[37m[─ ASIDE]\033[0m",
}


def print_institutional_report(results: list, arb_only: bool = False):
    W   = 120
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "═" * W)
    print(f"  INSTITUTIONAL FLOW + ARBITRAGE DETECTOR  —  {now}")
    print(f"  {'Pure Flow: CVD · OI · Funding · Order Book · Liq Map · Cross-Exchange Arb':^{W-4}}")
    print("═" * W)

    valid = [r for r in results if "verdict" in r]
    ranked = sorted(valid, key=lambda x: abs(x["verdict"]["total_score"]), reverse=True)

    if not arb_only:
        # ── Overview Table ─────────────────────────────────────────────────
        print(f"\n  {'Symbol':<12} {'Price':>12} {'Phase':>12} {'Action':>12} "
              f"{'Score':>7} {'Conv':>6} {'OI%/hr':>8} {'Fund':>8} {'TopLS':>7} {'InstFlow':>12}")
        print("─" * W)

        for r in ranked:
            v     = r["verdict"]
            pos   = r["positioning"]
            oi    = r["oi_flow"]
            sym   = r["symbol"]
            spot  = r["spot"]
            sc    = v["total_score"]
            conv  = v["conviction"]
            phase = PHASE_ICONS.get(v["phase"], v["phase"])
            act   = ACTION_ICONS.get(v["action"], v["action"])
            fund  = pos.get("funding_bps", 0)
            tls   = pos.get("top_position_ls", 1.0)
            oivel = oi.get("oi_velocity_1h", 0)
            inst  = r["flow"].get("net_inst_usd", 0)
            inst_str = f"{inst/1e6:>+.2f}M"
            inst_c = "\033[92m" if inst > 0 else ("\033[91m" if inst < 0 else "\033[37m")

            print(f"  {sym:<12} ${spot:>11.4f} {phase:>12}  "
                  f"{act:>12}  {_c(sc, f'{sc:+.3f}'):>7}  "
                  f"{conv:>6}  {oivel:>+6.2f}%  {fund:>+6.2f}  {tls:>6.3f}  "
                  f"{inst_c}{inst_str}\033[0m")

        # ── Deep Breakdown for top 5 ──────────────────────────────────────
        print("\n" + "─" * W)
        print("  INSTITUTIONAL FLOW DETAIL")
        print("─" * W)

        for r in ranked[:5]:
            v    = r["verdict"]
            pos  = r["positioning"]
            oi   = r["oi_flow"]
            flow = r["flow"]
            ob   = r["ob"]
            liq  = r["liq_risk"]
            sym  = r["symbol"]
            spot = r["spot"]

            phase_str = PHASE_ICONS.get(v["phase"], v["phase"])
            act_str   = ACTION_ICONS.get(v["action"], v["action"])

            print(f"\n  ┌─── {_b(sym)}  ${spot:.4f}  {phase_str}  {act_str}  "
                  f"Conviction={_b(v['conviction'])} ───")
            print(f"  │  {v['what_is_happening'][:W-6]}")
            if len(v["what_is_happening"]) > W - 6:
                print(f"  │  ...{v['what_is_happening'][W-6:2*(W-6)]}")

            # Action detail
            print(f"  │  → {v['action_detail']}")
            print(f"  │")

            # Order flow
            cum = flow.get("cum_delta", 0)
            buy = flow.get("buy_usd", 0)
            sel = flow.get("sell_usd", 0)
            cvdr = flow.get("cvd_ratio", 1.0)
            vel  = flow.get("velocity_accel", 1.0)
            nb   = len(flow.get("block_trades", []))
            nw   = len(flow.get("whale_trades", []))
            ni   = len(flow.get("iceberg_levels", []))
            inst = flow.get("net_inst_usd", 0)
            inst_c = "\033[92m" if inst > 0 else "\033[91m"

            smart_cvd  = flow.get("smart_cvd", 0)
            retail_cvd = flow.get("retail_cvd", 0)
            srd        = flow.get("smart_retail_div", "ALIGNED")
            exhausted  = flow.get("delta_exhaustion", False)
            vel_str    = "↑ accel" if vel > 1.3 else ("↓ decel" if vel < 0.7 else "→ steady")
            if exhausted:
                vel_str += " ⚠EXHAUSTED"
            print(f"  │  ORDER FLOW:  CVD={_c(cum, f'{cum/1e6:+.2f}M')}  "
                  f"Buy/Sell={cvdr:.3f}x  Velocity={vel_str}  "
                  f"Net Inst={inst_c}{inst/1e6:+.2f}M\033[0m  "
                  f"Blocks={nb}  Whales={nw}  Icebergs={ni}")

            # Smart money vs retail divergence
            if srd != "ALIGNED" or abs(smart_cvd) > 1000:
                srd_c = "\033[92m" if "BUYING" in srd else ("\033[91m" if "SELLING" in srd else "\033[37m")
                print(f"  │    SmartCVD={_c(smart_cvd, f'{smart_cvd/1e6:+.2f}M')}  "
                      f"RetailCVD={_c(retail_cvd, f'{retail_cvd/1e6:+.2f}M')}  "
                      f"→ {srd_c}{srd}\033[0m")

            # Iceberg levels
            for ice in flow.get("iceberg_levels", [])[:3]:
                side_c = "\033[92m" if ice["side"] == "BUY" else "\033[91m"
                hit_str = f"≥{ice['hit_count']}" if ice['hit_count'] >= 1000 else str(ice['hit_count'])
                print(f"  │    Iceberg {side_c}{ice['side']:4s}\033[0m  "
                      f"${ice['total_usd']/1e6:.2f}M @ {ice['price']:.4f}  "
                      f"({hit_str} hits — hidden order detected)")

            # Block / whale trades
            for t in (flow.get("block_trades", []) + flow.get("whale_trades", []))[:3]:
                side_c = "\033[92m" if not t["is_maker"] else "\033[91m"
                label  = "BLOCK" if t["usd"] >= BLOCK_TRADE_USD else "WHALE"
                ts_str = datetime.fromtimestamp(t["time"] / 1000, tz=timezone.utc).strftime("%H:%M:%S")
                print(f"  │    {label} {side_c}{'BUY' if not t['is_maker'] else 'SELL':4s}\033[0m  "
                      f"${t['usd']/1e6:.2f}M  @ {t['price']:.4f}  UTC {ts_str}")

            # Session flow
            sess = flow.get("session_delta", {})
            if any(abs(v2) > 0 for v2 in sess.values()):
                parts = []
                for sname, sval in sess.items():
                    if abs(sval) > 0:
                        parts.append(f"{sname}={_c(sval, f'{sval/1e6:+.2f}M')}")
                print(f"  │  SESSION FLOW:  {' | '.join(parts)}")

            # Order book
            ob_walls_b = ob.get("bid_walls", [])
            ob_walls_a = ob.get("ask_walls", [])
            d05 = ob.get("depth_buckets", {}).get("0.5pct", {})
            bid_d = d05.get("bid_usd", 0); ask_d = d05.get("ask_usd", 0)
            imb   = d05.get("imbalance", 1.0)
            spread = ob.get("spread_bps", 0)
            print(f"  │  ORDER BOOK:  Spread={spread:.1f}bps  "
                  f"Depth±0.5%: Bid=${bid_d/1e6:.1f}M vs Ask=${ask_d/1e6:.1f}M "
                  f"(imb={_c(imb-1, f'{imb:.2f}x')})  "
                  f"Bid walls={len(ob_walls_b)}  Ask walls={len(ob_walls_a)}")

            # Key walls
            if ob_walls_b:
                p, q, usd = ob_walls_b[0]
                print(f"  │    Largest BID WALL: ${usd/1e6:.2f}M @ {p:.4f}  "
                      f"({abs(p-spot)/spot*100:.2f}% below spot) ← institutional support")
            if ob_walls_a:
                p, q, usd = ob_walls_a[0]
                print(f"  │    Largest ASK WALL: ${usd/1e6:.2f}M @ {p:.4f}  "
                      f"({abs(p-spot)/spot*100:.2f}% above spot) ← institutional resistance")

            # Vacuums
            for vac in ob.get("ask_vacuums", [])[:1]:
                print(f"  │    ASK VACUUM: {vac['from']:.4f}→{vac['to']:.4f} "
                      f"({vac['gap_pct']:.3f}% gap) ← thin air above, fast move if breached")
            for vac in ob.get("bid_vacuums", [])[:1]:
                print(f"  │    BID VACUUM: {vac['from']:.4f}→{vac['to']:.4f} "
                      f"({vac['gap_pct']:.3f}% gap) ← thin air below, fast move if broken")

            # OI + positioning
            print(f"  │  OI FLOW:  {oi['flow_type']:30s}  "
                  f"velocity={oi['oi_velocity_1h']:+.2f}%/hr  "
                  f"24h={oi['oi_24h_pct']:+.1f}%  trend={oi['oi_trend']}")

            fund_bps  = pos.get("funding_bps", 0)
            fund_ann  = pos.get("funding_ann_pct", 0)
            fund_mom  = pos.get("funding_momentum", "stable")
            fund_z    = pos.get("funding_z_score", 0)
            sq_risk   = pos.get("squeeze_risk", "low")
            div       = pos.get("divergence", "ALIGNED")
            tls_val = pos.get('top_position_ls', 1.0)
            tls_str = _c(tls_val - 1, f"{tls_val:.3f}")
            print(f"  │  POSITIONING:  TopTrader L/S={tls_str}  "
                  f"Fund={_c(fund_bps, f'{fund_bps:+.2f}bps')} ({fund_ann:+.1f}%/yr, {fund_mom})  "
                  f"Z={fund_z:+.1f}σ  Squeeze={sq_risk}")
            if div != "ALIGNED":
                print(f"  │  ⚡ SMART/RETAIL DIVERGENCE: {div}")

            # Liquidation map
            na = liq.get("nearest_above")
            nb2 = liq.get("nearest_below")
            mm  = liq.get("mm_push_direction", "?")
            ll  = liq.get("long_liq_usd", 0)
            sl  = liq.get("short_liq_usd", 0)
            cu  = liq.get("cascade_up_prob", 0)
            cd  = liq.get("cascade_down_prob", 0)
            print(f"  │  LIQ MAP:  Long liq=${ll/1e6:.1f}M  Short liq=${sl/1e6:.1f}M  "
                  f"MM push={_c(1 if mm=='UP' else -1, mm)}  "
                  f"Cascade ▲={cu:.0f}%  ▼={cd:.0f}%")
            if na:
                dist = (na['level'] - spot) / spot * 100
                print(f"  │    NEXT LIQ ▲: ${na['usd']/1e6:.1f}M SHORT liq @ {na['level']:.4f}  "
                      f"(+{dist:.2f}% from spot) → price magnet if shorts get squeezed")
            if nb2:
                dist = (spot - nb2['level']) / spot * 100
                print(f"  │    NEXT LIQ ▼: ${nb2['usd']/1e6:.1f}M LONG liq @ {nb2['level']:.4f}  "
                      f"(-{dist:.2f}% from spot) → price magnet if longs get liquidated")

            print(f"  └{'─' * (W - 4)}")

    # ══════════════════════════════════════════════════════════════════════
    # ARBITRAGE SECTION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * W)
    print("  ARBITRAGE OPPORTUNITIES")
    print("═" * W)

    # 1) CARRY ARB (best yield first)
    print(f"\n  1. FUNDING CARRY ARBITRAGE  (Long/Short spot + hedge perp)")
    print(f"  {'Symbol':<12} {'Strategy':<25} {'Funding/8h':>12} {'Gross Ann':>10} "
          f"{'Net Ann':>10} {'Momentum':>12} {'Squeeze':>10} {'OK':>5}")
    print(f"  {'─'*103}")

    carry_all = [r["arb"]["carry"] for r in valid if r["arb"].get("carry")]
    carry_all.sort(key=lambda x: -abs(x.get("net_yield_ann", 0)))

    for c in carry_all:
        sym   = c["symbol"]
        strat = "SHORT_PERP" if c["strategy"] == "LONG_SPOT_SHORT_PERP" else "LONG_PERP"
        fbps  = c["funding_bps"]
        gross = c["gross_yield_ann"]
        net   = c["net_yield_ann"]
        mom   = c["momentum"]
        sq    = c["squeeze_risk"].split(" ")[0]
        ok    = "\033[92m YES\033[0m" if c["viable"] else "\033[91m  NO\033[0m"
        net_c = "\033[92m" if net > 0 else "\033[91m"
        print(f"  {sym:<12} {strat:<25} {_c(fbps, f'{fbps:>+10.3f}bps')}  "
              f"{gross:>+9.2f}%  {net_c}{net:>+9.2f}%\033[0m  {mom:>12}  {sq:>10}  {ok}")

    # 2) BASIS ARB
    print(f"\n\n  2. SPOT-PERP BASIS ARBITRAGE  (Perp premium/discount convergence)")
    print(f"  {'Symbol':<12} {'Basis':>10} {'Strategy':<30} {'Gross':>8} {'Net':>8} "
          f"{'Hours to settle':>16} {'OK':>5}")
    print(f"  {'─'*95}")

    basis_all = [r["arb"]["basis"] for r in valid
                 if r["arb"].get("basis") and r["arb"]["basis"].get("basis_bps", 0) != 0]
    basis_all.sort(key=lambda x: -abs(x.get("basis_bps", 0)))

    for b in basis_all:
        sym   = b["symbol"]
        bps   = b.get("basis_bps", 0)
        strat = b.get("strategy", "N/A")[:29]
        gross = b.get("gross_yield_pct", 0)
        net   = b.get("net_yield_pct", 0)
        h2f   = b.get("hours_to_funding", 0)
        ok    = "\033[92m YES\033[0m" if b.get("viable") else "\033[91m  NO\033[0m"
        print(f"  {sym:<12} {_c(bps, f'{bps:>+8.2f}bps')}  {strat:<30} "
              f"{gross:>+7.4f}%  {net:>+7.4f}%  {h2f:>14.2f}h  {ok}")

    # 3) CROSS-EXCHANGE ARB
    print(f"\n\n  3. CROSS-EXCHANGE ARBITRAGE  (Binance / OKX / Bybit price diff)")
    print(f"  {'Symbol':<12} {'Cheap Buy':<22} {'Expensive Sell':<22} "
          f"{'Spread':>8} {'Net':>8} {'OK':>5}")
    print(f"  {'─'*85}")

    cross_all = [r["arb"]["cross_exchange"] for r in valid
                 if r["arb"].get("cross_exchange") and r["arb"]["cross_exchange"].get("spread_bps", 0) > 0]
    cross_all.sort(key=lambda x: -x.get("spread_bps", 0))

    for cx in cross_all:
        sym    = cx["symbol"]
        cheap  = cx.get("cheapest", {})
        dear   = cx.get("dearest", {})
        spread = cx.get("spread_bps", 0)
        net    = cx.get("net_bps", 0)
        ok     = "\033[92m YES\033[0m" if cx.get("viable") else "\033[91m  NO\033[0m"
        cheap_str = f"{cheap.get('exchange','?')} @ {cheap.get('price',0):.4f}"
        dear_str  = f"{dear.get('exchange','?')} @ {dear.get('price',0):.4f}"
        print(f"  {sym:<12} {cheap_str:<22} {dear_str:<22} "
              f"{_c(spread, f'{spread:>+7.2f}bps')}  {_c(net, f'{net:>+6.2f}'):>8}  {ok}")

        # Show all prices for context
        if cx.get("all_prices"):
            price_line = "  " + " | ".join(
                f"{ex}={p:.4f}" for ex, p in cx["all_prices"]
            )
            print(f"  {'':12}  {_dim(price_line[2:])}")

    # 4) TRIANGULAR ARB
    if any(r.get("tri_arb") for r in results):
        print(f"\n\n  4. TRIANGULAR ARBITRAGE  (3-asset loop on Binance)")
        print(f"  {'Loop':<35} {'Profit%':>10}  {'Viable':>8}")
        print(f"  {'─'*60}")
        # deduplicate — tri_arb is the same list attached to every result
        seen_loops = set()
        all_tri = []
        for r in results:
            for t in r.get("tri_arb", []):
                if t["loop"] not in seen_loops:
                    seen_loops.add(t["loop"])
                    all_tri.append(t)
        all_tri.sort(key=lambda x: -x["profit_pct"])
        for t in all_tri[:10]:
            ok = "\033[92m YES\033[0m" if t["viable"] else "\033[37m  marginal\033[0m"
            tp = t["profit_pct"]; tp_str = _c(tp, f"{tp:>+9.4f}%")
            print(f"  {t['loop']:<35} {tp_str}  {ok}")
        if not all_tri:
            print(f"  No profitable triangular arb detected at current prices.")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("  INSTITUTIONAL FLOW SUMMARY")
    print("─" * W)

    action_long  = [r for r in ranked if r["verdict"]["action"] == "LONG"]
    action_short = [r for r in ranked if r["verdict"]["action"] == "SHORT"]
    action_wait  = [r for r in ranked if r["verdict"]["action"] == "WAIT"]
    best_carry   = max(carry_all, key=lambda x: x.get("net_yield_ann", 0)) if carry_all else None
    best_cross   = max(cross_all, key=lambda x: x.get("net_bps", 0)) if cross_all else None

    if action_long:
        print(f"\n  \033[92m[LONG SIGNALS]\033[0m")
        for r in action_long:
            v = r["verdict"]; pos = r["positioning"]
            print(f"    ▲ {r['symbol']:<12} Conv={v['conviction']:6}  "
                  f"Phase={v['phase']:15}  Next={v['next_move']:12}  "
                  f"Detail: {v['action_detail']}")

    if action_short:
        print(f"\n  \033[91m[SHORT SIGNALS]\033[0m")
        for r in action_short:
            v = r["verdict"]; pos = r["positioning"]
            print(f"    ▼ {r['symbol']:<12} Conv={v['conviction']:6}  "
                  f"Phase={v['phase']:15}  Next={v['next_move']:12}  "
                  f"Detail: {v['action_detail']}")

    if action_wait:
        print(f"\n  \033[93m[WATCH — POSITION BUILDUP DETECTED]\033[0m")
        for r in action_wait:
            v = r["verdict"]
            print(f"    ◈ {r['symbol']:<12} OI velocity={r['oi_flow'].get('oi_velocity_1h',0):+.2f}%/hr  "
                  f"Detail: {v['action_detail']}")

    if not action_long and not action_short and not action_wait:
        print(f"\n  No high-conviction directional trades. Best opportunity:")
        if ranked:
            r = ranked[0]; v = r["verdict"]
            print(f"    {r['symbol']} — {v['phase']} (score={v['total_score']:+.3f}) {v['conviction']}")

    if best_carry and best_carry.get("viable"):
        print(f"\n  \033[96m[BEST CARRY ARB]\033[0m  "
              f"{best_carry['symbol']} → {best_carry['net_yield_ann']:+.2f}%/yr net  "
              f"({best_carry['strategy'].replace('_',' ')})  "
              f"Fund={best_carry['funding_bps']:+.2f}bps/8h")

    if best_cross and best_cross.get("viable"):
        print(f"  \033[96m[BEST CROSS-EXCHANGE ARB]\033[0m  "
              f"{best_cross['symbol']} → {best_cross['net_bps']:+.2f}bps net  "
              f"Buy {best_cross.get('cheapest',{}).get('exchange','?')}, "
              f"Sell {best_cross.get('dearest',{}).get('exchange','?')}")

    print("\n" + "═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# TRIANGULAR ARB SCAN (global — needs all spot prices together)
# ─────────────────────────────────────────────────────────────────────────────

def scan_triangular_arb(all_results: list) -> List[dict]:
    prices_map = {}
    for r in all_results:
        if r.get("spot", 0) > 0:
            prices_map[f"{r['symbol']}_spot"] = r["spot"]
    return compute_triangular_arb(prices_map)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Institutional Flow + Arbitrage Detector — Pure Flow, No Indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python crypto_institutional_flow.py                         -- full 8-symbol scan
  python crypto_institutional_flow.py --quick                 -- BTC + ETH only
  python crypto_institutional_flow.py --symbols BTCUSDT SOLUSDT
  python crypto_institutional_flow.py --arb-only              -- arb trades only
  python crypto_institutional_flow.py --json                  -- save results
        """
    )
    parser.add_argument("--symbols",  nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--quick",    action="store_true", help="BTC+ETH only")
    parser.add_argument("--arb-only", action="store_true", help="Show arb trades only")
    parser.add_argument("--json",     action="store_true", help="Save to JSON")
    parser.add_argument("--watch",    type=int, default=0, metavar="SEC",
                        help="Continuous mode: re-scan every N seconds (e.g. --watch 120)")
    args = parser.parse_args()

    if args.quick:
        args.symbols = ["BTCUSDT", "ETHUSDT"]

    print(f"\n  {'═'*62}")
    print(f"  INSTITUTIONAL FLOW + ARBITRAGE DETECTOR")
    print(f"  Scanning {len(args.symbols)} symbols")
    print(f"  Signals: CVD · Block Trades · Icebergs · OI Velocity · Funding")
    print(f"           Order Book Depth · Liq Cascade · Top-Trader L/S")
    print(f"  Arb:     Carry · Basis · Cross-Exchange · Triangular")
    print(f"  {'═'*62}\n")

    def _run_scan():
        results = []
        for sym in args.symbols:
            try:
                r = scan_symbol_deep(sym)
                results.append(r)
            except Exception as e:
                import traceback
                print(f"  ERROR on {sym}: {e}")
                traceback.print_exc()
                results.append({"symbol": sym, "error": str(e)})

        # Compute triangular arb across all symbols
        tri_arb = scan_triangular_arb(results)
        for r in results:
            r["tri_arb"] = tri_arb  # attach to all results so report shows once

        print_institutional_report(results, arb_only=args.arb_only)
        return results

    results = _run_scan()

    if args.watch > 0:
        print(f"  [WATCH MODE] Re-scanning every {args.watch}s. Ctrl+C to stop.\n")
        while True:
            try:
                time.sleep(args.watch)
                print(f"\n\033[2J\033[H")  # clear screen
                results = _run_scan()
            except KeyboardInterrupt:
                print("\n  Watch mode stopped.")
                break

    if args.json:
        def _jx(obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray):     return obj.tolist()
            raise TypeError(type(obj))

        # Remove heavy fields
        clean = []
        for r in results:
            rc = {k: v for k, v in r.items() if k not in ("tri_arb",)}
            rc.pop("fund_hist", None)
            clean.append(rc)
        with open("flow_scan_results.json", "w") as f:
            json.dump(clean, f, default=_jx, indent=2)
        print(f"  Results saved to flow_scan_results.json\n")


if __name__ == "__main__":
    main()
