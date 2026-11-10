#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crypto_institutional_scanner.py
================================
Comprehensive crypto institutional-position detector + quant scanner.

Ports the full India Trading Terminal strategy suite to crypto:
  - Hurst exponent (regime: trending vs mean-reverting)
  - MACD(12,26,9) histogram
  - ICT Order Blocks (last opposite-color candle before impulse)
  - BOS / CHoCH market structure (from fw_smart_money.py logic)
  - Wyckoff phase (Accumulation / Distribution / Markup / Markdown)
  - Supply/Demand zones (base-then-impulse)
  - IPDA liquidity pools (prev-day highs/lows + equal highs/lows)
  - Volume Profile (POC / VAH / VAL)
  - CVD from real Binance aggTrades (taker buy vs sell volume)
  - A/D line + OBV cross-check

PLUS crypto-specific institutional signals:
  - Open Interest divergence (OI vs price direction)
  - Funding rate extremes (squeeze risk)
  - Order book walls (large bid/ask clusters = institutional zones)
  - Top-trader long/short ratio (Binance global / top-trader)
  - Liquidation cluster detection
  - Whale aggTrade detection (single trades > $500k)

Usage:
    python crypto_institutional_scanner.py
    python crypto_institutional_scanner.py --symbols BTCUSDT ETHUSDT SOLUSDT
    python crypto_institutional_scanner.py --tf 1h --limit 150
"""

import argparse
import sys
import time
import math
import json
import statistics
from datetime import datetime, timezone
from typing import Optional

# -- Optional dependencies -- #
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
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_SPOT      = "https://api.binance.com"
BINANCE_FAPI      = "https://fapi.binance.com"   # USDT-M futures
BINANCE_DAPI      = "https://dapi.binance.com"   # COIN-M futures

DEFAULT_SYMBOLS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                     "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "ADAUSDT"]

DEFAULT_TF        = "1h"      # 1h gives good SMC structure
DEFAULT_LIMIT     = 150       # ~6 days of hourly bars

# Hurst thresholds
HURST_TREND_THRESH   = 0.55
HURST_MR_THRESH      = 0.45

# Institutional signal thresholds
FUNDING_EXTREME_BPS  = 5.0    # >5bps per 8h = extreme (may squeeze)
FUNDING_HIGH_BPS     = 2.5    # >2.5bps = bullish bias
OI_CHANGE_PCT        = 2.0    # >2% OI change since last bar = institutional move
ORDERBOOK_WALL_MULT  = 5.0    # bid/ask > 5× avg depth = wall
WHALE_TRADE_USD      = 500_000  # single trade > $500k = whale
TOP_TRADER_LONG_HIGH = 1.8    # top-trader L/S > 1.8 = heavily long
TOP_TRADER_LONG_LOW  = 0.6    # top-trader L/S < 0.6 = heavily short

# Impulse detection (crypto is more volatile; use 1.8× avg range)
IMPULSE_MULT         = 1.8
IMPULSE_BODY_FRAC    = 0.55

# Signal weights (sum should be 1.0 for clean confluence)
WEIGHTS = {
    "hurst":           0.08,
    "macd":            0.08,
    "ict_ob":          0.08,
    "bos_choch":       0.10,
    "wyckoff":         0.07,
    "supply_demand":   0.07,
    "volume_profile":  0.07,
    "cvd":             0.08,
    "ad_line":         0.05,
    "oi_divergence":   0.10,
    "funding":         0.07,
    "ob_walls":        0.07,
    "top_trader_ls":   0.08,
}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP HELPERS
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "Mozilla/5.0 CryptoScanner/1.0"}

def _get(url, params=None, timeout=10, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, tf: str = "1h", limit: int = 150) -> dict:
    """Fetch OHLCV from Binance USDT-M futures (more liquid than spot for large caps)."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/klines",
                {"symbol": symbol, "interval": tf, "limit": limit})
    if not data or not isinstance(data, list):
        # fallback to spot
        data = _get(f"{BINANCE_SPOT}/api/v3/klines",
                    {"symbol": symbol, "interval": tf, "limit": limit})
    if not data:
        return {}

    opens, highs, lows, closes, vols = [], [], [], [], []
    timestamps = []
    for k in data:
        try:
            timestamps.append(int(k[0]))
            opens.append(float(k[1]))
            highs.append(float(k[2]))
            lows.append(float(k[3]))
            closes.append(float(k[4]))
            vols.append(float(k[5]))
        except Exception:
            pass

    return {
        "symbol": symbol,
        "tf": tf,
        "timestamps": timestamps,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "vols": vols,
        "spot": closes[-1] if closes else 0.0,
        "day_high": max(highs[-24:]) if len(highs) >= 24 else max(highs) if highs else 0.0,
        "day_low":  min(lows[-24:]) if len(lows) >= 24 else min(lows) if lows else 0.0,
        "prev_high": max(highs[-48:-24]) if len(highs) >= 48 else 0.0,
        "prev_low":  min(lows[-48:-24]) if len(lows) >= 48 else 0.0,
        "prev_close": closes[-25] if len(closes) >= 25 else 0.0,
    }


def fetch_funding_rate(symbol: str) -> dict:
    """Get current funding rate + predicted next funding from USDT-M."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    if not isinstance(data, dict):
        return {}
    try:
        return {
            "funding_rate": float(data.get("lastFundingRate", 0)),
            "mark_price":   float(data.get("markPrice", 0)),
            "index_price":  float(data.get("indexPrice", 0)),
            "next_funding_time": int(data.get("nextFundingTime", 0)),
        }
    except Exception:
        return {}


def fetch_open_interest(symbol: str) -> dict:
    """Fetch current OI + 5m/1h change via OI hist endpoint."""
    cur = _get(f"{BINANCE_FAPI}/fapi/v1/openInterest", {"symbol": symbol})
    # Historical OI for change calculation (last 5 data points at 5min)
    hist = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
               {"symbol": symbol, "period": "5m", "limit": 24})  # ~2h

    cur_oi = 0.0
    if isinstance(cur, dict):
        try:
            cur_oi = float(cur.get("openInterest", 0))
        except Exception:
            pass

    oi_series = []
    if isinstance(hist, list):
        for item in hist:
            try:
                oi_series.append(float(item.get("sumOpenInterest", 0)))
            except Exception:
                pass

    oi_change_pct = 0.0
    if len(oi_series) >= 2:
        old = oi_series[0]
        if old > 0:
            oi_change_pct = (oi_series[-1] - old) / old * 100

    return {
        "open_interest": cur_oi,
        "oi_series": oi_series,
        "oi_change_pct_2h": oi_change_pct,
    }


def fetch_agg_trades_cvd(symbol: str, limit: int = 1000) -> dict:
    """
    Real CVD from Binance aggTrades.
    isBuyerMaker=False → taker BUY (aggressive buy) → delta positive
    isBuyerMaker=True  → taker SELL (aggressive sell) → delta negative
    """
    data = _get(f"{BINANCE_FAPI}/fapi/v1/aggTrades",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        # fallback to spot
        data = _get(f"{BINANCE_SPOT}/api/v3/aggTrades",
                    {"symbol": symbol, "limit": limit})

    if not isinstance(data, list):
        return {"cum_delta": 0, "recent_delta": 0, "buy_volume": 0, "sell_volume": 0,
                "whale_trades": [], "total_trades": 0}

    buy_vol = 0.0
    sell_vol = 0.0
    deltas = []
    whale_trades = []

    for t in data:
        try:
            qty = float(t.get("q", 0))
            price = float(t.get("p", 0))
            usd_val = qty * price
            is_maker = bool(t.get("m", False))
            delta = -usd_val if is_maker else usd_val  # maker=sell; taker=buy
            deltas.append(delta)
            if is_maker:
                sell_vol += usd_val
            else:
                buy_vol += usd_val

            if usd_val >= WHALE_TRADE_USD:
                whale_trades.append({
                    "usd": round(usd_val),
                    "price": price,
                    "side": "SELL" if is_maker else "BUY",
                    "time": t.get("T", 0),
                })
        except Exception:
            pass

    cum_delta = sum(deltas)
    # recent = last 20% of trades
    k = max(1, len(deltas) // 5)
    recent_delta = sum(deltas[-k:])

    return {
        "cum_delta": cum_delta,
        "recent_delta": recent_delta,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "whale_trades": sorted(whale_trades, key=lambda x: -x["usd"])[:5],
        "total_trades": len(deltas),
        "cvd_ratio": buy_vol / sell_vol if sell_vol > 0 else 1.0,
    }


def fetch_order_book(symbol: str, depth: int = 50) -> dict:
    """Get L2 order book snapshot. Detect large walls."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/depth",
                {"symbol": symbol, "limit": depth})
    if not isinstance(data, dict):
        data = _get(f"{BINANCE_SPOT}/api/v3/depth",
                    {"symbol": symbol, "limit": depth})
    if not isinstance(data, dict):
        return {}

    bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in data.get("asks", [])]

    if not bids or not asks:
        return {}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid       = (best_bid + best_ask) / 2

    # Volume-weighted average across all levels
    bid_qtys = [q for _, q in bids]
    ask_qtys = [q for _, q in asks]
    avg_bid_qty = statistics.mean(bid_qtys) if bid_qtys else 1.0
    avg_ask_qty = statistics.mean(ask_qtys) if ask_qtys else 1.0

    # Find walls: levels > ORDERBOOK_WALL_MULT × average
    bid_walls = [(p, q) for p, q in bids if q >= avg_bid_qty * ORDERBOOK_WALL_MULT]
    ask_walls = [(p, q) for p, q in asks if q >= avg_ask_qty * ORDERBOOK_WALL_MULT]

    # Compute bid/ask ratio by USD value
    bid_usd = sum(p * q for p, q in bids[:20])
    ask_usd = sum(p * q for p, q in asks[:20])
    imbalance = bid_usd / ask_usd if ask_usd > 0 else 1.0

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": (best_ask - best_bid) / mid * 10000,
        "bid_walls": sorted(bid_walls, key=lambda x: -x[1])[:3],
        "ask_walls": sorted(ask_walls, key=lambda x: -x[1])[:3],
        "bid_ask_imbalance": imbalance,  # >1 = more bid pressure
        "bid_usd": bid_usd,
        "ask_usd": ask_usd,
    }


def fetch_top_trader_ratio(symbol: str) -> dict:
    """Binance top-trader long/short position ratio."""
    # Global long/short ratio
    global_data = _get(f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio",
                       {"symbol": symbol, "period": "1h", "limit": 5})
    # Top trader accounts ratio
    top_data = _get(f"{BINANCE_FAPI}/futures/data/topLongShortAccountRatio",
                    {"symbol": symbol, "period": "1h", "limit": 5})
    # Top trader positions ratio
    pos_data = _get(f"{BINANCE_FAPI}/futures/data/topLongShortPositionRatio",
                    {"symbol": symbol, "period": "1h", "limit": 5})

    result = {"global_ls": 1.0, "top_account_ls": 1.0, "top_position_ls": 1.0}
    for key, data in [("global_ls", global_data), ("top_account_ls", top_data),
                      ("top_position_ls", pos_data)]:
        if isinstance(data, list) and data:
            try:
                latest = data[-1]
                result[key] = float(latest.get("longShortRatio",
                                   latest.get("longAccount", 1.0)))
            except Exception:
                pass

    return result


def fetch_liquidations(symbol: str) -> dict:
    """Recent liquidations from Binance force orders endpoint."""
    data = _get(f"{BINANCE_FAPI}/fapi/v1/allForceOrders",
                {"symbol": symbol, "limit": 100})
    if not isinstance(data, list):
        return {"long_liq_usd": 0, "short_liq_usd": 0, "total_liq_usd": 0,
                "dominant_liq": "none", "levels": []}

    long_liq = 0.0
    short_liq = 0.0
    levels = []

    for item in data:
        try:
            side = str(item.get("side", ""))
            qty = float(item.get("origQty", 0))
            price = float(item.get("price", 0))
            usd = qty * price
            if side == "SELL":  # Long position liquidated
                long_liq += usd
            else:              # Short position liquidated
                short_liq += usd
            levels.append(price)
        except Exception:
            pass

    return {
        "long_liq_usd": long_liq,
        "short_liq_usd": short_liq,
        "total_liq_usd": long_liq + short_liq,
        "dominant_liq": "LONGS" if long_liq > short_liq else "SHORTS",
        "levels": sorted(set(round(p, 0) for p in levels)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PORTED INDIA QUANT STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return default


def _res(name, score, label, note):
    score = int(max(-2, min(2, int(score))))
    return {"name": name, "score": score, "label": label, "note": note}


# ── Hurst Exponent ────────────────────────────────────────────────────────────

def hurst_exponent(closes: list) -> dict:
    """
    Hurst exponent on price LEVELS via polyfit(log(lags), log(tau)).
    H >= 0.55 = trending/persistent
    H < 0.45  = mean-reverting
    0.45-0.55 = random walk
    Exactly as in India Terminal fw_quant.py hurst_factor().
    """
    name = "Hurst Exponent"
    try:
        c = np.asarray([_safe_float(x) for x in closes], dtype=float)
        n = len(c)
        if n < 20:
            return _res(name, 0, "Insufficient bars", f"Need ≥20 bars, have {n}")

        max_lag = min(20, n // 2)
        lags = list(range(2, max_lag))
        if len(lags) < 3:
            return _res(name, 0, "Too few lags", "")

        tau = [float(np.std(c[lag:] - c[:-lag])) for lag in lags]
        tau = [t for t in tau if t > 0]
        if len(tau) < 3:
            return _res(name, 0, "Tau degenerate", "")

        log_lags = np.log([float(l) for l in lags[:len(tau)]])
        log_tau  = np.log(tau)
        H = float(np.polyfit(log_lags, log_tau, 1)[0])

        if H >= HURST_TREND_THRESH:
            score = 1  # trending — direction from recent price slope
            slope = c[-1] - c[-5] if len(c) >= 5 else 0
            score = 1 if slope > 0 else -1
            if H >= 0.65:
                score *= 2  # strong persistence
            lbl = f"Trending {'UP' if slope > 0 else 'DOWN'} (H={H:.3f})"
            note = f"Hurst={H:.3f} ≥ {HURST_TREND_THRESH}: persistent trend, {'bullish' if slope > 0 else 'bearish'}."
        elif H < HURST_MR_THRESH:
            score = 0  # mean-reverting: position depends on where we are in range
            lbl = f"Mean-Reverting (H={H:.3f})"
            note = f"Hurst={H:.3f} < {HURST_MR_THRESH}: mean-reverting regime, range-trade."
        else:
            score = 0
            lbl = f"Random Walk (H={H:.3f})"
            note = f"Hurst={H:.3f}: no edge from regime."

        return _res(name, score, lbl, note)
    except Exception as e:
        return _res(name, 0, "Hurst error", str(e))


# ── MACD ─────────────────────────────────────────────────────────────────────

def macd_signal(closes: list) -> dict:
    """MACD(12,26,9) histogram sign + slope. From fw_quant.py momentum_12_26."""
    name = "MACD(12,26,9)"
    try:
        c = np.asarray([_safe_float(x) for x in closes], dtype=float)
        if len(c) < 35:
            return _res(name, 0, "Insufficient bars", f"Need ≥35 bars")

        def ema(arr, span):
            k = 2.0 / (span + 1)
            out = [arr[0]]
            for x in arr[1:]:
                out.append(x * k + out[-1] * (1 - k))
            return np.array(out)

        ema12  = ema(c, 12)
        ema26  = ema(c, 26)
        macd   = ema12 - ema26
        signal = ema(macd, 9)
        hist   = macd - signal

        cur = hist[-1]
        prv = hist[-2] if len(hist) >= 2 else cur

        if cur > 0 and cur > prv:
            score, lbl = 2, "MACD expanding bullish"
            note = f"Histogram +{cur:.4f} and rising — accelerating buy momentum."
        elif cur > 0:
            score, lbl = 1, "MACD positive"
            note = f"Histogram +{cur:.4f} but contracting — weakening bullish."
        elif cur < 0 and cur < prv:
            score, lbl = -2, "MACD expanding bearish"
            note = f"Histogram {cur:.4f} and falling — accelerating sell momentum."
        elif cur < 0:
            score, lbl = -1, "MACD negative"
            note = f"Histogram {cur:.4f} but contracting — weakening bearish."
        else:
            score, lbl = 0, "MACD at zero"
            note = "Histogram near zero — no momentum edge."

        return _res(name, score, lbl, note)
    except Exception as e:
        return _res(name, 0, "MACD error", str(e))


# ── Swing Pivots (shared helper) ──────────────────────────────────────────────

def _swing_pivots(highs: list, lows: list, k: int = 3) -> list:
    """k-bar fractal pivots. Returns list of (idx, price, 'H'|'L')."""
    n = min(len(highs), len(lows))
    piv = []
    for i in range(k, n - k):
        win_h = highs[i - k:i + k + 1]
        win_l = lows[i - k:i + k + 1]
        if highs[i] >= max(win_h):
            piv.append((i, highs[i], "H"))
        elif lows[i] <= min(win_l):
            piv.append((i, lows[i], "L"))
    # Collapse consecutive same-type, keep more extreme
    out = []
    for p in piv:
        if out and out[-1][2] == p[2]:
            if (p[2] == "H" and p[1] >= out[-1][1]) or (p[2] == "L" and p[1] <= out[-1][1]):
                out[-1] = p
        else:
            out.append(p)
    return out


def _impulses(highs, lows, opens, closes, mult=IMPULSE_MULT, body_frac=IMPULSE_BODY_FRAC):
    """Indices of impulse bars: range ≥ mult×avg_range AND body ≥ body_frac×range."""
    n = min(len(highs), len(lows), len(opens), len(closes))
    if n < 10:
        return []
    ranges = [highs[i] - lows[i] for i in range(n)]
    avg_r  = statistics.mean(ranges) if ranges else 0
    if avg_r <= 0:
        return []
    out = []
    for i in range(n):
        r = highs[i] - lows[i]
        b = abs(closes[i] - opens[i])
        if r >= mult * avg_r and b >= body_frac * r:
            out.append(i)
    return out


# ── ICT Order Blocks ──────────────────────────────────────────────────────────

def ict_order_blocks(ohlcv: dict) -> dict:
    """
    Last opposite-color candle before an impulse ≥ IMPULSE_MULT × avg_range
    with body ≥ IMPULSE_BODY_FRAC × range.
    Pad = 0.25 × avg_range.
    Invalidated if price closes THROUGH the OB after formation.
    Ported from fw_smart_money.py ict_order_block().
    """
    name = "ICT Order Block"
    try:
        o = ohlcv["opens"]; h = ohlcv["highs"]; l = ohlcv["lows"]
        c = ohlcv["closes"]; spot = ohlcv["spot"]
        n = min(len(o), len(h), len(l), len(c))
        if n < 20:
            return _res(name, 0, "Insufficient bars", "")

        avg_r = statistics.mean([h[i] - l[i] for i in range(n)]) or 0.01
        pad   = 0.25 * avg_r
        imps  = _impulses(h[:n], l[:n], o[:n], c[:n])

        bull_obs, bear_obs = [], []

        for imp_i in imps:
            if imp_i < 1:
                continue
            is_bull_impulse = c[imp_i] > o[imp_i]  # bullish impulse candle

            # Look back for last opposite-color candle before impulse
            for j in range(imp_i - 1, max(0, imp_i - 10), -1):
                if is_bull_impulse and c[j] < o[j]:  # bearish candle before bull impulse
                    ob_lo = l[j] - pad
                    ob_hi = c[j] + pad  # top of bearish OB (close of bearish candle)
                    # Check not invalidated (no close below OB low after formation)
                    invalidated = any(c[k] < ob_lo for k in range(j + 1, n))
                    if not invalidated:
                        bull_obs.append({"hi": ob_hi, "lo": ob_lo, "idx": j})
                    break
                elif not is_bull_impulse and c[j] > o[j]:  # bullish candle before bear impulse
                    ob_hi = h[j] + pad
                    ob_lo = o[j] - pad  # bottom of bullish OB (open of bull candle)
                    invalidated = any(c[k] > ob_hi for k in range(j + 1, n))
                    if not invalidated:
                        bear_obs.append({"hi": ob_hi, "lo": ob_lo, "idx": j})
                    break

        # Score based on spot vs nearest active OB
        # Bullish OB: spot entering OB from above (bounce expected)
        # Bearish OB: spot entering OB from below (rejection expected)
        in_bull_ob = any(ob["lo"] <= spot <= ob["hi"] for ob in bull_obs)
        in_bear_ob = any(ob["lo"] <= spot <= ob["hi"] for ob in bear_obs)

        near_bull = any(ob["lo"] <= spot <= ob["hi"] * 1.005 for ob in bull_obs)
        near_bear = any(ob["hi"] >= spot >= ob["lo"] * 0.995 for ob in bear_obs)

        if in_bull_ob:
            return _res(name, 2, "Inside Bullish OB",
                       f"Spot {spot:.2f} inside bullish order block — institutional buy zone.")
        if in_bear_ob:
            return _res(name, -2, "Inside Bearish OB",
                       f"Spot {spot:.2f} inside bearish order block — institutional sell zone.")
        if near_bull:
            return _res(name, 1, "Approaching Bullish OB",
                       f"Spot near bullish OB support. {len(bull_obs)} bull OBs active.")
        if near_bear:
            return _res(name, -1, "Approaching Bearish OB",
                       f"Spot near bearish OB resistance. {len(bear_obs)} bear OBs active.")

        return _res(name, 0, "No active OB",
                   f"Spot {spot:.2f} not near any order block. Bull OBs={len(bull_obs)}, Bear OBs={len(bear_obs)}.")
    except Exception as e:
        return _res(name, 0, "ICT OB error", str(e))


# ── BOS / CHoCH Market Structure ──────────────────────────────────────────────

def bos_choch(ohlcv: dict) -> dict:
    """
    Break of Structure (BOS) and Change of Character (CHoCH).
    Swing pivots k=3; HH+HL=bullish, LH+LL=bearish; BOS on breakout, CHoCH on first counter.
    Ported from fw_smart_money.py ict_market_structure().
    """
    name = "BOS/CHoCH Structure"
    try:
        h = ohlcv["highs"]; l = ohlcv["lows"]; c = ohlcv["closes"]
        n = min(len(h), len(l), len(c))
        if n < 20:
            return _res(name, 0, "Insufficient bars", "")

        piv = _swing_pivots(list(h[:n]), list(l[:n]), k=3)
        if len(piv) < 4:
            return _res(name, 0, "Too few pivots", f"Only {len(piv)} swing pivots.")

        ph = [p for p in piv if p[2] == "H"]
        pl = [p for p in piv if p[2] == "L"]
        if len(ph) < 2 or len(pl) < 2:
            return _res(name, 0, "Sparse pivots", "")

        # Determine trend structure
        hh = ph[-1][1] > ph[-2][1]   # higher high
        hl = pl[-1][1] > pl[-2][1]   # higher low
        lh = ph[-1][1] < ph[-2][1]   # lower high
        ll = pl[-1][1] < pl[-2][1]   # lower low

        spot = ohlcv["spot"]
        last_swing_hi = ph[-1][1]
        last_swing_lo = pl[-1][1]

        if hh and hl:
            structure = "BULLISH"
            # BOS = clean break above last pivot high
            if spot > last_swing_hi:
                return _res(name, 2, "BOS Bullish — New High",
                           f"BOS: Spot {spot:.2f} broke above swing high {last_swing_hi:.2f}. Bullish structure intact.")
            return _res(name, 1, "Bullish Structure (HH+HL)",
                       f"Higher highs + higher lows. Last swing hi={last_swing_hi:.2f}, lo={last_swing_lo:.2f}.")
        elif lh and ll:
            structure = "BEARISH"
            if spot < last_swing_lo:
                return _res(name, -2, "BOS Bearish — New Low",
                           f"BOS: Spot {spot:.2f} broke below swing low {last_swing_lo:.2f}. Bearish structure intact.")
            return _res(name, -1, "Bearish Structure (LH+LL)",
                       f"Lower highs + lower lows. Last swing hi={last_swing_hi:.2f}, lo={last_swing_lo:.2f}.")
        elif hh and ll:
            # CHoCH: bullish then bearish = character change
            return _res(name, -1, "CHoCH — Bearish Shift",
                       f"Change of Character: HH followed by LL — potential reversal down.")
        elif lh and hl:
            return _res(name, 1, "CHoCH — Bullish Shift",
                       f"Change of Character: LH followed by HL — potential reversal up.")
        else:
            return _res(name, 0, "Choppy/Consolidation",
                       f"No clear HH/HL or LH/LL pattern — market consolidating.")
    except Exception as e:
        return _res(name, 0, "BOS/CHoCH error", str(e))


# ── Wyckoff Phase ─────────────────────────────────────────────────────────────

def wyckoff_phase(ohlcv: dict) -> dict:
    """
    Wyckoff phase: pos in day range, vol_rising (late_v > 1.15×early_v),
    range_narrowing (late_r < 0.80×early_r).
    Markup/Markdown/Accumulation/Distribution.
    Ported from fw_smart_money.py wyckoff_phase().
    """
    name = "Wyckoff Phase"
    try:
        h = ohlcv["highs"]; l = ohlcv["lows"]
        c = ohlcv["closes"]; v = ohlcv["vols"]
        n = min(len(h), len(l), len(c), len(v))
        if n < 20:
            return _res(name, 0, "Insufficient bars", "")

        # Use last 24 bars (1 day on 1h chart) as the "session"
        window = min(24, n)
        h_w = h[n - window:n]; l_w = l[n - window:n]
        c_w = c[n - window:n]; v_w = v[n - window:n]

        d_hi = max(h_w); d_lo = min(l_w)
        rng  = d_hi - d_lo
        if rng <= 0:
            return _res(name, 0, "Flat range", "")

        spot = ohlcv["spot"]
        pos  = (spot - d_lo) / rng  # 0=low, 1=high

        half = window // 2
        early_v = sum(v_w[:half]) / max(half, 1)
        late_v  = sum(v_w[half:]) / max(window - half, 1)
        early_r = statistics.mean([h_w[i] - l_w[i] for i in range(half)]) or 0.001
        late_r  = statistics.mean([h_w[i] - l_w[i] for i in range(half, window)]) or 0.001

        vol_rising     = late_v > 1.15 * early_v
        range_narrowing = late_r < 0.80 * early_r

        if pos >= 0.70 and not range_narrowing:
            if vol_rising:
                return _res(name, 2, "Markup Phase",
                           f"Spot at {pos:.0%} of range, vol expanding — Wyckoff Markup (bullish).")
            return _res(name, 1, "Distribution Risk",
                       f"Spot at {pos:.0%} of range; vol not expanding — possible Distribution.")
        elif pos <= 0.30 and not range_narrowing:
            if vol_rising:
                return _res(name, -2, "Markdown Phase",
                           f"Spot at {pos:.0%} of range, vol expanding — Wyckoff Markdown (bearish).")
            return _res(name, -1, "Accumulation Starting",
                       f"Spot at {pos:.0%} of range; vol drying up — possible Accumulation bottom.")
        elif range_narrowing and 0.30 <= pos <= 0.70:
            if vol_rising:
                return _res(name, 1, "Spring/Test (Accumulation)",
                           f"Range narrowing + vol rising mid-range — potential Wyckoff Spring.")
            return _res(name, 0, "Consolidation",
                       f"Range narrowing, mid-price — neutral Wyckoff consolidation.")
        else:
            return _res(name, 0, "Transitional",
                       f"Spot {pos:.0%} of range, mixed signals.")
    except Exception as e:
        return _res(name, 0, "Wyckoff error", str(e))


# ── Supply / Demand Zones ────────────────────────────────────────────────────

def supply_demand_zones(ohlcv: dict) -> dict:
    """
    Base-then-impulse pattern detection.
    impulse = body ≥ 0.6×range AND range ≥ 2×avg_range.
    base ≤ 1.5×avg_range.
    pad = 0.25×avg_range.
    Ported from fw_smart_money.py supply_demand().
    """
    name = "Supply/Demand Zones"
    try:
        o = ohlcv["opens"]; h = ohlcv["highs"]
        l = ohlcv["lows"];  c = ohlcv["closes"]
        n = min(len(o), len(h), len(l), len(c))
        if n < 10:
            return _res(name, 0, "Insufficient bars", "")

        avg_r = statistics.mean([h[i] - l[i] for i in range(n)]) or 0.01
        pad   = 0.25 * avg_r
        spot  = ohlcv["spot"]
        imps  = _impulses(list(h[:n]), list(l[:n]), list(o[:n]), list(c[:n]))

        demand_zones, supply_zones = [], []
        for imp_i in imps:
            if imp_i < 1:
                continue
            is_bull = c[imp_i] > o[imp_i]

            # Find base candles before this impulse
            for j in range(imp_i - 1, max(0, imp_i - 6), -1):
                if h[j] - l[j] <= 1.5 * avg_r:
                    if is_bull:  # Demand zone = base before bull impulse
                        demand_zones.append({
                            "lo": l[j] - pad,
                            "hi": h[j] + pad,
                            "idx": j,
                        })
                    else:         # Supply zone = base before bear impulse
                        supply_zones.append({
                            "lo": l[j] - pad,
                            "hi": h[j] + pad,
                            "idx": j,
                        })
                    break

        in_demand = any(z["lo"] <= spot <= z["hi"] for z in demand_zones)
        in_supply = any(z["lo"] <= spot <= z["hi"] for z in supply_zones)
        near_demand = any(z["lo"] <= spot <= z["hi"] * 1.003 for z in demand_zones)
        near_supply = any(z["hi"] >= spot >= z["lo"] * 0.997 for z in supply_zones)

        if in_demand:
            return _res(name, 2, "Inside Demand Zone",
                       f"Spot {spot:.2f} in demand zone — high-probability long area.")
        if in_supply:
            return _res(name, -2, "Inside Supply Zone",
                       f"Spot {spot:.2f} in supply zone — high-probability short area.")
        if near_demand:
            return _res(name, 1, "Approaching Demand",
                       f"Demand={len(demand_zones)} zones, Supply={len(supply_zones)} zones.")
        if near_supply:
            return _res(name, -1, "Approaching Supply",
                       f"Demand={len(demand_zones)} zones, Supply={len(supply_zones)} zones.")

        return _res(name, 0, "Between zones",
                   f"Demand={len(demand_zones)}, Supply={len(supply_zones)} zones. No confluence.")
    except Exception as e:
        return _res(name, 0, "S/D error", str(e))


# ── IPDA Liquidity Pools ──────────────────────────────────────────────────────

def ipda_liquidity(ohlcv: dict) -> dict:
    """
    Prev-day highs/lows + equal highs/lows from swing pivots (k=3 fractals).
    tol = max(0.5×avg_range, 0.0005×spot).
    Ported from fw_smart_money.py ipda_liquidity().
    """
    name = "IPDA Liquidity"
    try:
        h = ohlcv["highs"]; l = ohlcv["lows"]
        c = ohlcv["closes"]
        n = min(len(h), len(l), len(c))
        if n < 25:
            return _res(name, 0, "Insufficient bars", "")

        spot  = ohlcv["spot"]
        avg_r = statistics.mean([h[i] - l[i] for i in range(n)]) or spot * 0.001
        tol   = max(0.5 * avg_r, 0.0005 * spot)

        # Previous 24h liquidity levels
        prev_hi = max(h[n - 48:n - 24]) if n >= 48 else max(h[:n // 2])
        prev_lo = min(l[n - 48:n - 24]) if n >= 48 else min(l[:n // 2])

        # Swing pivots for equal highs/lows (stop clusters)
        piv = _swing_pivots(list(h[:n]), list(l[:n]), k=3)
        ph = [p for p in piv if p[2] == "H"]
        pl = [p for p in piv if p[2] == "L"]

        # Equal highs: within tol of each other → stop cluster above
        eq_highs, eq_lows = [], []
        for i in range(len(ph)):
            for j in range(i + 1, len(ph)):
                if abs(ph[i][1] - ph[j][1]) <= tol:
                    eq_highs.append((ph[i][1] + ph[j][1]) / 2)
        for i in range(len(pl)):
            for j in range(i + 1, len(pl)):
                if abs(pl[i][1] - pl[j][1]) <= tol:
                    eq_lows.append((pl[i][1] + pl[j][1]) / 2)

        # Score: spot sweeping or approaching a liquidity pool
        all_liq_above = [prev_hi] + eq_highs
        all_liq_below = [prev_lo] + eq_lows

        # "Sweeping" = within tol of a pool
        sweeping_above = any(abs(spot - lvl) <= tol for lvl in all_liq_above)
        sweeping_below = any(abs(spot - lvl) <= tol for lvl in all_liq_below)
        # Approaching = within 2×tol
        approaching_above = any(0 < lvl - spot <= 3 * tol for lvl in all_liq_above)
        approaching_below = any(0 < spot - lvl <= 3 * tol for lvl in all_liq_below)

        above_levels = sorted(set(round(x, 2) for x in all_liq_above))
        below_levels = sorted(set(round(x, 2) for x in all_liq_below))

        if sweeping_above:
            return _res(name, -1, "Sweeping Liquidity Above",
                       f"Spot {spot:.2f} sweeping equal highs/PDH at {above_levels} — stop-hunt risk.")
        if sweeping_below:
            return _res(name, 1, "Sweeping Liquidity Below",
                       f"Spot {spot:.2f} sweeping equal lows/PDL at {below_levels[:3]} — potential reversal up.")
        if approaching_above:
            return _res(name, -1, "Liquidity Pool Above",
                       f"Stop cluster above at {above_levels[:3]}. PDH={prev_hi:.2f}.")
        if approaching_below:
            return _res(name, 1, "Liquidity Pool Below",
                       f"Demand + stop cluster below at {below_levels[:3]}. PDL={prev_lo:.2f}.")

        return _res(name, 0, "No nearby liquidity",
                   f"PDH={prev_hi:.2f}, PDL={prev_lo:.2f}. EqHighs={len(eq_highs)}, EqLows={len(eq_lows)}.")
    except Exception as e:
        return _res(name, 0, "IPDA error", str(e))


# ── Volume Profile ────────────────────────────────────────────────────────────

def volume_profile(ohlcv: dict, n_bins: int = 24) -> dict:
    """
    Intraday volume profile: POC / 70% VAH/VAL.
    Ported from fw_volume_flow.py volume_profile().
    """
    name = "Volume Profile"
    try:
        h = np.asarray(ohlcv["highs"]); l = np.asarray(ohlcv["lows"])
        c = np.asarray(ohlcv["closes"]); v = np.asarray(ohlcv["vols"])
        n = min(len(h), len(l), len(c), len(v))
        if n < 10:
            return _res(name, 0, "Insufficient bars", "")

        h, l, c, v = h[:n], l[:n], c[:n], v[:n]
        tp = (h + l + c) / 3.0
        lo_p, hi_p = float(np.min(tp)), float(np.max(tp))
        if hi_p - lo_p <= 0:
            return _res(name, 0, "Flat profile", "")

        idx  = np.clip(((tp - lo_p) / (hi_p - lo_p) * n_bins).astype(int), 0, n_bins - 1)
        prof = np.zeros(n_bins)
        np.add.at(prof, idx, v)
        centers = lo_p + (np.arange(n_bins) + 0.5) * (hi_p - lo_p) / n_bins

        poc_i = int(np.argmax(prof))
        poc   = float(centers[poc_i])

        total = float(np.sum(prof))
        acc = prof[poc_i]; lo_i = hi_i = poc_i
        while acc < 0.70 * total and (lo_i > 0 or hi_i < n_bins - 1):
            below = prof[lo_i - 1] if lo_i > 0 else -1.0
            above = prof[hi_i + 1] if hi_i < n_bins - 1 else -1.0
            if above >= below:
                hi_i += 1; acc += prof[hi_i]
            else:
                lo_i -= 1; acc += prof[lo_i]
        val = float(centers[lo_i])
        vah = float(centers[hi_i])

        spot = ohlcv["spot"]
        band = max((vah - val) * 0.10, (hi_p - lo_p) / n_bins * 0.5)
        k = min(5, n - 1)
        recent = float(c[-1] - c[-1 - k]) if k > 0 else 0.0
        avg_v  = float(np.mean(v))
        rec_v  = float(np.mean(v[-k:])) if k > 0 else avg_v
        levels = f"POC={poc:.2f} VAH={vah:.2f} VAL={val:.2f}"

        if spot > vah + band:
            score = 2 if rec_v > avg_v else 1
            return _res(name, score, "Above Value Area",
                       f"Spot {spot:.2f} accepted above value area on {'strong' if rec_v > avg_v else 'light'} vol. {levels}")
        elif spot < val - band:
            score = -2 if rec_v > avg_v else -1
            return _res(name, score, "Below Value Area",
                       f"Spot {spot:.2f} trading below value area. {levels}")
        elif abs(spot - vah) <= band:
            return _res(name, -1 if recent <= 0 else 1, "At VAH",
                       f"Spot at value area high {'being rejected' if recent <= 0 else 'breaking through'}. {levels}")
        elif abs(spot - val) <= band:
            return _res(name, 1 if recent >= 0 else -1, "At VAL",
                       f"Spot at value area low {'bouncing' if recent >= 0 else 'leaking'}. {levels}")
        elif abs(spot - poc) <= band:
            return _res(name, 0, "At POC — Balanced",
                       f"Spot at POC {poc:.2f} — two-way trade.")
        else:
            return _res(name, 1 if spot > poc else -1, "Inside Value",
                       f"Spot inside value area, {'above' if spot > poc else 'below'} POC. {levels}")
    except Exception as e:
        return _res(name, 0, "Volume Profile error", str(e))


# ── CVD from aggTrades ────────────────────────────────────────────────────────

def cvd_analysis(agg_data: dict) -> dict:
    """Score CVD from real aggTrades data (fetched separately)."""
    name = "Order Flow CVD"
    try:
        cum = _safe_float(agg_data.get("cum_delta", 0))
        rec = _safe_float(agg_data.get("recent_delta", 0))
        buy = _safe_float(agg_data.get("buy_volume", 1))
        sel = _safe_float(agg_data.get("sell_volume", 1))
        total = buy + sel
        thresh = max(total * 0.005, 1.0)

        if abs(cum) < thresh:
            return _res(name, 0, "Delta balanced",
                       f"CVD near zero ({cum:.0f}); balanced order flow.")

        bull = cum > 0
        ratio = agg_data.get("cvd_ratio", 1.0)

        if bull:
            if rec > 0:
                score = 2 if ratio > 1.3 else 1
                lbl = "Strong Buy Flow" if score == 2 else "Bullish CVD"
            else:
                score = 1
                lbl = "CVD Bullish, Flow Fading"
            note = f"CVD={cum:.0f}, recent={rec:.0f}, buy/sell ratio={ratio:.2f}."
        else:
            if rec < 0:
                score = -2 if ratio < 0.77 else -1
                lbl = "Strong Sell Flow" if score == -2 else "Bearish CVD"
            else:
                score = -1
                lbl = "CVD Bearish, Flow Fading"
            note = f"CVD={cum:.0f}, recent={rec:.0f}, buy/sell ratio={ratio:.2f}."

        return _res(name, score, lbl, note)
    except Exception as e:
        return _res(name, 0, "CVD error", str(e))


# ── A/D Line ──────────────────────────────────────────────────────────────────

def ad_line(ohlcv: dict) -> dict:
    """Chaikin A/D line + OBV cross-check. Ported from fw_volume_flow.py."""
    name = "A/D Line"
    try:
        h = np.asarray(ohlcv["highs"]); l = np.asarray(ohlcv["lows"])
        c = np.asarray(ohlcv["closes"]); v = np.asarray(ohlcv["vols"])
        n = min(len(h), len(l), len(c), len(v))
        if n < 10:
            return _res(name, 0, "Insufficient bars", "")
        h, l, c, v = h[:n], l[:n], c[:n], v[:n]

        rng = h - l
        mfm = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng > 0, rng, 1.0), 0.0)
        ad  = np.cumsum(mfm * v)
        obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v)

        def slope_norm(y):
            nn = len(y)
            if nn < 3: return 0.0
            sl = float(np.polyfit(np.arange(nn), y.astype(float), 1)[0])
            scale = max(abs(float(np.max(y)) - float(np.min(y))), 1e-9)
            return sl * nn / scale

        ad_sl  = slope_norm(ad)
        obv_sl = slope_norm(obv)
        px_sl  = slope_norm(c)

        confirmed = (np.sign(obv_sl) == np.sign(ad_sl)) and abs(obv_sl) > 0.1
        diverging = np.sign(px_sl) != np.sign(ad_sl) or abs(px_sl) < 0.15

        if ad_sl > 0.25:
            if diverging and confirmed:
                return _res(name, 2, "Hidden Accumulation",
                           "A/D rising vs flat/down price + OBV confirmed — smart money accumulating.")
            score = 2 if (confirmed and ad_sl > 0.6) else 1
            return _res(name, score, "Accumulation",
                       f"A/D trending up {'+ OBV confirmed' if confirmed else ''}.")
        elif ad_sl < -0.25:
            if diverging and confirmed:
                return _res(name, -2, "Hidden Distribution",
                           "A/D falling vs flat/up price + OBV confirmed — smart money distributing.")
            score = -2 if (confirmed and ad_sl < -0.6) else -1
            return _res(name, score, "Distribution",
                       f"A/D trending down {'+ OBV confirmed' if confirmed else ''}.")
        else:
            return _res(name, 0, "A/D Neutral", f"A/D line flat (slope={ad_sl:.3f}).")
    except Exception as e:
        return _res(name, 0, "A/D error", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CRYPTO-SPECIFIC INSTITUTIONAL SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

def oi_divergence_signal(ohlcv: dict, oi_data: dict) -> dict:
    """
    Open Interest divergence with price.
    OI rising + price rising = new longs opening = bullish conviction
    OI rising + price falling = new shorts opening = bearish conviction
    OI falling + price rising = short covering (less conviction)
    OI falling + price falling = long liquidation (capitulation)
    """
    name = "OI Divergence"
    try:
        c = ohlcv["closes"]
        n = len(c)
        if n < 5:
            return _res(name, 0, "Insufficient bars", "")

        oi_change_pct = _safe_float(oi_data.get("oi_change_pct_2h", 0))
        price_change_pct = (c[-1] - c[-5]) / c[-5] * 100 if c[-5] > 0 else 0

        oi_rising = oi_change_pct > OI_CHANGE_PCT
        oi_falling = oi_change_pct < -OI_CHANGE_PCT
        price_rising = price_change_pct > 0.3
        price_falling = price_change_pct < -0.3

        if oi_rising and price_rising:
            score = 2
            lbl = "Institutional Long Build"
            note = f"OI +{oi_change_pct:.1f}% + Price +{price_change_pct:.1f}% — new longs opening."
        elif oi_rising and price_falling:
            score = -2
            lbl = "Institutional Short Build"
            note = f"OI +{oi_change_pct:.1f}% + Price {price_change_pct:.1f}% — new shorts opening."
        elif oi_falling and price_rising:
            score = 1
            lbl = "Short Covering Rally"
            note = f"OI {oi_change_pct:.1f}% + Price +{price_change_pct:.1f}% — shorts covering, less conviction."
        elif oi_falling and price_falling:
            score = -1
            lbl = "Long Liquidation"
            note = f"OI {oi_change_pct:.1f}% + Price {price_change_pct:.1f}% — longs getting stopped out."
        else:
            score = 0
            lbl = "OI Neutral"
            note = f"OI change={oi_change_pct:.1f}%, Price change={price_change_pct:.1f}% — no strong divergence."

        return _res(name, score, lbl, note)
    except Exception as e:
        return _res(name, 0, "OI error", str(e))


def funding_rate_signal(fund_data: dict) -> dict:
    """
    Funding rate as institutional sentiment gauge.
    >FUNDING_EXTREME_BPS = market over-leveraged long → squeeze risk
    <-FUNDING_EXTREME_BPS = market over-leveraged short → squeeze risk
    """
    name = "Funding Rate"
    try:
        rate = _safe_float(fund_data.get("funding_rate", 0))
        rate_bps = rate * 10000
        rate_ann_pct = rate * 1095 * 100

        if rate_bps >= FUNDING_EXTREME_BPS:
            return _res(name, -1, "Extreme Long Funding",
                       f"Funding={rate_bps:.2f}bps/8h ({rate_ann_pct:.1f}% ann) — market over-leveraged LONG, squeeze risk.")
        elif rate_bps >= FUNDING_HIGH_BPS:
            return _res(name, 1, "Positive Funding — Bull Bias",
                       f"Funding={rate_bps:.2f}bps/8h ({rate_ann_pct:.1f}% ann) — longs paying, net bullish sentiment.")
        elif rate_bps <= -FUNDING_EXTREME_BPS:
            return _res(name, 1, "Extreme Short Funding",
                       f"Funding={rate_bps:.2f}bps/8h ({rate_ann_pct:.1f}% ann) — market over-leveraged SHORT, squeeze risk.")
        elif rate_bps <= -FUNDING_HIGH_BPS:
            return _res(name, -1, "Negative Funding — Bear Bias",
                       f"Funding={rate_bps:.2f}bps/8h ({rate_ann_pct:.1f}% ann) — shorts paying, net bearish sentiment.")
        else:
            return _res(name, 0, "Neutral Funding",
                       f"Funding={rate_bps:.2f}bps/8h — neutral, no leverage extreme.")
    except Exception as e:
        return _res(name, 0, "Funding error", str(e))


def order_book_wall_signal(ob_data: dict) -> dict:
    """
    Order book imbalance + institutional walls.
    Large bid wall below = institutional support
    Large ask wall above = institutional resistance
    """
    name = "Order Book Walls"
    try:
        if not ob_data:
            return _res(name, 0, "No OB data", "")

        imbalance = _safe_float(ob_data.get("bid_ask_imbalance", 1.0), 1.0)
        bid_walls = ob_data.get("bid_walls", [])
        ask_walls = ob_data.get("ask_walls", [])
        spread_bps = _safe_float(ob_data.get("spread_bps", 0))

        bid_usd = _safe_float(ob_data.get("bid_usd", 0))
        ask_usd = _safe_float(ob_data.get("ask_usd", 0))

        if imbalance > 2.0 and bid_walls:
            return _res(name, 2, "Strong Bid Wall (Inst. Support)",
                       f"Bid/Ask={imbalance:.2f}x — institutional bid wall absorbing sells. Walls at {[round(p,0) for p,q in bid_walls[:2]]}.")
        elif imbalance > 1.5:
            return _res(name, 1, "Bid-Heavy Book",
                       f"Bid/Ask={imbalance:.2f}x — more buy-side pressure. Spread={spread_bps:.1f}bps.")
        elif imbalance < 0.5 and ask_walls:
            return _res(name, -2, "Strong Ask Wall (Inst. Resistance)",
                       f"Bid/Ask={imbalance:.2f}x — institutional ask wall capping rallies. Walls at {[round(p,0) for p,q in ask_walls[:2]]}.")
        elif imbalance < 0.67:
            return _res(name, -1, "Ask-Heavy Book",
                       f"Bid/Ask={imbalance:.2f}x — more sell-side pressure.")
        else:
            return _res(name, 0, "Balanced Book",
                       f"Bid/Ask={imbalance:.2f}x — balanced order flow. Spread={spread_bps:.1f}bps.")
    except Exception as e:
        return _res(name, 0, "OB error", str(e))


def top_trader_signal(ls_data: dict) -> dict:
    """
    Binance top-trader long/short position ratio.
    >TOP_TRADER_LONG_HIGH = institutions heavily long
    <TOP_TRADER_LONG_LOW  = institutions heavily short
    """
    name = "Top Trader L/S"
    try:
        if not ls_data:
            return _res(name, 0, "No L/S data", "")

        top_pos = _safe_float(ls_data.get("top_position_ls", 1.0), 1.0)
        top_acc = _safe_float(ls_data.get("top_account_ls", 1.0), 1.0)
        global_ls = _safe_float(ls_data.get("global_ls", 1.0), 1.0)

        # Top positions ratio is most indicative of "whale" intent
        if top_pos >= TOP_TRADER_LONG_HIGH and top_acc >= TOP_TRADER_LONG_HIGH:
            return _res(name, 2, "Institutions Heavily Long",
                       f"Top positions L/S={top_pos:.2f}, accounts={top_acc:.2f} — institutions positioned long.")
        elif top_pos >= 1.3:
            return _res(name, 1, "Institutions Leaning Long",
                       f"Top positions L/S={top_pos:.2f} — slight institutional long bias.")
        elif top_pos <= TOP_TRADER_LONG_LOW and top_acc <= TOP_TRADER_LONG_LOW:
            return _res(name, -2, "Institutions Heavily Short",
                       f"Top positions L/S={top_pos:.2f}, accounts={top_acc:.2f} — institutions positioned short.")
        elif top_pos <= 0.77:
            return _res(name, -1, "Institutions Leaning Short",
                       f"Top positions L/S={top_pos:.2f} — slight institutional short bias.")
        else:
            return _res(name, 0, "Institutions Neutral",
                       f"Top positions L/S={top_pos:.2f} — neutral positioning.")
    except Exception as e:
        return _res(name, 0, "L/S error", str(e))


# ─────────────────────────────────────────────────────────────────────────────
# CONFLUENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_confluence(signals: dict) -> dict:
    """
    Weighted sum of all signal scores → final conviction score.
    Returns: {score: float, bias: str, confidence: str, details: list}
    """
    total_weight = 0.0
    weighted_sum = 0.0
    signal_list  = []

    for key, weight in WEIGHTS.items():
        sig = signals.get(key)
        if sig is None:
            continue
        s = int(sig.get("score", 0))
        total_weight  += weight
        weighted_sum  += s * weight
        signal_list.append({
            "key":    key,
            "name":   sig.get("name", key),
            "score":  s,
            "label":  sig.get("label", ""),
            "weight": weight,
        })

    if total_weight <= 0:
        return {"score": 0.0, "bias": "NEUTRAL", "confidence": "NONE", "signals": signal_list}

    final = weighted_sum / total_weight
    bull_count  = sum(1 for s in signal_list if s["score"] > 0)
    bear_count  = sum(1 for s in signal_list if s["score"] < 0)
    strong_bull = sum(1 for s in signal_list if s["score"] == 2)
    strong_bear = sum(1 for s in signal_list if s["score"] == -2)

    if final >= 1.2:
        bias = "STRONG LONG"
    elif final >= 0.5:
        bias = "LONG"
    elif final <= -1.2:
        bias = "STRONG SHORT"
    elif final <= -0.5:
        bias = "SHORT"
    else:
        bias = "NEUTRAL"

    agreement = max(bull_count, bear_count) / max(len(signal_list), 1)
    if agreement >= 0.75 and (strong_bull >= 3 or strong_bear >= 3):
        confidence = "HIGH"
    elif agreement >= 0.6:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "score":       round(final, 3),
        "bias":        bias,
        "confidence":  confidence,
        "bull_signals": bull_count,
        "bear_signals": bear_count,
        "strong_bull":  strong_bull,
        "strong_bear":  strong_bear,
        "total_signals": len(signal_list),
        "signals":      sorted(signal_list, key=lambda x: abs(x["score"]), reverse=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL SYMBOL SCAN
# ─────────────────────────────────────────────────────────────────────────────

def scan_symbol(symbol: str, tf: str = DEFAULT_TF, limit: int = DEFAULT_LIMIT,
                verbose: bool = False) -> dict:
    """Full institutional + quant scan on one symbol."""
    print(f"  Scanning {symbol}...", end=" ", flush=True)
    t0 = time.time()

    # -- Fetch all data -- #
    ohlcv   = fetch_ohlcv(symbol, tf, limit)
    if not ohlcv or len(ohlcv.get("closes", [])) < 20:
        print(f"NO DATA ({time.time()-t0:.1f}s)")
        return {"symbol": symbol, "error": "no_data"}

    fund    = fetch_funding_rate(symbol)
    oi      = fetch_open_interest(symbol)
    agg_cvd = fetch_agg_trades_cvd(symbol, limit=500)
    ob      = fetch_order_book(symbol, depth=50)
    ls      = fetch_top_trader_ratio(symbol)

    # -- Run all signals in order -- #
    signals = {}

    # Quant / Technical
    signals["hurst"]         = hurst_exponent(ohlcv["closes"])
    signals["macd"]          = macd_signal(ohlcv["closes"])
    signals["ict_ob"]        = ict_order_blocks(ohlcv)
    signals["bos_choch"]     = bos_choch(ohlcv)
    signals["wyckoff"]       = wyckoff_phase(ohlcv)
    signals["supply_demand"] = supply_demand_zones(ohlcv)
    signals["volume_profile"]= volume_profile(ohlcv)
    signals["cvd"]           = cvd_analysis(agg_cvd)
    signals["ad_line"]       = ad_line(ohlcv)

    # Crypto Institutional
    signals["oi_divergence"] = oi_divergence_signal(ohlcv, oi)
    signals["funding"]       = funding_rate_signal(fund)
    signals["ob_walls"]      = order_book_wall_signal(ob)
    signals["top_trader_ls"] = top_trader_signal(ls)

    confluence = compute_confluence(signals)
    elapsed = time.time() - t0
    print(f"{confluence['bias']:14s} score={confluence['score']:+.3f}  ({elapsed:.1f}s)")

    # Whale summary
    whale_trades = agg_cvd.get("whale_trades", [])

    result = {
        "symbol":        symbol,
        "spot":          ohlcv["spot"],
        "tf":            tf,
        "signals":       signals,
        "confluence":    confluence,
        "whale_trades":  whale_trades,
        "funding_bps":   fund.get("funding_rate", 0) * 10000 if fund else 0,
        "oi_change_2h":  oi.get("oi_change_pct_2h", 0),
        "cvd_ratio":     agg_cvd.get("cvd_ratio", 1.0),
        "top_trader_ls": ls.get("top_position_ls", 1.0),
        "scan_ts":       datetime.now(timezone.utc).isoformat(),
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def _color(score, text):
    """ANSI color: green=bull, red=bear, white=neutral."""
    if score > 0:
        return f"\033[92m{text}\033[0m"
    elif score < 0:
        return f"\033[91m{text}\033[0m"
    return f"\033[37m{text}\033[0m"


def print_report(results: list):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "═" * 110)
    print(f"  CRYPTO INSTITUTIONAL SCANNER  —  {now}")
    print("═" * 110)

    # Sort by absolute confluence score
    ranked = sorted([r for r in results if "confluence" in r],
                    key=lambda x: abs(x["confluence"]["score"]), reverse=True)

    # Header
    print(f"\n{'Symbol':<12} {'Price':>12} {'Bias':>14} {'Score':>7} {'Conf':>6} "
          f"{'Bull/Bear':>10} {'FundBps':>8} {'OI%':>7} {'CVD Ratio':>10} {'TopLS':>7}")
    print("─" * 110)

    for r in ranked:
        c    = r["confluence"]
        sym  = r["symbol"]
        spot = r["spot"]
        bias = c["bias"]
        score = c["score"]
        conf = c["confidence"]
        bull = c["bull_signals"]
        bear = c["bear_signals"]
        fbps = r.get("funding_bps", 0)
        oipct= r.get("oi_change_2h", 0)
        cvdr = r.get("cvd_ratio", 1.0)
        tls  = r.get("top_trader_ls", 1.0)

        bias_colored = _color(score, f"{bias:>14}")
        score_str    = _color(score, f"{score:+.3f}")
        print(f"{sym:<12} ${spot:>11.2f} {bias_colored} {score_str} {conf:>6} "
              f"  {bull}B/{bear}S     {fbps:>+6.2f}  {oipct:>+5.1f}%  {cvdr:>9.3f}  {tls:>6.3f}")

    # ─── Detail view for top 3 ──────────────────────────────────────────────
    print("\n" + "─" * 110)
    print("  TOP SIGNALS BREAKDOWN (highest conviction first)")
    print("─" * 110)

    for r in ranked[:5]:
        c    = r["confluence"]
        print(f"\n  {r['symbol']}  ${r['spot']:.2f}  ➤  {c['bias']}  "
              f"(score={c['score']:+.3f}, {c['confidence']} confidence, "
              f"{c['bull_signals']} bull / {c['bear_signals']} bear signals)")
        for sig in c["signals"][:8]:  # top 8 by absolute score
            if sig["score"] != 0:
                bar  = "▲" * abs(sig["score"]) if sig["score"] > 0 else "▼" * abs(sig["score"])
                line = f"    {_color(sig['score'], bar)} [{sig['name']:<22}] {sig['label']}"
                print(line[:100])

        # Whale trades
        wt = r.get("whale_trades", [])
        if wt:
            print(f"\n    🐋 Whale Trades:")
            for w in wt[:3]:
                side_col = "\033[92m" if w["side"] == "BUY" else "\033[91m"
                print(f"       {side_col}{w['side']:4s}\033[0m  ${w['usd']:>12,.0f}  @ {w['price']:.2f}")

    # ─── Actionable Summary ──────────────────────────────────────────────────
    print("\n" + "─" * 110)
    print("  ACTIONABLE SUMMARY")
    print("─" * 110)

    strong_long  = [r for r in ranked if r["confluence"]["bias"] == "STRONG LONG"
                    and r["confluence"]["confidence"] in ("HIGH", "MEDIUM")]
    strong_short = [r for r in ranked if r["confluence"]["bias"] == "STRONG SHORT"
                    and r["confluence"]["confidence"] in ("HIGH", "MEDIUM")]

    if strong_long:
        syms = ", ".join(r["symbol"] for r in strong_long)
        print(f"\n  \033[92m[LONG BIAS]\033[0m  {syms}")
        for r in strong_long:
            fr = r.get("funding_bps", 0)
            oi = r.get("oi_change_2h", 0)
            print(f"    {r['symbol']}: {r['confluence']['bull_signals']} bullish signals, "
                  f"funding={fr:+.2f}bps, OI{oi:+.1f}%, CVD={r.get('cvd_ratio',1):.2f}x buy")

    if strong_short:
        syms = ", ".join(r["symbol"] for r in strong_short)
        print(f"\n  \033[91m[SHORT BIAS]\033[0m  {syms}")
        for r in strong_short:
            fr = r.get("funding_bps", 0)
            oi = r.get("oi_change_2h", 0)
            print(f"    {r['symbol']}: {r['confluence']['bear_signals']} bearish signals, "
                  f"funding={fr:+.2f}bps, OI{oi:+.1f}%")

    if not strong_long and not strong_short:
        print(f"\n  No high-conviction setups right now. Best opportunity:")
        if ranked:
            r = ranked[0]
            print(f"    {r['symbol']} — {r['confluence']['bias']} "
                  f"({r['confluence']['score']:+.3f}) with {r['confluence']['confidence']} confidence.")

    # ─── Carry Trade (Funding Harvest) Summary ──────────────────────────────
    print("\n" + "─" * 110)
    print("  FUNDING CARRY OPPORTUNITIES (best yield-to-capture)")
    print("─" * 110)
    carry = [(r["symbol"], r.get("funding_bps", 0), r["spot"])
             for r in results if "confluence" in r]
    carry = sorted(carry, key=lambda x: abs(x[1]), reverse=True)
    for sym, fbps, price in carry[:5]:
        ann = fbps * 1095 / 100
        direction = "Short perp + Long spot" if fbps > 0 else "Long perp + Short spot"
        print(f"    {sym:<12} {fbps:+.3f} bps/8h  ({ann:+.2f}% ann)  → {direction}")

    print("\n" + "═" * 110)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crypto Institutional Scanner")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--tf",     default=DEFAULT_TF,
                        help="Timeframe: 1m 5m 15m 1h 4h 1d (default: 1h)")
    parser.add_argument("--limit",  type=int, default=DEFAULT_LIMIT,
                        help="Number of OHLCV bars (default: 150)")
    parser.add_argument("--json",   action="store_true",
                        help="Also dump JSON results to crypto_scan_results.json")
    args = parser.parse_args()

    print(f"\n  Crypto Institutional Scanner")
    print(f"  Scanning {len(args.symbols)} symbols on {args.tf} timeframe ({args.limit} bars each)")
    print(f"  Strategies: Hurst · MACD · ICT OB · BOS/CHoCH · Wyckoff · S/D Zones · "
          f"IPDA · VP · CVD · AD · OI · Funding · OB Walls · Top Trader L/S\n")

    results = []
    for sym in args.symbols:
        try:
            r = scan_symbol(sym, args.tf, args.limit)
            results.append(r)
        except Exception as e:
            print(f"  ERROR on {sym}: {e}")
            results.append({"symbol": sym, "error": str(e)})

    print_report(results)

    if args.json:
        # Serialise (remove numpy types)
        def jsonify(obj):
            if isinstance(obj, (np.integer,)):   return int(obj)
            if isinstance(obj, (np.floating,)):  return float(obj)
            if isinstance(obj, (np.ndarray,)):   return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        out_path = "crypto_scan_results.json"
        # Strip raw ohlcv from json (too large)
        for r in results:
            r.pop("ohlcv_raw", None)
        with open(out_path, "w") as f:
            json.dump(results, f, default=jsonify, indent=2)
        print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
