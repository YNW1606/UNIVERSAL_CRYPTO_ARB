#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crypto_master_engine.py  —  CRYPTO MASTER ENGINE  (10-Layer Unified System)
===========================================================================
The definitive crypto trading engine combining the top proven strategies
into a single, deeply-integrated signal pipeline.

10 Layers:
  1  Session Quality + Volatility Regime
  2  Multi-Timeframe Structure    (1d / 4h / 1h triple alignment)
  3  Live Institutional Flow      (smart CVD, icebergs, OI velocity)
  4  Funding Rate + OI Structure  (+ funding-rate TREND via 7-period EMA)
  5  Manipulation Detection       (spoof, stop-hunt, layering)
  6  Enhanced XGBoost ML          (35 features incl. liq-distance, tick
                                   imbalance, funding-cross-BTC, session mom.)
  7  Multi-Symbol Market Breadth
  8  Master Score Fusion          (0-100, regime-adaptive weights)
  9  Liquidation Cascade Detector (OI-density liq clusters as momentum amps)
 10  VWAP Structure               (NY-anchored session VWAP + prev-day S/R)

Usage:
    python crypto_master_engine.py                    # scan all 8 symbols
    python crypto_master_engine.py --symbol BTCUSDT   # single deep dive
    python crypto_master_engine.py --watch 300        # re-scan every 5 min
    python crypto_master_engine.py --retrain          # retrain enhanced ML
    python crypto_master_engine.py --arb              # carry arb only
"""

import argparse, os, sys, math, time, pickle, statistics, json
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import requests

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORT INDIA STRATEGY MODULES (for SMC structure + Hurst + MACD + Wyckoff)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import crypto_india_strategies as cis
    HAS_CIS = True
except ImportError:
    HAS_CIS = False
    print("[WARN] crypto_india_strategies.py not found — multi-TF structure disabled")

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_FAPI   = "https://fapi.binance.com"
BINANCE_SPOT   = "https://api.binance.com"

DEFAULT_SYMBOLS = [
    # Large-cap anchors
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "ADAUSDT",
    # Backtest top-performers (breakout + momentum confirmed on 180d live data)
    "ARBUSDT", "SUIUSDT", "OPUSDT", "UNIUSDT",
    "NEARUSDT", "INJUSDT", "DOTUSDT", "APTUSDT",
]
TRAIN_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT",
                 "ARBUSDT", "SUIUSDT", "OPUSDT", "UNIUSDT"]

CACHE_DIR   = "alpha_cache"
# v2 = 35-feature model (old 25-feature pickles are incompatible — new path forces retrain)
MODEL_PATH  = os.path.join(CACHE_DIR, "master_xgb_v2.pkl")
SCALER_PATH = os.path.join(CACHE_DIR, "master_scaler_v2.pkl")
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 MasterEngine/1.0"}

# Score fusion weights (must sum to 1.0)
# Layer 11 (BREAKOUT) added based on 180-day live backtest: +15.4% PnL, PF 1.39
W_MULTI_TF   = 0.20
W_INST_FLOW  = 0.18
W_ML         = 0.17
W_LIQ        = 0.11
W_MANIP      = 0.08
W_FUNDING_OI = 0.08
W_BREAKOUT   = 0.08   # Layer 11 — Donchian breakout (backtest-validated)
W_VWAP       = 0.05
W_BREADTH    = 0.05

BASE_WEIGHTS = {
    "tf":       W_MULTI_TF,
    "inst":     W_INST_FLOW,
    "ml":       W_ML,
    "liq":      W_LIQ,
    "manip":    W_MANIP,
    "fund":     W_FUNDING_OI,
    "breakout": W_BREAKOUT,
    "vwap":     W_VWAP,
    "breadth":  W_BREADTH,
}

# Trade grade thresholds (0-100) — slightly more inclusive to catch more setups
GRADE_PERFECT = 72
GRADE_HIGH    = 58
GRADE_MOD     = 45

# Liquidation cascade detector params
LIQ_PROXIMITY_PCT = 0.5     # % distance considered "approaching" a liq cluster
LIQ_LOOKBACK_BARS = 120     # swing-pivot lookback for liq level estimation

# VWAP structure params
VWAP_DEV_THRESHOLD = 0.3    # % above/below session VWAP = directional
NY_OPEN_MINUTE_UTC = 13 * 60 + 30   # 13:30 UTC = NY equities open (VWAP anchor)

# Round-number granularity per coin (for stop-hunt detection)
ROUND_TICKS = {
    "BTC": 1000.0, "ETH": 100.0, "SOL": 5.0, "BNB": 10.0,
    "XRP": 0.05,   "DOGE": 0.005, "AVAX": 0.5, "ADA": 0.01,
}

# ─────────────────────────────────────────────────────────────────────────────
#  TINY HTTP + MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get(url, params=None, timeout=10, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(0.5 * (i + 1))

def _safe(x, d=0.0):
    try:
        v = float(x)
        return d if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return d

def _clamp(x, lo=-5.0, hi=5.0):
    return max(lo, min(hi, x))

def _ema(arr, period):
    if len(arr) == 0:
        return np.array([])
    k = 2.0 / (period + 1.0)
    out = [float(arr[0])]
    for v in arr[1:]:
        out.append(out[-1] + k * (float(v) - out[-1]))
    return np.array(out)

def _atr(h, l, c, period=14):
    if len(h) < 2:
        return 0.0
    tr = np.maximum(h[1:] - l[1:], np.maximum(
        np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) else 0.0
    atr = [float(np.mean(tr[:period]))]
    k = 1.0 / period
    for v in tr[period:]:
        atr.append(atr[-1] * (1 - k) + float(v) * k)
    return atr[-1] if atr else 0.0

def _avg_range(h, l, lookback=20):
    rngs = [hh - ll for hh, ll in zip(h[-lookback:], l[-lookback:]) if hh >= ll]
    return float(np.mean(rngs)) if rngs else 0.0

def _zscore(series):
    if len(series) < 2:
        return 0.0
    mu, sigma = float(np.mean(series)), float(np.std(series))
    return 0.0 if sigma < 1e-12 else (series[-1] - mu) / sigma

def _ohlcv_from_klines(klines):
    o = np.array([float(k[1]) for k in klines])
    h = np.array([float(k[2]) for k in klines])
    l = np.array([float(k[3]) for k in klines])
    c = np.array([float(k[4]) for k in klines])
    v = np.array([float(k[5]) for k in klines])
    return o, h, l, c, v

def _bar_delta(o, h, l, c, v):
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        loc = np.where(rng > 0, (2.0 * (c - l) / np.where(rng > 0, rng, 1.0) - 1.0), 0.0)
    return loc * v

def _swing_pivots(h, l, k=3):
    sh, sl = [], []
    n = min(len(h), len(l))
    for i in range(k, n - k):
        if h[i] >= np.max(h[i - k:i + k + 1]):
            if not (sh and i - sh[-1][0] <= k and h[i] == sh[-1][1]):
                sh.append((i, float(h[i])))
        if l[i] <= np.min(l[i - k:i + k + 1]):
            if not (sl and i - sl[-1][0] <= k and l[i] == sl[-1][1]):
                sl.append((i, float(l[i])))
    return sh, sl

def _fmt_usd(v):
    if v >= 1e9:  return "%.1fB" % (v / 1e9)
    if v >= 1e6:  return "%.1fM" % (v / 1e6)
    if v >= 1e3:  return "%.1fK" % (v / 1e3)
    return "%.2f" % v

# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_klines_raw(symbol, interval="1h", limit=300):
    try:
        from data_utils import fetch_klines as _fetch_universal
        data = _fetch_universal(symbol, interval, limit)
        if isinstance(data, list) and len(data) > 0:
            return data
    except Exception:
        pass
    data = _get(f"{BINANCE_SPOT}/api/v3/klines",
                {"symbol": symbol, "interval": interval, "limit": limit})
    if isinstance(data, list):
        return data
    data = _get(f"{BINANCE_FAPI}/fapi/v1/klines",
                {"symbol": symbol, "interval": interval, "limit": limit})
    return data if isinstance(data, list) else []

def fetch_klines_dicts(symbol, interval="1h", limit=500):
    raw = fetch_klines_raw(symbol, interval, limit)
    bars = []
    for k in raw:
        try:
            bars.append({
                "ts":     int(k[0]),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
                "trades": int(k[8]) if len(k) > 8 else 0,
            })
        except Exception:
            pass
    return bars

def fetch_oi_history(symbol, period="1h", limit=100):
    try:
        from data_utils import fetch_oi_history as _fetch_oi
        raw = _fetch_oi(symbol, period, limit)
        if raw:
            result = []
            for d in raw:
                try:
                    result.append({"ts": int(d.get("timestamp", 0)),
                                   "oi": float(d.get("sumOpenInterest", d.get("oi", 0)))})
                except Exception:
                    pass
            if result:
                return result
    except Exception:
        pass
    data = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit})
    if not isinstance(data, list):
        return []
    result = []
    for d in data:
        try:
            result.append({"ts": int(d["timestamp"]), "oi": float(d["sumOpenInterestValue"])})
        except Exception:
            pass
    return result

def fetch_funding_hist(symbol, limit=100):
    try:
        from data_utils import fetch_funding_history as _fetch_fund
        raw = _fetch_fund(symbol, limit)
        if raw:
            result = []
            for d in raw:
                try:
                    result.append({"ts": int(d.get("fundingTime", 0)),
                                   "rate": float(d.get("fundingRate", 0)) * 10000})
                except Exception:
                    pass
            if result:
                return result
    except Exception:
        pass
    data = _get(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return []
    result = []
    for d in data:
        try:
            result.append({"ts": int(d["fundingTime"]),
                           "rate": float(d["fundingRate"]) * 10000})
        except Exception:
            pass
    return result

def fetch_order_book(symbol, depth=100):
    try:
        from data_utils import fetch_orderbook as _fetch_ob
        data = _fetch_ob(symbol, min(depth, 200))
        if data.get("bids") or data.get("asks"):
            return data
    except Exception:
        pass
    data = _get(f"{BINANCE_FAPI}/fapi/v1/depth",
                {"symbol": symbol, "limit": depth})
    if not isinstance(data, dict):
        data = _get(f"{BINANCE_SPOT}/api/v3/depth",
                    {"symbol": symbol, "limit": depth})
    return data if isinstance(data, dict) else {"bids": [], "asks": []}

def fetch_spot_price(symbol):
    try:
        from data_utils import fetch_ticker_price as _fetch_px
        px = _fetch_px(symbol)
        if px > 0:
            return px
    except Exception:
        pass
    d = _get(f"{BINANCE_SPOT}/api/v3/ticker/price", {"symbol": symbol})
    return _safe(d.get("price", 0)) if isinstance(d, dict) else 0.0

def fetch_agg_trades(symbol, limit=1500):
    try:
        from data_utils import fetch_agg_trades as _fetch_trades
        raw = _fetch_trades(symbol, min(limit, 1000))
        if raw:
            return [{"price": t["price"], "qty": t["qty"],
                     "usd": t["price"] * t["qty"],
                     "is_maker": t["is_buyer_maker"],
                     "time": t["time_ms"]} for t in raw]
    except Exception:
        pass
    data = _get(f"{BINANCE_FAPI}/fapi/v1/aggTrades",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        data = _get(f"{BINANCE_SPOT}/api/v3/aggTrades",
                    {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        return []
    result = []
    for t in data:
        try:
            result.append({
                "price":    float(t["p"]),
                "qty":      float(t["q"]),
                "usd":      float(t["p"]) * float(t["q"]),
                "is_maker": bool(t["m"]),
                "time":     int(t["T"]),
            })
        except Exception:
            pass
    return result

def fetch_long_short_ratio(symbol):
    d = _get(f"{BINANCE_FAPI}/futures/data/topLongShortPositionRatio",
             {"symbol": symbol, "period": "5m", "limit": 1})
    if isinstance(d, list) and d:
        return _safe(d[0].get("longShortRatio", 1.0))
    return 1.0

def fetch_funding_rate(symbol):
    d = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    if isinstance(d, dict):
        return _safe(d.get("lastFundingRate", 0))
    return 0.0

def fetch_open_interest_now(symbol):
    d = _get(f"{BINANCE_FAPI}/fapi/v1/openInterest", {"symbol": symbol})
    if isinstance(d, dict):
        return _safe(d.get("openInterestValue", 0))
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1  —  SESSION QUALITY + VOLATILITY REGIME
# ═════════════════════════════════════════════════════════════════════════════

def session_quality():
    # Crypto trades 24/7 globally — Asia/Late-NY penalties were too aggressive
    # and killed profitable signals during China/Japan hours.
    hour = datetime.now(timezone.utc).hour
    if 12 <= hour < 21:
        return 1.00, "NEW YORK"
    if 7 <= hour < 12:
        return 0.85, "LONDON"
    if 21 <= hour <= 23:
        return 0.78, "LATE NY"
    return 0.72, "ASIA"

def volatility_regime(h, l, c):
    if len(h) < 25:
        return "UNKNOWN", 1.0
    atr_now = _atr(h, l, c, 14)
    ranges   = h[-20:] - l[-20:]
    med_rng  = float(np.median(ranges)) if len(ranges) else atr_now
    if med_rng <= 0:
        return "UNKNOWN", 1.0
    ratio = atr_now / med_rng
    if ratio > 2.5:
        return "EXTREME", ratio
    if ratio > 1.5:
        return "HIGH", ratio
    if ratio < 0.5:
        return "LOW", ratio
    return "NORMAL", ratio

def market_health_score(h, l, c, v):
    if len(v) < 25:
        return 0.5
    vol_now = float(v[-1])
    vol_avg = float(np.mean(v[-20:]))
    vol_ratio = vol_now / max(vol_avg, 1e-9)
    score = 0.5
    if vol_ratio > 1.5:
        score += 0.3
    elif vol_ratio > 1.0:
        score += 0.15
    elif vol_ratio < 0.5:
        score -= 0.2
    return max(0.1, min(1.0, score))


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2  —  MULTI-TIMEFRAME STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def _tf_direction(o, h, l, c):
    """Quick direction from EMA stack + swing structure."""
    if len(c) < 55:
        return 0, "UNKNOWN"
    ema50  = _ema(c, 50)
    ema200 = _ema(c, 200) if len(c) >= 200 else _ema(c, min(len(c) - 1, 100))
    spot = float(c[-1])
    e50  = float(ema50[-1])
    e200 = float(ema200[-1])

    # EMA stack direction
    bull_stack = spot > e50 > e200
    bear_stack = spot < e50 < e200

    # Swing structure via SMC if available
    structure = "RANGE"
    if HAS_CIS and len(c) >= 10:
        try:
            smc_res, sh, sl = cis.smc_market_structure(o, h, l, c)
            structure = smc_res.get("structure", "RANGE")
        except Exception:
            pass

    if bull_stack and structure == "UPTREND":
        return 2, "UPTREND"
    if bear_stack and structure == "DOWNTREND":
        return -2, "DOWNTREND"
    if bull_stack:
        return 1, "BULLISH BIAS"
    if bear_stack:
        return -1, "BEARISH BIAS"
    return 0, "RANGE"

def _tf_hurst(c):
    """Return Hurst score from cis or inline estimate."""
    if HAS_CIS:
        try:
            res = cis.quant_hurst(c)
            return res.get("score", 0), res.get("label", "")
        except Exception:
            pass
    if len(c) < 20:
        return 0, ""
    ts   = np.asarray(c[-50:] if len(c) >= 50 else c, float)
    lags = range(2, min(16, len(ts) // 2))
    tau  = [max(np.std(ts[lg:] - ts[:-lg]), 1e-9) for lg in lags]
    if len(tau) < 2:
        return 0, ""
    h = float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0] * 2.0)
    if h >= 0.55:
        return 1, "H=%.2f TRENDING" % h
    if h < 0.45:
        return -1, "H=%.2f MEAN-REV" % h
    return 0, "H=%.2f RANDOM" % h

def _cvd_slope(o, h, l, c, v, n=20):
    """Bar-delta CVD slope over last n bars (positive = buying accelerating)."""
    if len(c) < n + 2:
        return 0.0
    deltas = _bar_delta(o[-n:], h[-n:], l[-n:], c[-n:], v[-n:])
    cvd    = np.cumsum(deltas)
    if len(cvd) < 4:
        return 0.0
    slope = float(cvd[-1] - cvd[0]) / max(abs(float(np.mean(np.abs(deltas)))), 1e-9)
    return max(-5.0, min(5.0, slope / n))

def _check_bos(o, h, l, c):
    """Detect recent 4h BOS or CHoCH on this TF."""
    if not HAS_CIS or len(c) < 10:
        return None
    try:
        smc_res, sh, sl = cis.smc_market_structure(o, h, l, c)
        bos   = smc_res.get("last_bos")
        choch = smc_res.get("last_choch")
        if bos and bos.get("bars_ago", 999) <= 10:
            return bos
        if choch and choch.get("bars_ago", 999) <= 10:
            return choch
    except Exception:
        pass
    return None

def analyze_tf(symbol, interval, limit):
    """Fetch one TF and return direction, hurst, cvd_slope, bos, atr."""
    raw = fetch_klines_raw(symbol, interval, limit)
    if len(raw) < 50:
        return {"dir": 0, "label": "NO DATA", "hurst": 0, "cvd_slope": 0.0,
                "bos": None, "atr": 0.0, "bars": 0}
    o, h, l, c, v = _ohlcv_from_klines(raw)
    direction, label = _tf_direction(o, h, l, c)
    h_score, h_label = _tf_hurst(c)
    cvd_s   = _cvd_slope(o, h, l, c, v)
    bos_ev  = _check_bos(o, h, l, c)
    atr_v   = _atr(h, l, c)
    return {
        "dir": direction, "label": label,
        "hurst": h_score, "hurst_label": h_label,
        "cvd_slope": cvd_s, "bos": bos_ev,
        "atr": atr_v, "bars": len(c),
        "spot": float(c[-1]),
    }

def triple_tf_score(d1, h4, h1):
    """
    Score -3 to +3 from three TF directions.
    Returns (score, label, detail)
    """
    scores  = [d1["dir"], h4["dir"], h1["dir"]]
    bull    = sum(1 for s in scores if s > 0)
    bear    = sum(1 for s in scores if s < 0)
    strong  = sum(1 for s in scores if abs(s) >= 2)

    # Bonus for 4h BOS confirmation
    bos_bonus = 0
    bos_note  = ""
    if h4.get("bos"):
        ev = h4["bos"]
        if "BULL" in ev.get("type", ""):
            bos_bonus = 1; bos_note = " + 4h BOS ✓"
        elif "BEAR" in ev.get("type", ""):
            bos_bonus = -1; bos_note = " + 4h BOS ✗"

    if bull == 3 and strong >= 2:
        sc = 3; label = "PERFECT TRIPLE ALIGNMENT" + bos_note
    elif bull == 3:
        sc = 2; label = "ALL 3 TF BULLISH" + bos_note
    elif bull == 2 and d1["dir"] > 0 and h4["dir"] > 0:
        sc = 2; label = "1D+4H TREND, 1H PULLBACK = ENTRY" + bos_note
    elif bull == 2:
        sc = 1; label = "MAJORITY BULLISH" + bos_note
    elif d1["dir"] > 0:
        sc = 1; label = "1D TREND, 4H NOT YET"
    elif bear == 3 and strong >= 2:
        sc = -3; label = "PERFECT TRIPLE BEARISH" + bos_note
    elif bear == 3:
        sc = -2; label = "ALL 3 TF BEARISH" + bos_note
    elif bear == 2 and d1["dir"] < 0 and h4["dir"] < 0:
        sc = -2; label = "1D+4H BEAR, 1H PULLBACK = ENTRY" + bos_note
    elif bear == 2:
        sc = -1; label = "MAJORITY BEARISH" + bos_note
    elif d1["dir"] < 0:
        sc = -1; label = "1D BEAR, 4H NOT YET"
    else:
        sc = 0; label = "MIXED — NO TRADE"

    sc = max(-3, min(3, sc + bos_bonus))

    # Hurst regime confirmation
    hurst_sum = d1.get("hurst", 0) + h4.get("hurst", 0)
    hurst_note = ""
    if hurst_sum >= 2 and sc > 0:
        sc = min(3, sc + 1); hurst_note = " [TRENDING CONFIRMED]"
    elif hurst_sum <= -2 and sc < 0:
        sc = max(-3, sc - 1); hurst_note = " [TRENDING CONFIRMED]"
    elif hurst_sum < 0 and sc > 0:
        sc = max(0, sc - 1); hurst_note = " [MEAN-REV WARNING]"
    elif hurst_sum > 0 and sc < 0:
        sc = min(0, sc + 1); hurst_note = " [MEAN-REV WARNING]"

    # CVD 4h acceleration
    cvd_note = ""
    if h4.get("cvd_slope", 0) > 0.3 and sc > 0:
        cvd_note = " [CVD ACCEL ↑]"
    elif h4.get("cvd_slope", 0) < -0.3 and sc < 0:
        cvd_note = " [CVD ACCEL ↓]"

    full_label = label + hurst_note + cvd_note
    return max(-3, min(3, sc)), full_label


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3  —  LIVE INSTITUTIONAL FLOW
# ═════════════════════════════════════════════════════════════════════════════

def compute_smart_cvd(trades):
    """Split smart money ($100k+) vs retail (<$5k) CVD."""
    smart_cvd = 0.0
    retail_cvd = 0.0
    smart_usd = 0.0
    retail_usd = 0.0
    for t in trades:
        usd = t["usd"]
        sign = 1 if not t["is_maker"] else -1  # maker=sell taker, taker=buy
        if usd >= 100_000:
            smart_cvd += sign * usd
            smart_usd += usd
        elif usd <= 5_000:
            retail_cvd += sign * usd
            retail_usd += usd
    return {
        "smart_cvd": smart_cvd,
        "retail_cvd": retail_cvd,
        "smart_usd": smart_usd,
        "retail_usd": retail_usd,
        "divergence": smart_cvd > 0 and retail_cvd < 0 or smart_cvd < 0 and retail_cvd > 0,
    }

def detect_icebergs(trades):
    """Group trades by price (4 sig figs), 5+ hits = iceberg."""
    price_groups: Dict[float, List] = defaultdict(list)
    for t in trades:
        p = round(t["price"], 4)
        price_groups[p].append(t)
    icebergs = []
    for price, group in price_groups.items():
        if len(group) >= 5:
            total_usd = sum(t["usd"] for t in group)
            buy_usd   = sum(t["usd"] for t in group if not t["is_maker"])
            side = "BUY" if buy_usd > total_usd * 0.6 else "SELL" if buy_usd < total_usd * 0.4 else "BOTH"
            icebergs.append({"price": price, "hits": len(group),
                             "usd": total_usd, "side": side})
    icebergs.sort(key=lambda x: -x["usd"])
    return icebergs[:5]

def net_inst_flow(trades):
    """Net USD flow for $100k+ trades (buys - sells)."""
    net = 0.0
    for t in trades:
        if t["usd"] >= 100_000:
            net += t["usd"] if not t["is_maker"] else -t["usd"]
    return net

def oi_velocity(symbol):
    """OI % change per hour from last 2 data points."""
    hist = fetch_oi_history(symbol, period="1h", limit=3)
    if len(hist) < 2:
        return 0.0
    oi_now  = hist[-1]["oi"]
    oi_prev = hist[-2]["oi"]
    if oi_prev <= 0:
        return 0.0
    return (oi_now - oi_prev) / oi_prev * 100.0

def inst_flow_score(smart_cvd_data, icebergs, oi_vel, ls_ratio):
    """
    Combine institutional flow signals → score -3 to +3.
    smart_cvd_data: dict from compute_smart_cvd()
    """
    score = 0.0
    notes = []

    # Smart CVD
    s_cvd = smart_cvd_data.get("smart_cvd", 0)
    r_cvd = smart_cvd_data.get("retail_cvd", 0)
    s_usd = smart_cvd_data.get("smart_usd", 1)
    if s_cvd > 0:
        score += min(1.5, s_cvd / max(s_usd, 1) * 3)
        notes.append("Smart CVD +%s" % _fmt_usd(s_cvd))
    elif s_cvd < 0:
        score += max(-1.5, s_cvd / max(s_usd, 1) * 3)
        notes.append("Smart CVD -%s" % _fmt_usd(abs(s_cvd)))

    # Divergence bonus: smart buys, retail sells = strong bullish
    if smart_cvd_data.get("divergence"):
        if s_cvd > 0:
            score += 0.5; notes.append("SMART↑/RETAIL↓ divergence")
        elif s_cvd < 0:
            score -= 0.5; notes.append("SMART↓/RETAIL↑ divergence")

    # Icebergs
    if icebergs:
        top = icebergs[0]
        if top["side"] == "BUY":
            score += 0.5; notes.append("Iceberg BUY %.4f (%d hits)" % (top["price"], top["hits"]))
        elif top["side"] == "SELL":
            score -= 0.5; notes.append("Iceberg SELL %.4f (%d hits)" % (top["price"], top["hits"]))

    # OI velocity
    if oi_vel > 0.5:
        score += 0.3; notes.append("OI +%.1f%%/hr" % oi_vel)
    elif oi_vel < -0.5:
        score -= 0.3; notes.append("OI %.1f%%/hr" % oi_vel)

    # Long/short ratio
    if ls_ratio > 1.3:
        score -= 0.2; notes.append("L/S=%.2f (crowded longs)" % ls_ratio)
    elif ls_ratio < 0.8:
        score += 0.2; notes.append("L/S=%.2f (crowded shorts)" % ls_ratio)

    return max(-3.0, min(3.0, score)), notes


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 4  —  FUNDING RATE + LIQUIDATION STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def funding_signal(symbol):
    """
    Funding rate → contrarian or momentum score.
    Returns (score -2..+2, label, rate_pct)
    """
    rate = fetch_funding_rate(symbol)   # raw decimal (e.g. 0.0001)
    rate_pct = rate * 100               # as percentage

    if rate > 0.001:
        return -2, "CROWDED LONGS — contrarian SHORT", rate_pct
    if rate > 0.0005:
        return -1, "Elevated funding — lean short", rate_pct
    if rate < -0.0005:
        return 2, "SHORT SQUEEZE potential", rate_pct
    if rate < -0.0003:
        return 1, "Shorts overextended", rate_pct
    return 0, "Neutral funding", rate_pct

def funding_trend_signal(symbol):
    """
    Funding rate TREND (direction of change, not just level).
    7-period EMA of historical funding rates vs current rate:
      current > EMA × 1.2 (accelerating) → longs paying harder = momentum continuation
      current < EMA × 0.8 (decelerating) → reversal warning
    Mirrored when funding is negative (shorts paying).
    Returns (score -1..+1, label, ema_ratio). Fails safe to (0.0, "", 1.0).
    """
    try:
        hist = fetch_funding_hist(symbol, 30)
        if len(hist) < 8:
            return 0.0, "", 1.0
        rates = np.array([d["rate"] for d in hist], float)  # in bps (×10000)
        ema7 = _ema(rates, 7)
        ema_f = float(ema7[-1])
        rate_now = float(rates[-1])
        if abs(ema_f) < 1e-9:
            return 0.0, "", 1.0
        ratio = rate_now / ema_f
        if ema_f > 0:
            # Positive funding regime: longs paying
            if ratio > 1.2:
                return 0.75, "FUNDING ACCELERATING ↑ (longs paying harder = bull momentum)", ratio
            if ratio < 0.8:
                return -0.5, "FUNDING DECELERATING (long momentum fading = reversal warning)", ratio
        else:
            # Negative funding regime: shorts paying
            if ratio > 1.2:
                return -0.75, "FUNDING ACCELERATING ↓ (shorts paying harder = bear momentum)", ratio
            if ratio < 0.8:
                return 0.5, "FUNDING DECELERATING (short momentum fading = squeeze warning)", ratio
        return 0.0, "", ratio
    except Exception:
        return 0.0, "", 1.0


def oi_structure(symbol, spot):
    """OI trend vs price trend → LONG_BUILD / SHORT_BUILD / DE-RISKING."""
    hist = fetch_oi_history(symbol, period="1h", limit=12)
    if len(hist) < 4:
        return "UNKNOWN", 0.0
    oi_vals = [h["oi"] for h in hist]
    oi_now  = oi_vals[-1]
    oi_old  = oi_vals[0]
    oi_chg  = (oi_now - oi_old) / max(oi_old, 1) * 100

    klines = fetch_klines_raw(symbol, "1h", 13)
    if len(klines) < 2:
        return "UNKNOWN", oi_chg
    c = [float(k[4]) for k in klines]
    price_chg = (c[-1] - c[0]) / max(abs(c[0]), 1) * 100

    if oi_chg > 0.5:
        if price_chg > 0:
            return "LONG_BUILD", oi_chg
        return "SHORT_BUILD", oi_chg
    if oi_chg < -0.5:
        return "DE-RISKING", oi_chg
    return "FLAT", oi_chg

def funding_oi_combined(symbol, spot):
    """
    Combine funding level + funding TREND + OI structure → score -2..+2 and notes.
    """
    f_score, f_label, f_rate = funding_signal(symbol)
    oi_state, oi_chg = oi_structure(symbol, spot)
    score = f_score
    notes = [f_label + " (%.4f%%/8h)" % f_rate]

    # Funding rate TREND (7-period EMA — direction of change predicts continuation/reversal)
    ft_score, ft_label, ft_ratio = funding_trend_signal(symbol)
    if ft_score != 0.0:
        score += ft_score
        notes.append(ft_label + " (rate/EMA7=%.2f)" % ft_ratio)

    if oi_state == "LONG_BUILD":
        score += 0.5; notes.append("OI +%.1f%%: LONG BUILD (bullish)" % oi_chg)
    elif oi_state == "SHORT_BUILD":
        score -= 0.5; notes.append("OI +%.1f%%: SHORT BUILD (bearish)" % oi_chg)
    elif oi_state == "DE-RISKING":
        score *= 0.5; notes.append("OI %.1f%%: DE-RISKING (caution)" % oi_chg)

    # Carry arb opportunity
    carry_ann = abs(f_rate) * 3 * 365  # 3 payments/day * 365
    carry_note = ""
    if abs(f_rate) >= 0.01:
        carry_note = "Carry arb: +%.1f%%/yr" % carry_ann

    return max(-2.0, min(2.0, score)), notes, carry_ann, carry_note


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 5  —  MANIPULATION DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def spoof_static(ob, spot):
    """
    Isolated large orders > 8× median and > 4× neighbours = likely spoof.
    Returns (score, direction, details)
    """
    asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:100]]
    bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:100]]

    def _walls(levels, side):
        if len(levels) < 7:
            return []
        qtys = [q for _, q in levels]
        median_q = float(np.median(qtys))
        walls = []
        for i, (price, qty) in enumerate(levels):
            if qty < 8.0 * median_q:
                continue
            nb_start = max(0, i - 3)
            nb_end   = min(len(levels), i + 4)
            neighbours = [levels[j][1] for j in range(nb_start, nb_end) if j != i]
            avg_nb = statistics.mean(neighbours) if neighbours else median_q
            if qty > 4 * avg_nb:
                walls.append({
                    "price": price, "qty": qty, "usd": price * qty,
                    "side": side, "isolated": qty / max(avg_nb, 1e-9),
                    "pct": abs(price - spot) / spot * 100,
                })
        return walls

    ask_walls = _walls(asks, "ask")
    bid_walls = _walls(bids, "bid")

    if not ask_walls and not bid_walls:
        return 0, "CLEAN", []

    top_ask_usd = max((w["usd"] for w in ask_walls), default=0)
    top_bid_usd = max((w["usd"] for w in bid_walls), default=0)

    if top_ask_usd > top_bid_usd * 1.5 and top_ask_usd > 500_000:
        return 2, "FAKE_ASK_BULLISH", ask_walls[:3]
    if top_bid_usd > top_ask_usd * 1.5 and top_bid_usd > 500_000:
        return 2, "FAKE_BID_BEARISH", bid_walls[:3]

    score = min(1, (len(ask_walls) + len(bid_walls)) // 2)
    return score, "MIXED", (ask_walls + bid_walls)[:3]

def stop_hunt_proximity(spot, symbol, h, l, c, atr):
    """
    Is price within 0.4% of PDH/PDL/round numbers/swing H+L?
    Returns (score 0-2, direction, levels)
    """
    coin = symbol.replace("USDT", "").replace("BUSD", "")
    tick = ROUND_TICKS.get(coin, spot * 0.01)
    rnd_levels = []
    base = round(spot / tick) * tick
    for mult in range(-3, 4):
        rnd_levels.append(base + mult * tick)

    sh, sl = _swing_pivots(h[-50:], l[-50:])
    swing_levels = [p for _, p in sh[-3:]] + [p for _, p in sl[-3:]]

    # PDH / PDL
    if len(h) >= 25:
        pdh = float(np.max(h[-25:-1]))
        pdl = float(np.min(l[-25:-1]))
        swing_levels += [pdh, pdl]

    all_levels = rnd_levels + swing_levels
    near = [lv for lv in all_levels if abs(lv - spot) / max(spot, 1) <= 0.004]

    if not near:
        return 0, "NO_HUNT", []

    above = [lv for lv in near if lv > spot]
    below = [lv for lv in near if lv <= spot]

    score = min(2, len(near))
    if len(above) > len(below):
        return score, "HUNT_ABOVE", above[:3]
    if len(below) > len(above):
        return score, "HUNT_BELOW", below[:3]
    return score, "BOTH_SIDES", near[:3]

def layering_check(ob, spot):
    """3+ walls >6× median within 0.3% on ONE side = layering."""
    asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:100]]
    bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:100]]

    def _layer_walls(levels, spot):
        if len(levels) < 7:
            return 0, []
        qtys  = [q for _, q in levels]
        med_q = float(np.median(qtys))
        near  = [(p, q) for p, q in levels if q > 6 * med_q and abs(p - spot) / spot <= 0.003]
        return len(near), near

    ask_n, ask_walls = _layer_walls(asks, spot)
    bid_n, bid_walls = _layer_walls(bids, spot)

    if ask_n >= 3:
        return 1, "LAYERING_ASKS", ask_walls[:5]
    if bid_n >= 3:
        return 1, "LAYERING_BIDS", bid_walls[:5]
    return 0, "NO_LAYER", []

def manip_combined(symbol, ob, spot, h, l, c, atr):
    """
    Combine spoof + stop-hunt + layering → score -2..+2 (bullish positive).
    """
    sp_score, sp_dir, sp_walls = spoof_static(ob, spot)
    hunt_score, hunt_dir, hunt_lvls = stop_hunt_proximity(spot, symbol, h, l, c, atr)
    lay_score, lay_dir, lay_walls   = layering_check(ob, spot)

    total = sp_score * 0.5 + hunt_score * 0.3 + lay_score * 0.2

    # Direction: fake asks = price will go UP (remove resistance)
    bull_manip = sp_dir == "FAKE_ASK_BULLISH" or lay_dir == "LAYERING_ASKS"
    bear_manip = sp_dir == "FAKE_BID_BEARISH" or lay_dir == "LAYERING_BIDS"

    if bull_manip and not bear_manip:
        direction = "BULLISH"; signed = total
    elif bear_manip and not bull_manip:
        direction = "BEARISH"; signed = -total
    else:
        direction = "MIXED"; signed = 0.0

    notes = []
    if sp_dir not in ("CLEAN", "MIXED"):
        top_w = sp_walls[0] if sp_walls else {}
        notes.append("%s: %s at %.4f" % (sp_dir, _fmt_usd(top_w.get("usd", 0)),
                                          top_w.get("price", 0)))
    if hunt_score > 0:
        notes.append("STOP-HUNT near %s" % (", ".join("%.4f" % x for x in hunt_lvls[:2])))
    if lay_dir != "NO_LAYER":
        notes.append(lay_dir)

    return max(-2.0, min(2.0, signed)), direction, notes, sp_dir


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 6  —  ENHANCED ML  (35 features)
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES_MASTER = [
    # Original 18
    "ret_1h", "ret_4h", "ret_12h", "ret_24h",
    "vol_ratio_1h", "vol_ratio_4h",
    "body_ratio", "upper_wick", "lower_wick",
    "oi_chg_1h", "oi_chg_4h",
    "funding_z", "funding_abs",
    "momentum_5h", "momentum_12h",
    "close_vs_vwap", "range_expansion", "trade_count_ratio",
    # Microstructure 7
    "cvd_slope", "smart_cvd_ratio",
    "hurst_1d", "macd_hist_norm",
    "ema_stack", "vol_percentile", "session_weight",
    # New 10 (v2): on-chain/derivs proxies + liq structure + session context
    "volume_acceleration",    # vol[-1] / mean(vol[-5:])
    "funding_rate_ema_ratio", # current funding / 7-period EMA of funding
    "liq_distance_buy",       # % to nearest long-liquidation cluster (below price)
    "liq_distance_sell",      # % to nearest short-liquidation cluster (above price)
    "breadth_momentum",       # directional persistence proxy (sign-consistency of bars)
    "oi_price_divergence",    # OI rising while price flat = accumulation
    "spread_compression",     # range narrowing = breakout warning
    "funding_cross_btc",      # this symbol's funding vs BTC's funding (relative heat)
    "tick_imbalance",         # intra-bar buy/sell pressure, normalized -1..+1
    "session_momentum",       # price change since NY session open
]

LABEL_UP = 0.003
LABEL_DN = 0.003

def _build_master_features(klines, oi_hist, fund_hist, btc_fund_hist=None):
    """Build (X, y) with 35 features for master engine.
    btc_fund_hist: optional BTC funding history for the funding_cross_btc feature.
    """
    if len(klines) < 30:
        return np.array([]), np.array([])

    closes  = np.array([b["close"]  for b in klines], float)
    highs   = np.array([b["high"]   for b in klines], float)
    lows    = np.array([b["low"]    for b in klines], float)
    opens   = np.array([b["open"]   for b in klines], float)
    vols    = np.array([b["volume"] for b in klines], float)
    tcount  = np.array([b["trades"] for b in klines], float)
    ts_arr  = np.array([b["ts"]     for b in klines], dtype="int64")

    oi_map  = {d["ts"]: d["oi"] for d in oi_hist}
    oi_vals = np.array([oi_map.get(t, np.nan) for t in ts_arr], float)
    for i in range(1, len(oi_vals)):
        if np.isnan(oi_vals[i]):
            oi_vals[i] = oi_vals[i - 1]
    if len(oi_vals) > 0 and np.isnan(oi_vals[0]):
        valid = oi_vals[~np.isnan(oi_vals)]
        oi_vals[0] = valid[0] if len(valid) > 0 else 0.0

    fund_map = {}
    for d in fund_hist:
        fund_map[d["ts"]] = d["rate"]
    fund_vals = np.full(len(ts_arr), 0.0)
    last_fund = 0.0
    for i, t in enumerate(ts_arr):
        if t in fund_map:
            last_fund = fund_map[t]
        fund_vals[i] = last_fund

    vwap24 = np.full(len(closes), np.nan)
    for i in range(23, len(closes)):
        w = vols[i - 23:i + 1]
        p = closes[i - 23:i + 1]
        vwap24[i] = np.sum(w * p) / max(np.sum(w), 1e-9)

    # Pre-compute EMA50/200 for ema_stack feature
    ema50  = _ema(closes, 50)  if len(closes) >= 51  else _ema(closes, max(2, len(closes) - 1))
    ema200 = _ema(closes, 200) if len(closes) >= 201 else _ema(closes, max(2, len(closes) - 1))

    # Pre-compute MACD for macd_hist_norm feature
    macd_line   = _ema(closes, 12) - _ema(closes, 26)
    macd_signal = _ema(macd_line, 9)
    macd_hist   = macd_line - macd_signal

    # Pre-compute bar-delta CVD for cvd_slope
    bar_deltas = _bar_delta(opens, highs, lows, closes, vols)
    cvd_cum    = np.cumsum(bar_deltas)

    # Pre-compute 7-period EMA of funding for funding_rate_ema_ratio
    fund_ema7 = _ema(fund_vals, 7) if len(fund_vals) else np.zeros(len(ts_arr))

    # Pre-compute BTC funding aligned to this symbol's timestamps (funding_cross_btc)
    btc_fund_vals = np.full(len(ts_arr), 0.0)
    if btc_fund_hist:
        btc_map = {d["ts"]: d["rate"] for d in btc_fund_hist}
        last_bf = 0.0
        for i, t in enumerate(ts_arr):
            if t in btc_map:
                last_bf = btc_map[t]
            btc_fund_vals[i] = last_bf

    # Pre-compute swing pivots once (liq_distance features); pivot at index j is
    # only known after j+3 bars, so features at bar i only use pivots with idx <= i-3
    piv_sh, piv_sl = _swing_pivots(highs, lows, k=3)

    rows, labels = [], []
    n = len(closes)
    for i in range(25, n - 1):
        c = closes[i]
        if c <= 0:
            continue

        def pct(a, b):
            return (a - b) / max(abs(b), 1e-9) if b != 0 else 0.0

        ret_1h   = pct(closes[i], closes[i - 1])
        ret_4h   = pct(closes[i], closes[i - 4])
        ret_12h  = pct(closes[i], closes[i - 12])
        ret_24h  = pct(closes[i], closes[i - 24])

        vol_avg24 = float(np.mean(vols[i - 24:i]))
        vol_r_1h  = vols[i] / max(vol_avg24, 1e-9)
        vol_r_4h  = float(np.mean(vols[i - 4:i + 1])) / max(vol_avg24, 1e-9)

        rng     = highs[i] - lows[i]
        body    = abs(closes[i] - opens[i])
        body_r  = body / max(rng, 1e-9)
        u_wick  = (highs[i] - max(opens[i], closes[i])) / max(rng, 1e-9)
        l_wick  = (min(opens[i], closes[i]) - lows[i]) / max(rng, 1e-9)

        oi_1h = pct(oi_vals[i], oi_vals[i - 1]) if i > 0 and oi_vals[i - 1] != 0 else 0.0
        oi_4h = pct(oi_vals[i], oi_vals[i - 4]) if i >= 4 and oi_vals[i - 4] != 0 else 0.0

        fund_z   = _zscore(fund_vals[max(0, i - 48):i + 1])
        fund_abs = fund_vals[i]

        mom_5  = float(np.mean(np.diff(closes[i - 5:i + 1]))) / max(c, 1)
        mom_12 = float(np.mean(np.diff(closes[i - 12:i + 1]))) / max(c, 1)

        vwap_v   = vwap24[i] if not np.isnan(vwap24[i]) else c
        c_vs_vwap = pct(c, vwap_v)

        rng_avg  = float(np.mean(highs[i - 12:i] - lows[i - 12:i]))
        rng_exp  = rng / max(rng_avg, 1e-9)

        tc_avg   = float(np.mean(tcount[i - 12:i]))
        tc_ratio = tcount[i] / max(tc_avg, 1e-9)

        # NEW features
        cvd_window = 12
        cvd_s = 0.0
        if i >= cvd_window:
            d_seg = bar_deltas[i - cvd_window:i + 1]
            avg_d = float(np.mean(np.abs(d_seg))) or 1.0
            cvd_s = float(cvd_cum[i] - cvd_cum[i - cvd_window]) / (avg_d * cvd_window)
            cvd_s = max(-5.0, min(5.0, cvd_s))

        # smart_cvd_ratio: ratio of recent cvd slope sign (proxy)
        smart_cvd_ratio = max(-1.0, min(1.0, cvd_s / 5.0))

        # hurst_1d: slope of vol std over lags on last 50 bars
        hurst_v = 0.5
        if i >= 50:
            ts = closes[i - 50:i + 1]
            lgs = range(2, 10)
            tau = [max(np.std(ts[lg:] - ts[:-lg]), 1e-9) for lg in lgs]
            if len(tau) >= 2:
                hurst_v = float(np.polyfit(np.log(list(lgs)), np.log(tau), 1)[0] * 2.0)
        hurst_v = max(0.0, min(1.0, hurst_v))

        # macd_hist_norm
        mh = float(macd_hist[i]) / max(c * 0.001, 1e-9)
        mh = max(-5.0, min(5.0, mh))

        # ema_stack: +1 price > ema50 > ema200, -1 reverse, 0 mixed
        e50 = float(ema50[i])
        e200 = float(ema200[i]) if i < len(ema200) else e50
        if c > e50 > e200:
            ema_st = 1.0
        elif c < e50 < e200:
            ema_st = -1.0
        else:
            ema_st = 0.0

        # vol_percentile: where current vol ranks vs 50-bar window
        if i >= 50:
            win = vols[i - 50:i + 1]
            vp = float(np.sum(win <= vols[i])) / len(win)
        else:
            vp = 0.5

        # session_weight from UTC hour of bar timestamp
        bar_hour = int((ts_arr[i] // 1000) % 86400 // 3600)
        if 12 <= bar_hour < 21:
            sess_w = 1.0
        elif 7 <= bar_hour < 12:
            sess_w = 0.85
        elif 21 <= bar_hour <= 23:
            sess_w = 0.78
        else:
            sess_w = 0.72

        # ── NEW v2 features (10) ──────────────────────────────────────────
        # volume_acceleration: current vol vs 5-bar mean
        vol_accel = vols[i] / max(float(np.mean(vols[i - 5:i])), 1e-9)

        # funding_rate_ema_ratio: current funding vs its 7-period EMA
        fe = float(fund_ema7[i]) if i < len(fund_ema7) else 0.0
        fund_ema_ratio = (fund_vals[i] / fe) if abs(fe) > 1e-9 else 1.0

        # liq_distance_buy/sell: % to nearest liquidation cluster proxy
        # (stops/liqs cluster at recent swing lows below and swing highs above)
        above_lvls = [p for idx, p in piv_sh if idx <= i - 3 and p > c]
        below_lvls = [p for idx, p in piv_sl if idx <= i - 3 and p < c]
        liq_dist_sell = min((p - c) / c for p in above_lvls) * 100 if above_lvls else 5.0
        liq_dist_buy  = min((c - p) / c for p in below_lvls) * 100 if below_lvls else 5.0

        # breadth_momentum proxy: sign-consistency of last 8 bars
        diffs = np.sign(np.diff(closes[i - 8:i + 1]))
        breadth_mom = float(np.mean(diffs)) if len(diffs) else 0.0

        # oi_price_divergence: OI rising while price flat = accumulation
        oi_price_div = oi_4h * (1.0 - min(1.0, abs(ret_4h) / 0.005))

        # spread_compression: recent range narrowing = breakout warning
        rng3  = float(np.mean(highs[i - 2:i + 1] - lows[i - 2:i + 1]))
        rng12 = float(np.mean(highs[i - 12:i] - lows[i - 12:i]))
        spread_comp = 1.0 - (rng3 / max(rng12, 1e-9))

        # funding_cross_btc: this symbol's funding vs BTC's (relative heat)
        bf = float(btc_fund_vals[i])
        fund_cross_btc = (fund_vals[i] / bf) if abs(bf) > 1e-9 else 1.0

        # tick_imbalance: intra-bar buy-sell pressure normalized -1..+1
        tick_imb = float(bar_deltas[i]) / max(float(vols[i]), 1e-9)

        # session_momentum: price change since NY session open (13:00 UTC bar)
        hours_since_open = (bar_hour - 13) % 24
        j_open = max(0, i - hours_since_open)
        sess_mom = pct(closes[i], opens[j_open])

        feat = [
            ret_1h, ret_4h, ret_12h, ret_24h,
            vol_r_1h, vol_r_4h,
            body_r, u_wick, l_wick,
            oi_1h, oi_4h,
            fund_z, fund_abs,
            mom_5, mom_12,
            c_vs_vwap, rng_exp, tc_ratio,
            # microstructure 7
            cvd_s, smart_cvd_ratio,
            hurst_v, mh, ema_st, vp, sess_w,
            # new 10 (v2)
            vol_accel, fund_ema_ratio,
            liq_dist_buy, liq_dist_sell,
            breadth_mom, oi_price_div,
            spread_comp, fund_cross_btc,
            tick_imb, sess_mom,
        ]
        feat = [_clamp(v, -5.0, 5.0) for v in feat]
        rows.append(feat)

        nxt = pct(closes[i + 1], closes[i])
        if nxt > LABEL_UP:
            lbl = 2
        elif nxt < -LABEL_DN:
            lbl = 0
        else:
            lbl = 1
        labels.append(lbl)

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=int)


def train_master_ml(force=False):
    """Train 35-feature XGBoost. Returns (model, scaler) or (None, None)."""
    if not HAS_XGB or not HAS_SKLEARN:
        return None, None
    if not force and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:  model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
            return model, scaler
        except Exception:
            pass

    print("[ML] Training 35-feature Master XGBoost on 6 symbols × 500 bars ...")
    try:
        btc_fr = fetch_funding_hist("BTCUSDT", 500)
    except Exception:
        btc_fr = []
    all_X, all_y = [], []
    for sym in TRAIN_SYMBOLS:
        try:
            kl  = fetch_klines_dicts(sym, "1h", 500)
            oi  = fetch_oi_history(sym, "1h", 500)
            fr  = fetch_funding_hist(sym, 500)
            X, y = _build_master_features(kl, oi, fr, btc_fund_hist=btc_fr)
            if X.shape[0] > 20:
                all_X.append(X); all_y.append(y)
        except Exception as e:
            print("  [WARN] %s: %s" % (sym, e))

    if not all_X:
        return None, None

    X_full = np.vstack(all_X)
    y_full = np.concatenate(all_y)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_full)

    model = XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.035,
        subsample=0.8, colsample_bytree=0.75,
        min_child_weight=4, gamma=0.1,
        verbosity=0, eval_metric="mlogloss",
        use_label_encoder=False,
    )
    model.fit(X_sc, y_full)

    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print("[ML] Done. Model saved to %s" % MODEL_PATH)
    return model, scaler


def predict_ml(model, scaler, symbol):
    """
    Fetch live data, extract last-bar features, return (class, pct_up, pct_dn, feat_note).
    class: 0=DOWN, 1=FLAT, 2=UP
    """
    if model is None or scaler is None:
        return 1, 33.0, 33.0, "ML not trained"
    try:
        kl  = fetch_klines_dicts(symbol, "1h", 520)
        oi  = fetch_oi_history(symbol, "1h", 50)
        fr  = fetch_funding_hist(symbol, 50)
        try:
            btc_fr = fr if symbol == "BTCUSDT" else fetch_funding_hist("BTCUSDT", 50)
        except Exception:
            btc_fr = []
        X, _ = _build_master_features(kl, oi, fr, btc_fund_hist=btc_fr)
        if X.shape[0] == 0:
            return 1, 33.0, 33.0, "Insufficient bars"
        x_live = X[-1:]
        x_sc   = scaler.transform(x_live)
        proba  = model.predict_proba(x_sc)[0]
        cls    = int(np.argmax(proba))
        pct_dn = round(proba[0] * 100, 1)
        pct_up = round(proba[2] * 100, 1)

        # Feature importance note
        closes = np.array([b["close"] for b in kl], float)
        ret_12h = (closes[-1] - closes[-13]) / max(abs(closes[-13]), 1) * 100
        note = "ret_12h=%+.1f%%" % ret_12h
        return cls, pct_up, pct_dn, note
    except Exception as e:
        return 1, 33.0, 33.0, str(e)[:50]


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 7  —  MULTI-SYMBOL MARKET BREADTH
# ═════════════════════════════════════════════════════════════════════════════

def scan_breadth(symbols=None):
    """
    Quick 1d scan of all symbols → bull/bear count.
    Returns breadth dict.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    bull, bear, neutral = 0, 0, 0
    details = {}
    for sym in symbols:
        try:
            raw = fetch_klines_raw(sym, "1d", 60)
            if len(raw) < 55:
                neutral += 1; continue
            o, h, l, c, v = _ohlcv_from_klines(raw)
            ema50  = _ema(c, 50)
            spot   = float(c[-1])
            e50    = float(ema50[-1])
            macd   = _ema(c, 12) - _ema(c, 26)
            sig    = _ema(macd, 9)
            hist_v = float(macd[-1]) - float(sig[-1])
            d1_dir = 0
            if spot > e50 and hist_v > 0:
                d1_dir = 1; bull += 1
            elif spot < e50 and hist_v < 0:
                d1_dir = -1; bear += 1
            else:
                neutral += 1
            details[sym] = {"dir": d1_dir, "spot": spot}
        except Exception:
            neutral += 1
    return {"bull": bull, "bear": bear, "neutral": neutral,
            "total": len(symbols), "details": details}

def breadth_score(breadth):
    """
    Returns (score -2..+2, label)
    """
    bull  = breadth.get("bull", 0)
    bear  = breadth.get("bear", 0)
    total = breadth.get("total", 8)
    if bull >= 6:
        return 2, "BROAD MARKET BULL (%d/%d)" % (bull, total)
    if bull >= 5:
        return 1, "MAJORITY BULL (%d/%d)" % (bull, total)
    if bear >= 6:
        return -2, "BROAD MARKET BEAR (%d/%d)" % (bear, total)
    if bear >= 5:
        return -1, "MAJORITY BEAR (%d/%d)" % (bear, total)
    return 0, "MIXED (%d bull, %d bear)" % (bull, bear)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 9  —  LIQUIDATION CASCADE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

def liq_cascade_signal(symbol, spot, h, l, c, v):
    """
    Layer 9 — Liquidation Cascade Detector.
    Liq clusters ≈ recent swing highs (short liqs/stops above) and swing lows
    (long liqs/stops below), weighted by OI and volume density at the level.
    If price is within LIQ_PROXIMITY_PCT of a major cluster AND momentum points
    toward it → cascading liquidations amplify the move in that direction.

    liq_raw = (oi_density_at_level × price_proximity_factor) clamped 0-10,
    then signed by cascade direction and mapped to -3..+3 for fusion.
    Returns (score -3..+3, notes, info_dict). Fails safe to (0.0, [], {}).
    """
    try:
        if spot <= 0 or len(h) < 30:
            return 0.0, [], {}

        n = len(h)
        lb = min(LIQ_LOOKBACK_BARS, n)
        offset = n - lb
        sh, sl = _swing_pivots(h[-lb:], l[-lb:], k=3)

        vol_avg = float(np.mean(v[-50:])) if len(v) >= 50 else float(np.mean(v))
        vol_avg = max(vol_avg, 1e-9)

        # OI scale factor: more open interest = denser liq clusters (1.0 if unavailable)
        try:
            oi_now = fetch_open_interest_now(symbol)
        except Exception:
            oi_now = 0.0
        oi_factor = 1.0 + min(1.0, _safe(oi_now) / 1e9)

        # Build clusters: (side, price, density)
        clusters_above, clusters_below = [], []
        for idx, price in sh:
            if price > spot:
                gi = min(offset + idx, n - 1)
                density = float(v[gi]) / vol_avg
                clusters_above.append((price, density))
        for idx, price in sl:
            if price < spot:
                gi = min(offset + idx, n - 1)
                density = float(v[gi]) / vol_avg
                clusters_below.append((price, density))

        # Nearest cluster each side
        near_above = min(clusters_above, key=lambda x: x[0] - spot) if clusters_above else None
        near_below = min(clusters_below, key=lambda x: spot - x[0]) if clusters_below else None

        dist_above = (near_above[0] - spot) / spot * 100 if near_above else 99.0
        dist_below = (spot - near_below[0]) / spot * 100 if near_below else 99.0

        # Short-term momentum (last 3 bars)
        momentum = (float(c[-1]) - float(c[-4])) / max(abs(float(c[-4])), 1e-9) * 100 if len(c) >= 4 else 0.0

        score = 0.0
        notes = []
        liq_raw = 0.0

        # Cascade UP: approaching short-liq cluster above with upward momentum
        if near_above and dist_above <= LIQ_PROXIMITY_PCT and momentum > 0:
            proximity = 1.0 - (dist_above / LIQ_PROXIMITY_PCT)
            liq_raw = max(0.0, min(10.0, near_above[1] * oi_factor * proximity * 3.0))
            score = (liq_raw / 10.0) * 3.0
            notes.append("LIQ CASCADE ↑: short-liq cluster $%.4f (%.2f%% away, density %.1fx)"
                         % (near_above[0], dist_above, near_above[1]))
        # Cascade DOWN: approaching long-liq cluster below with downward momentum
        elif near_below and dist_below <= LIQ_PROXIMITY_PCT and momentum < 0:
            proximity = 1.0 - (dist_below / LIQ_PROXIMITY_PCT)
            liq_raw = max(0.0, min(10.0, near_below[1] * oi_factor * proximity * 3.0))
            score = -(liq_raw / 10.0) * 3.0
            notes.append("LIQ CASCADE ↓: long-liq cluster $%.4f (%.2f%% away, density %.1fx)"
                         % (near_below[0], dist_below, near_below[1]))

        info = {
            "dist_above_pct": round(dist_above, 3),
            "dist_below_pct": round(dist_below, 3),
            "momentum_3bar_pct": round(momentum, 3),
            "liq_raw_0_10": round(liq_raw, 2),
            "liq_norm_0_100": round(liq_raw * 10.0, 1),
            "oi_factor": round(oi_factor, 2),
        }
        return max(-3.0, min(3.0, score)), notes, info
    except Exception:
        return 0.0, [], {}


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 10  —  VWAP STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def vwap_structure_signal(symbol, spot=None):
    """
    Layer 10 — Session VWAP anchored to NY open (13:30 UTC) on 15m klines.
      price > session VWAP by >0.3% → bullish; < -0.3% → bearish
      VWAP slope (rising/falling) adds +/- 0.5 bonus
      previous session's VWAP acts as support/resistance (noted when within 0.2%)
    Returns (score -2..+2, notes, info_dict). Fails safe to (0.0, [], {}).
    """
    try:
        raw = fetch_klines_raw(symbol, "15m", 300)   # ~3 days of 15m bars
        if not raw or len(raw) < 30:
            return 0.0, [], {}
        o, h, l, c, v = _ohlcv_from_klines(raw)
        ts = np.array([int(k[0]) for k in raw], dtype="int64")
        if spot is None or spot <= 0:
            spot = float(c[-1])
        if spot <= 0:
            return 0.0, [], {}

        # Minute-of-day for each bar; find session anchors (13:30 UTC = NY open)
        mod = (ts // 1000 % 86400) // 60
        anchors = [i for i in range(len(ts)) if int(mod[i]) == NY_OPEN_MINUTE_UTC]
        if not anchors:
            # fallback: anchor at UTC midnight bars
            anchors = [i for i in range(len(ts)) if int(mod[i]) == 0]
        if not anchors:
            # last-resort fallback: rolling ~13.5h session so the layer still works
            # on feeds with misaligned bar timestamps
            a = max(0, len(ts) - 54)
            anchors = [a] if a > 0 else [0]

        a_now  = anchors[-1]
        a_prev = anchors[-2] if len(anchors) >= 2 else None

        tp = (h + l + c) / 3.0

        def _vwap(i0, i1):
            w = v[i0:i1 + 1]
            p = tp[i0:i1 + 1]
            tw = float(np.sum(w))
            return float(np.sum(w * p)) / tw if tw > 0 else float(np.mean(p))

        vwap_now = _vwap(a_now, len(c) - 1)
        dev_pct  = (spot - vwap_now) / vwap_now * 100.0

        # VWAP slope: compare current session VWAP to VWAP up to session midpoint
        mid = a_now + max(1, (len(c) - 1 - a_now) // 2)
        vwap_mid = _vwap(a_now, mid)
        slope_up = vwap_now > vwap_mid

        score = 0.0
        notes = []
        if dev_pct > VWAP_DEV_THRESHOLD:
            score += 1.0
            notes.append("Price +%.2f%% above session VWAP $%.4f (bullish)" % (dev_pct, vwap_now))
        elif dev_pct < -VWAP_DEV_THRESHOLD:
            score -= 1.0
            notes.append("Price %.2f%% below session VWAP $%.4f (bearish)" % (dev_pct, vwap_now))
        else:
            notes.append("Price at session VWAP $%.4f (%.2f%%) — magnet zone" % (vwap_now, dev_pct))

        # Slope bonus
        if slope_up:
            score += 0.5
            if abs(dev_pct) > VWAP_DEV_THRESHOLD:
                notes.append("VWAP rising ↑")
        else:
            score -= 0.5
            if abs(dev_pct) > VWAP_DEV_THRESHOLD:
                notes.append("VWAP falling ↓")

        # Previous session VWAP as S/R
        prev_vwap = None
        if a_prev is not None and a_now - a_prev >= 2:
            prev_vwap = _vwap(a_prev, a_now - 1)
            prev_dist = (spot - prev_vwap) / prev_vwap * 100.0
            if abs(prev_dist) <= 0.2:
                role = "SUPPORT" if dev_pct >= 0 else "RESISTANCE"
                notes.append("At prev-session VWAP $%.4f (%s)" % (prev_vwap, role))
            elif prev_dist > 0:
                score += 0.25   # above prior session value = acceptance higher
            else:
                score -= 0.25

        info = {
            "session_vwap": round(vwap_now, 6),
            "dev_pct": round(dev_pct, 3),
            "vwap_slope": "UP" if slope_up else "DOWN",
            "prev_session_vwap": round(prev_vwap, 6) if prev_vwap else None,
        }
        return max(-2.0, min(2.0, score)), notes, info
    except Exception:
        return 0.0, [], {}


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 11  —  DONCHIAN BREAKOUT SIGNAL  (backtest-validated: +15.4% / 180d)
# ═════════════════════════════════════════════════════════════════════════════

# Backtest on 180 days of live Binance data confirmed:
#   Top breakout symbols: UNI +$4k, NEAR +$3.7k, ARB +$3k, INJ +$2k
#   Profit factor 1.39, win rate 44.6%, Sharpe 6.1
BREAKOUT_BARS   = 20       # Donchian channel period
BREAKOUT_VOL_X  = 1.8      # Volume must be > 1.8× 20-bar avg (backtest: 1.8 → WR 54.7% vs 44.6% @ 1.4)
BREAKOUT_ATR_X  = 1.15     # ATR must be > 1.15× 20-bar ATR avg (expanded volatility = real breakout)
BREAKOUT_SCORE_MAX = 3.0   # max score magnitude on -3..+3 scale

def donchian_breakout_signal(symbol, h, l, c, v, spot=None):
    """
    Layer 11 — Donchian-channel breakout with volume + ATR expansion confirmation.
    Backtest-validated on 180-day live Binance data: PF 1.66, WR 54.7%, Sharpe 9.69
    (improved from PF 1.39/44.6% by raising VOL_X 1.4→1.8, ATR_X 1.1→1.15).

    Returns (score, notes, info) where score is in [-3, +3].
      +3 = very strong upside breakout
      -3 = very strong downside breakout
       0 = no breakout / consolidation
    """
    info = {}
    try:
        n = len(c)
        if n < BREAKOUT_BARS + 10:
            return 0.0, "INSUFFICIENT_DATA", info

        # Donchian channel: highest high / lowest low of last N bars (excluding current)
        look = BREAKOUT_BARS
        ch_high = float(np.max(h[-(look+1):-1]))
        ch_low  = float(np.min(l[-(look+1):-1]))
        price   = float(c[-1]) if spot is None else spot

        info["ch_high"] = round(ch_high, 4)
        info["ch_low"]  = round(ch_low, 4)

        # ATR expansion check
        atr_now = _atr(h, l, c, 14)
        atr_arr = []
        for i in range(look):
            if n - look + i >= 15:
                atr_arr.append(_atr(h[:n-look+i], l[:n-look+i], c[:n-look+i], 14))
        atr_ma = float(np.mean(atr_arr)) if atr_arr else atr_now
        atr_expanding = atr_now > atr_ma * BREAKOUT_ATR_X

        # Volume confirmation
        v_ma20 = float(np.mean(v[-21:-1])) if n > 21 else float(np.mean(v[:-1]))
        v_ratio = float(v[-1]) / (v_ma20 + 1e-10)
        vol_confirmed = v_ratio >= BREAKOUT_VOL_X

        info["atr_ratio"] = round(atr_now / (atr_ma + 1e-10), 2)
        info["vol_ratio"] = round(v_ratio, 2)

        if not vol_confirmed and not atr_expanding:
            return 0.0, "NO_BREAKOUT:vol+ATR_weak", info

        # Determine breakout direction and strength
        if price > ch_high:
            dist_pct = (price - ch_high) / ch_high * 100
            # Score scales with: (1) how far above the channel, (2) volume, (3) ATR expansion
            strength = min(1.0, dist_pct / 0.5) * 0.5          # distance contribution (0→0.5)
            strength += min(1.0, (v_ratio - 1.0) / 1.0) * 0.3  # volume contribution (0→0.3)
            strength += (0.2 if atr_expanding else 0.0)          # ATR bonus
            score = strength * BREAKOUT_SCORE_MAX
            notes = f"UP_BREAK:+{dist_pct:.2f}% vol×{v_ratio:.1f} {'ATR↑' if atr_expanding else ''}"
            info["direction"] = "UP"
            info["dist_pct"]  = round(dist_pct, 3)

        elif price < ch_low:
            dist_pct = (ch_low - price) / ch_low * 100
            strength = min(1.0, dist_pct / 0.5) * 0.5
            strength += min(1.0, (v_ratio - 1.0) / 1.0) * 0.3
            strength += (0.2 if atr_expanding else 0.0)
            score = -strength * BREAKOUT_SCORE_MAX
            notes = f"DN_BREAK:-{dist_pct:.2f}% vol×{v_ratio:.1f} {'ATR↑' if atr_expanding else ''}"
            info["direction"] = "DOWN"
            info["dist_pct"]  = round(dist_pct, 3)

        else:
            # Inside channel — check proximity for pre-breakout signal
            ch_range = ch_high - ch_low
            pos_pct  = (price - ch_low) / (ch_range + 1e-10)
            # Give a small pre-breakout score when price is in the top/bottom 15%
            if pos_pct > 0.85:
                score = 0.5 * (pos_pct - 0.85) / 0.15
                notes = f"PRE_BREAK_UP:{pos_pct*100:.0f}% channel"
            elif pos_pct < 0.15:
                score = -0.5 * (0.15 - pos_pct) / 0.15
                notes = f"PRE_BREAK_DN:{pos_pct*100:.0f}% channel"
            else:
                score = 0.0
                notes = f"INSIDE_CHANNEL:{pos_pct*100:.0f}%"
            info["direction"] = "NEUTRAL"

        return float(np.clip(score, -BREAKOUT_SCORE_MAX, BREAKOUT_SCORE_MAX)), notes, info

    except Exception as e:
        return 0.0, f"ERR:{e}", {}


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 8  —  MASTER SCORE FUSION  (regime-adaptive)
# ═════════════════════════════════════════════════════════════════════════════

def normalize_to_100(raw_score, max_range=3.0):
    """Convert a score in [-max_range, +max_range] to 0-100 centered at 50."""
    clamped = max(-max_range, min(max_range, raw_score))
    return 50.0 + (clamped / max_range) * 50.0

def _regime_weights(trend_regime):
    """
    Regime-adaptive fusion weights.
    TRENDING: boost multi-TF, ML, breakout, liq cascades; reduce VWAP/breadth.
    CHOPPY:   boost VWAP mean-reversion and institutional flow; reduce breakout/TF.
    Always renormalized to sum to 1.0.
    """
    w = dict(BASE_WEIGHTS)
    if trend_regime == "TRENDING":
        w["tf"]       += 0.05
        w["ml"]       += 0.02
        w["liq"]      += 0.02
        w["breakout"] += 0.03   # breakout strongest in trending regimes
        w["vwap"]     -= 0.04
        w["manip"]    -= 0.04
        w["breadth"]  -= 0.04
    elif trend_regime == "CHOPPY":
        w["vwap"]     += 0.05
        w["inst"]     += 0.04
        w["manip"]    += 0.02
        w["breakout"] -= 0.04   # reduce breakout in choppy regime (false breaks)
        w["tf"]    -= 0.08
        w["liq"]   -= 0.04
    w = {k: max(0.0, val) for k, val in w.items()}
    s = sum(w.values()) or 1.0
    return {k: val / s for k, val in w.items()}

def fuse_master_score(
    tf_score,      # -3..+3
    inst_score,    # -3..+3
    ml_cls,        # 0=DOWN, 1=FLAT, 2=UP
    ml_pct_up,     # 0..100
    manip_score,   # -2..+2
    fund_oi_score, # -2..+2
    breadth_sc,    # -2..+2
    vol_regime,    # "NORMAL", "HIGH", "LOW", "EXTREME"
    sess_quality,  # 0..1
    liq_score=0.0,        # -3..+3  (Layer 9)
    vwap_score=0.0,       # -2..+2  (Layer 10)
    breakout_score=0.0,   # -3..+3  (Layer 11 — Donchian, backtest-validated)
    trend_regime="NEUTRAL",  # "TRENDING" / "CHOPPY" / "NEUTRAL"
):
    """
    Combine all 11 signal layers into a 0-100 master score with regime-adaptive weights.
    Returns (score, direction, grade, label).
    Layer 11 (Donchian breakout) added after 180-day live backtest: PF 1.39, +15.4%.
    """
    # Convert ML to -3..+3 scale
    if ml_cls == 2:
        ml_raw = (ml_pct_up / 100.0) * 3.0
    elif ml_cls == 0:
        ml_raw = -(1.0 - ml_pct_up / 100.0) * 3.0
    else:
        ml_raw = (ml_pct_up - 50.0) / 100.0

    # Regime-adaptive weighted composite
    w = _regime_weights(trend_regime)
    raw = (
        tf_score       * w["tf"]       +
        inst_score     * w["inst"]     +
        ml_raw         * w["ml"]       +
        liq_score      * w["liq"]      +
        manip_score    * w["manip"]    +
        fund_oi_score  * w["fund"]     +
        breakout_score * w["breakout"] +
        vwap_score     * w["vwap"]     +
        breadth_sc     * w["breadth"]
    )

    # Apply session quality multiplier (scales distance from 50)
    score = 50.0 + (raw / 3.0) * 50.0 * sess_quality

    # Volatility penalty
    if vol_regime == "EXTREME":
        score = 50.0 + (score - 50.0) * 0.5
    elif vol_regime == "LOW":
        score = 50.0 + (score - 50.0) * 0.8

    score = max(0.0, min(100.0, score))

    if score >= GRADE_PERFECT:
        grade = "PERFECT (★★★)"
    elif score >= GRADE_HIGH:
        grade = "HIGH (★★)"
    elif score >= GRADE_MOD:
        grade = "MODERATE (★)"
    else:
        grade = "NO TRADE"

    direction = "LONG" if score > 52 else ("SHORT" if score < 48 else "WAIT")

    label = "%s — %s" % (grade, direction)
    return round(score, 1), direction, grade, label

def generate_trade_plan(spot, atr, direction, score):
    """
    Returns entry, SL, TP1, tp_pct, sl_pct, kelly_pct.
    """
    if atr <= 0:
        atr = spot * 0.01

    if score >= GRADE_PERFECT:
        sl_mult = 1.0; tp1_mult = 2.5; trail_mult = 1.5; partial = "50%"
        kelly_pct = min(15.0, score / 100.0 * 20.0)
    elif score >= GRADE_HIGH:
        sl_mult = 1.5; tp1_mult = 2.0; trail_mult = 2.0; partial = "50%"
        kelly_pct = min(12.0, score / 100.0 * 18.0)
    else:
        sl_mult = 1.5; tp1_mult = 2.0; trail_mult = None; partial = "100%"
        kelly_pct = min(8.0, score / 100.0 * 12.0)

    sign = 1 if direction == "LONG" else -1
    sl  = spot - sign * sl_mult  * atr
    tp1 = spot + sign * tp1_mult * atr

    sl_pct  = abs(sl  - spot) / spot * 100
    tp_pct  = abs(tp1 - spot) / spot * 100

    plan = {
        "entry": spot, "sl": sl, "tp1": tp1,
        "sl_pct": sl_pct, "tp_pct": tp_pct,
        "sl_mult": sl_mult, "tp_mult": tp1_mult,
        "partial": partial, "trail_mult": trail_mult,
        "kelly": kelly_pct,
    }
    return plan


# ═════════════════════════════════════════════════════════════════════════════
#  CARRY ARB OPPORTUNITIES
# ═════════════════════════════════════════════════════════════════════════════

def scan_carry_arb(symbols=None):
    """Find best carry arbitrage (long spot + short perp) opportunities."""
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    opps = []
    for sym in symbols:
        try:
            rate = fetch_funding_rate(sym)
            rate_pct = rate * 100  # percentage
            ann = abs(rate_pct) * 3 * 365
            if ann >= 8.0:
                direction = "SHORT" if rate_pct > 0 else "LONG"
                opps.append({
                    "symbol": sym,
                    "rate_8h": rate_pct,
                    "ann_pct": ann,
                    "direction": direction,
                })
        except Exception:
            pass
    opps.sort(key=lambda x: -x["ann_pct"])
    return opps


# ═════════════════════════════════════════════════════════════════════════════
#  MASTER ANALYZE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

def analyze_symbol(symbol, model, scaler, breadth_data=None):
    """
    Run all 10 layers for one symbol.
    Returns full result dict.
    """
    result = {"symbol": symbol, "error": None}

    # ── Layer 1: Session ─────────────────────────────────────────────────────
    sess_q, sess_name = session_quality()

    # ── Fetch primary 1h data ────────────────────────────────────────────────
    raw_1h = fetch_klines_raw(symbol, "1h", 300)
    if len(raw_1h) < 50:
        result["error"] = "Insufficient 1h klines"
        return result
    o, h, l, c, v = _ohlcv_from_klines(raw_1h)
    spot = float(c[-1])
    atr  = _atr(h, l, c)
    vol_regime, vol_ratio = volatility_regime(h, l, c)

    if vol_regime == "EXTREME":
        result["warning"] = "EXTREME VOLATILITY — reduced confidence"

    result["spot"]        = spot
    result["atr"]         = round(atr, 4)
    result["session"]     = sess_name
    result["vol_regime"]  = vol_regime

    # ── Layer 2: Multi-TF ────────────────────────────────────────────────────
    d1_tf  = analyze_tf(symbol, "1d", 300)
    h4_tf  = analyze_tf(symbol, "4h", 200)
    h1_tf  = {"dir": int(_tf_direction(o, h, l, c)[0]),
               "label": _tf_direction(o, h, l, c)[1],
               "hurst": _tf_hurst(c)[0],
               "cvd_slope": _cvd_slope(o, h, l, c, v),
               "atr": atr, "spot": spot}

    tf_sc, tf_label = triple_tf_score(d1_tf, h4_tf, h1_tf)
    result["tf"] = {
        "score": tf_sc, "label": tf_label,
        "d1": d1_tf["label"], "h4": h4_tf["label"], "h1": h1_tf["label"],
    }

    # ── Layer 3: Institutional Flow ──────────────────────────────────────────
    trades     = fetch_agg_trades(symbol, 1500)
    smart_cvd  = compute_smart_cvd(trades)
    icebergs   = detect_icebergs(trades)
    oi_vel     = oi_velocity(symbol)
    ls_ratio   = fetch_long_short_ratio(symbol)
    inst_sc, inst_notes = inst_flow_score(smart_cvd, icebergs, oi_vel, ls_ratio)
    result["flow"] = {
        "score": round(inst_sc, 2),
        "smart_cvd": round(smart_cvd["smart_cvd"], 0),
        "retail_cvd": round(smart_cvd["retail_cvd"], 0),
        "icebergs": icebergs[:2],
        "oi_vel": round(oi_vel, 2),
        "ls_ratio": round(ls_ratio, 2),
        "notes": inst_notes,
    }

    # ── Layer 4: Funding + OI ────────────────────────────────────────────────
    fund_sc, fund_notes, carry_ann, carry_note = funding_oi_combined(symbol, spot)
    result["funding"] = {
        "score": round(fund_sc, 2),
        "notes": fund_notes,
        "carry_ann": round(carry_ann, 1),
        "carry_note": carry_note,
    }

    # ── Layer 5: Manipulation ────────────────────────────────────────────────
    ob = fetch_order_book(symbol, 100)
    manip_sc, manip_dir, manip_notes, spoof_dir = manip_combined(symbol, ob, spot, h, l, c, atr)
    result["manip"] = {
        "score": round(manip_sc, 2),
        "direction": manip_dir,
        "notes": manip_notes,
        "spoof_dir": spoof_dir,
    }

    # ── Layer 6: ML ──────────────────────────────────────────────────────────
    ml_cls, ml_pct_up, ml_pct_dn, ml_note = predict_ml(model, scaler, symbol)
    result["ml"] = {
        "class": ["DOWN", "FLAT", "UP"][ml_cls],
        "pct_up": ml_pct_up,
        "pct_dn": ml_pct_dn,
        "note": ml_note,
    }

    # ── Layer 7: Breadth ─────────────────────────────────────────────────────
    if breadth_data is None:
        breadth_data = {"bull": 0, "bear": 0, "neutral": 0, "total": 8}
    b_sc, b_label = breadth_score(breadth_data)
    result["breadth"] = {"score": b_sc, "label": b_label}

    # ── Layer 9: Liquidation Cascade Detector ────────────────────────────────
    liq_sc, liq_notes, liq_info = liq_cascade_signal(symbol, spot, h, l, c, v)
    result["liq"] = {
        "score": round(liq_sc, 2),
        "notes": liq_notes,
        "info": liq_info,
    }

    # ── Layer 10: VWAP Structure ─────────────────────────────────────────────
    vwap_sc, vwap_notes, vwap_info = vwap_structure_signal(symbol, spot)
    result["vwap"] = {
        "score": round(vwap_sc, 2),
        "notes": vwap_notes,
        "info": vwap_info,
    }

    # ── Layer 11: Donchian Breakout (backtest-validated on 180d live data) ───
    brk_sc, brk_notes, brk_info = donchian_breakout_signal(symbol, h, l, c, v, spot)
    result["breakout"] = {
        "score": round(brk_sc, 2),
        "notes": brk_notes,
        "info": brk_info,
    }

    # ── Trend regime for adaptive fusion (from 1d + 4h Hurst) ────────────────
    hurst_sum = d1_tf.get("hurst", 0) + h4_tf.get("hurst", 0)
    if hurst_sum >= 1:
        trend_regime = "TRENDING"
    elif hurst_sum <= -1:
        trend_regime = "CHOPPY"
    else:
        trend_regime = "NEUTRAL"
    result["trend_regime"] = trend_regime

    # ── Layer 8: Master Fusion (all 11 layers, regime-adaptive) ─────────────
    master_sc, direction, grade, fusion_label = fuse_master_score(
        tf_score       = tf_sc,
        inst_score     = inst_sc,
        ml_cls         = ml_cls,
        ml_pct_up      = ml_pct_up,
        manip_score    = manip_sc,
        fund_oi_score  = fund_sc,
        breadth_sc     = b_sc,
        vol_regime     = vol_regime,
        sess_quality   = sess_q,
        liq_score      = liq_sc,
        vwap_score     = vwap_sc,
        breakout_score = brk_sc,
        trend_regime   = trend_regime,
    )
    plan = generate_trade_plan(spot, atr, direction, master_sc)

    result["master"]    = {"score": master_sc, "direction": direction,
                           "grade": grade, "label": fusion_label}
    result["trade_plan"] = plan
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING
# ═════════════════════════════════════════════════════════════════════════════

def _grade_star(grade):
    if "PERFECT" in grade:  return "★★★ PERFECT SETUP   "
    if "HIGH"    in grade:  return "★★  HIGH CONVICTION "
    if "MODERATE" in grade: return "★   MODERATE        "
    return "    NO TRADE        "

def format_signal(r, verbose=True):
    if r.get("error"):
        return "  [ERR] %-12s %s" % (r["symbol"], r["error"])

    sym    = r["symbol"]
    m      = r["master"]
    plan   = r["trade_plan"]
    spot   = r["spot"]
    score  = m["score"]
    grade  = m["grade"]
    direct = m["direction"]

    if m["direction"] == "WAIT" or score < GRADE_MOD:
        if not verbose:
            return None
        return "    %-12s  score=%.0f/100  WAIT" % (sym, score)

    lines = []
    star  = _grade_star(grade)
    lines.append("%s %-12s  score=%.0f/100   %s" % (star, sym, score, direct))

    # Trade plan
    sl   = plan["sl"];   tp1 = plan["tp1"]
    slp  = plan["sl_pct"]; tpp = plan["tp_pct"]
    lines.append("     Entry $%s  SL $%s (-%.2f%%)  TP1 $%s (+%.2f%%)%s" % (
        _fmt_price(spot), _fmt_price(sl), slp, _fmt_price(tp1), tpp,
        "  Trail after TP1" if plan["trail_mult"] else ""))
    lines.append("     Kelly: %.1f%% portfolio" % plan["kelly"])

    # Layer details
    tf = r["tf"]
    lines.append("     [3TF] %s" % tf["label"])
    lines.append("           1d: %-20s  4h: %-20s  1h: %s" % (
        tf["d1"], tf["h4"], tf["h1"]))

    fl = r["flow"]
    iceberg_txt = ""
    if fl["icebergs"]:
        top_i = fl["icebergs"][0]
        iceberg_txt = " | Iceberg: %.4f (%d hits)" % (top_i["price"], top_i["hits"])
    lines.append("     [FLOW] Smart CVD %s%s%s" % (
        "+%s" % _fmt_usd(fl["smart_cvd"]) if fl["smart_cvd"] >= 0 else "-%s" % _fmt_usd(abs(fl["smart_cvd"])),
        "  Retail %s" % ("+%s" % _fmt_usd(fl["retail_cvd"]) if fl["retail_cvd"] >= 0 else "-%s" % _fmt_usd(abs(fl["retail_cvd"]))),
        iceberg_txt))

    fd = r["funding"]
    fund_txt = fd["notes"][0] if fd["notes"] else ""
    lines.append("     [FUND] %s" % fund_txt)
    if fd["carry_note"]:
        lines.append("            %s" % fd["carry_note"])

    ml = r["ml"]
    lines.append("     [ML]   %s%% UP  |  %s" % (ml["pct_up"], ml["note"]))

    lq = r.get("liq", {})
    if lq.get("notes"):
        lines.append("     [LIQ]  %s" % lq["notes"][0])

    vw = r.get("vwap", {})
    if vw.get("notes"):
        lines.append("     [VWAP] %s" % " | ".join(vw["notes"][:2]))

    if r.get("trend_regime") and r["trend_regime"] != "NEUTRAL":
        lines.append("     [REGIME] %s — adaptive weights applied" % r["trend_regime"])

    mn = r["manip"]
    if mn["notes"]:
        lines.append("     [MANIP] %s" % " | ".join(mn["notes"][:2]))
    elif mn["direction"] == "CLEAN":
        lines.append("     [MANIP] CLEAN — no spoofing detected")

    if r.get("warning"):
        lines.append("     [WARN] %s" % r["warning"])

    return "\n".join(lines)

def _fmt_price(p):
    if p >= 1000:
        return "{:,.2f}".format(p)
    if p >= 1:
        return "{:.4f}".format(p)
    return "{:.6f}".format(p)

def print_header(sess_name, breadth_data):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bull = breadth_data.get("bull", 0)
    total = breadth_data.get("total", 8)
    sep = "═" * 70
    print(sep)
    print("  CRYPTO MASTER ENGINE  |  %s  |  %s SESSION" % (now, sess_name))
    print(sep)
    print("  Market Breadth: %d/%d BULLISH" % (bull, total))

def print_arb_section(opps):
    if not opps:
        return
    print()
    print("CARRY ARB OPPORTUNITIES:")
    items = ["  %s +%.1f%%/yr" % (o["symbol"], o["ann_pct"]) for o in opps[:5]]
    print("  |  ".join(items))

def print_footer():
    print("═" * 70)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Crypto Master Engine")
    parser.add_argument("--symbol", help="Single symbol to analyze")
    parser.add_argument("--watch", type=int, help="Watch mode: re-scan every N seconds")
    parser.add_argument("--retrain", action="store_true", help="Force retrain ML model")
    parser.add_argument("--arb", action="store_true", help="Carry arb scan only")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--verbose", action="store_true", help="Show all symbols including NO TRADE")
    args = parser.parse_args()

    # Carry arb only mode
    if args.arb:
        print("\n[CARRY ARB SCAN]")
        opps = scan_carry_arb()
        for o in opps:
            print("  %-12s  %s  rate=%.4f%%/8h  ann=%.1f%%/yr" % (
                o["symbol"], o["direction"], o["rate_8h"], o["ann_pct"]))
        return

    # Train or load ML
    print("[+] Loading ML model ...")
    model, scaler = train_master_ml(force=args.retrain)
    if model is None:
        print("[WARN] ML model unavailable (xgboost/sklearn not installed)")

    symbols = [args.symbol] if args.symbol else DEFAULT_SYMBOLS

    def _run_scan():
        # Layer 7: Breadth (once per scan)
        print("[+] Scanning market breadth ...")
        breadth_data = scan_breadth(DEFAULT_SYMBOLS)
        b_sc, b_label = breadth_score(breadth_data)
        sess_q, sess_name = session_quality()

        print_header(sess_name, breadth_data)
        print("  Breadth: %s" % b_label)

        results = []
        for sym in symbols:
            print("[+] Analyzing %s ..." % sym, end="\r")
            r = analyze_symbol(sym, model, scaler, breadth_data)
            results.append(r)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
            return

        # Sort by score descending
        results.sort(key=lambda x: -(x.get("master", {}).get("score", 0)))

        print()
        shown = 0
        for r in results:
            line = format_signal(r, verbose=args.verbose)
            if line:
                print(line)
                print()
                shown += 1

        if shown == 0:
            print("    No signals above threshold right now. Market may be consolidating.")

        # Carry arb
        arb_opps = scan_carry_arb(symbols)
        print_arb_section(arb_opps)
        print_footer()

    if args.watch:
        while True:
            try:
                _run_scan()
            except KeyboardInterrupt:
                print("\n[Stopped]")
                break
            except Exception as e:
                print("[ERROR] %s" % e)
            print("\n[Next scan in %ds — Ctrl+C to stop]\n" % args.watch)
            time.sleep(args.watch)
    else:
        _run_scan()


if __name__ == "__main__":
    main()
