#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crypto_alpha_engine.py  —  INSTITUTIONAL ALPHA ENGINE v3
=========================================================
Layer 1 : Manipulation Detection  (spoofing · layering · stop-hunt · wash-trade)
Layer 2 : ML Ensemble             (XGBoost trained on live historical microstructure)
Layer 3 : RL Trading Agent        (Q-learning, 300 discrete states)
Layer 4 : Signal Fusion           (entry / SL / TP / Kelly sizing)

Usage:
    python crypto_alpha_engine.py                 # full scan + train
    python crypto_alpha_engine.py --quick         # BTC+ETH only
    python crypto_alpha_engine.py --watch 120     # continuous every 2 min
    python crypto_alpha_engine.py --deep          # +8s OB spoof wait per symbol
    python crypto_alpha_engine.py --train-only    # retrain models then exit
"""

import argparse, sys, os, json, math, time, pickle, statistics
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import requests
import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_SPOT = "https://api.binance.com"

DEFAULT_SYMBOLS  = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
                    "XRPUSDT","DOGEUSDT","AVAXUSDT","ADAUSDT"]
TRAIN_SYMBOLS    = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
                    "XRPUSDT","AVAXUSDT"]   # symbols used to train ML

# ── Manipulation ─────────────────────────────────────────────────
SPOOF_WALL_MULT        = 8.0    # level qty > 8× median = candidate wall
SPOOF_EVAPORATE_THR    = 0.35   # < 35% remaining after 8s = spoof
SPOOF_CHECK_DELAY      = 8      # seconds between OB snapshots (--deep only)
LAYER_MIN_WALLS        = 3      # ≥3 walls within range = layering
LAYER_RANGE_PCT        = 0.005  # 0.5% of spot
WASH_WINDOW_MS         = 1500   # 1.5s window for same-price buy+sell
STOP_HUNT_RANGE_PCT    = 0.004  # within 0.4% of a stop cluster = danger
ROUND_NUMBER_TICKS     = {      # coin → round-number granularity
    "BTC": 1000.0, "ETH": 100.0, "SOL": 5.0, "BNB": 10.0,
    "XRP": 0.05,   "DOGE": 0.005,"AVAX": 0.5,"ADA": 0.01,
}

# ── ML ───────────────────────────────────────────────────────────
TRAIN_BARS      = 500     # 1h klines for each training symbol
LABEL_UP_PCT    = 0.003   # > +0.3% next 1h → class 2 (UP)
LABEL_DN_PCT    = 0.003   # < -0.3% → class 0 (DOWN); else class 1 (FLAT)
XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    verbosity=0, eval_metric="mlogloss",
)

# ── RL ───────────────────────────────────────────────────────────
RL_ALPHA    = 0.12
RL_GAMMA    = 0.90
RL_EPS_INIT = 0.30
RL_EPS_MIN  = 0.05
RL_EPS_DECAY= 0.997
RL_TRAIN_EP = 3          # episodes over the historical data

# ── Signal Fusion ────────────────────────────────────────────────
W_FLOW  = 0.30   # institutional flow weight
W_MANIP = 0.20   # manipulation signal weight
W_ML    = 0.30   # ML prediction weight
W_RL    = 0.20   # RL agent weight

CACHE_DIR = "alpha_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 AlphaEngine/3.0"}

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
            if i == retries - 1: return None
            time.sleep(0.4 * (i + 1))

def _safe(x, d=0.0):
    try:
        v = float(x)
        return d if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return d

def _zscore(series):
    """Return z-score of last value vs entire series."""
    if len(series) < 2: return 0.0
    mu, sigma = float(np.mean(series)), float(np.std(series))
    if sigma < 1e-12: return 0.0
    return (series[-1] - mu) / sigma

def _clamp(x, lo=-5.0, hi=5.0):
    return max(lo, min(hi, x))

# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHERS  (lean — only what alpha engine needs beyond flow)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str = "1h", limit: int = 500) -> List[dict]:
    data = _get(f"{BINANCE_FAPI}/fapi/v1/klines",
                {"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(data, list): return []
    bars = []
    for k in data:
        try:
            bars.append({
                "ts":     int(k[0]),
                "open":   float(k[1]),
                "high":   float(k[2]),
                "low":    float(k[3]),
                "close":  float(k[4]),
                "volume": float(k[5]),
                "trades": int(k[8]),
            })
        except Exception: pass
    return bars

def fetch_oi_history(symbol: str, period: str = "1h", limit: int = 500) -> List[dict]:
    data = _get(f"{BINANCE_FAPI}/futures/data/openInterestHist",
                {"symbol": symbol, "period": period, "limit": limit})
    if not isinstance(data, list): return []
    result = []
    for d in data:
        try:
            result.append({"ts": int(d["timestamp"]), "oi": float(d["sumOpenInterestValue"])})
        except Exception: pass
    return result

def fetch_funding_hist(symbol: str, limit: int = 500) -> List[dict]:
    data = _get(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list): return []
    result = []
    for d in data:
        try:
            result.append({"ts": int(d["fundingTime"]), "rate": float(d["fundingRate"]) * 10000})
        except Exception: pass
    return result

def fetch_agg_trades(symbol: str, limit: int = 1500) -> List[dict]:
    data = _get(f"{BINANCE_FAPI}/fapi/v1/aggTrades", {"symbol": symbol, "limit": limit})
    if not isinstance(data, list):
        data = _get(f"{BINANCE_SPOT}/api/v3/aggTrades", {"symbol": symbol, "limit": limit})
    if not isinstance(data, list): return []
    result = []
    for t in data:
        try:
            result.append({
                "price": float(t["p"]), "qty": float(t["q"]),
                "usd": float(t["p"]) * float(t["q"]),
                "is_maker": bool(t["m"]), "time": int(t["T"]),
            })
        except Exception: pass
    return result

def fetch_order_book(symbol: str, depth: int = 500) -> dict:
    data = _get(f"{BINANCE_FAPI}/fapi/v1/depth", {"symbol": symbol, "limit": depth})
    if not isinstance(data, dict):
        data = _get(f"{BINANCE_SPOT}/api/v3/depth", {"symbol": symbol, "limit": depth})
    return data if isinstance(data, dict) else {"bids": [], "asks": []}

def fetch_spot_price(symbol: str) -> float:
    d = _get(f"{BINANCE_SPOT}/api/v3/ticker/price", {"symbol": symbol})
    return _safe(d.get("price", 0)) if isinstance(d, dict) else 0.0

def fetch_long_short_ratio(symbol: str) -> float:
    d = _get(f"{BINANCE_FAPI}/futures/data/topLongShortPositionRatio",
             {"symbol": symbol, "period": "5m", "limit": 1})
    if isinstance(d, list) and d:
        return _safe(d[0].get("longShortRatio", 1.0))
    return 1.0

def fetch_funding_rate(symbol: str) -> float:
    d = _get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex", {"symbol": symbol})
    if isinstance(d, dict):
        return _safe(d.get("lastFundingRate", 0)) * 10000
    return 0.0

def fetch_open_interest(symbol: str) -> float:
    d = _get(f"{BINANCE_FAPI}/fapi/v1/openInterest", {"symbol": symbol})
    if isinstance(d, dict):
        return _safe(d.get("openInterestValue", 0))
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — MANIPULATION DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_spoofing_static(ob: dict, spot: float) -> dict:
    """
    Fast spoof proxy from a single OB snapshot.
    Isolated large orders (level qty >> neighbours) = likely spoofs.
    No wait required — good approximation for real-time.
    """
    asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:100]]
    bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:100]]

    def _find_isolated_walls(levels, side):
        if len(levels) < 7: return []
        qtys = [q for _, q in levels]
        median_q = float(np.median(qtys)) if qtys else 1.0
        walls = []
        for i, (price, qty) in enumerate(levels):
            if qty < SPOOF_WALL_MULT * median_q: continue
            # Check neighbours  ±3 levels
            neighbour_start = max(0, i - 3)
            neighbour_end   = min(len(levels), i + 4)
            neighbours = [levels[j][1] for j in range(neighbour_start, neighbour_end) if j != i]
            avg_nb = statistics.mean(neighbours) if neighbours else median_q
            if qty > 4 * avg_nb:   # 4× more isolated than neighbours
                walls.append({
                    "price":    price,
                    "qty":      qty,
                    "usd":      price * qty,
                    "side":     side,
                    "isolated": qty / max(avg_nb, 1e-9),
                    "pct_from_spot": abs(price - spot) / spot * 100,
                })
        return sorted(walls, key=lambda x: -x["usd"])[:5]

    ask_walls = _find_isolated_walls(asks, "ASK")
    bid_walls = _find_isolated_walls(bids, "BID")

    spoof_ask_usd = sum(w["usd"] for w in ask_walls)
    spoof_bid_usd = sum(w["usd"] for w in bid_walls)
    total = len(ask_walls) + len(bid_walls)

    # Net spoof direction:
    # Fake ASK wall = spoofing resistance = price actually wants to go UP
    # Fake BID wall = spoofing support   = price actually wants to go DOWN
    if spoof_ask_usd > spoof_bid_usd * 1.5:
        direction = "FAKE_RESISTANCE_BULLISH"   # shorts will squeeze
    elif spoof_bid_usd > spoof_ask_usd * 1.5:
        direction = "FAKE_SUPPORT_BEARISH"      # longs will liquidate
    else:
        direction = "MIXED" if total > 0 else "CLEAN"

    return {
        "ask_spoofs":    ask_walls,
        "bid_spoofs":    bid_walls,
        "spoof_ask_usd": spoof_ask_usd,
        "spoof_bid_usd": spoof_bid_usd,
        "direction":     direction,
        "total_spoofs":  total,
        "score":         min(10, total * 2 + (spoof_ask_usd + spoof_bid_usd) / 100_000),
    }


def detect_spoofing_deep(symbol: str, ob1: dict, spot: float) -> dict:
    """
    Deep spoof: compare OB snapshot before vs 8s later.
    Large orders that evaporate without trading = confirmed spoofs.
    Use --deep flag to enable (adds 8s per symbol).
    """
    time.sleep(SPOOF_CHECK_DELAY)
    ob2 = fetch_order_book(symbol, depth=100)

    asks1 = {float(p): float(q) for p, q in ob1.get("asks", [])[:100]}
    bids1 = {float(p): float(q) for p, q in ob1.get("bids", [])[:100]}
    asks2 = {float(p): float(q) for p, q in ob2.get("asks", [])[:100]}
    bids2 = {float(p): float(q) for p, q in ob2.get("bids", [])[:100]}

    def _evaporated(side1, side2, side_label):
        if not side1: return []
        med = float(np.median(list(side1.values())))
        confirmed = []
        for price, qty in side1.items():
            if qty < SPOOF_WALL_MULT * med: continue
            new_qty = side2.get(price, 0)
            evap = (qty - new_qty) / max(qty, 1e-9)
            if evap > (1 - SPOOF_EVAPORATE_THR):
                confirmed.append({
                    "price": price, "original_qty": qty, "new_qty": new_qty,
                    "usd": price * qty, "side": side_label,
                    "evap_pct": round(evap * 100, 1),
                    "pct_from_spot": abs(price - spot) / spot * 100,
                })
        return sorted(confirmed, key=lambda x: -x["usd"])[:5]

    ask_spoofs = _evaporated(asks1, asks2, "ASK")
    bid_spoofs = _evaporated(bids1, bids2, "BID")
    total = len(ask_spoofs) + len(bid_spoofs)
    spoof_ask_usd = sum(s["usd"] for s in ask_spoofs)
    spoof_bid_usd = sum(s["usd"] for s in bid_spoofs)

    if spoof_ask_usd > spoof_bid_usd * 1.5:
        direction = "FAKE_RESISTANCE_BULLISH"
    elif spoof_bid_usd > spoof_ask_usd * 1.5:
        direction = "FAKE_SUPPORT_BEARISH"
    else:
        direction = "MIXED" if total > 0 else "CLEAN"

    return {
        "ask_spoofs":    ask_spoofs,
        "bid_spoofs":    bid_spoofs,
        "spoof_ask_usd": spoof_ask_usd,
        "spoof_bid_usd": spoof_bid_usd,
        "direction":     direction,
        "total_spoofs":  total,
        "score":         min(10, total * 3),
        "confirmed":     True,  # deep = confirmed
    }


def detect_layering(ob: dict, spot: float) -> dict:
    """
    Layering = 3+ large walls stacked on ONE side within 0.5% of spot.
    Creates artificial support/resistance to fool algos.

    LAYERING_ASKS = fake ceiling → real move is UP
    LAYERING_BIDS = fake floor   → real move is DOWN
    """
    asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:100]]
    bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:100]]

    def _layer_count(levels, spot_ref, is_ask):
        if len(levels) < 4: return 0, 0.0, []
        qtys = [q for _, q in levels]
        med  = float(np.median(qtys)) if qtys else 1.0
        # Only look within 0.3% of spot for layering (wider = too many false positives)
        rng  = spot_ref * 0.003
        limit = spot_ref + rng if is_ask else spot_ref - rng
        layers = [(p, q) for p, q in levels
                  if q > 6 * med and
                  (p <= limit if is_ask else p >= limit)]
        return len(layers), sum(p * q for p, q in layers), layers

    ask_count, ask_usd, ask_layers = _layer_count(asks, spot, True)
    bid_count, bid_usd, bid_layers = _layer_count(bids, spot, False)

    if ask_count >= LAYER_MIN_WALLS and ask_count > bid_count:
        direction = "LAYERING_ASKS"    # bearish manipulation → actually bullish squeeze
    elif bid_count >= LAYER_MIN_WALLS and bid_count > ask_count:
        direction = "LAYERING_BIDS"    # bullish manipulation → actually bearish trap
    elif ask_count >= LAYER_MIN_WALLS and bid_count >= LAYER_MIN_WALLS:
        direction = "LAYERING_BOTH"
    else:
        direction = "CLEAN"

    return {
        "ask_layer_count": ask_count,
        "bid_layer_count": bid_count,
        "ask_layer_usd":   ask_usd,
        "bid_layer_usd":   bid_usd,
        "direction":       direction,
        "score":           min(10, (ask_count + bid_count) * 1.5),
    }


def detect_stop_hunt(spot: float, symbol: str, klines: List[dict]) -> dict:
    """
    Stop hunts target clustered stop-loss orders near key price levels.
    Key levels: round numbers, swing highs/lows, prev-day high/low.

    If spot is within 0.4% of a stop cluster AND price was away from it
    5+ bars ago → potential stop hunt setup.
    """
    base = symbol.replace("USDT", "")
    tick = ROUND_NUMBER_TICKS.get(base, spot * 0.01)

    # Round number levels within 2% of spot
    low_round  = math.floor(spot / tick) * tick
    round_levels = [low_round - tick, low_round, low_round + tick, low_round + 2 * tick]

    # Previous day high/low from klines
    day_bars = klines[-24:] if len(klines) >= 24 else klines
    pdh = max((b["high"] for b in day_bars), default=spot)
    pdl = min((b["low"]  for b in day_bars), default=spot)

    # Recent swing highs/lows (last 50 bars, local extrema)
    recent = klines[-50:] if len(klines) >= 50 else klines
    swing_highs, swing_lows = [], []
    for i in range(2, len(recent) - 2):
        if (recent[i]["high"] > recent[i-1]["high"] and
                recent[i]["high"] > recent[i-2]["high"] and
                recent[i]["high"] > recent[i+1]["high"] and
                recent[i]["high"] > recent[i+2]["high"]):
            swing_highs.append(recent[i]["high"])
        if (recent[i]["low"] < recent[i-1]["low"] and
                recent[i]["low"] < recent[i-2]["low"] and
                recent[i]["low"] < recent[i+1]["low"] and
                recent[i]["low"] < recent[i+2]["low"]):
            swing_lows.append(recent[i]["low"])

    all_levels = round_levels + [pdh, pdl] + swing_highs[-3:] + swing_lows[-3:]
    all_levels = [lvl for lvl in all_levels if lvl > 0]

    # Find levels within stop-hunt range
    near = []
    for lvl in all_levels:
        pct_away = abs(spot - lvl) / spot
        if pct_away <= STOP_HUNT_RANGE_PCT:
            label = ("ROUND_NUM" if lvl in round_levels else
                     "PDH" if lvl == pdh else
                     "PDL" if lvl == pdl else
                     "SWING_HIGH" if lvl in swing_highs else "SWING_LOW")
            near.append({
                "level":    round(lvl, 6),
                "label":    label,
                "pct_away": round(pct_away * 100, 3),
                "above":    spot < lvl,
            })

    near.sort(key=lambda x: x["pct_away"])

    hunt_above = sum(1 for n in near if n["above"])
    hunt_below = sum(1 for n in near if not n["above"])

    if hunt_above >= 2:
        direction = "HUNT_ABOVE"   # stops above will get triggered → price spike up then reverse
    elif hunt_below >= 2:
        direction = "HUNT_BELOW"
    elif near:
        direction = f"NEAR_{near[0]['label']}"
    else:
        direction = "CLEAR"

    return {
        "near_levels":   near[:5],
        "hunt_above":    hunt_above,
        "hunt_below":    hunt_below,
        "direction":     direction,
        "score":         min(10, len(near) * 2.5),
    }


def detect_wash_trading(trades: List[dict]) -> dict:
    """
    Wash trading = coordinated buy + sell at same price within short window.
    Creates fake volume to manipulate perceived liquidity.
    In Binance aggTrades: same rounded price, buy+sell within 1.5 seconds.
    """
    if not trades: return {"wash_pct": 0, "wash_usd": 0, "score": 0}

    buy_times:  Dict[float, List[int]] = defaultdict(list)
    sell_times: Dict[float, List[int]] = defaultdict(list)

    for t in trades:
        p = round(t["price"], 3)
        if not t["is_maker"]:
            buy_times[p].append(t["time"])
        else:
            sell_times[p].append(t["time"])

    wash_prices, wash_usd = [], 0.0
    matched_pairs: set = set()
    for price in buy_times:
        if price not in sell_times: continue
        for bt in buy_times[price]:
            for st in sell_times[price]:
                if abs(bt - st) <= WASH_WINDOW_MS:
                    pair_key = (price, min(bt, st), max(bt, st))
                    if pair_key not in matched_pairs:
                        matched_pairs.add(pair_key)
                        wash_prices.append(price)
                        # Average trade size at this price level
                        level_usd = statistics.mean(
                            t["usd"] for t in trades if round(t["price"], 3) == price)
                        wash_usd += level_usd * 2  # 1 buy + 1 sell
                    break

    wash_prices = list(set(wash_prices))
    total_vol = sum(t["usd"] for t in trades)
    # Cap at 50% — above that it's a data/precision artifact
    wash_pct  = min(50.0, wash_usd / max(total_vol, 1) * 100)

    return {
        "wash_prices": wash_prices[:10],
        "wash_count":  len(wash_prices),
        "wash_usd":    wash_usd,
        "wash_pct":    wash_pct,
        "score":       min(10, wash_pct * 2),
    }


def detect_quote_stuffing(trades: List[dict]) -> dict:
    """
    Quote stuffing = flood of tiny orders to slow competitor algos.
    Signal: unusually high trade count with unusually small average size.
    """
    if len(trades) < 50:
        return {"detected": False, "score": 0}

    avg_usd = statistics.mean(t["usd"] for t in trades)
    total   = len(trades)

    # Compare last 20% of trades vs first 80%
    split    = int(total * 0.8)
    recent   = trades[split:]
    r_avg    = statistics.mean(t["usd"] for t in recent) if recent else avg_usd
    r_count  = len(recent)
    expected_count = max(1, (total - split))

    # High count + tiny size = stuffing
    tiny_pct    = sum(1 for t in recent if t["usd"] < avg_usd * 0.1) / max(r_count, 1) * 100
    count_ratio = r_count / max(expected_count, 1)

    detected = tiny_pct > 60 and r_avg < avg_usd * 0.3
    score    = min(10, tiny_pct / 10) if detected else 0

    return {
        "detected":    detected,
        "tiny_pct":    round(tiny_pct, 1),
        "size_ratio":  round(r_avg / max(avg_usd, 1), 3),
        "score":       score,
    }


def compute_manipulation_score(spoof: dict, layer: dict, stop_hunt: dict,
                                wash: dict, stuffing: dict, spot: float) -> dict:
    """
    Fuse all manipulation signals into a single score and direction bias.
    Returns: manip_score (0-10), direction_bias (BULLISH/BEARISH/MIXED/CLEAN),
             manip_state_idx (for RL state)
    """
    s_spoof   = spoof.get("score", 0)
    s_layer   = layer.get("score", 0)
    s_hunt    = stop_hunt.get("score", 0)
    s_wash    = wash.get("score", 0)
    s_stuff   = stuffing.get("score", 0)

    manip_score = (s_spoof * 0.35 + s_layer * 0.25 + s_hunt * 0.20 +
                   s_wash * 0.10 + s_stuff * 0.10)

    # Direction bias from manipulation:
    bullish_manip = (
        spoof.get("direction") == "FAKE_RESISTANCE_BULLISH" or
        layer.get("direction") == "LAYERING_ASKS" or
        stop_hunt.get("direction") == "HUNT_ABOVE"
    )
    bearish_manip = (
        spoof.get("direction") == "FAKE_SUPPORT_BEARISH" or
        layer.get("direction") == "LAYERING_BIDS" or
        stop_hunt.get("direction") == "HUNT_BELOW"
    )

    if bullish_manip and not bearish_manip:
        direction = "BULLISH"
    elif bearish_manip and not bullish_manip:
        direction = "BEARISH"
    elif bullish_manip and bearish_manip:
        direction = "MIXED"
    else:
        direction = "CLEAN"

    # RL state index: 0=CLEAN, 1=FAKE_RESISTANCE(bull), 2=FAKE_SUPPORT(bear), 3=STOP_HUNT
    if direction == "BULLISH":
        state_idx = 1
    elif direction == "BEARISH":
        state_idx = 2
    elif stop_hunt.get("score", 0) > 3:
        state_idx = 3
    else:
        state_idx = 0

    return {
        "manip_score":   round(manip_score, 2),
        "direction":     direction,
        "state_idx":     state_idx,
        "components":    {"spoof": s_spoof, "layer": s_layer, "hunt": s_hunt,
                          "wash": s_wash, "stuffing": s_stuff},
    }


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — ML FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "ret_1h","ret_4h","ret_12h","ret_24h",
    "vol_ratio_1h","vol_ratio_4h",
    "body_ratio","upper_wick","lower_wick",
    "oi_chg_1h","oi_chg_4h",
    "funding_z","funding_abs",
    "momentum_5h","momentum_12h",
    "close_vs_vwap",
    "range_expansion",
    "trade_count_ratio",
]


def build_feature_matrix(klines: List[dict], oi_hist: List[dict],
                          fund_hist: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from historical klines + OI + funding.
    y: 0=DOWN, 1=FLAT, 2=UP  (next 1h price direction)
    """
    if len(klines) < 30:
        return np.array([]), np.array([])

    closes  = np.array([b["close"]  for b in klines], dtype=float)
    highs   = np.array([b["high"]   for b in klines], dtype=float)
    lows    = np.array([b["low"]    for b in klines], dtype=float)
    opens   = np.array([b["open"]   for b in klines], dtype=float)
    vols    = np.array([b["volume"] for b in klines], dtype=float)
    tcount  = np.array([b["trades"] for b in klines], dtype=float)
    ts      = np.array([b["ts"]     for b in klines], dtype="int64")

    # Align OI to klines by timestamp
    oi_map  = {d["ts"]: d["oi"] for d in oi_hist}
    oi_vals = np.array([oi_map.get(t, np.nan) for t in ts], dtype=float)
    # Forward-fill NaN
    for i in range(1, len(oi_vals)):
        if np.isnan(oi_vals[i]):
            oi_vals[i] = oi_vals[i-1]
    if np.isnan(oi_vals[0]): oi_vals[0] = oi_vals[~np.isnan(oi_vals)][0] if np.any(~np.isnan(oi_vals)) else 0

    # Align funding to klines (funding every 8h → forward-fill)
    fund_map = {}
    for d in fund_hist:
        fund_map[d["ts"]] = d["rate"]
    fund_vals = np.full(len(ts), 0.0)
    last_fund = 0.0
    for i, t in enumerate(ts):
        if t in fund_map: last_fund = fund_map[t]
        fund_vals[i] = last_fund

    # Compute VWAP over rolling 24 bars
    vwap24 = np.full(len(closes), np.nan)
    for i in range(23, len(closes)):
        w = vols[i-23:i+1]
        p = closes[i-23:i+1]
        vwap24[i] = np.sum(w * p) / max(np.sum(w), 1e-9)

    rows, labels = [], []
    n = len(closes)

    for i in range(25, n - 1):
        c = closes[i]
        if c <= 0: continue

        def pct(a, b): return (a - b) / max(abs(b), 1e-9) if b != 0 else 0.0

        ret_1h   = pct(closes[i],   closes[i-1])
        ret_4h   = pct(closes[i],   closes[i-4])
        ret_12h  = pct(closes[i],   closes[i-12])
        ret_24h  = pct(closes[i],   closes[i-24])

        vol_avg24 = float(np.mean(vols[i-24:i])) if i >= 24 else vols[i]
        vol_avg4  = float(np.mean(vols[i-4:i]))   if i >= 4  else vols[i]
        vol_r_1h  = vols[i] / max(vol_avg24, 1e-9)
        vol_r_4h  = float(np.mean(vols[i-4:i+1])) / max(vol_avg24, 1e-9)

        rng = highs[i] - lows[i]
        body = abs(closes[i] - opens[i])
        body_ratio   = body / max(rng, 1e-9)
        upper_wick   = (highs[i] - max(opens[i], closes[i])) / max(rng, 1e-9)
        lower_wick   = (min(opens[i], closes[i]) - lows[i])  / max(rng, 1e-9)

        oi_1h = pct(oi_vals[i], oi_vals[i-1])  if oi_vals[i-1] != 0 else 0.0
        oi_4h = pct(oi_vals[i], oi_vals[i-4])  if oi_vals[i-4] != 0 else 0.0

        fund_z   = _zscore(fund_vals[max(0, i-48):i+1])
        fund_abs = fund_vals[i]

        mom_5  = float(np.mean(np.diff(closes[i-5:i+1]))) / max(c, 1)
        mom_12 = float(np.mean(np.diff(closes[i-12:i+1]))) / max(c, 1)

        vwap_v = vwap24[i] if not np.isnan(vwap24[i]) else c
        c_vs_vwap = pct(c, vwap_v)

        rng_avg = float(np.mean(highs[i-12:i] - lows[i-12:i]))
        rng_exp = rng / max(rng_avg, 1e-9)

        tc_avg = float(np.mean(tcount[i-12:i]))
        tc_ratio = tcount[i] / max(tc_avg, 1e-9)

        feat = [
            ret_1h, ret_4h, ret_12h, ret_24h,
            vol_r_1h, vol_r_4h,
            body_ratio, upper_wick, lower_wick,
            oi_1h, oi_4h,
            fund_z, fund_abs,
            mom_5, mom_12,
            c_vs_vwap,
            rng_exp,
            tc_ratio,
        ]

        # Clip extreme values
        feat = [_clamp(v, -5, 5) for v in feat]
        rows.append(feat)

        # Label: next 1h return
        nxt = pct(closes[i+1], closes[i])
        if   nxt >  LABEL_UP_PCT:  lbl = 2  # UP
        elif nxt < -LABEL_DN_PCT:  lbl = 0  # DOWN
        else:                       lbl = 1  # FLAT

        labels.append(lbl)

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=int)


def extract_live_features(klines: List[dict], oi_hist: List[dict],
                           fund_hist: List[dict]) -> Optional[np.ndarray]:
    """Extract features from the most recent bar for live prediction."""
    if len(klines) < 30:
        return None
    X, _ = build_feature_matrix(klines, oi_hist, fund_hist)
    if X.shape[0] == 0:
        return None
    return X[-1:]   # last row = current bar's features


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — ML MODEL  (XGBoost ensemble)
# ═════════════════════════════════════════════════════════════════════════════

MODEL_PATH  = os.path.join(CACHE_DIR, "xgb_model.pkl")
SCALER_PATH = os.path.join(CACHE_DIR, "xgb_scaler.pkl")


class MLEnsemble:
    """
    XGBoost classifier trained on historical microstructure features.
    Predicts: 0=DOWN, 1=FLAT, 2=UP  for next 1h.
    """
    def __init__(self):
        self.model   = None
        self.scaler  = None
        self.trained = False
        self.feature_importance: Dict[str, float] = {}

    def train(self, symbols: List[str] = TRAIN_SYMBOLS):
        if not HAS_XGB or not HAS_SKLEARN:
            print("  [ML] XGBoost/sklearn not available — skipping ML training")
            return False

        print(f"  [ML] Fetching training data for {len(symbols)} symbols × {TRAIN_BARS} bars...")
        all_X, all_y = [], []

        for sym in symbols:
            try:
                klines   = fetch_klines(sym, "1h", TRAIN_BARS)
                oi_hist  = fetch_oi_history(sym, "1h", TRAIN_BARS)
                fund_h   = fetch_funding_hist(sym, 500)
                X, y     = build_feature_matrix(klines, oi_hist, fund_h)
                if X.shape[0] > 20:
                    all_X.append(X); all_y.append(y)
                    print(f"      {sym}: {X.shape[0]} samples  "
                          f"(UP={int((y==2).sum())} FLAT={int((y==1).sum())} DN={int((y==0).sum())})")
            except Exception as e:
                print(f"      {sym}: error — {e}")

        if not all_X:
            print("  [ML] No training data collected — skipping")
            return False

        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)

        self.scaler = StandardScaler()
        X_scaled    = self.scaler.fit_transform(X_all)

        self.model  = XGBClassifier(**XGB_PARAMS)
        self.model.fit(X_scaled, y_all)
        self.trained = True

        # Feature importance
        imp = self.model.feature_importances_
        self.feature_importance = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in imp]))

        # Save
        with open(MODEL_PATH,  "wb") as f: pickle.dump(self.model,  f)
        with open(SCALER_PATH, "wb") as f: pickle.dump(self.scaler, f)
        print(f"  [ML] Trained on {X_all.shape[0]} samples. "
              f"Top feature: {max(self.feature_importance, key=self.feature_importance.get)}")
        return True

    def load(self) -> bool:
        try:
            with open(MODEL_PATH,  "rb") as f: self.model  = pickle.load(f)
            with open(SCALER_PATH, "rb") as f: self.scaler = pickle.load(f)
            self.trained = True
            return True
        except Exception:
            return False

    def predict(self, features: np.ndarray) -> Tuple[int, float, List[float]]:
        """
        Returns (class, confidence, [prob_down, prob_flat, prob_up])
        """
        if not self.trained or features is None:
            return 1, 0.33, [0.33, 0.34, 0.33]
        try:
            X_s = self.scaler.transform(features)
            probs = self.model.predict_proba(X_s)[0]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred])
            return pred, conf, [float(p) for p in probs]
        except Exception:
            return 1, 0.33, [0.33, 0.34, 0.33]

    def predict_live(self, symbol: str, klines=None, oi_hist=None, fund_hist=None
                     ) -> Tuple[int, float, List[float]]:
        if klines is None:
            klines   = fetch_klines(symbol, "1h", 60)
            oi_hist  = fetch_oi_history(symbol, "1h", 60)
            fund_hist= fetch_funding_hist(symbol, 60)
        feat = extract_live_features(klines, oi_hist, fund_hist)
        return self.predict(feat)


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3 — RL TRADING AGENT  (Q-Learning, 300 discrete states)
# ═════════════════════════════════════════════════════════════════════════════

RL_PATH = os.path.join(CACHE_DIR, "rl_qtable.pkl")
ACTIONS = {0: "HOLD", 1: "LONG", 2: "SHORT"}

class QLearningAgent:
    """
    State space: 4 × 5 × 3 × 5 = 300 discrete states
      - oi_state    (4): FAST_BUILD / BUILD / FLAT / SHRINK
      - cvd_state   (5): STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
      - fund_state  (3): HIGH / NORMAL / LOW
      - ml_state    (5): STRONG_UP / UP / FLAT / DOWN / STRONG_DOWN

    Actions: 0=HOLD, 1=LONG, 2=SHORT
    """
    N_STATES  = 4 * 5 * 3 * 5   # 300
    N_ACTIONS = 3
    STATE_DIMS = (4, 5, 3, 5)

    def __init__(self):
        self.Q       = np.zeros((self.N_STATES, self.N_ACTIONS), dtype=np.float32)
        self.epsilon = RL_EPS_INIT
        self.steps   = 0

    # ── State Encoding ────────────────────────────────────────────
    @staticmethod
    def encode_oi(oi_vel_pct_hr: float) -> int:
        """OI velocity % per hour → discrete state."""
        if   oi_vel_pct_hr >  2.0: return 0  # FAST_BUILD
        elif oi_vel_pct_hr >  0.3: return 1  # BUILD
        elif oi_vel_pct_hr > -0.3: return 2  # FLAT
        else:                       return 3  # SHRINK

    @staticmethod
    def encode_cvd(cvd_ratio: float) -> int:
        """Buy/Sell ratio → discrete state."""
        if   cvd_ratio > 1.6:  return 0  # STRONG_BUY
        elif cvd_ratio > 1.15: return 1  # BUY
        elif cvd_ratio > 0.87: return 2  # NEUTRAL
        elif cvd_ratio > 0.65: return 3  # SELL
        else:                   return 4  # STRONG_SELL

    @staticmethod
    def encode_fund(fund_bps: float) -> int:
        """Funding rate bps → discrete state."""
        if   fund_bps >  2.5: return 0  # HIGH (long squeeze risk)
        elif fund_bps > -1.0: return 1  # NORMAL
        else:                  return 2  # LOW (short squeeze risk)

    @staticmethod
    def encode_ml(ml_class: int, ml_conf: float) -> int:
        """ML prediction class + confidence → discrete state."""
        if   ml_class == 2 and ml_conf > 0.55: return 0  # STRONG_UP
        elif ml_class == 2:                      return 1  # UP
        elif ml_class == 1:                      return 2  # FLAT
        elif ml_class == 0:                      return 3  # DOWN
        else:                                     return 4  # STRONG_DOWN

    def state_index(self, oi_vel: float, cvd_ratio: float,
                    fund_bps: float, ml_class: int, ml_conf: float) -> int:
        s1 = self.encode_oi(oi_vel)
        s2 = self.encode_cvd(cvd_ratio)
        s3 = self.encode_fund(fund_bps)
        s4 = self.encode_ml(ml_class, ml_conf)
        d  = self.STATE_DIMS
        return s1 * (d[1]*d[2]*d[3]) + s2 * (d[2]*d[3]) + s3 * d[3] + s4

    # ── Training ──────────────────────────────────────────────────
    def train(self, symbols: List[str] = TRAIN_SYMBOLS, episodes: int = RL_TRAIN_EP):
        """Train Q-table on historical data via simulated trades."""
        print(f"  [RL]  Training Q-agent on {len(symbols)} symbols × {episodes} episodes...")

        for sym in symbols:
            klines   = fetch_klines(sym, "1h", TRAIN_BARS)
            oi_hist  = fetch_oi_history(sym, "1h", TRAIN_BARS)
            fund_h   = fetch_funding_hist(sym, 500)

            if len(klines) < 30: continue

            closes = [b["close"] for b in klines]
            oi_map = {d["ts"]: d["oi"] for d in oi_hist}
            fund_map: Dict = {}
            for d in fund_h: fund_map[d["ts"]] = d["rate"]

            # Build per-bar features for state
            last_fund = 0.0
            bars_data = []
            prev_oi   = None
            for b in klines:
                oi_v = oi_map.get(b["ts"], 0)
                if b["ts"] in fund_map: last_fund = fund_map[b["ts"]]
                if prev_oi and prev_oi > 0:
                    oi_vel = (oi_v - prev_oi) / prev_oi * 100
                else:
                    oi_vel = 0.0
                prev_oi = oi_v
                bars_data.append({"close": b["close"], "oi_vel": oi_vel, "fund": last_fund * 10000})

            X_all, y_all = build_feature_matrix(klines, oi_hist, fund_h)

            for ep in range(episodes):
                self.epsilon = max(RL_EPS_MIN, RL_EPS_INIT * (RL_EPS_DECAY ** self.steps))

                for i in range(1, len(bars_data) - 1):
                    close_now  = bars_data[i]["close"]
                    close_next = bars_data[i+1]["close"] if i+1 < len(bars_data) else close_now

                    oi_vel   = bars_data[i]["oi_vel"]
                    fund_bps = bars_data[i]["fund"]

                    # Use volume as CVD proxy for historical (no actual CVD in history)
                    cvd_ratio = klines[i]["volume"] / max(
                        float(np.mean([k["volume"] for k in klines[max(0,i-12):i+1]])), 1e-9)
                    cvd_ratio = max(0.5, min(2.0, cvd_ratio))

                    # ML class from feature matrix if available
                    idx = i - 25
                    if 0 <= idx < len(X_all) if len(X_all) > 0 else False:
                        ml_class, ml_conf, _ = (1, 0.33, [])
                    else:
                        ml_class, ml_conf = 1, 0.33

                    s = self.state_index(oi_vel, cvd_ratio, fund_bps, ml_class, ml_conf)

                    # ε-greedy action
                    if np.random.rand() < self.epsilon:
                        a = np.random.randint(self.N_ACTIONS)
                    else:
                        a = int(np.argmax(self.Q[s]))

                    # Reward = PnL from holding 1 bar
                    ret = (close_next - close_now) / max(close_now, 1e-9)
                    if   a == 1: reward = ret   * 100   # LONG
                    elif a == 2: reward = -ret  * 100   # SHORT
                    else:        reward = -0.001          # HOLD: tiny holding cost

                    # Next state (approximate)
                    if i + 1 < len(bars_data):
                        n_oi  = bars_data[i+1]["oi_vel"]
                        n_cvd = klines[i+1]["volume"] / max(
                            float(np.mean([k["volume"] for k in klines[max(0,i-11):i+2]])), 1e-9)
                        n_cvd = max(0.5, min(2.0, n_cvd))
                        s_next = self.state_index(n_oi, n_cvd, fund_bps, ml_class, ml_conf)
                    else:
                        s_next = s

                    # Q update
                    td = reward + RL_GAMMA * np.max(self.Q[s_next]) - self.Q[s, a]
                    self.Q[s, a] += RL_ALPHA * td
                    self.steps += 1

        with open(RL_PATH, "wb") as f: pickle.dump((self.Q, self.steps), f)
        print(f"  [RL]  Trained. Steps={self.steps}, ε={self.epsilon:.3f}")

    def load(self) -> bool:
        try:
            with open(RL_PATH, "rb") as f:
                self.Q, self.steps = pickle.load(f)
            self.epsilon = max(RL_EPS_MIN, RL_EPS_INIT * (RL_EPS_DECAY ** self.steps))
            return True
        except Exception:
            return False

    def predict(self, oi_vel: float, cvd_ratio: float, fund_bps: float,
                ml_class: int, ml_conf: float) -> Tuple[int, float]:
        """Return (best_action, q_value_confidence)."""
        s   = self.state_index(oi_vel, cvd_ratio, fund_bps, ml_class, ml_conf)
        q   = self.Q[s]
        a   = int(np.argmax(q))
        # Confidence = softmax of Q-values
        q_exp = np.exp(q - np.max(q))
        conf  = float(q_exp[a] / q_exp.sum())
        return a, conf

    def update_online(self, oi_vel, cvd_ratio, fund_bps, ml_class, ml_conf,
                      action: int, reward: float,
                      next_oi_vel, next_cvd, next_fund, next_ml_class, next_ml_conf):
        """Online update from a completed paper trade."""
        s      = self.state_index(oi_vel, cvd_ratio, fund_bps, ml_class, ml_conf)
        s_next = self.state_index(next_oi_vel, next_cvd, next_fund, next_ml_class, next_ml_conf)
        td = reward + RL_GAMMA * np.max(self.Q[s_next]) - self.Q[s, action]
        self.Q[s, action] += RL_ALPHA * td


# ═════════════════════════════════════════════════════════════════════════════
#  INSTITUTIONAL FLOW (import from crypto_institutional_flow)
# ═════════════════════════════════════════════════════════════════════════════

def _get_inst_flow(symbol: str) -> dict:
    """
    Import the deep scan from crypto_institutional_flow if available,
    else run a lightweight inline version.
    """
    try:
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location(
            "cif", os.path.join(os.path.dirname(__file__), "crypto_institutional_flow.py"))
        cif  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cif)
        return cif.scan_symbol_deep(symbol)
    except Exception:
        pass

    # Fallback: lightweight inline flow
    trades = fetch_agg_trades(symbol)
    ob     = fetch_order_book(symbol, 200)
    spot   = _safe(_get(f"{BINANCE_FAPI}/fapi/v1/ticker/price", {"symbol": symbol},
                        ).get("price", 0) if isinstance(
                _get(f"{BINANCE_FAPI}/fapi/v1/ticker/price", {"symbol": symbol}), dict) else {})
    if spot <= 0: spot = fetch_spot_price(symbol)

    buy_usd  = sum(t["usd"] for t in trades if not t["is_maker"])
    sell_usd = sum(t["usd"] for t in trades if t["is_maker"])
    cum_delta = buy_usd - sell_usd
    cvd_ratio = buy_usd / max(sell_usd, 1e-9)

    asks = [(float(p), float(q)) for p, q in ob.get("asks", [])[:50]]
    bids = [(float(p), float(q)) for p, q in ob.get("bids", [])[:50]]
    bid_usd = sum(p*q for p,q in bids[:20])
    ask_usd = sum(p*q for p,q in asks[:20])

    top_ls  = fetch_long_short_ratio(symbol)
    fund    = fetch_funding_rate(symbol)

    return {
        "symbol": symbol, "spot": spot,
        "flow": {"cum_delta": cum_delta, "cvd_ratio": cvd_ratio,
                 "buy_usd": buy_usd, "sell_usd": sell_usd},
        "ob": {"ob_bias": "BID_HEAVY" if bid_usd > ask_usd * 1.2 else
                          ("ASK_HEAVY" if ask_usd > bid_usd * 1.2 else "BALANCED")},
        "oi_flow": {"oi_velocity_1h": 0.5, "flow_type": "NEUTRAL_FLOW"},
        "positioning": {"funding_bps": fund, "top_position_ls": top_ls,
                        "pos_score": 1 if top_ls > 1.5 else (-1 if top_ls < 0.7 else 0)},
        "liq_risk": {}, "verdict": {"action": "HOLD", "conviction": "LOW",
                                     "phase": "NEUTRAL", "total_score": 0},
    }


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — SIGNAL FUSION + TRADE RECOMMENDATION
# ═════════════════════════════════════════════════════════════════════════════

def _flow_score(result: dict) -> float:
    """Convert institutional flow verdict to -1..+1 score."""
    v = result.get("verdict", {})
    s = _safe(v.get("total_score", 0))
    return _clamp(s / 3.0, -1.0, 1.0)


def _ml_score(ml_class: int, ml_conf: float) -> float:
    """Convert ML prediction to -1..+1 score."""
    if   ml_class == 2: return  ml_conf      # UP
    elif ml_class == 0: return -ml_conf      # DOWN
    else:               return  0.0           # FLAT


def _rl_score(rl_action: int, rl_conf: float) -> float:
    """Convert RL action to -1..+1 score."""
    if   rl_action == 1: return  rl_conf     # LONG
    elif rl_action == 2: return -rl_conf     # SHORT
    else:                return  0.0


def _manip_score_normalized(manip: dict) -> float:
    """Manipulation direction as -1..+1 score."""
    d = manip.get("direction", "CLEAN")
    s = manip.get("manip_score", 0) / 10.0  # 0..1
    if d == "BULLISH": return  s
    if d == "BEARISH": return -s
    return 0.0


def _kelly_fraction(win_prob: float, win_loss_ratio: float = 2.0,
                    max_kelly: float = 0.10) -> float:
    """
    Kelly criterion position size.
    win_prob: probability of win (0..1)
    win_loss_ratio: typical risk-reward
    Returns: fraction of portfolio (capped at max_kelly)
    """
    q   = 1 - win_prob
    k   = win_prob - q / win_loss_ratio
    return round(max(0.0, min(max_kelly, k)), 4)


def _atr_estimate(klines: List[dict], period: int = 14) -> float:
    """Simple ATR from klines."""
    if len(klines) < period + 1: return 0.0
    trs = []
    for i in range(1, len(klines)):
        h, l, pc = klines[i]["high"], klines[i]["low"], klines[i-1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs[-period:]))


def fuse_signals(
        symbol: str, spot: float,
        inst_result: dict,
        manip:       dict,
        ml_class:    int, ml_conf: float, ml_probs: List[float],
        rl_action:   int, rl_conf: float,
        klines:      List[dict],
) -> dict:
    """
    Combine all 4 signal layers into one alpha score and trade plan.
    """
    fs = _flow_score(inst_result)
    ms = _ml_score(ml_class, ml_conf)
    rs = _rl_score(rl_action, rl_conf)
    ps = _manip_score_normalized(manip)

    # Weighted combination
    alpha = fs * W_FLOW + ms * W_ML + rs * W_RL + ps * W_MANIP
    alpha = round(_clamp(alpha, -1.0, 1.0), 4)

    # Direction
    if   alpha >= 0.35:  direction = "STRONG_LONG"
    elif alpha >= 0.15:  direction = "LONG"
    elif alpha <= -0.35: direction = "STRONG_SHORT"
    elif alpha <= -0.15: direction = "SHORT"
    else:                direction = "HOLD"

    # Conviction = agreement between signals
    signs = [
        1 if fs > 0.1 else (-1 if fs < -0.1 else 0),
        1 if ms > 0.1 else (-1 if ms < -0.1 else 0),
        1 if rs > 0.1 else (-1 if rs < -0.1 else 0),
        1 if ps > 0.1 else (-1 if ps < -0.1 else 0),
    ]
    agree = abs(sum(signs))
    if   agree == 4: conviction = "MAXIMUM"
    elif agree == 3: conviction = "HIGH"
    elif agree == 2: conviction = "MEDIUM"
    elif agree == 1: conviction = "LOW"
    else:            conviction = "CONFLICTED"

    # Trade plan
    atr   = _atr_estimate(klines) if klines else spot * 0.005
    if atr <= 0: atr = spot * 0.005

    if direction in ("STRONG_LONG", "LONG"):
        entry  = spot
        stop   = round(spot - 1.5 * atr, 6)
        tp1    = round(spot + 2.0 * atr, 6)
        tp2    = round(spot + 3.5 * atr, 6)
        rr     = (tp1 - entry) / max(entry - stop, 1e-9)
    elif direction in ("STRONG_SHORT", "SHORT"):
        entry  = spot
        stop   = round(spot + 1.5 * atr, 6)
        tp1    = round(spot - 2.0 * atr, 6)
        tp2    = round(spot - 3.5 * atr, 6)
        rr     = (entry - tp1) / max(stop - entry, 1e-9)
    else:
        entry = stop = tp1 = tp2 = spot
        rr    = 0.0

    # Kelly position sizing
    win_prob   = 0.5 + abs(alpha) * 0.3
    kelly_frac = _kelly_fraction(win_prob, max(rr, 1.5))

    return {
        "symbol":        symbol,
        "spot":          spot,
        "alpha":         alpha,
        "direction":     direction,
        "conviction":    conviction,

        # Component scores
        "flow_score":    round(fs, 3),
        "ml_score":      round(ms, 3),
        "rl_score":      round(rs, 3),
        "manip_score":   round(ps, 3),

        # ML details
        "ml_class":      ml_class,
        "ml_conf":       round(ml_conf, 3),
        "ml_probs":      [round(p, 3) for p in ml_probs],

        # RL details
        "rl_action":     ACTIONS[rl_action],
        "rl_conf":       round(rl_conf, 3),

        # Manipulation
        "manip_direction": manip.get("direction", "CLEAN"),
        "manip_raw_score": manip.get("manip_score", 0),

        # Trade plan
        "entry":         entry,
        "stop_loss":     stop,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "risk_reward":   round(rr, 2),
        "kelly_size":    kelly_frac,
        "atr":           round(atr, 6),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  FULL ALPHA SCAN  (per symbol)
# ═════════════════════════════════════════════════════════════════════════════

def alpha_scan_symbol(symbol: str, ml: MLEnsemble, rl: QLearningAgent,
                      deep: bool = False) -> dict:
    """
    Run all 4 layers for one symbol. Returns fused alpha result.
    """
    t0 = time.time()

    # Fetch core data
    trades = fetch_agg_trades(symbol)
    ob     = fetch_order_book(symbol, 500)
    klines = fetch_klines(symbol, "1h", 60)
    oi_h   = fetch_oi_history(symbol, "1h", 60)
    fund_h = fetch_funding_hist(symbol, 60)
    spot   = klines[-1]["close"] if klines else fetch_spot_price(symbol)

    # Layer 1: Manipulation
    spoof     = detect_spoofing_deep(symbol, ob, spot) if deep else detect_spoofing_static(ob, spot)
    layering  = detect_layering(ob, spot)
    stop_hunt = detect_stop_hunt(spot, symbol, klines)
    wash      = detect_wash_trading(trades)
    stuffing  = detect_quote_stuffing(trades)
    manip     = compute_manipulation_score(spoof, layering, stop_hunt, wash, stuffing, spot)

    # Layer 2: ML
    ml_class, ml_conf, ml_probs = ml.predict_live(symbol, klines, oi_h, fund_h)

    # Layer 3: RL
    oi_vel   = 0.5   # default; will be replaced from inst flow
    cvd_ratio = 1.0
    fund_bps  = fetch_funding_rate(symbol)

    if trades:
        buy_usd  = sum(t["usd"] for t in trades if not t["is_maker"])
        sell_usd = sum(t["usd"] for t in trades if t["is_maker"])
        cvd_ratio = buy_usd / max(sell_usd, 1e-9)

    if len(oi_h) >= 2:
        oi_vel = (oi_h[-1]["oi"] - oi_h[-2]["oi"]) / max(oi_h[-2]["oi"], 1) * 100

    rl_action, rl_conf = rl.predict(oi_vel, cvd_ratio, fund_bps, ml_class, ml_conf)

    # Layer 4: Institutional flow (import or inline)
    inst = _get_inst_flow(symbol)

    # Fuse
    result = fuse_signals(symbol, spot, inst, manip,
                          ml_class, ml_conf, ml_probs,
                          rl_action, rl_conf, klines)
    result["elapsed"]    = round(time.time() - t0, 1)
    result["spoof"]      = spoof
    result["layering"]   = layering
    result["stop_hunt"]  = stop_hunt
    result["wash"]       = wash
    result["stuffing"]   = stuffing
    result["inst"]       = inst
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

W = 120

LABEL_CLS  = {2: "UP  ", 1: "FLAT", 0: "DOWN"}
LABEL_ACT  = {0: "HOLD", 1: "LONG", 2: "SHRT"}
COL_G  = "\033[92m"; COL_R = "\033[91m"; COL_Y = "\033[93m"
COL_C  = "\033[96m"; COL_W = "\033[37m"; RESET = "\033[0m"
BOLD   = "\033[1m";  DIM   = "\033[2m"

def _c(val, s): return (COL_G if val > 0 else (COL_R if val < 0 else COL_W)) + s + RESET
def _b(s):      return BOLD + s + RESET

def print_alpha_report(results: List[dict]):
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("\n" + "═" * W)
    print(f"  {BOLD}INSTITUTIONAL ALPHA ENGINE v3{RESET}  —  {now}")
    print(f"  {'Layer 1: Manipulation':30}  {'Layer 2: ML XGBoost':30}  {'Layer 3: RL Agent':20}  {'Layer 4: Flow':15}")
    print("═" * W)

    # Summary table
    hdr = (f"  {'Symbol':<12} {'Price':>12} {'Alpha':>7} {'Dir':>14} {'Conv':>10}"
           f"  {'ML':>8} {'RL':>8} {'Manip':>12}  {'Entry':>12} {'SL':>12} {'TP1':>12}  {'R:R':>5} {'Kelly':>6}")
    print(hdr)
    print("─" * W)

    ranked = sorted(results, key=lambda x: -abs(x.get("alpha", 0)))
    for r in ranked:
        alpha = r.get("alpha", 0)
        dir_  = r.get("direction", "HOLD")
        conv  = r.get("conviction", "LOW")
        price = r.get("spot", 0)
        ml_c  = LABEL_CLS.get(r.get("ml_class", 1), "FLAT")
        ml_p  = r.get("ml_conf", 0)
        rl_a  = r.get("rl_action", "HOLD")
        manip = r.get("manip_direction", "CLEAN")
        entry = r.get("entry", 0)
        sl    = r.get("stop_loss", 0)
        tp1   = r.get("take_profit_1", 0)
        rr    = r.get("risk_reward", 0)
        kelly = r.get("kelly_size", 0)

        dir_col = COL_G if "LONG" in dir_ else (COL_R if "SHORT" in dir_ else COL_W)
        alpha_s = _c(alpha, f"{alpha:+.3f}")
        entry_s = f"${entry:,.2f}" if price > 100 else f"${entry:.5f}"
        sl_s    = f"${sl:,.2f}"   if price > 100 else f"${sl:.5f}"
        tp1_s   = f"${tp1:,.2f}"  if price > 100 else f"${tp1:.5f}"

        price_s = f"${price:,.2f}" if price > 100 else f"${price:.4f}"
        manip_c = COL_G if manip == "BULLISH" else (COL_R if manip == "BEARISH" else COL_W)

        print(f"  {r['symbol']:<12} {price_s:>12} {alpha_s:>7}  "
              f"{dir_col}{dir_:>14}{RESET}  {conv:>10}"
              f"  {ml_c}@{ml_p:.0%}  {rl_a:>8}  "
              f"{manip_c}{manip:>12}{RESET}"
              f"  {entry_s:>12} {sl_s:>12} {tp1_s:>12}"
              f"  {rr:>5.1f}  {kelly:>5.1%}")

    # Detailed breakdown
    print("\n" + "─" * W)
    print(f"  DETAILED SIGNAL BREAKDOWN")
    print("─" * W)

    for r in ranked:
        alpha = r.get("alpha", 0)
        dir_  = r.get("direction", "HOLD")
        if abs(alpha) < 0.10 and dir_ == "HOLD": continue   # skip flat noise

        dir_col = COL_G if "LONG" in dir_ else (COL_R if "SHORT" in dir_ else COL_Y)
        price_s = f"${r['spot']:,.4f}"

        print(f"\n  ┌─── {_b(r['symbol'])}  {price_s}  "
              f"{dir_col}[{dir_}]{RESET}  alpha={_c(alpha, f'{alpha:+.3f}')}  "
              f"Conv={_b(r['conviction'])} ───")

        # Score breakdown bar
        fs = r.get("flow_score", 0); ms = r.get("ml_score", 0)
        rs = r.get("rl_score",  0); ps = r.get("manip_score", 0)
        print(f"  │  Scores:  Flow={_c(fs,f'{fs:+.3f}')}({W_FLOW:.0%})  "
              f"ML={_c(ms,f'{ms:+.3f}')}({W_ML:.0%})  "
              f"RL={_c(rs,f'{rs:+.3f}')}({W_RL:.0%})  "
              f"Manip={_c(ps,f'{ps:+.3f}')}({W_MANIP:.0%})")

        # ML prediction
        probs = r.get("ml_probs", [0.33, 0.34, 0.33])
        p_dn, p_fl, p_up = probs[0], probs[1], probs[2]
        print(f"  │  ML Pred: {_c(p_up-p_dn, LABEL_CLS[r['ml_class']])}  "
              f"DN={COL_R}{p_dn:.1%}{RESET} FLAT={COL_W}{p_fl:.1%}{RESET} UP={COL_G}{p_up:.1%}{RESET}  "
              f"Conf={r['ml_conf']:.1%}")

        # RL
        rl_act = r.get("rl_action", "HOLD"); rl_c = r.get("rl_conf", 0)
        rl_col = COL_G if rl_act == "LONG" else (COL_R if rl_act == "SHORT" else COL_W)
        print(f"  │  RL Agent: {rl_col}{rl_act}{RESET} (Q-conf={rl_c:.1%})")

        # Manipulation
        spoof    = r.get("spoof", {})
        layer    = r.get("layering", {})
        s_hunt   = r.get("stop_hunt", {})
        wash     = r.get("wash", {})
        stuffing = r.get("stuffing", {})
        m_dir    = r.get("manip_direction", "CLEAN")
        m_score = r.get("manip_raw_score", 0)
        m_col   = COL_G if m_dir == "BULLISH" else (COL_R if m_dir == "BEARISH" else COL_W)
        print(f"  │  Manipulation: {m_col}{m_dir}{RESET}  score={m_score:.1f}/10")

        total_spoofs = spoof.get("total_spoofs", 0)
        if total_spoofs > 0:
            sp_dir = spoof.get("direction", "")
            sp_col = COL_G if "BULLISH" in sp_dir else (COL_R if "BEARISH" in sp_dir else COL_W)
            print(f"  │    SPOOFING  {sp_col}[{sp_dir}]{RESET}  "
                  f"FakeAsks=${spoof.get('spoof_ask_usd',0)/1e6:.2f}M  "
                  f"FakeBids=${spoof.get('spoof_bid_usd',0)/1e6:.2f}M  "
                  f"({total_spoofs} isolated walls)")
            # Show top spoofs
            for s in (spoof.get("ask_spoofs", []) + spoof.get("bid_spoofs", []))[:2]:
                print(f"  │      {s['side']} wall ${s['usd']/1e3:.0f}k @ {s['price']:.4f} "
                      f"({s['isolated']:.0f}× neighbours, {s['pct_from_spot']:.2f}% from spot)")

        l_dir = layer.get("direction", "CLEAN")
        if l_dir != "CLEAN":
            l_col = COL_Y
            print(f"  │    LAYERING  {l_col}[{l_dir}]{RESET}  "
                  f"AskWalls={layer['ask_layer_count']}(${layer['ask_layer_usd']/1e3:.0f}k)  "
                  f"BidWalls={layer['bid_layer_count']}(${layer['bid_layer_usd']/1e3:.0f}k)")

        near = s_hunt.get("near_levels", [])
        if near:
            h_col = COL_Y
            print(f"  │    STOP-HUNT  {h_col}[{s_hunt['direction']}]{RESET}  "
                  f"Clusters above={s_hunt['hunt_above']}  below={s_hunt['hunt_below']}")
            for n in near[:3]:
                side_s = "▲ ABOVE" if n["above"] else "▼ BELOW"
                print(f"  │      {side_s}  {n['label']} @ {n['level']:.4f}  ({n['pct_away']:.3f}% away)")

        if wash.get("wash_count", 0) > 0:
            print(f"  │    WASH TRADE  score={wash['score']:.1f}  "
                  f"volume={wash['wash_pct']:.1f}% fake  "
                  f"${wash['wash_usd']/1e3:.0f}k wash")

        if stuffing.get("detected", False):
            print(f"  │    QUOTE STUFFING  tiny_pct={stuffing['tiny_pct']:.0f}%  "
                  f"size_ratio={stuffing['size_ratio']:.2f}×")

        # Trade plan
        entry = r.get("entry", 0); sl = r.get("stop_loss", 0)
        tp1 = r.get("take_profit_1", 0); tp2 = r.get("take_profit_2", 0)
        rr = r.get("risk_reward", 0); kelly = r.get("kelly_size", 0)
        atr = r.get("atr", 0)
        if dir_ not in ("HOLD",):
            dir_col = COL_G if "LONG" in dir_ else COL_R
            fmt = lambda v: f"${v:,.2f}" if v > 100 else f"${v:.5f}"
            print(f"  │")
            print(f"  │  ⚡ TRADE PLAN ({dir_col}{dir_}{RESET})")
            print(f"  │    Entry:   {fmt(entry)}")
            print(f"  │    Stop:    {COL_R}{fmt(sl)}{RESET}  ({abs(entry-sl)/max(entry,1)*100:.2f}% risk)")
            print(f"  │    TP1:     {COL_G}{fmt(tp1)}{RESET}  R:R = {rr:.1f}×")
            print(f"  │    TP2:     {COL_G}{fmt(tp2)}{RESET}  (runner)")
            print(f"  │    Kelly:   {COL_C}{kelly:.1%}{RESET} of portfolio  ATR={fmt(atr)}")

        print(f"  └{'─'*(W-4)}")

    # Global summary
    strong = [r for r in results if r.get("conviction") in ("MAXIMUM","HIGH")
              and r.get("direction") not in ("HOLD",)]
    print(f"\n{'─'*W}")
    print(f"  ALPHA SUMMARY")
    print(f"{'─'*W}")
    if strong:
        for r in sorted(strong, key=lambda x: -abs(x["alpha"])):
            dir_col = COL_G if "LONG" in r["direction"] else COL_R
            print(f"  {dir_col}{'▲' if 'LONG' in r['direction'] else '▼'}{RESET} "
                  f"{r['symbol']:<10}  alpha={r['alpha']:+.3f}  conv={r['conviction']}"
                  f"  ML={LABEL_CLS[r['ml_class']]}@{r['ml_conf']:.0%}"
                  f"  RL={r['rl_action']:4}  Manip={r['manip_direction']}"
                  f"  Kelly={r['kelly_size']:.1%}")
    else:
        print(f"  No high-conviction signals. Market in consolidation.")

    best_ml = max(results, key=lambda x: x["ml_conf"])
    print(f"\n  Best ML confidence: {best_ml['symbol']} → "
          f"{LABEL_CLS[best_ml['ml_class']]} @ {best_ml['ml_conf']:.1%}")
    print("═" * W + "\n")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Crypto Alpha Engine v3 — Manipulation + ML + RL + Flow")
    parser.add_argument("--symbols",    nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--quick",      action="store_true", help="BTC+ETH only")
    parser.add_argument("--deep",       action="store_true", help="8s OB spoof confirmation per symbol")
    parser.add_argument("--watch",      type=int, default=0,  metavar="SEC")
    parser.add_argument("--train-only", action="store_true",  help="Retrain models and exit")
    parser.add_argument("--retrain",    action="store_true",  help="Force retrain even if cache exists")
    parser.add_argument("--json",       action="store_true",  help="Save results to JSON")
    args = parser.parse_args()

    if args.quick:
        args.symbols = ["BTCUSDT", "ETHUSDT"]

    # ── Init models ──────────────────────────────────────────────
    ml = MLEnsemble()
    rl = QLearningAgent()

    ml_loaded = (not args.retrain) and ml.load()
    rl_loaded = (not args.retrain) and rl.load()

    if not ml_loaded:
        ml.train(TRAIN_SYMBOLS)
    else:
        print(f"  [ML] Loaded cached model from {MODEL_PATH}")

    if not rl_loaded:
        rl.train(TRAIN_SYMBOLS, RL_TRAIN_EP)
    else:
        print(f"  [RL] Loaded Q-table ({rl.steps} steps, ε={rl.epsilon:.3f})")

    if args.train_only:
        print("  Training complete. Exiting (--train-only).")
        return

    # ── Scan loop ────────────────────────────────────────────────
    def _run():
        print(f"\n  {'═'*62}")
        print(f"  INSTITUTIONAL ALPHA ENGINE v3 — Scanning {len(args.symbols)} symbols")
        print(f"  {'═'*62}\n")

        results = []
        for sym in args.symbols:
            try:
                sys.stdout.write(f"  [{sym}] Scanning all 4 layers...")
                sys.stdout.flush()
                r = alpha_scan_symbol(sym, ml, rl, deep=args.deep)
                sys.stdout.write(f" done ({r['elapsed']}s)\n")
                sys.stdout.flush()
                results.append(r)
            except Exception as e:
                import traceback
                print(f"\n  ERROR [{sym}]: {e}")
                traceback.print_exc()

        if results:
            print_alpha_report(results)

        if args.json:
            def _jx(o):
                if isinstance(o, (np.integer,)):  return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, np.ndarray):     return o.tolist()
                raise TypeError(type(o))
            # Strip heavy nested dicts for JSON
            clean = []
            for r in results:
                rc = {k: v for k, v in r.items()
                      if k not in ("inst", "spoof", "layering", "stop_hunt", "wash", "stuffing")}
                clean.append(rc)
            with open("alpha_results.json", "w") as f:
                json.dump(clean, f, default=_jx, indent=2)
            print("  Results → alpha_results.json\n")

        return results

    _run()

    if args.watch > 0:
        print(f"  [WATCH] Re-scanning every {args.watch}s. Ctrl+C to stop.\n")
        while True:
            try:
                time.sleep(args.watch)
                print("\033[2J\033[H")
                _run()
            except KeyboardInterrupt:
                print("\n  Watch mode stopped.")
                break


if __name__ == "__main__":
    main()
