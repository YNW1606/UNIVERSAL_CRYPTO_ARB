"""
Institutional Intelligence Layer
=================================
Fetches open interest, top-trader positioning, global L/S ratio, and funding
from Binance Futures (all free, no API key needed) to build a composite
institutional_score (-100..+100) that gates directional signals.

A POSITIVE score = institutional flow is bullish (supports LONG).
A NEGATIVE score = institutional flow is bearish (supports SHORT).

Usage:
    from institutional_intel import get_institutional_signal, get_btc_regime
    intel = get_institutional_signal("BTCUSDT")
    regime = get_btc_regime()   # "BULL" | "BEAR" | "SIDEWAYS"
"""

import time
import requests
from datetime import datetime, timezone

FAPI = "https://fapi.binance.com"
BAPI = "https://api.binance.com"
_TIMEOUT = 7

# In-process cache: symbol → (timestamp, result)
_CACHE: dict = {}
_CACHE_TTL = 300          # seconds — one result per 5-min scan window
_REGIME_CACHE: dict = {}  # {"ts": float, "regime": str, "pct4h": float}
_REGIME_TTL  = 600        # 10 min

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _neutral() -> dict:
    """Return a fully-neutral result (all APIs failed)."""
    return {
        "oi_trend":           "FLAT",
        "oi_pct_1h":          0.0,
        "top_trader_bias":    "NEUTRAL",
        "top_long_ratio":     1.0,
        "global_long_ratio":  0.5,
        "funding_trend":      "STABLE",
        "latest_funding":     0.0,
        "institutional_score": 0,
        "confidence":         "LOW",
        "data_quality":       "API_FAIL",
    }


# ── Open Interest ─────────────────────────────────────────────────────────────

def _get_oi(symbol: str) -> dict:
    """
    Returns {"oi_trend": "RISING|FALLING|FLAT", "oi_pct_1h": float}
    Uses /futures/data/openInterestHist (free, no key).
    """
    data = _get(f"{FAPI}/futures/data/openInterestHist",
                {"symbol": symbol, "period": "1h", "limit": 5})
    if not data or len(data) < 3:
        return {"oi_trend": "FLAT", "oi_pct_1h": 0.0}

    try:
        ois = [float(d["sumOpenInterest"]) for d in data]
        # Latest vs average of earlier 3
        latest   = ois[-1]
        avg_prev = sum(ois[:3]) / 3
        pct_chg  = (latest - avg_prev) / avg_prev * 100 if avg_prev else 0.0

        if pct_chg > 0.5:
            trend = "RISING"
        elif pct_chg < -0.5:
            trend = "FALLING"
        else:
            trend = "FLAT"

        return {"oi_trend": trend, "oi_pct_1h": round(pct_chg, 3)}
    except Exception:
        return {"oi_trend": "FLAT", "oi_pct_1h": 0.0}


# ── Top Trader Positioning ────────────────────────────────────────────────────

def _get_top_traders(symbol: str) -> dict:
    """
    Returns {"top_trader_bias": "LONG|SHORT|NEUTRAL", "top_long_ratio": float}
    Uses /futures/data/topLongShortAccountRatio.
    """
    data = _get(f"{FAPI}/futures/data/topLongShortAccountRatio",
                {"symbol": symbol, "period": "1h", "limit": 3})
    if not data or len(data) < 1:
        return {"top_trader_bias": "NEUTRAL", "top_long_ratio": 1.0}

    try:
        # Average of recent prints
        ratios = [float(d.get("longShortRatio", 1.0)) for d in data]
        avg_ratio = sum(ratios) / len(ratios)

        if avg_ratio >= 1.15:
            bias = "LONG"
        elif avg_ratio <= 0.85:
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        return {"top_trader_bias": bias, "top_long_ratio": round(avg_ratio, 3)}
    except Exception:
        return {"top_trader_bias": "NEUTRAL", "top_long_ratio": 1.0}


# ── Global Retail L/S Ratio ───────────────────────────────────────────────────

def _get_global_ratio(symbol: str) -> dict:
    """
    Returns {"global_long_ratio": float}  (fraction longs, e.g. 0.60 = 60% long)
    Uses /futures/data/globalLongShortAccountRatio.
    """
    data = _get(f"{FAPI}/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": "1h", "limit": 3})
    if not data or len(data) < 1:
        return {"global_long_ratio": 0.5}

    try:
        # Latest ratio (longShortRatio = long/short, NOT long fraction)
        ratio = float(data[-1].get("longShortRatio", 1.0))
        # Convert to long fraction: ratio = long/short → long_frac = ratio / (1 + ratio)
        long_frac = ratio / (1.0 + ratio)
        return {"global_long_ratio": round(long_frac, 3)}
    except Exception:
        return {"global_long_ratio": 0.5}


# ── Funding Rate Trend ────────────────────────────────────────────────────────

def _get_funding(symbol: str) -> dict:
    """
    Returns {"funding_trend": str, "latest_funding": float}
    Uses /fapi/v1/fundingRate.
    """
    data = _get(f"{FAPI}/fapi/v1/fundingRate",
                {"symbol": symbol, "limit": 5})
    if not data or len(data) < 3:
        return {"funding_trend": "STABLE", "latest_funding": 0.0}

    try:
        rates = [float(d["fundingRate"]) for d in data]
        latest   = rates[-1]
        avg_prev = sum(rates[:3]) / 3

        if latest > avg_prev * 1.3 and latest > 0.0001:
            trend = "RISING"
        elif latest < avg_prev * 0.7 or (latest < -0.0001 and avg_prev >= 0):
            trend = "FALLING"
        else:
            trend = "STABLE"

        return {"funding_trend": trend, "latest_funding": round(latest, 7)}
    except Exception:
        return {"funding_trend": "STABLE", "latest_funding": 0.0}


# ── Composite Score ───────────────────────────────────────────────────────────

def _compute_score(oi: dict, top: dict, global_r: dict, fund: dict) -> tuple:
    """
    Returns (institutional_score: int, confidence: str)

    Score components (all additive, clamped -100..+100):
    +30 / -30 : OI rising + top traders LONG (confirms buying) / SHORT (confirms selling)
    +20 / -20 : OI rising + top traders SHORT (squeeze risk for shorts → bullish) /
                OI falling + top traders SHORT (confirms sell-off)
    -20 / +20 : Funding rising strongly (overleveraged longs → bearish for LONG)
    +15 / -15 : Retail EXTREME short (contrarian bullish) / retail EXTREME long (contrarian bearish)
    """
    score = 0
    components_hit = 0

    oi_trend = oi.get("oi_trend", "FLAT")
    top_bias = top.get("top_trader_bias", "NEUTRAL")
    global_lr = global_r.get("global_long_ratio", 0.5)
    fund_rate = fund.get("latest_funding", 0.0)
    fund_trend = fund.get("funding_trend", "STABLE")

    # --- OI + Top Trader alignment ---
    if oi_trend == "RISING":
        if top_bias == "LONG":
            score += 30   # OI rising + top traders long = strong buying
            components_hit += 1
        elif top_bias == "SHORT":
            score += 15   # OI rising + top traders short = potential squeeze → mild bullish
            components_hit += 1
    elif oi_trend == "FALLING":
        if top_bias == "SHORT":
            score -= 25   # OI falling + top traders short = confirmed sell-off
            components_hit += 1
        elif top_bias == "LONG":
            score -= 10   # OI falling + top traders long = mixed/uncertain
            components_hit += 1

    # --- Funding signal ---
    # High positive funding = longs overleveraged → bearish (fade longs)
    # High negative funding = shorts overleveraged → bullish (fade shorts)
    if fund_rate > 0.0003 and fund_trend == "RISING":
        score -= 20       # Longs too crowded → mean reversion bearish
        components_hit += 1
    elif fund_rate > 0.0002:
        score -= 10       # Moderately crowded longs
        components_hit += 1
    elif fund_rate < -0.0002 and fund_trend != "RISING":
        score += 20       # Shorts too crowded → potential short squeeze
        components_hit += 1

    # --- Retail contrarian ---
    # Retail extreme long = contrarian bearish
    if global_lr > 0.65:
        score -= 15
        components_hit += 1
    elif global_lr > 0.60:
        score -= 8
        components_hit += 1
    # Retail extreme short = contrarian bullish
    elif global_lr < 0.35:
        score += 15
        components_hit += 1
    elif global_lr < 0.40:
        score += 8
        components_hit += 1

    score = max(-100, min(100, score))

    # Confidence based on data quality and signal clarity
    if components_hit >= 3:
        confidence = "HIGH"
    elif components_hit >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return int(score), confidence


# ── Public API ────────────────────────────────────────────────────────────────

def get_institutional_signal(symbol: str) -> dict:
    """
    Returns institutional flow intelligence for a symbol.
    Cached for _CACHE_TTL seconds to prevent rate-limit hammering.

    institutional_score: -100..+100
      positive = flow supports LONG
      negative = flow supports SHORT
      near 0   = neutral / insufficient data
    """
    now = time.time()
    cached = _CACHE.get(symbol)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    try:
        oi       = _get_oi(symbol)
        top      = _get_top_traders(symbol)
        global_r = _get_global_ratio(symbol)
        fund     = _get_funding(symbol)

        score, conf = _compute_score(oi, top, global_r, fund)

        result = {
            "oi_trend":            oi["oi_trend"],
            "oi_pct_1h":           oi["oi_pct_1h"],
            "top_trader_bias":     top["top_trader_bias"],
            "top_long_ratio":      top["top_long_ratio"],
            "global_long_ratio":   global_r["global_long_ratio"],
            "funding_trend":       fund["funding_trend"],
            "latest_funding":      fund["latest_funding"],
            "institutional_score": score,
            "confidence":          conf,
            "data_quality":        "OK",
        }
    except Exception:
        result = _neutral()

    _CACHE[symbol] = (now, result)
    return result


# ── BTC Regime ────────────────────────────────────────────────────────────────

def get_btc_regime() -> dict:
    """
    Returns BTC market regime from 4h klines.
    {"regime": "BULL"|"BEAR"|"SIDEWAYS", "btc_4h_pct": float}

    BULL  : BTC 4h close ≥ +1.5% above 4 bars ago
    BEAR  : BTC 4h close ≤ -1.5%
    SIDEWAYS: in between
    """
    now = time.time()
    if _REGIME_CACHE and (now - _REGIME_CACHE.get("ts", 0)) < _REGIME_TTL:
        return {k: v for k, v in _REGIME_CACHE.items() if k != "ts"}

    try:
        data = _get(f"{BAPI}/api/v3/klines",
                    {"symbol": "BTCUSDT", "interval": "4h", "limit": 8})
        if not data or len(data) < 5:
            raise ValueError("insufficient klines")

        # close 4 bars ago vs current close
        ref_close     = float(data[-5][4])
        current_close = float(data[-1][4])
        pct4h = (current_close - ref_close) / ref_close * 100

        if pct4h >= 1.5:
            regime = "BULL"
        elif pct4h <= -1.5:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"

        _REGIME_CACHE.update(ts=now, regime=regime, btc_4h_pct=round(pct4h, 2))
    except Exception:
        regime = _REGIME_CACHE.get("regime", "SIDEWAYS")
        pct4h  = _REGIME_CACHE.get("btc_4h_pct", 0.0)

    return {"regime": regime, "btc_4h_pct": round(_REGIME_CACHE.get("btc_4h_pct", 0), 2)}


# ── Direction Compatibility Check ─────────────────────────────────────────────

def is_signal_confirmed(direction: str, intel: dict, regime: dict) -> tuple:
    """
    Returns (confirmed: bool, reason: str)

    Gates:
    1. Regime gate: SHORT blocked in BULL regime; LONG blocked in BEAR regime
    2. Institutional gate: LONG blocked if inst_score < -20; SHORT blocked if inst_score > +20
    """
    inst_score = intel.get("institutional_score", 0)
    reg        = regime.get("regime", "SIDEWAYS")

    # Regime gate
    if direction == "SHORT" and reg == "BULL":
        return False, f"REGIME_BLOCK: BTC 4h={regime.get('btc_4h_pct',0):+.1f}% (BULL, no shorts)"
    if direction == "LONG" and reg == "BEAR":
        return False, f"REGIME_BLOCK: BTC 4h={regime.get('btc_4h_pct',0):+.1f}% (BEAR, no longs)"

    # Institutional gate
    if direction == "LONG" and inst_score < -20:
        return False, f"INST_BLOCK: inst_score={inst_score} (bearish flow, skip LONG)"
    if direction == "SHORT" and inst_score > 20:
        return False, f"INST_BLOCK: inst_score={inst_score} (bullish flow, skip SHORT)"

    return True, f"CONFIRMED: inst_score={inst_score}, regime={reg}"


if __name__ == "__main__":
    print("Testing institutional intel...")
    btc = get_btc_regime()
    print(f"BTC regime: {btc}")
    for sym in ["BTCUSDT", "SOLUSDT", "ETHUSDT"]:
        intel = get_institutional_signal(sym)
        confirmed, reason = is_signal_confirmed("LONG", intel, btc)
        print(f"{sym}: score={intel['institutional_score']:+d}  conf={intel['confidence']}  "
              f"oi={intel['oi_trend']}  top_bias={intel['top_trader_bias']}  "
              f"{'✓ CONFIRMED' if confirmed else '✗ BLOCKED'}")
        print(f"  → {reason}")
