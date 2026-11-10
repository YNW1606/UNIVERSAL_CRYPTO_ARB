#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crypto_india_strategies.py  —  ALL India Trading Terminal strategies ported to crypto
======================================================================================
14 fully-ported strategy modules from INDIA_TRADING_TERMINAL working on Binance data:

MODULE 1  — SMC / ICT         : market structure (BOS/CHoCH), order blocks, FVGs,
                                 liquidity pools, stop-hunt detection, premium/discount OTE
MODULE 2  — Wyckoff           : phase classification (Accum/Markup/Distrib/Markdown),
                                 event detection (SC,AR,ST,SPRING,UTAD,SOS,SOW,BC,LPS,LPSY),
                                 effort-vs-result (Wyckoff 3rd law), volume analysis
MODULE 3  — Order Flow        : CVD from klines, delta divergence, absorption detection,
                                 volume profile (POC/VAH/VAL/HVN/LVN), manipulation detection
MODULE 4  — Price Action      : candlestick patterns (engulfing/hammer/star),
                                 chart patterns (double top/bottom, channel, breakout),
                                 Elliott wave 5th-wave exhaustion heuristic
MODULE 5  — Volume Flow       : Chaikin A/D line, OBV, CVD proxy
MODULE 6  — Quant             : MACD 12/26/9 momentum, Hurst exponent regime
MODULE 7  — Smart Money       : supply/demand zones, IPDA draw on liquidity,
                                 ICT market structure, ICT order blocks
MODULE 8  — Conviction Fusion : all 14 framework scores fused → final verdict + trade plan

Data source: Binance REST API (klines + aggTrades optional)
No Kite / NSE / Indian market data required.

Run: C:\\Users\\Yash\\anaconda3\\python.exe crypto_india_strategies.py
     C:\\Users\\Yash\\anaconda3\\python.exe crypto_india_strategies.py --symbol ETHUSDT
     C:\\Users\\Yash\\anaconda3\\python.exe crypto_india_strategies.py --all
     C:\\Users\\Yash\\anaconda3\\python.exe crypto_india_strategies.py --watch 300
"""

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple, Optional

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BINANCE_FAPI  = "https://fapi.binance.com"
BINANCE_SPOT  = "https://api.binance.com"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
           "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "ADAUSDT"]

KLINE_INTERVAL  = "1h"    # primary analysis interval
KLINE_LIMIT     = 300     # bars to fetch (Wyckoff/SMC need 60+, Hurst needs 50+)
PIVOT_K         = 3       # fractal half-width for swing detection
IMPULSE_MULT    = 1.7     # displacement bar: range ≥ mult × avg range
SESSION         = requests.Session()


# ─────────────────────────────────────────────────────────────────────────────
# DATA UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str = KLINE_INTERVAL,
                  limit: int = KLINE_LIMIT) -> List:
    """Fetch Binance spot klines. Returns raw list."""
    url = f"{BINANCE_SPOT}/api/v3/klines"
    r = SESSION.get(url, params={"symbol": symbol, "interval": interval,
                                  "limit": limit}, timeout=10)
    r.raise_for_status()
    return r.json()


def _ohlcv(klines) -> Tuple:
    """Parse Binance klines → (o, h, l, c, v) numpy arrays."""
    o = np.array([float(k[1]) for k in klines])
    h = np.array([float(k[2]) for k in klines])
    l = np.array([float(k[3]) for k in klines])
    c = np.array([float(k[4]) for k in klines])
    v = np.array([float(k[5]) for k in klines])
    return o, h, l, c, v


def _safe(v, default=0.0):
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _r(v, d=2):
    try:
        if v is None: return None
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, d)
    except Exception:
        return None


def _ema(arr, period):
    k = 2.0 / (period + 1.0)
    out = [float(arr[0])]
    for v in arr[1:]:
        out.append(out[-1] + k * (float(v) - out[-1]))
    return np.array(out)


def _atr(h, l, c, period=14):
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


def _swing_pivots(h, l, k=PIVOT_K):
    """Fractal swing highs/lows → (sh, sl) as [(idx, price)]."""
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


def _impulses(o, h, l, c, mult=IMPULSE_MULT, lookback=30):
    """Displacement bars: range ≥ mult×avg AND body dominates. Returns [(idx, ±1)]."""
    ar = _avg_range(h, l, lookback)
    if ar <= 0:
        return []
    out = []
    for i in range(len(c)):
        rng = h[i] - l[i]
        body = c[i] - o[i]
        if rng >= mult * ar and abs(body) >= 0.5 * rng:
            out.append((i, 1 if body > 0 else -1))
    return out


def _result(name, score, label, note):
    score = int(max(-2, min(2, int(score))))
    return {"name": name, "score": score, "label": str(label), "note": str(note)}


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — SMC / ICT
# ─────────────────────────────────────────────────────────────────────────────

def smc_market_structure(o, h, l, c):
    """Classify structure + find last BOS and CHoCH."""
    sh, sl = _swing_pivots(h, l)
    n = len(c)
    out = {
        "structure": "RANGE", "sequence": "",
        "last_bos": None, "last_choch": None,
        "swing_highs": [{"idx": i, "price": _r(p)} for i, p in sh[-6:]],
        "swing_lows":  [{"idx": i, "price": _r(p)} for i, p in sl[-6:]],
    }
    if len(sh) < 2 or len(sl) < 2:
        return out, sh, sl

    hh = sh[-1][1] > sh[-2][1]; hl = sl[-1][1] > sl[-2][1]
    lh = sh[-1][1] < sh[-2][1]; ll = sl[-1][1] < sl[-2][1]

    if hh and hl:
        out["structure"] = "UPTREND"; out["sequence"] = "HH / HL"
    elif lh and ll:
        out["structure"] = "DOWNTREND"; out["sequence"] = "LH / LL"
    else:
        out["structure"] = "RANGE"
        out["sequence"] = ("HH" if hh else "LH") + " / " + ("HL" if hl else "LL")

    merged = sorted(
        [(i, p, "H") for i, p in sh] + [(i, p, "L") for i, p in sl],
        key=lambda x: x[0])
    prevailing = 0
    last_hi = last_lo = None
    for idx, price, kind in merged:
        if kind == "H":
            if last_hi is not None and price > last_hi[1]:
                ev = {"type": "BULLISH_BOS" if prevailing >= 0 else "BULLISH_CHOCH",
                      "idx": idx, "price": _r(price), "broke": _r(last_hi[1]),
                      "bars_ago": int(n - 1 - idx)}
                if prevailing >= 0:
                    out["last_bos"] = ev
                else:
                    out["last_choch"] = ev
                prevailing = 1
            last_hi = (idx, price)
        else:
            if last_lo is not None and price < last_lo[1]:
                ev = {"type": "BEARISH_BOS" if prevailing <= 0 else "BEARISH_CHOCH",
                      "idx": idx, "price": _r(price), "broke": _r(last_lo[1]),
                      "bars_ago": int(n - 1 - idx)}
                if prevailing <= 0:
                    out["last_bos"] = ev
                else:
                    out["last_choch"] = ev
                prevailing = -1
            last_lo = (idx, price)
    return out, sh, sl


def smc_order_blocks(o, h, l, c, spot, ar, max_out=8):
    """Last opposite-direction candle before each displacement impulse."""
    obs = []
    pad = 0.15 * ar if ar > 0 else 0.0
    n = len(c)
    seen = set()
    for idx, direction in _impulses(o, h, l, c):
        for j in range(idx - 1, max(-1, idx - 6), -1):
            body = c[j] - o[j]
            if direction > 0 and body < 0:
                key = ("B", round(l[j], 6))
                if key not in seen:
                    obs.append((1, float(l[j]), float(h[j]), idx, j))
                    seen.add(key)
                break
            if direction < 0 and body > 0:
                key = ("S", round(h[j], 6))
                if key not in seen:
                    obs.append((-1, float(l[j]), float(h[j]), idx, j))
                    seen.add(key)
                break

    rows = []
    for kind, zlo, zhi, imp_idx, ob_idx in obs:
        mitigated = any(l[k] <= zhi + pad and h[k] >= zlo - pad
                        for k in range(imp_idx + 1, n))
        violated = any((kind > 0 and c[k] < zlo - pad) or (kind < 0 and c[k] > zhi + pad)
                       for k in range(imp_idx + 1, n))
        rows.append({
            "type": "BULLISH" if kind > 0 else "BEARISH",
            "high": _r(zhi), "low": _r(zlo), "mid": _r((zhi + zlo) / 2.0),
            "idx": ob_idx, "bars_ago": int(n - 1 - ob_idx),
            "mitigated": bool(mitigated), "violated": bool(violated),
            "in_zone": bool(zlo <= spot <= zhi),
            "_imp": imp_idx,
        })
    rows.sort(key=lambda x: (x["violated"], -x["_imp"]))
    for x in rows:
        x.pop("_imp", None)
    return rows[:max_out]


def smc_fvgs(h, l, c, spot, max_out=8):
    """3-candle fair value gaps (imbalances)."""
    rows = []
    n = len(c)
    for i in range(2, n):
        if h[i - 2] < l[i]:      # bullish FVG
            g_lo, g_hi, kind = float(h[i - 2]), float(l[i]), "BULLISH"
        elif l[i - 2] > h[i]:    # bearish FVG
            g_lo, g_hi, kind = float(h[i]), float(l[i - 2]), "BEARISH"
        else:
            continue
        mid = (g_lo + g_hi) / 2.0
        filled = any(l[k] <= mid <= h[k] for k in range(i + 1, n))
        rows.append({
            "type": kind, "high": _r(g_hi), "low": _r(g_lo), "mid": _r(mid),
            "idx": i - 1, "bars_ago": int(n - 1 - (i - 1)),
            "filled": bool(filled), "in_zone": bool(g_lo <= spot <= g_hi),
            "_i": i,
        })
    rows.sort(key=lambda x: (x["filled"], -x["_i"]))
    for x in rows:
        x.pop("_i", None)
    return rows[:max_out]


def smc_liquidity(sh, sl, h, l, spot, ar):
    """Buy-side (equal highs) / sell-side (equal lows) liquidity pools."""
    tol = max(ar * 0.35, spot * 0.0008) if ar > 0 else spot * 0.0008

    def _cluster(pivots):
        pools = []
        used = set()
        for i in range(len(pivots)):
            if i in used: continue
            grp = [pivots[i][1]]; idxs = [pivots[i][0]]
            for j in range(i + 1, len(pivots)):
                if j in used: continue
                if abs(pivots[j][1] - pivots[i][1]) <= tol:
                    grp.append(pivots[j][1]); idxs.append(pivots[j][0]); used.add(j)
            pools.append({"price": _r(float(np.mean(grp))), "touches": len(grp),
                          "last_idx": max(idxs)})
        return pools

    buy_side = sorted(_cluster(sh), key=lambda x: (-x["touches"], abs((x["price"] or spot) - spot)))
    sell_side = sorted(_cluster(sl), key=lambda x: (-x["touches"], abs((x["price"] or spot) - spot)))
    return buy_side[:8], sell_side[:8]


def smc_stop_hunts(o, h, l, c, sh, sl, ar, n_dates=None):
    """Detect liquidity grabs: spike beyond a prior swing, then reversal."""
    events = []
    n = len(c)
    tol = ar * 0.1 if ar > 0 else 0.0
    _SWEEP = 4

    for si, sp in sh:
        for k in range(si + 1, min(n, si + 12)):
            if h[k] > sp + tol and c[k] < sp:
                rev = any(c[m] < sp - tol for m in range(k, min(n, k + _SWEEP + 1)))
                if rev:
                    events.append({
                        "type": "STOP_HUNT_HIGH", "idx": k,
                        "price": _r(h[k]), "swept_level": _r(sp),
                        "bars_ago": int(n - 1 - k),
                        "note": "Buy-side liquidity grabbed above %.4f then rejected — bearish." % sp,
                    })
                break

    for si, sp in sl:
        for k in range(si + 1, min(n, si + 12)):
            if l[k] < sp - tol and c[k] > sp:
                rev = any(c[m] > sp + tol for m in range(k, min(n, k + _SWEEP + 1)))
                if rev:
                    events.append({
                        "type": "STOP_HUNT_LOW", "idx": k,
                        "price": _r(l[k]), "swept_level": _r(sp),
                        "bars_ago": int(n - 1 - k),
                        "note": "Sell-side liquidity grabbed below %.4f then reclaimed — bullish." % sp,
                    })
                break

    seen = set()
    uniq = []
    for e in sorted(events, key=lambda x: -x["idx"]):
        key = (e["type"], e["idx"])
        if key not in seen:
            seen.add(key); uniq.append(e)
    return uniq


def smc_premium_discount(h, l, spot, lookback=40):
    """Current dealing range → equilibrium, premium/discount, OTE band."""
    hh = h[-lookback:] if len(h) >= lookback else h
    ll = l[-lookback:] if len(l) >= lookback else l
    rng_hi = float(np.max(hh)); rng_lo = float(np.min(ll))
    span = rng_hi - rng_lo
    if span <= 0:
        return {"range_high": _r(rng_hi), "range_low": _r(rng_lo),
                "eq": _r(rng_hi), "zone": "EQUILIBRIUM", "pct": None, "ote_zone": None}
    pct = (spot - rng_lo) / span
    eq = (rng_hi + rng_lo) / 2.0
    zone = "PREMIUM" if pct >= 0.55 else ("DISCOUNT" if pct <= 0.45 else "EQUILIBRIUM")
    ote_discount = {"low": _r(rng_lo + span * 0.21), "high": _r(rng_lo + span * 0.38)}
    ote_premium  = {"low": _r(rng_lo + span * 0.62), "high": _r(rng_lo + span * 0.79)}
    return {
        "range_high": _r(rng_hi), "range_low": _r(rng_lo),
        "eq": _r(eq), "zone": zone, "pct": _r(pct * 100.0, 1),
        "ote_zone": {"discount_long": ote_discount, "premium_short": ote_premium},
    }


def run_smc(o, h, l, c, spot, ar):
    """Run full SMC suite → (score -4..+4, label, details)."""
    ms, sh, sl = smc_market_structure(o, h, l, c)
    obs        = smc_order_blocks(o, h, l, c, spot, ar)
    fvgs       = smc_fvgs(h, l, c, spot)
    buy_side, sell_side = smc_liquidity(sh, sl, h, l, spot, ar)
    hunts      = smc_stop_hunts(o, h, l, c, sh, sl, ar)
    pd         = smc_premium_discount(h, l, spot)

    score = 0
    struct = ms["structure"]
    if struct == "UPTREND":   score += 2
    elif struct == "DOWNTREND": score -= 2

    choch = ms.get("last_choch")
    bos   = ms.get("last_bos")
    if choch and (choch.get("bars_ago") or 99) <= 20:
        score += 1 if choch["type"].startswith("BULLISH") else -1
    elif bos and (bos.get("bars_ago") or 99) <= 20:
        score += 0.5 if bos["type"].startswith("BULLISH") else -0.5

    if pd["zone"] == "PREMIUM": score -= 1
    elif pd["zone"] == "DISCOUNT": score += 1

    fresh_grabs = [m for m in hunts if (m.get("bars_ago") or 99) <= 5]
    if fresh_grabs:
        score += 1 if fresh_grabs[0]["type"] == "STOP_HUNT_LOW" else -1

    score = int(max(-4, min(4, round(score))))
    if score >= 2:    label = "SMC BULLISH"
    elif score <= -2: label = "SMC BEARISH"
    else:             label = "SMC NEUTRAL"

    return score, label, {
        "structure": struct, "sequence": ms["sequence"],
        "last_bos": bos, "last_choch": choch,
        "premium_discount": pd, "order_blocks": obs[:4],
        "fvgs": fvgs[:4], "liquidity": {"buy_side": buy_side[:4], "sell_side": sell_side[:4]},
        "manipulation": hunts[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — WYCKOFF
# ─────────────────────────────────────────────────────────────────────────────

def wyckoff_volume_analysis(o, h, l, c, v):
    """Up-vol vs down-vol, volume trend, climactic-volume flag."""
    n = len(c)
    lb = min(n, 60)
    o2, c2, v2 = o[-lb:], c[-lb:], v[-lb:]
    up_vol   = float(np.sum(v2[c2 >= o2]))
    down_vol = float(np.sum(v2[c2 < o2]))
    tot = up_vol + down_vol
    up_share = (up_vol / tot * 100.0) if tot > 0 else 50.0

    v_avg   = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v)) if n else 0.0
    recent_v = float(np.mean(v[-5:])) if n >= 5 else v_avg
    ratio = (recent_v / v_avg) if v_avg > 0 else 1.0
    vtrend = "EXPANDING" if ratio >= 1.25 else ("CONTRACTING" if ratio <= 0.8 else "STEADY")

    ref = float(np.mean(v[-30:])) if n >= 30 else v_avg
    climactic = False; climax_note = ""
    if ref > 0:
        recent = v[-5:]
        idx = int(np.argmax(recent))
        peak = float(recent[idx])
        if peak >= 2.2 * ref:
            climactic = True
            bar  = c[-5:][idx]; obar = o[-5:][idx]
            climax_note = "Climactic %s volume spike (%.1f× normal)" % (
                "up" if bar >= obar else "down", peak / ref)

    return {
        "up_share_pct": _r(up_share, 1), "trend": vtrend, "vol_ratio": _r(ratio, 2),
        "climactic": climactic, "climax_note": climax_note,
        "bias": "buying" if up_share >= 56 else ("selling" if up_share <= 44 else "balanced"),
    }


def wyckoff_effort_vs_result(o, h, l, c, v):
    """Wyckoff 3rd law: big volume + small range = absorption."""
    n = len(c)
    if n < 20:
        return {"note": "insufficient bars", "state": "—"}
    rng = h - l
    avg_rng = float(np.mean(rng[-20:]))
    avg_vol = float(np.mean(v[-20:]))
    if avg_rng <= 0 or avg_vol <= 0:
        return {"note": "flat", "state": "—"}
    rng_x = float(rng[-1]) / avg_rng
    vol_x = float(v[-1]) / avg_vol
    up = c[-1] >= o[-1]
    state = "NORMAL"; note = ""
    if vol_x >= 1.6 and rng_x <= 0.8:
        state = "ABSORPTION"
        note = "High effort (%.1f× vol) into %.1f× range — institutions absorbing; a move is being prepared." % (vol_x, rng_x)
    elif vol_x <= 0.7 and rng_x >= 1.4:
        state = "NO_SUPPLY" if up else "NO_DEMAND"
        note = "Wide %.1f× range on light %.1f× vol — %s; path of least resistance is %s." % (
            rng_x, vol_x, "no supply above" if up else "no demand below", "up" if up else "down")
    elif vol_x >= 1.5 and rng_x >= 1.5:
        state = "CLIMACTIC"
        note = "Effort and result both large (%.1f× vol, %.1f× range) — climactic %s bar." % (vol_x, rng_x, "up" if up else "down")
    else:
        note = "Effort and result in proportion — orderly, no divergence."
    return {"state": state, "vol_x": _r(vol_x, 2), "range_x": _r(rng_x, 2), "up_bar": bool(up), "note": note}


def wyckoff_trading_range(h, l, c, lookback=60):
    """Support/resistance of the recent consolidation range."""
    n = len(c)
    lb = min(n, lookback)
    hh, ll = h[-lb:], l[-lb:]
    resistance = float(np.percentile(hh, 90))
    support    = float(np.percentile(ll, 10))
    hard_res   = float(np.max(hh))
    hard_sup   = float(np.min(ll))
    last = float(c[-1])
    width = resistance - support
    pos = ((last - support) / width) if width > 0 else 0.5
    return ({"support": _r(support), "resistance": _r(resistance),
             "range_low": _r(hard_sup), "range_high": _r(hard_res),
             "position": _r(max(0.0, min(1.0, pos)), 2),
             "width_pct": _r((width / last * 100.0) if last else None, 2)},
            support, resistance, hard_sup, hard_res)


def wyckoff_classify_phase(o, h, l, c, v, tr):
    """Return (phase, confidence 0-100, sub_phase, bias)."""
    n = len(c)
    last = float(c[-1])
    ema20 = _ema(c, 20)[-1]
    ema50 = _ema(c, 50)[-1] if n >= 50 else ema20

    # slope (per-bar % change)
    def _slope(arr, nn):
        nn = min(nn, len(arr))
        if nn < 5: return 0.0
        y = arr[-nn:]
        x = np.arange(nn, dtype=float)
        b = np.polyfit(x, y, 1)[0]
        base = float(np.mean(y))
        return float(b / base) if base else 0.0

    slope_s = _slope(c, 20); slope_l = _slope(c, 50)

    # ADX proxy: ratio of directional movement to total range
    rng = h - l
    lb = min(n, 20)
    adx_proxy = float(np.std(np.diff(c[-lb:])) / np.mean(rng[-lb:])) * 10 if np.mean(rng[-lb:]) > 0 else 0

    prior_trend = (float(c[-20]) / float(c[-60]) - 1.0) if n >= 60 else 0.0

    if n >= 40:
        early_r = float(np.mean(rng[-40:-20]))
        late_r  = float(np.mean(rng[-10:]))
        compressed = late_r < 0.85 * early_r if early_r > 0 else False
    else:
        compressed = False

    lb2 = min(n, 30)
    o2, c2, v2 = o[-lb2:], c[-lb2:], v[-lb2:]
    up_mask = c2 >= o2
    up_v = float(np.mean(v2[up_mask])) if up_mask.any() else 0.0
    dn_v = float(np.mean(v2[~up_mask])) if (~up_mask).any() else 0.0

    pos = tr["position"] if tr["position"] is not None else 0.5
    strong_up = last > ema20 > ema50 and slope_s > 0 and adx_proxy >= 2.0
    strong_dn = last < ema20 < ema50 and slope_s < 0 and adx_proxy >= 2.0
    vol_favors_up = up_v > dn_v * 1.1
    vol_favors_dn = dn_v > up_v * 1.1

    conf = 40.0; sub = ""
    if strong_up:
        phase = "MARKUP"; bias = "BULLISH"
        conf = 55 + min(30, adx_proxy * 3) + (10 if vol_favors_up else 0)
        sub = "Trend up on rally volume" if vol_favors_up else "Trend up"
    elif strong_dn:
        phase = "MARKDOWN"; bias = "BEARISH"
        conf = 55 + min(30, adx_proxy * 3) + (10 if vol_favors_dn else 0)
        sub = "Trend down on decline volume" if vol_favors_dn else "Trend down"
    else:
        if prior_trend <= -0.04 or (prior_trend < 0 and pos <= 0.5):
            phase = "ACCUMULATION"; bias = "BULLISH"
            conf = 45 + (18 if compressed else 0) + (17 if not vol_favors_dn else 0) + (10 if pos <= 0.4 else 0)
            sub = "Basing after decline"
            if compressed and up_v < dn_v:
                sub = "Volume drying on declines — supply exhausting"
        elif prior_trend >= 0.04 or (prior_trend > 0 and pos >= 0.5):
            phase = "DISTRIBUTION"; bias = "BEARISH"
            conf = 45 + (18 if compressed else 0) + (17 if not vol_favors_up else 0) + (10 if pos >= 0.6 else 0)
            sub = "Churning after advance"
            if vol_favors_up and slope_s <= 0:
                sub = "Rally volume fading at highs — supply absorbing"
        else:
            if slope_s >= 0 and pos <= 0.5:
                phase = "ACCUMULATION"; bias = "BULLISH"; conf = 42; sub = "Range, lower half"
            elif slope_s < 0 and pos >= 0.5:
                phase = "DISTRIBUTION"; bias = "BEARISH"; conf = 42; sub = "Range, upper half"
            elif slope_s >= 0:
                phase = "MARKUP"; bias = "BULLISH"; conf = 40; sub = "Early advance"
            else:
                phase = "MARKDOWN"; bias = "BEARISH"; conf = 40; sub = "Early decline"

    return phase, float(max(30, min(92, conf))), sub, bias


def wyckoff_detect_events(o, h, l, c, v, support, resistance, max_events=10):
    """Detect SC, AR, ST, SPRING, UTAD, SOS, SOW, BC, LPS, LPSY."""
    n = len(c)
    if n < 30: return []
    rng = h - l
    events = []
    scan_from = max(25, n - 90)

    for i in range(scan_from, n):
        vref = float(np.mean(v[max(0, i - 30):i])) if i >= 5 else 0.0
        rref = float(np.mean(rng[max(0, i - 20):i])) if i >= 5 else 0.0
        if vref <= 0 or rref <= 0: continue
        vol_x = v[i] / vref; rng_x = rng[i] / rref
        up = c[i] >= o[i]; body = c[i] - o[i]
        prior_hi = float(np.max(h[max(0, i - 20):i])) if i >= 2 else h[i]
        prior_lo = float(np.min(l[max(0, i - 20):i])) if i >= 2 else l[i]

        ev = None
        if l[i] < support and c[i] > support and vol_x >= 1.0:
            ev = ("SPRING", "BULLISH", "Price undercut support %.4f then closed back — bullish shakeout." % support)
        elif h[i] > resistance and c[i] < resistance and vol_x >= 1.0:
            ev = ("UPTHRUST", "BEARISH", "Price poked above resistance %.4f then failed — bearish trap." % resistance)
        elif (not up) and vol_x >= 2.2 and rng_x >= 1.5 and l[i] <= prior_lo * 1.01:
            ev = ("SC", "BULLISH", "Selling Climax — panic selling (%.1f× vol), likely capitulation." % vol_x)
        elif up and vol_x >= 2.2 and rng_x >= 1.5 and h[i] >= prior_hi * 0.99:
            ev = ("BC", "BEARISH", "Buying Climax — euphoric buying (%.1f× vol), likely exhaustion." % vol_x)
        elif up and c[i] > resistance and vol_x >= 1.4 and body >= 0.6 * rng[i]:
            ev = ("SOS", "BULLISH", "Sign of Strength — decisive break of resistance %.4f on volume." % resistance)
        elif (not up) and c[i] < support and vol_x >= 1.4 and abs(body) >= 0.6 * rng[i]:
            ev = ("SOW", "BEARISH", "Sign of Weakness — decisive break of support %.4f on volume." % support)
        elif up and rng_x >= 1.4 and i >= 3 and np.min(l[max(0,i-3):i]) <= prior_lo * 1.01 and vol_x >= 1.1:
            ev = ("AR", "NEUTRAL", "Automatic Rally — snap-back bounce defines range top.")
        elif l[i] <= support * 1.01 and l[i] >= prior_lo * 0.99 and vol_x <= 0.8 and c[i] > l[i]:
            ev = ("ST", "BULLISH", "Secondary Test — retest of support %.4f on shrinking volume." % support)
        elif (up and l[i] > support and l[i] > prior_lo and vol_x <= 1.0
              and c[i] > o[i] and i >= 8 and c[i] > c[i - 8]
              and l[i] <= support + 0.35 * max(1e-9, resistance - support)):
            ev = ("LPS", "BULLISH", "Last Point of Support — higher low just above support; markup launchpad.")
        elif ((not up) and h[i] < resistance and h[i] < prior_hi and vol_x <= 1.0
              and i >= 8 and c[i] < c[i - 8]
              and h[i] >= resistance - 0.35 * max(1e-9, resistance - support)):
            ev = ("LPSY", "BEARISH", "Last Point of Supply — lower high just below resistance; markdown launchpad.")

        if ev:
            events.append({"event": ev[0], "implication": ev[1], "note": ev[2],
                           "price": _r(float(c[i])),
                           "volume_note": "%.1f× avg vol, %.1f× avg range" % (vol_x, rng_x),
                           "_i": i, "bars_ago": int(n - 1 - i)})

    dedup = []
    for e in events:
        if dedup and dedup[-1]["event"] == e["event"] and e["_i"] - dedup[-1]["_i"] <= 2:
            dedup[-1] = e
        else:
            dedup.append(e)
    for e in dedup:
        e.pop("_i", None)
    dedup.reverse()
    return dedup[:max_events]


def run_wyckoff(o, h, l, c, v, spot):
    """Run full Wyckoff suite → (score -4..+4, label, details)."""
    tr, support, resistance, hard_sup, hard_res = wyckoff_trading_range(h, l, c)
    phase, conf, sub, bias = wyckoff_classify_phase(o, h, l, c, v, tr)
    vol_analysis = wyckoff_volume_analysis(o, h, l, c, v)
    evr = wyckoff_effort_vs_result(o, h, l, c, v)
    events = wyckoff_detect_events(o, h, l, c, v, support, resistance)

    score = 0
    if phase == "MARKUP":       score = 2
    elif phase == "ACCUMULATION": score = 1
    elif phase == "DISTRIBUTION": score = -1
    elif phase == "MARKDOWN":   score = -2

    # boost if key event confirms
    if events:
        ev0 = events[0]
        if ev0["event"] in ("SPRING", "SC", "ST", "LPS", "SOS") and (ev0.get("bars_ago") or 99) <= 10:
            score = min(4, score + 1)
        elif ev0["event"] in ("UPTHRUST", "BC", "LPSY", "SOW") and (ev0.get("bars_ago") or 99) <= 10:
            score = max(-4, score - 1)

    if evr["state"] == "ABSORPTION":  score = int(score * 0.7)  # wait for direction
    elif evr["state"] == "NO_SUPPLY": score = min(4, score + 1)
    elif evr["state"] == "NO_DEMAND": score = max(-4, score - 1)

    if vol_analysis["climactic"]: score = int(score * 0.5)

    score = int(max(-4, min(4, score)))
    if score >= 2:    label = "WYCKOFF BULLISH (%s)" % phase
    elif score <= -2: label = "WYCKOFF BEARISH (%s)" % phase
    else:             label = "WYCKOFF NEUTRAL (%s)" % phase

    return score, label, {
        "phase": phase, "confidence": _r(conf, 0), "sub_phase": sub,
        "trading_range": tr, "volume_analysis": vol_analysis,
        "effort_vs_result": evr, "recent_events": events[:5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — ORDER FLOW
# ─────────────────────────────────────────────────────────────────────────────

def _bar_delta(o, h, l, c, v):
    """Per-bar approximate CVD from close-location in range."""
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        loc = np.where(rng > 0, (2.0 * (c - l) / np.where(rng > 0, rng, 1.0) - 1.0), 0.0)
    return loc * v


def orderflow_analyze(o, h, l, c, v):
    """CVD trend, delta divergence, absorption bars, effort/result."""
    n = len(c)
    deltas = _bar_delta(o, h, l, c, v)
    cvd = np.cumsum(deltas)
    cvd_last = float(cvd[-1])

    k = min(20, n)
    y = cvd[-k:]; x = np.arange(k, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if k >= 3 else 0.0
    scale = max(abs(float(np.max(cvd)) - float(np.min(cvd))), 1e-9)
    slope_norm = slope * k / scale
    cvd_trend = "RISING" if slope_norm > 0.12 else ("FALLING" if slope_norm < -0.12 else "FLAT")

    # delta divergence
    def _swings(arr, left=2, right=2):
        highs, lows = [], []
        nn = len(arr)
        for i in range(left, nn - right):
            seg = arr[i - left:i + right + 1]
            if arr[i] == seg.max() and (seg == arr[i]).sum() == 1: highs.append(i)
            if arr[i] == seg.min() and (seg == arr[i]).sum() == 1: lows.append(i)
        return highs, lows

    ph, pl = _swings(c)
    div = {"type": "NONE", "note": "No clear delta divergence."}
    if len(ph) >= 2:
        i1, i2 = ph[-2], ph[-1]
        if c[i2] > c[i1] and cvd[i2] < cvd[i1]:
            div = {"type": "BEARISH",
                   "note": "Price HH (%.4f > %.4f) but CVD LH — distribution / bearish divergence." % (c[i2], c[i1])}
    if div["type"] == "NONE" and len(pl) >= 2:
        j1, j2 = pl[-2], pl[-1]
        if c[j2] < c[j1] and cvd[j2] > cvd[j1]:
            div = {"type": "BULLISH",
                   "note": "Price LL (%.4f < %.4f) but CVD HL — absorption at lows / bullish divergence." % (c[j2], c[j1])}

    # absorption bars
    rng = h - l
    med_rng = float(np.median(rng[rng > 0])) if np.any(rng > 0) else 0.0
    avg_v = float(np.mean(v)) if n else 0.0
    absorption = []
    for i in range(max(0, n - 40), n):
        if v[i] > avg_v * 1.6 and med_rng > 0 and rng[i] < med_rng * 0.7:
            side = "supply" if deltas[i] < 0 else "demand"
            absorption.append({"note": "Heavy vol (%.1f×) tiny range — %s absorbed." % (
                v[i] / avg_v if avg_v else 0, side)})

    score = 0
    if cvd_trend == "RISING": score += 1
    elif cvd_trend == "FALLING": score -= 1
    if div["type"] == "BULLISH": score += 2
    elif div["type"] == "BEARISH": score -= 2
    score = int(max(-4, min(4, score)))

    bias = "BULLISH" if score >= 2 else ("BEARISH" if score <= -2 else "NEUTRAL")
    return score, bias, {
        "cvd_last": _r(cvd_last, 0), "cvd_trend": cvd_trend,
        "delta_divergence": div, "absorption": absorption[-3:],
        "slope_norm": _r(slope_norm, 3),
    }


def orderflow_volume_profile(h, l, c, v, n_bins=36):
    """Volume profile: POC, VAH, VAL, HVN/LVN classification."""
    n = len(c)
    if n < 20 or float(np.sum(v)) <= 0: return None
    tp = (h + l + c) / 3.0
    lo, hi = float(np.min(tp)), float(np.max(tp))
    if hi - lo <= 0: return None

    idx = np.clip(((tp - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    prof = np.zeros(n_bins)
    np.add.at(prof, idx, v)
    centers = lo + (np.arange(n_bins) + 0.5) * (hi - lo) / n_bins

    poc_i = int(np.argmax(prof)); poc = float(centers[poc_i])
    total = float(np.sum(prof)); acc = prof[poc_i]; lo_i = hi_i = poc_i
    while acc < 0.70 * total and (lo_i > 0 or hi_i < n_bins - 1):
        below = prof[lo_i - 1] if lo_i > 0 else -1.0
        above = prof[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if above >= below:
            hi_i += 1; acc += prof[hi_i]
        else:
            lo_i -= 1; acc += prof[lo_i]
    val = float(centers[lo_i]); vah = float(centers[hi_i])

    last = float(c[-1])
    if last > vah:    pvp = "ABOVE_VALUE"
    elif last < val:  pvp = "BELOW_VALUE"
    elif last > poc:  pvp = "UPPER_VALUE"
    elif last < poc:  pvp = "LOWER_VALUE"
    else:             pvp = "AT_POC"

    mean_v = float(np.mean(prof[prof > 0])) if np.any(prof > 0) else 0.0
    nodes = [{"price": _r(float(centers[i])),
              "type": "HVN" if prof[i] >= mean_v * 1.5 else ("LVN" if prof[i] <= mean_v * 0.4 else "MID")}
             for i in range(n_bins) if prof[i] > 0]

    score = 1 if pvp in ("ABOVE_VALUE", "UPPER_VALUE") else (-1 if pvp in ("BELOW_VALUE", "LOWER_VALUE") else 0)
    return score, {"poc": _r(poc), "vah": _r(vah), "val": _r(val),
                   "price_vs_poc": pvp, "nodes_count": len(nodes)}


def orderflow_manipulation(o, h, l, c, v):
    """Stop-hunts, false breakouts, volume climax, weak breakouts."""
    n = len(c); rng = h - l; body = np.abs(c - o)
    avg_v = float(np.mean(v)) if n else 0.0
    med_rng = float(np.median(rng[rng > 0])) if np.any(rng > 0) else 0.0
    events = []
    look = min(30, n - 6)

    for i in range(n - look, n):
        if i < 6: continue
        prior_hi = float(np.max(h[i - 5:i])); prior_lo = float(np.min(l[i - 5:i]))
        upper_wick = h[i] - max(o[i], c[i]); lower_wick = min(o[i], c[i]) - l[i]
        b = body[i]; bar_rng = rng[i]

        if (h[i] > prior_hi and c[i] < prior_hi and bar_rng > 0
                and upper_wick > b * 1.5 and upper_wick > med_rng * 0.6 and v[i] > avg_v * 1.2):
            events.append({"kind": "STOP_HUNT", "direction": "BEARISH",
                           "note": "Wick above %.4f closed back — buy-side grab; bearish." % prior_hi})
            continue
        if (l[i] < prior_lo and c[i] > prior_lo and bar_rng > 0
                and lower_wick > b * 1.5 and lower_wick > med_rng * 0.6 and v[i] > avg_v * 1.2):
            events.append({"kind": "STOP_HUNT", "direction": "BULLISH",
                           "note": "Wick below %.4f closed back — sell-side grab; bullish." % prior_lo})
            continue
        if i <= n - 2:
            if c[i] > prior_hi and any(c[j] < prior_hi for j in range(i + 1, min(i + 3, n))):
                events.append({"kind": "FALSE_BREAKOUT", "direction": "BEARISH",
                               "note": "Broke %.4f then back inside — bull trap." % prior_hi}); continue
            if c[i] < prior_lo and any(c[j] > prior_lo for j in range(i + 1, min(i + 3, n))):
                events.append({"kind": "FALSE_BREAKOUT", "direction": "BULLISH",
                               "note": "Broke %.4f then back inside — bear trap." % prior_lo}); continue
        if v[i] > avg_v * 2.2 and bar_rng > med_rng * 1.5 and bar_rng > 0:
            close_pos = (c[i] - l[i]) / bar_rng
            if close_pos < 0.35 and c[i] > c[i - 1]:
                events.append({"kind": "VOLUME_CLIMAX", "direction": "BEARISH",
                               "note": "Climactic vol (%.1f×) closed near low — buying climax." % (v[i]/avg_v if avg_v else 0)}); continue
            if close_pos > 0.65 and c[i] < c[i - 1]:
                events.append({"kind": "VOLUME_CLIMAX", "direction": "BULLISH",
                               "note": "Climactic vol (%.1f×) closed near high — selling climax." % (v[i]/avg_v if avg_v else 0)}); continue

    return events


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — PRICE ACTION
# ─────────────────────────────────────────────────────────────────────────────

def pa_candlestick(o, h, l, c):
    """Engulfing, hammer, shooting star — location-aware."""
    NAME = "Candlestick"
    n = min(len(o), len(h), len(l), len(c))
    if n < 3: return _result(NAME, 0, "Too few bars", "Need 3+ bars.")
    o1, h1, l1, c1 = o[n-1], h[n-1], l[n-1], c[n-1]
    o2, h2, l2, c2 = o[n-2], h[n-2], l[n-2], c[n-2]
    body1 = abs(c1 - o1); body2 = abs(c2 - o2)
    up1 = h1 - max(o1, c1); lo1 = min(o1, c1) - l1; rng1 = h1 - l1
    if rng1 <= 0: return _result(NAME, 0, "Degenerate bar", "")
    if body1 <= 0.12 * rng1: return _result(NAME, 0, "Doji indecision", "Last bar is a doji.")

    day_h = float(np.max(h)); day_l = float(np.min(l)); day_rng = max(day_h - day_l, c1 * 0.001, 1e-9)
    near_low  = (min(l1, l2) - day_l) <= 0.25 * day_rng
    near_high = (day_h - max(h1, h2)) <= 0.25 * day_rng

    meaningful = body1 >= 0.5 * rng1 and body2 >= 0.25 * max(h2 - l2, 1e-9)
    if meaningful and c1 > o1 and c2 < o2 and c1 >= max(o2, c2) and o1 <= min(o2, c2):
        if near_low:
            strong = body1 >= 1.2 * body2
            return _result(NAME, 2 if strong else 1, "Bullish engulfing at low",
                           "Bullish engulfing near day low (body %.4f engulfs %.4f)." % (body1, body2))
        return _result(NAME, 0, "Bullish engulfing mid-range", "Location weak.")
    if meaningful and c1 < o1 and c2 > o2 and o1 >= max(o2, c2) and c1 <= min(o2, c2):
        if near_high:
            strong = body1 >= 1.2 * body2
            return _result(NAME, -2 if strong else -1, "Bearish engulfing at high",
                           "Bearish engulfing near day high.")
        return _result(NAME, 0, "Bearish engulfing mid-range", "Location weak.")
    if lo1 >= 2.0 * body1 and lo1 >= 0.55 * rng1 and up1 <= 0.6 * body1 + 0.15 * rng1:
        if near_low:
            return _result(NAME, 1, "Hammer at day low",
                           "Pin-bar lower wick %.1f× body." % (lo1 / max(body1, 1e-9)))
        return _result(NAME, 0, "Hammer but not at low", "Ignored.")
    if up1 >= 2.0 * body1 and up1 >= 0.55 * rng1 and lo1 <= 0.6 * body1 + 0.15 * rng1:
        if near_high:
            return _result(NAME, -1, "Shooting star at high",
                           "Pin-bar upper wick %.1f× body." % (up1 / max(body1, 1e-9)))
        return _result(NAME, 0, "Shooting star but not at high", "Ignored.")
    return _result(NAME, 0, "No reversal candle", "No classic pattern on last bars.")


def pa_chart_pattern(h, l, c):
    """Double top/bottom, range breakout, channel — ported from fw_price_action."""
    NAME = "Chart Pattern"
    n = min(len(h), len(l), len(c))
    if n < 15: return _result(NAME, 0, "Too few bars", "")
    h = h[:n]; l = l[:n]; c = c[:n]
    spot = c[-1]
    day_h = float(np.max(h)); day_l = float(np.min(l))
    day_rng = max(day_h - day_l, spot * 0.001, 1e-9)

    def _near_a(a, b, ref, frac=0.0025):
        return abs(a - b) <= frac * ref if ref > 0 else False

    piv = []
    k = 3; nn = n
    for i in range(k, nn - k):
        win_h = h[i - k:i + k + 1]; win_l = l[i - k:i + k + 1]
        if h[i] >= max(win_h): piv.append((i, h[i], "H"))
        elif l[i] <= min(win_l): piv.append((i, l[i], "L"))
    out = []
    for p in piv:
        if out and out[-1][2] == p[2]:
            if (p[2] == "H" and p[1] >= out[-1][1]) or (p[2] == "L" and p[1] <= out[-1][1]):
                out[-1] = p
        else:
            out.append(p)
    ph = [(i, v) for i, v, t in out if t == "H"]
    pl = [(i, v) for i, v, t in out if t == "L"]

    if len(ph) >= 2:
        (i1, p1), (i2, p2) = ph[-2], ph[-1]
        if (i2 - i1) >= 5 and _near_a(p1, p2, spot, 0.0025) and (day_h - max(p1, p2)) <= 0.10 * day_rng:
            if c[-1] < min(p1, p2) - 0.15 * day_rng:
                return _result(NAME, -2, "Double top at high", "Two matched highs breaking away down.")
            if c[-1] < min(p1, p2):
                return _result(NAME, -1, "Possible double top", "Two matched highs; early rejection.")
    if len(pl) >= 2:
        (i1, p1), (i2, p2) = pl[-2], pl[-1]
        if (i2 - i1) >= 5 and _near_a(p1, p2, spot, 0.0025) and (min(p1, p2) - day_l) <= 0.10 * day_rng:
            if c[-1] > max(p1, p2) + 0.15 * day_rng:
                return _result(NAME, 2, "Double bottom at low", "Two matched lows breaking away up.")
            if c[-1] > max(p1, p2):
                return _result(NAME, 1, "Possible double bottom", "Two matched lows; early bounce.")

    N = min(30, n - 5)
    if N >= 10:
        rng_hi = max(h[-(N + 5):-5]); rng_lo = min(l[-(N + 5):-5]); rng_w = rng_hi - rng_lo
        tight = rng_w <= 0.45 * day_rng or rng_w <= 0.004 * spot
        if tight and rng_w > 0:
            if c[-1] > rng_hi + 0.25 * rng_w and c[-2] > rng_hi:
                return _result(NAME, 2, "Range breakout up", "Closed above %d-bar range high %.4f." % (N, rng_hi))
            if c[-1] > rng_hi:
                return _result(NAME, 1, "Testing range breakout", "")
            if c[-1] < rng_lo - 0.25 * rng_w and c[-2] < rng_lo:
                return _result(NAME, -2, "Range breakdown", "Closed below %d-bar range low %.4f." % (N, rng_lo))
            if c[-1] < rng_lo:
                return _result(NAME, -1, "Testing range breakdown", "")

    def _rising(seq): return len(seq) >= 2 and all(seq[i+1][1] > seq[i][1] for i in range(len(seq)-1))
    def _falling(seq): return len(seq) >= 2 and all(seq[i+1][1] < seq[i][1] for i in range(len(seq)-1))
    hh = _rising(ph[-3:]) if len(ph) >= 2 else False
    hl = _rising(pl[-3:]) if len(pl) >= 2 else False
    lh = _falling(ph[-3:]) if len(ph) >= 2 else False
    ll = _falling(pl[-3:]) if len(pl) >= 2 else False
    if hh and hl: return _result(NAME, 2 if (len(ph)>=3 and len(pl)>=3) else 1, "Up channel intact", "HH and HL.")
    if lh and ll: return _result(NAME, -2 if (len(ph)>=3 and len(pl)>=3) else -1, "Down channel intact", "LH and LL.")

    if n >= 20 and c[0] > 0:
        drift = (c[-1] - c[0]) / c[0]
        thirds = n // 3; seg = [c[thirds-1], c[2*thirds-1], c[-1]]
        if drift > 0.004 and seg[0] < seg[1] < seg[2]:
            return _result(NAME, 1, "Steady uptrend", "Drift %.2f%%." % (100*drift))
        if drift < -0.004 and seg[0] > seg[1] > seg[2]:
            return _result(NAME, -1, "Steady downtrend", "Drift %.2f%%." % (100*drift))

    return _result(NAME, 0, "No clear structure", "Choppy — no channel/breakout/double top.")


def pa_elliott_wave(c):
    """5th-wave exhaustion heuristic — ported from fw_price_action."""
    NAME = "Elliott Wave (5)"
    if len(c) < 20: return _result(NAME, 0, "Too few bars", "")

    def _zigzag(closes, pct=0.004):
        n = len(closes)
        if n < 3: return []
        zz = [(0, closes[0])]; last_p = closes[0]; direction = 0
        ext_i, ext_p = 0, closes[0]
        for i in range(1, n):
            p = closes[i]
            if direction >= 0 and p > ext_p: ext_i, ext_p = i, p
            if direction <= 0 and p < ext_p: ext_i, ext_p = i, p
            if direction == 0:
                if last_p > 0 and abs(p - last_p) / last_p >= pct:
                    direction = 1 if p > last_p else -1; ext_i, ext_p = i, p
            elif direction == 1:
                if ext_p > 0 and (ext_p - p) / ext_p >= pct:
                    zz.append((ext_i, ext_p)); last_p = ext_p; direction = -1; ext_i, ext_p = i, p
            else:
                if ext_p > 0 and (p - ext_p) / ext_p >= pct:
                    zz.append((ext_i, ext_p)); last_p = ext_p; direction = 1; ext_i, ext_p = i, p
        if not zz or zz[-1][0] != ext_i: zz.append((ext_i, ext_p))
        return zz

    zz = _zigzag(c, pct=0.004)
    if len(zz) < 6: return _result(NAME, 0, "No 5-swing zigzag", "Elliott subjective — neutral.")
    pts = zz[-6:]
    legs = [pts[i+1][1] - pts[i][1] for i in range(5)]
    if any(abs(x) <= 0 for x in legs): return _result(NAME, 0, "Degenerate swings", "")
    signs = [1 if x > 0 else -1 for x in legs]
    if signs == [1,-1,1,-1,1] and pts[-1][1] > pts[0][1]:
        imp = [abs(legs[0]), abs(legs[2]), abs(legs[4])]
        if imp[1] >= min(imp) and c[-1] >= pts[-2][1]:
            return _result(NAME, -1, "Wave 5 up — exhaustion risk", "5-swing upside impulse; mild warning.")
    if signs == [-1,1,-1,1,-1] and pts[-1][1] < pts[0][1]:
        imp = [abs(legs[0]), abs(legs[2]), abs(legs[4])]
        if imp[1] >= min(imp) and c[-1] <= pts[-2][1]:
            return _result(NAME, 1, "Wave 5 down — exhaustion risk", "5-swing downside impulse; mild bounce warning.")
    return _result(NAME, 0, "No clean 5-wave count", "Elliott subjective — neutral.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — VOLUME FLOW (Chaikin A/D + OBV)
# ─────────────────────────────────────────────────────────────────────────────

def volume_accumulation_distribution(o, h, l, c, v):
    """Chaikin A/D line slope + OBV cross-check + price divergence."""
    NAME = "Accum/Distrib"
    n = len(c)
    if n < 5 or float(np.sum(v)) <= 0:
        return _result(NAME, 0, "Insufficient data", "")
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng > 0, rng, 1.0), 0.0)
    ad = np.cumsum(mfm * v)
    dirn = np.sign(np.diff(c, prepend=c[0]))
    obv = np.cumsum(dirn * v)

    def _slope_norm(y):
        nn = len(y)
        if nn < 3: return 0.0
        x = np.arange(nn, dtype=float)
        sl = np.polyfit(x, np.asarray(y, dtype=float), 1)[0]
        scale = max(abs(float(np.max(y)) - float(np.min(y))), 1e-9)
        return float(sl * nn / scale)

    ad_sl = _slope_norm(ad); obv_sl = _slope_norm(obv); px_sl = _slope_norm(c)
    if ad_sl > 0.25: score = 1
    elif ad_sl < -0.25: score = -1
    else: return _result(NAME, 0, "A/D flat neutral", "No clear accumulation/distribution.")

    confirmed = (np.sign(obv_sl) == np.sign(ad_sl)) and abs(obv_sl) > 0.1
    diverging = np.sign(px_sl) != np.sign(ad_sl) or abs(px_sl) < 0.15

    if score > 0:
        if diverging and confirmed:
            return _result(NAME, 2, "Hidden accumulation detected",
                           "A/D rising while price flat/down and OBV confirms.")
        return _result(NAME, 2 if (confirmed and ad_sl > 0.6) else 1, "Accumulation in progress",
                       "A/D rising%s." % (", OBV confirms" if confirmed else ""))
    else:
        if diverging and confirmed:
            return _result(NAME, -2, "Hidden distribution detected",
                           "A/D falling while price flat/up and OBV confirms.")
        return _result(NAME, -2 if (confirmed and ad_sl < -0.6) else -1, "Distribution in progress",
                       "A/D falling%s." % (", OBV confirms" if confirmed else ""))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 6 — QUANT (MACD + Hurst)
# ─────────────────────────────────────────────────────────────────────────────

def quant_macd_momentum(c):
    """MACD 12/26/9 histogram sign + slope."""
    NAME = "Quant MACD Momentum"
    closes = [float(x) for x in c]
    if len(closes) < 26:
        return _result(NAME, 0, "Insufficient bars", "Need 26+ bars.")
    macd   = [a - b for a, b in zip(_ema(closes, 12), _ema(closes, 26))]
    signal = list(_ema(macd, 9))
    hist   = [m - s for m, s in zip(macd, signal)]
    h_now = hist[-1]; h_prev = hist[-2] if len(hist) > 1 else hist[-1]
    px = closes[-1] if closes[-1] else 1.0
    h_pct = 100.0 * h_now / px; rising = h_now > h_prev
    note = "MACD histogram %+.3f%% and %s." % (h_pct, "rising" if rising else "falling")
    if abs(h_pct) < 0.005: return _result(NAME, 0, "Momentum flat", "")
    if h_now > 0:
        return _result(NAME, 2 if rising else 1,
                       "Strong bullish momentum" if rising else "Bullish momentum fading", note)
    return _result(NAME, -2 if not rising else -1,
                   "Strong bearish momentum" if not rising else "Bearish momentum fading", note)


def quant_hurst(c, spot=None):
    """Hurst exponent regime: trending → WITH drift; mean-reverting → FADE drift."""
    NAME = "Hurst Exponent"
    closes = [float(x) for x in c]
    if len(closes) < 8:
        return _result(NAME, 0, "Insufficient data", "")
    ts = np.asarray(closes, float)
    lags = range(2, min(20, len(ts) // 2))
    if len(list(lags)) < 2:
        return _result(NAME, 0, "Cannot estimate Hurst", "")
    tau = [max(np.std(ts[lag:] - ts[:-lag]), 1e-9) for lag in lags]
    h = float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0] * 2.0)
    sp = float(spot) if spot is not None else closes[-1]
    day_open = closes[0]
    drift = (sp - day_open) / day_open if day_open else 0.0
    up = drift > 0; dpct = 100.0 * drift
    if h >= 0.55:
        if drift == 0: return _result(NAME, 0, "Trending but flat", "H=%.2f" % h)
        return _result(NAME, 2 if up else -2,
                       "Strong trend %s regime" % ("up" if up else "down"),
                       "H=%.2f persistent, drift %+.2f%%." % (h, dpct))
    if 0.52 <= h < 0.55:
        if drift == 0: return _result(NAME, 0, "Mild trend flat", "H=%.2f" % h)
        return _result(NAME, 1 if up else -1, "Mild trend %s" % ("up" if up else "down"),
                       "H=%.2f mildly persistent." % h)
    if h < 0.45:
        if drift == 0: return _result(NAME, 0, "Mean reversion flat", "H=%.2f" % h)
        return _result(NAME, -1 if up else 1, "Fade move mean-reversion",
                       "H=%.2f mean-reverting; %.2f%% session move likely fades." % (h, dpct))
    return _result(NAME, 0, "Random walk", "H=%.2f in random-walk band." % h)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 7 — SMART MONEY FRAMEWORKS (from fw_smart_money.py)
# ─────────────────────────────────────────────────────────────────────────────

def sm_supply_demand(o, h, l, c, spot):
    """Base-then-impulse supply/demand zones."""
    NAME = "Supply/Demand"
    n = len(c)
    if n < 15: return _result(NAME, 0, "Insufficient data", "")
    ar = _avg_range(h, l)
    if ar <= 0 or spot is None: return _result(NAME, 0, "No range", "Flat session.")
    zones = []
    for idx, direction in _impulses(o, h, l, c):
        b0, b1 = max(0, idx - 5), idx
        if b1 - b0 < 2: continue
        base_h = max(h[b0:b1]); base_l = min(l[b0:b1])
        if base_h - base_l <= 1.5 * ar:
            zones.append((direction, float(base_l), float(base_h), idx))
    if not zones: return _result(NAME, 0, "No fresh zones", "")
    pad = 0.25 * ar
    for direction, zlo, zhi, idx in sorted(zones, key=lambda z: -z[3]):
        violated = any((direction > 0 and c[j] < zlo - pad) or (direction < 0 and c[j] > zhi + pad)
                       for j in range(idx + 1, n))
        if violated: continue
        if zlo - pad <= spot <= zhi + pad:
            if direction > 0:
                return _result(NAME, 2, "At demand zone",
                               "Retesting fresh demand base %.4f-%.4f; bounce favored." % (zlo, zhi))
            return _result(NAME, -2, "At supply zone",
                           "Retesting fresh supply base %.4f-%.4f; rejection favored." % (zlo, zhi))
    return _result(NAME, 0, "Between zones", "Price not in any fresh zone.")


def sm_ipda_liquidity(h, l, c, spot):
    """IPDA draw on liquidity: nearest unswept equal-highs/lows pool."""
    NAME = "IPDA Draw on Liquidity"
    n = len(c)
    if n < 10 or spot is None: return _result(NAME, 0, "Insufficient data", "")
    ar = _avg_range(h, l)
    tol = max(ar * 0.5, spot * 0.0005)
    day_h = float(np.max(h)); day_l = float(np.min(l))
    pools = []
    sh, sl = _swing_pivots(h, l)
    for piv, side, pname in ((sh, 1, "equal highs"), (sl, -1, "equal lows")):
        for i in range(len(piv) - 1):
            for j in range(i + 1, len(piv)):
                if abs(piv[i][1] - piv[j][1]) <= tol * 0.6:
                    px = (piv[i][1] + piv[j][1]) / 2.0
                    if (side > 0 and px > spot + tol * 0.3) or (side < 0 and px < spot - tol * 0.3):
                        pools.append((pname, px, side))
    if not pools: return _result(NAME, 0, "No unswept pools", "All nearby liquidity taken.")
    above = [p for p in pools if p[2] > 0 and p[1] > spot]
    below = [p for p in pools if p[2] < 0 and p[1] < spot]
    near_up = min(above, key=lambda p: p[1] - spot) if above else None
    near_dn = min(below, key=lambda p: spot - p[1]) if below else None
    trend = c[-1] - c[-min(20, n)]
    if near_up and not near_dn:
        sc = 2 if trend > 0 else 1
        return _result(NAME, sc, "Draw on %s" % near_up[0],
                       "Nearest unswept pool %s at %.4f above; price drawn up." % (near_up[0], near_up[1]))
    if near_dn and not near_up:
        sc = -2 if trend < 0 else -1
        return _result(NAME, sc, "Draw on %s" % near_dn[0],
                       "Nearest unswept pool %s at %.4f below; price drawn down." % (near_dn[0], near_dn[1]))
    if near_up and near_dn:
        du, dd = near_up[1] - spot, spot - near_dn[1]
        if du < dd * 0.7 and trend > 0:
            return _result(NAME, 1, "Draw on %s" % near_up[0], "Closer pool at %.4f with upward momentum." % near_up[1])
        if dd < du * 0.7 and trend < 0:
            return _result(NAME, -1, "Draw on %s" % near_dn[0], "Closer pool at %.4f with downward momentum." % near_dn[1])
    return _result(NAME, 0, "Liquidity balanced", "Pools both sides at similar distance.")


def sm_ict_market_structure(h, l, c, spot):
    """ICT-style market structure: HH/HL vs LH/LL, BOS, CHoCH."""
    NAME = "ICT Market Structure"
    if len(c) < 15: return _result(NAME, 0, "Insufficient data", "")
    sh, sl = _swing_pivots(h, l)
    if len(sh) < 2 or len(sl) < 2: return _result(NAME, 0, "No structure", "")
    hh = sh[-1][1] > sh[-2][1]; hl = sl[-1][1] > sl[-2][1]
    lh = sh[-1][1] < sh[-2][1]; ll = sl[-1][1] < sl[-2][1]
    last_sh, last_sl = sh[-1][1], sl[-1][1]
    if hh and hl:
        if spot is not None and spot > last_sh:
            return _result(NAME, 2, "Bullish BOS", "HH/HL and price breaking last swing high.")
        return _result(NAME, 1, "Bullish structure", "HH and HL intact.")
    if lh and ll:
        if spot is not None and spot < last_sl:
            return _result(NAME, -2, "Bearish BOS", "LH/LL and price breaking last swing low.")
        return _result(NAME, -1, "Bearish structure", "LH and LL intact.")
    if hh and ll and spot is not None and spot < last_sl:
        return _result(NAME, -1, "Bearish CHoCH", "First break below HL sequence.")
    if lh and hl and spot is not None and spot > last_sh:
        return _result(NAME, 1, "Bullish CHoCH", "First break above LH sequence.")
    return _result(NAME, 0, "Structure mixed", "Pivots not aligned.")


def sm_ict_order_block(o, h, l, c, spot, ar):
    """ICT order block: last opposite candle before displacement, retesting."""
    NAME = "ICT Order Block"
    if len(c) < 15 or ar <= 0 or spot is None:
        return _result(NAME, 0, "Insufficient data", "")
    pad = 0.25 * ar; n = len(c); obs = []
    for idx, direction in _impulses(o, h, l, c, mult=1.8):
        for j in range(idx - 1, max(-1, idx - 6), -1):
            body = c[j] - o[j]
            if direction > 0 and body < 0:
                obs.append((1, float(l[j]), float(h[j]), idx)); break
            if direction < 0 and body > 0:
                obs.append((-1, float(l[j]), float(h[j]), idx)); break
    if not obs: return _result(NAME, 0, "No order blocks", "")
    for kind, zlo, zhi, idx in sorted(obs, key=lambda z: -z[3]):
        violated = any((kind > 0 and c[k] < zlo - pad) or (kind < 0 and c[k] > zhi + pad)
                       for k in range(idx + 1, n))
        if violated: continue
        if zlo - pad <= spot <= zhi + pad and idx < n - 1:
            if kind > 0:
                return _result(NAME, 2, "Retesting bullish OB",
                               "Price at bullish OB %.4f-%.4f; support expected." % (zlo, zhi))
            return _result(NAME, -2, "Retesting bearish OB",
                           "Price at bearish OB %.4f-%.4f; resistance expected." % (zlo, zhi))
    return _result(NAME, 0, "Away from OBs", "Not retesting any valid order block.")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 8 — CONVICTION FUSION
# ─────────────────────────────────────────────────────────────────────────────

FRAMEWORK_WEIGHTS = {
    # ICT / smart money (highest weight — these are institutional footprints)
    "ICT Market Structure":    1.5,
    "ICT Order Block":         1.5,
    "Supply/Demand":           1.2,
    "IPDA Draw on Liquidity":  1.2,
    # Wyckoff + SMC (structural)
    "Wyckoff":                 1.4,
    "SMC":                     1.4,
    # Order flow (live microstructure)
    "Order Flow CVD":          1.3,
    "Volume Profile":          0.8,
    "Accum/Distrib":           1.0,
    # Price action (objective)
    "Chart Pattern":           1.0,
    "Candlestick":             0.7,
    # Quant (derived)
    "Quant MACD Momentum":     0.9,
    "Hurst Exponent":          0.9,
    "Elliott Wave (5)":        0.5,
}


def conviction_fusion(scores: list) -> dict:
    """Fuse all framework scores (name, score) → combined verdict + trade grade."""
    total_w = 0.0; total_wsc = 0.0
    bull = 0; bear = 0; neutral = 0
    for name, score in scores:
        w = FRAMEWORK_WEIGHTS.get(name, 1.0)
        total_w   += w
        total_wsc += w * score
        if score > 0: bull += 1
        elif score < 0: bear += 1
        else: neutral += 1

    combined = total_wsc / total_w if total_w > 0 else 0.0
    agree = max(bull, bear)
    n_total = len(scores)
    agree_pct = agree / n_total if n_total else 0

    if combined >= 1.5 and agree_pct >= 0.6:
        grade = "PERFECT TRADE LONG"
    elif combined >= 1.0 and agree_pct >= 0.5:
        grade = "HIGH-CONVICTION LONG"
    elif combined >= 0.4:
        grade = "MODERATE LONG"
    elif combined <= -1.5 and agree_pct >= 0.6:
        grade = "PERFECT TRADE SHORT"
    elif combined <= -1.0 and agree_pct >= 0.5:
        grade = "HIGH-CONVICTION SHORT"
    elif combined <= -0.4:
        grade = "MODERATE SHORT"
    else:
        grade = "NO-TRADE (mixed/weak)"

    if combined >= 0.5: direction = "BULLISH"
    elif combined <= -0.5: direction = "BEARISH"
    else: direction = "NEUTRAL"

    return {
        "combined_score": _r(combined, 3),
        "direction": direction,
        "grade": grade,
        "bull_count": bull,
        "bear_count": bear,
        "neutral_count": neutral,
        "agree_pct": _r(agree_pct * 100, 1),
        "frameworks_used": n_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYZER
# ─────────────────────────────────────────────────────────────────────────────

def analyze_symbol(symbol: str, interval: str = KLINE_INTERVAL,
                   limit: int = KLINE_LIMIT) -> dict:
    """Full 14-framework analysis for one crypto symbol."""
    klines = _fetch_klines(symbol, interval, limit)
    if not klines or len(klines) < 50:
        return {"symbol": symbol, "error": "insufficient kline data", "available": False}

    o, h, l, c, v = _ohlcv(klines)
    spot = float(c[-1])
    ar = _avg_range(h, l)
    atr_val = _atr(h, l, c)

    # ── run all 14 frameworks ──────────────────────────────────────────────
    # MODULE 1: SMC
    smc_score, smc_label, smc_det = run_smc(o, h, l, c, spot, ar)

    # MODULE 2: Wyckoff
    wy_score, wy_label, wy_det = run_wyckoff(o, h, l, c, v, spot)

    # MODULE 3: Order Flow (CVD + divergence)
    of_score, of_bias, of_det = orderflow_analyze(o, h, l, c, v)

    # MODULE 3b: Volume Profile
    vp_result = orderflow_volume_profile(h, l, c, v)
    vp_score = vp_result[0] if vp_result else 0
    vp_det   = vp_result[1] if vp_result else {}

    # MODULE 4: Price Action
    pa_candle = pa_candlestick(o, h, l, c)
    pa_chart  = pa_chart_pattern(h, l, c)
    pa_elliott = pa_elliott_wave(c)

    # MODULE 5: Volume Flow
    vf_ad = volume_accumulation_distribution(o, h, l, c, v)

    # MODULE 6: Quant
    q_macd  = quant_macd_momentum(c)
    q_hurst = quant_hurst(c, spot)

    # MODULE 7: Smart Money Frameworks
    sm_sd   = sm_supply_demand(o, h, l, c, spot)
    sm_ipda = sm_ipda_liquidity(h, l, c, spot)
    sm_ict_ms = sm_ict_market_structure(h, l, c, spot)
    sm_ict_ob = sm_ict_order_block(o, h, l, c, spot, ar)

    # MODULE 3c: Orderflow manipulation events
    manip_events = orderflow_manipulation(o, h, l, c, v)

    # ── build scores list for fusion ───────────────────────────────────────
    scores = [
        ("SMC",                    smc_score),
        ("Wyckoff",                wy_score),
        ("Order Flow CVD",         of_score),
        ("Volume Profile",         vp_score),
        ("Accum/Distrib",          vf_ad["score"]),
        ("Chart Pattern",          pa_chart["score"]),
        ("Candlestick",            pa_candle["score"]),
        ("Elliott Wave (5)",       pa_elliott["score"]),
        ("Quant MACD Momentum",    q_macd["score"]),
        ("Hurst Exponent",         q_hurst["score"]),
        ("Supply/Demand",          sm_sd["score"]),
        ("IPDA Draw on Liquidity", sm_ipda["score"]),
        ("ICT Market Structure",   sm_ict_ms["score"]),
        ("ICT Order Block",        sm_ict_ob["score"]),
    ]

    conviction = conviction_fusion(scores)

    # ── trade plan ────────────────────────────────────────────────────────
    sl_dist  = 1.5 * atr_val
    tp1_dist = 2.0 * atr_val
    tp2_dist = 3.5 * atr_val
    if conviction["direction"] == "BULLISH":
        trade_plan = {
            "direction": "LONG",
            "entry": _r(spot), "sl": _r(spot - sl_dist), "tp1": _r(spot + tp1_dist),
            "tp2": _r(spot + tp2_dist), "rr": _r(tp1_dist / sl_dist, 2) if sl_dist > 0 else None,
            "atr": _r(atr_val, 4),
        }
    elif conviction["direction"] == "BEARISH":
        trade_plan = {
            "direction": "SHORT",
            "entry": _r(spot), "sl": _r(spot + sl_dist), "tp1": _r(spot - tp1_dist),
            "tp2": _r(spot - tp2_dist), "rr": _r(tp1_dist / sl_dist, 2) if sl_dist > 0 else None,
            "atr": _r(atr_val, 4),
        }
    else:
        trade_plan = {"direction": "HOLD", "atr": _r(atr_val, 4)}

    # ── dominant signals (top 3 bullish + top 3 bearish frameworks) ───────
    bull_fw = sorted([(n, s) for n, s in scores if s > 0], key=lambda x: -x[1])[:3]
    bear_fw = sorted([(n, s) for n, s in scores if s < 0], key=lambda x: x[1])[:3]

    return {
        "available": True,
        "symbol": symbol,
        "spot": _r(spot),
        "interval": interval,
        "bars": len(klines),
        "conviction": conviction,
        "trade_plan": trade_plan,
        "bull_signals": [{"name": n, "score": s} for n, s in bull_fw],
        "bear_signals": [{"name": n, "score": s} for n, s in bear_fw],
        "all_scores": [{"name": n, "score": s} for n, s in scores],
        "details": {
            "smc": {"label": smc_label, "structure": smc_det.get("structure"),
                    "sequence": smc_det.get("sequence"),
                    "bos": smc_det.get("last_bos"), "choch": smc_det.get("last_choch"),
                    "premium_discount": smc_det.get("premium_discount"),
                    "order_blocks": smc_det.get("order_blocks", [])[:2],
                    "fvgs": smc_det.get("fvgs", [])[:2],
                    "manipulation": smc_det.get("manipulation", [])[:2]},
            "wyckoff": {"label": wy_label, "phase": wy_det.get("phase"),
                        "confidence": wy_det.get("confidence"),
                        "sub_phase": wy_det.get("sub_phase"),
                        "events": wy_det.get("recent_events", [])[:3],
                        "effort_vs_result": wy_det.get("effort_vs_result"),
                        "volume": wy_det.get("volume_analysis")},
            "orderflow": {"cvd": of_det, "volume_profile": vp_det,
                          "manipulation": manip_events[-3:]},
            "price_action": {"candlestick": pa_candle, "chart_pattern": pa_chart,
                             "elliott": pa_elliott},
            "volume_flow": {"ad": vf_ad},
            "quant": {"macd": q_macd, "hurst": q_hurst},
            "smart_money": {"supply_demand": sm_sd, "ipda": sm_ipda,
                            "ict_structure": sm_ict_ms, "ict_ob": sm_ict_ob},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(res: dict, verbose: bool = False):
    if not res.get("available"):
        print("  %-12s  ERROR: %s" % (res.get("symbol","?"), res.get("error","")))
        return

    sym = res["symbol"]; spot = res["spot"]
    cv  = res["conviction"]
    tp  = res["trade_plan"]
    grade = cv["grade"]

    if "PERFECT" in grade or "HIGH" in grade:
        star = ">>>"
    elif "MODERATE" in grade:
        star = " > "
    else:
        star = "   "

    print("\n%s %s  $%-12s  [%s]  score=%.3f  %d/%d agree  %s" % (
        star, sym, spot, grade, cv["combined_score"],
        max(cv["bull_count"], cv["bear_count"]), cv["frameworks_used"],
        cv["direction"]))

    # top signals
    if res["bull_signals"]:
        print("     BULL: " + " | ".join("%s(+%d)" % (n, s) for n, s in
              [(r["name"], r["score"]) for r in res["bull_signals"]]))
    if res["bear_signals"]:
        print("     BEAR: " + " | ".join("%s(%d)" % (n, s) for n, s in
              [(r["name"], r["score"]) for r in res["bear_signals"]]))

    # trade plan
    if tp["direction"] != "HOLD":
        if tp["direction"] == "LONG":
            print("     LONG  entry=%-10s  SL=%-10s  TP1=%-10s  TP2=%-10s  R:R=%.1f×" % (
                tp["entry"], tp["sl"], tp["tp1"], tp["tp2"], tp.get("rr") or 0))
        else:
            print("     SHORT entry=%-10s  SL=%-10s  TP1=%-10s  TP2=%-10s  R:R=%.1f×" % (
                tp["entry"], tp["sl"], tp["tp1"], tp["tp2"], tp.get("rr") or 0))

    if verbose:
        # Wyckoff phase
        wy = res["details"]["wyckoff"]
        print("     Wyckoff: %s (%s) conf=%s  EVR=%s" % (
            wy.get("phase"), wy.get("sub_phase",""), wy.get("confidence"),
            (wy.get("effort_vs_result") or {}).get("state", "—")))
        # SMC
        sm = res["details"]["smc"]
        print("     SMC: %s  P/D=%s" % (
            sm.get("structure"), (sm.get("premium_discount") or {}).get("zone", "—")))
        # Wyckoff events
        evs = wy.get("events", [])
        if evs:
            e0 = evs[0]
            print("     Last Wyckoff event: %s (%s, %s bars ago)" % (
                e0.get("event"), e0.get("implication"), e0.get("bars_ago")))
        # CVD
        of = res["details"]["orderflow"]["cvd"]
        print("     CVD: %s  divergence=%s" % (
            of.get("cvd_trend"), (of.get("delta_divergence") or {}).get("type")))
        # all scores
        print("     All scores: " + "  ".join(
            "%s=%+d" % (r["name"][:8], r["score"]) for r in res["all_scores"]))


# ─────────────────────────────────────────────────────────────────────────────
# V2 IMPROVEMENTS — data-driven from 5-year backtest results
# ─────────────────────────────────────────────────────────────────────────────
# Backtest findings (1d, 5yr, 8 symbols):
#   POSITIVE EV frameworks: CVD +0.857%, A/D +0.453%, Hurst +0.393%,
#                           ICT-MS +0.361%, MACD +0.335%, SMC +0.161%
#   NEGATIVE EV frameworks: Supply/Demand -0.307%, ICT-OB -0.211%, VP -0.053%
#   Root cause of low WR: averaging signal quality + no regime gate

# Frameworks with proven negative EV — excluded from V2 conviction score
EXCLUDED_FRAMEWORKS_V2 = {
    "Supply/Demand",      # -0.307% 5d EV in 5yr backtest
    "ICT Order Block",    # -0.211% 5d EV
    "Volume Profile",     # -0.053% 5d EV
}

# Weights rebalanced based on backtest-measured expected value per signal
FRAMEWORK_WEIGHTS_V2 = {
    "Order Flow CVD":          3.0,   # +0.857% EV — #1 most accurate
    "Accum/Distrib":           2.5,   # +0.453% EV — #2
    "Hurst Exponent":          2.5,   # +0.393% EV — #3  (regime confirmation)
    "ICT Market Structure":    2.0,   # +0.361% EV — #4
    "Quant MACD Momentum":     1.8,   # +0.335% EV — #5
    "SMC":                     1.5,   # +0.161% EV
    "Wyckoff":                 1.5,   # +0.101% EV
    "Chart Pattern":           1.0,   # +0.114% EV
    "Candlestick":             0.8,   # +0.499% EV (rare, but high quality)
    "IPDA Draw on Liquidity":  0.5,   # +0.045% EV (barely positive, keep low)
    "Elliott Wave (5)":        0.3,   # -0.301% EV (keep at tiny weight as exhaustion hint)
}


def regime_gate_v2(c: np.ndarray, wy_phase: str, direction: str) -> bool:
    """
    Returns True only when direction aligns with the confirmed macro trend.
    Rules (in order of confidence):
      1. EMA50 > EMA200 stack + Wyckoff MARKUP → allow LONG (highest quality)
      2. EMA50 < EMA200 stack + Wyckoff MARKDOWN → allow SHORT
      3. EMA alignment without Wyckoff (medium quality)
      4. Wyckoff phase breakout entries even when EMA not fully stacked
    Block all others — no trades in sideways/distributing markets.
    """
    n = len(c)
    if n < 50:
        return True  # insufficient history → don't block
    ema50  = float(_ema(c, 50)[-1])
    ema200 = float(_ema(c, min(200, n))[-1])
    spot   = float(c[-1])

    bull_stack = spot > ema50 and ema50 > ema200 * 0.995   # clean uptrend stack
    bear_stack = spot < ema50 and ema50 < ema200 * 1.005   # clean downtrend stack
    is_long    = direction == "BULLISH"
    is_short   = direction == "BEARISH"

    # Case 1: Full EMA stack + Wyckoff phase confirmation (best quality)
    if bull_stack and is_long and wy_phase in ("MARKUP", "ACCUMULATION"):
        return True
    if bear_stack and is_short and wy_phase in ("MARKDOWN", "DISTRIBUTION"):
        return True

    # Case 2: EMA stack but neutral Wyckoff phase
    if bull_stack and is_long and wy_phase not in ("DISTRIBUTION",):
        return True
    if bear_stack and is_short and wy_phase not in ("ACCUMULATION",):
        return True

    # Case 3: Wyckoff MARKUP/MARKDOWN breakout without full EMA stack
    # (early trend — EMA50 hasn't crossed EMA200 yet)
    if is_long and wy_phase == "MARKUP" and spot > ema50:
        return True
    if is_short and wy_phase == "MARKDOWN" and spot < ema50:
        return True

    return False   # all other cases: sideways, against trend → NO TRADE


def volatility_gate_v2(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
    """
    True = volatility is in normal tradeable range.
    Skip if ATR > 2.5× 20-day avg (avoid extreme spikes / black-swan days)
    or if ATR < 0.3× 20-day avg (dead market, spread kills edge).
    """
    n = len(c)
    if n < 25:
        return True
    rng = h - l
    avg_rng = float(np.mean(rng[-20:]))
    if avg_rng <= 0:
        return True
    atr_now = _atr(h, l, c, period=14)
    ratio   = atr_now / avg_rng
    return 0.30 <= ratio <= 2.50


def conviction_fusion_v2(scores: list, wy_phase: str = "") -> dict:
    """
    V2 conviction fusion:
    — Excludes 3 negative-EV frameworks (Supply/Demand, ICT OB, Volume Profile)
    — Uses backtest-calibrated weights (CVD 3×, A/D 2.5×, Hurst 2.5×, ...)
    — Slightly lower grade thresholds since fewer frameworks means lower raw score range
    """
    total_w = 0.0; total_wsc = 0.0
    bull = 0; bear = 0; neutral = 0

    for name, score in scores:
        if name in EXCLUDED_FRAMEWORKS_V2:
            continue
        w = FRAMEWORK_WEIGHTS_V2.get(name, 1.0)
        total_w   += w
        total_wsc += w * score
        if score > 0: bull += 1
        elif score < 0: bear += 1
        else: neutral += 1

    combined  = total_wsc / total_w if total_w > 0 else 0.0
    n_total   = bull + bear + neutral
    agree_pct = max(bull, bear) / n_total if n_total else 0.0

    if combined >= 1.4 and agree_pct >= 0.55:
        grade = "PERFECT TRADE LONG"
    elif combined >= 0.9 and agree_pct >= 0.45:
        grade = "HIGH-CONVICTION LONG"
    elif combined >= 0.5:
        grade = "MODERATE LONG"
    elif combined <= -1.4 and agree_pct >= 0.55:
        grade = "PERFECT TRADE SHORT"
    elif combined <= -0.9 and agree_pct >= 0.45:
        grade = "HIGH-CONVICTION SHORT"
    elif combined <= -0.5:
        grade = "MODERATE SHORT"
    else:
        grade = "NO-TRADE"

    if combined >= 0.4:    direction = "BULLISH"
    elif combined <= -0.4: direction = "BEARISH"
    else:                  direction = "NEUTRAL"

    return {
        "combined_score": _r(combined, 3),
        "direction": direction,
        "grade": grade,
        "bull_count": bull, "bear_count": bear, "neutral_count": neutral,
        "agree_pct": _r(agree_pct * 100, 1),
        "frameworks_used": n_total,
        "version": "v2",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Crypto India Strategies — 14-framework analysis")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--all",    action="store_true", help="scan all 8 symbols")
    ap.add_argument("--interval", default=KLINE_INTERVAL)
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--watch",   type=int, default=0, metavar="SEC",
                    help="re-scan every N seconds (0=run once)")
    args = ap.parse_args()

    syms = SYMBOLS if args.all else [args.symbol.upper()]

    def run_once():
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print("\n" + "="*72)
        print("  CRYPTO INDIA STRATEGIES — 14 frameworks  |  %s" % ts)
        print("="*72)
        results = []
        for sym in syms:
            try:
                res = analyze_symbol(sym, args.interval)
            except Exception as e:
                res = {"symbol": sym, "available": False, "error": str(e)[:80]}
            print_report(res, verbose=args.verbose)
            results.append(res)

        # summary: best setups
        tradeable = [r for r in results if r.get("available") and
                     ("PERFECT" in r["conviction"]["grade"] or "HIGH" in r["conviction"]["grade"])]
        if tradeable:
            print("\n" + "="*72)
            print("  TRADEABLE SETUPS (%d):" % len(tradeable))
            for r in sorted(tradeable, key=lambda x: abs(x["conviction"]["combined_score"]), reverse=True):
                cv = r["conviction"]; tp = r["trade_plan"]
                print("  ★ %-10s  %s  score=%.3f  %s" % (
                    r["symbol"], cv["grade"], cv["combined_score"], tp.get("direction","")))
        else:
            print("\n  No PERFECT/HIGH-CONVICTION setups — stay flat or wait.")
        print()

    if args.watch > 0:
        while True:
            run_once()
            print("  [sleeping %ds...]" % args.watch)
            time.sleep(args.watch)
    else:
        run_once()


if __name__ == "__main__":
    main()
