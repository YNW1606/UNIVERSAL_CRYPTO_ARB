#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crypto_india_backtest.py — 5-year historical backtest  (v1 vs v2 comparison)
=============================================================================
v1 = original 14-framework combined conviction, threshold 0.4, fixed 1.5/2.0 ATR
v2 = backtest-calibrated:
     - 3 negative-EV frameworks EXCLUDED (Supply/Demand, ICT OB, Volume Profile)
     - Weights rebalanced from actual 5yr EV data (CVD 3×, A/D 2.5×, Hurst 2.5×…)
     - Regime gate  : only trade in EMA50>EMA200 (long) or EMA50<EMA200 (short)
     - Volatility gate: skip extreme ATR (black-swan / dead market days)
     - Dynamic SL/TP: HIGH score → SL 1.0×ATR, TP 4.0×ATR (R:R 4:1)
                      MOD  score → SL 1.5×ATR, TP 3.0×ATR (R:R 2:1)
     - Entry at NEXT BAR OPEN (realistic execution); skip if price already gapped >0.5%
     - Threshold raised to 0.5

Run:
  python crypto_india_backtest.py                   # v2 only (fastest)
  python crypto_india_backtest.py --version both    # v1 vs v2 comparison
  python crypto_india_backtest.py --symbols BTCUSDT ETHUSDT --version both
  python crypto_india_backtest.py --no-cache        # force re-download
"""

import argparse, csv, math, os, sys, time
from datetime import datetime, timezone
from typing import List, Dict

import numpy as np
import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS        = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT",
                  "XRPUSDT","DOGEUSDT","AVAXUSDT","ADAUSDT"]
BACKTEST_YEARS = 5
WINDOW         = 300
MAX_HOLD_V1    = 10
MAX_HOLD_V2    = 15       # let winners run longer in v2
STEP           = 1
SLIPPAGE       = 0.001    # 0.1% round-trip
RISK_PER_TRADE = 0.01     # 1% portfolio risk per trade (equity curve)
FWD_DAYS       = 5        # for per-framework accuracy
CACHE_DIR      = "bt_cache"
REPORT_PATH    = "CRYPTO_INDIA_BACKTEST_REPORT.html"
BINANCE_SPOT   = "https://api.binance.com"

# V1 params
V1_THRESHOLD   = 0.4
V1_SL_MULT     = 1.5
V1_TP_MULT     = 2.0

# V2 params
V2_THRESHOLD   = 0.5
# Dynamic TP/SL per conviction level (abs_score threshold → (sl_mult, tp_mult))
V2_TP_TABLE    = [(1.4, 1.0, 4.0), (0.9, 1.5, 3.0), (0.5, 1.5, 2.5)]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoBacktest/1.0"})

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT STRATEGY MODULE
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import crypto_india_strategies as cis
    print("[OK] Imported crypto_india_strategies.py")
except ImportError as e:
    print("[ERR]", e); sys.exit(1)

FRAMEWORK_NAMES = [
    "SMC","Wyckoff","Order Flow CVD","Volume Profile","Accum/Distrib",
    "Chart Pattern","Candlestick","Elliott Wave (5)","Quant MACD Momentum",
    "Hurst Exponent","Supply/Demand","IPDA Draw on Liquidity",
    "ICT Market Structure","ICT Order Block",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH + CACHE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_full_history(symbol, interval="1d", years=5):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{years}y.csv")
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 23*3600:
        rows = []
        with open(cache, newline="") as f:
            for row in csv.reader(f):
                rows.append([float(x) if i > 0 else int(x) for i, x in enumerate(row)])
        if rows:
            print(f"  [cache] {symbol}: {len(rows)} bars"); return rows

    bpd = {"1d":1,"4h":6,"1h":24}.get(interval,1)
    target = years*365*bpd; now_ms = int(time.time()*1000); cur_end = now_ms
    all_k = []
    print(f"  [download] {symbol}:", end="", flush=True)
    while len(all_k) < target:
        need = min(1000, target - len(all_k))
        try:
            r = SESSION.get(f"{BINANCE_SPOT}/api/v3/klines",
                            params={"symbol":symbol,"interval":interval,
                                    "endTime":cur_end,"limit":need}, timeout=20)
            r.raise_for_status(); batch = r.json()
        except Exception as ex:
            print(f"\n  [warn] {ex}"); break
        if not batch: break
        all_k = batch + all_k
        cur_end = int(batch[0][0]) - 1
        print(".", end="", flush=True)
        if len(batch) < need: break
        time.sleep(0.05)
    print(f" {len(all_k)} bars")
    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        for k in all_k: w.writerow(k[:7])
    return all_k


def to_arrays(klines):
    ts = np.array([int(k[0]) for k in klines])
    o  = np.array([float(k[1]) for k in klines])
    h  = np.array([float(k[2]) for k in klines])
    l  = np.array([float(k[3]) for k in klines])
    c  = np.array([float(k[4]) for k in klines])
    v  = np.array([float(k[5]) for k in klines])
    return ts, o, h, l, c, v

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_all_frameworks(o_w, h_w, l_w, c_w, v_w):
    """Run all 14 India frameworks on a window slice. No HTTP."""
    spot = float(c_w[-1])
    ar   = cis._avg_range(h_w, l_w)
    atr  = cis._atr(h_w, l_w, c_w)

    def _safe(fn, *a, **kw):
        try: return fn(*a, **kw)
        except Exception: return None

    smc_r  = _safe(cis.run_smc, o_w, h_w, l_w, c_w, spot, ar)
    wy_r   = _safe(cis.run_wyckoff, o_w, h_w, l_w, c_w, v_w, spot)
    of_r   = _safe(cis.orderflow_analyze, o_w, h_w, l_w, c_w, v_w)
    vp_r   = _safe(cis.orderflow_volume_profile, h_w, l_w, c_w, v_w)
    ad_r   = _safe(cis.volume_accumulation_distribution, o_w, h_w, l_w, c_w, v_w)
    cn_r   = _safe(cis.pa_candlestick, o_w, h_w, l_w, c_w)
    cp_r   = _safe(cis.pa_chart_pattern, h_w, l_w, c_w)
    ew_r   = _safe(cis.pa_elliott_wave, c_w)
    mc_r   = _safe(cis.quant_macd_momentum, c_w)
    hu_r   = _safe(cis.quant_hurst, c_w, spot)
    sd_r   = _safe(cis.sm_supply_demand, o_w, h_w, l_w, c_w, spot)
    ip_r   = _safe(cis.sm_ipda_liquidity, h_w, l_w, c_w, spot)
    ms_r   = _safe(cis.sm_ict_market_structure, h_w, l_w, c_w, spot)
    ob_r   = _safe(cis.sm_ict_order_block, o_w, h_w, l_w, c_w, spot, ar)

    def _s(res, idx=0): return res[idx] if res is not None else 0
    def _d(res): return res["score"] if isinstance(res, dict) else 0

    scores = [
        ("SMC",                    _s(smc_r, 0)),
        ("Wyckoff",                _s(wy_r,  0)),
        ("Order Flow CVD",         _s(of_r,  0)),
        ("Volume Profile",         _s(vp_r,  0) if vp_r else 0),
        ("Accum/Distrib",          _d(ad_r)),
        ("Chart Pattern",          _d(cp_r)),
        ("Candlestick",            _d(cn_r)),
        ("Elliott Wave (5)",       _d(ew_r)),
        ("Quant MACD Momentum",    _d(mc_r)),
        ("Hurst Exponent",         _d(hu_r)),
        ("Supply/Demand",          _d(sd_r)),
        ("IPDA Draw on Liquidity", _d(ip_r)),
        ("ICT Market Structure",   _d(ms_r)),
        ("ICT Order Block",        _d(ob_r)),
    ]

    # Extract Wyckoff phase for regime gate
    wy_phase = ""
    if wy_r is not None:
        try: wy_phase = wy_r[2].get("phase", "")
        except Exception: pass

    return scores, atr, spot, wy_phase


def build_signal_rows(o, h, l, c, v, window, step):
    """Pre-compute all signal rows for a symbol (no trade simulation yet)."""
    n = len(c)
    rows = []  # (bar_i, scores, wy_phase, atr, regime_ok_v2, vol_ok_v2)
    t0 = time.time()
    testable = n - window - MAX_HOLD_V2

    for idx, i in enumerate(range(window, n - MAX_HOLD_V2, step)):
        o_w, h_w, l_w, c_w, v_w = o[i-window:i], h[i-window:i], l[i-window:i], c[i-window:i], v[i-window:i]
        try:
            scores, atr, spot, wy_phase = _run_all_frameworks(o_w, h_w, l_w, c_w, v_w)
        except Exception:
            continue

        # V2 gates (computed once per bar, used in v2 simulation)
        cv2 = cis.conviction_fusion_v2(scores, wy_phase)
        direction_v2 = cv2["direction"]
        regime_ok = cis.regime_gate_v2(c_w, wy_phase, direction_v2) if direction_v2 != "NEUTRAL" else False
        vol_ok    = cis.volatility_gate_v2(h_w, l_w, c_w)

        rows.append((i, scores, wy_phase, atr, regime_ok, vol_ok))

        if idx % 200 == 0:
            eta = (time.time()-t0)/max(idx,1)*(testable-idx)
            print(f"\r    {idx}/{testable}  ETA {eta:.0f}s   ", end="", flush=True)

    print(f"\r    {len(rows)} signal bars computed in {time.time()-t0:.0f}s          ")
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATORS
# ─────────────────────────────────────────────────────────────────────────────

def simulate_v1(signal_rows, o, h, l, c):
    """V1: original fixed SL/TP, no regime gate, entry at close of signal bar."""
    trades = []; next_ok = 0; n = len(c)
    for bar_i, scores, wy_phase, atr, _, _ in signal_rows:
        if bar_i < next_ok: continue
        cv = cis.conviction_fusion(scores)
        sc = cv["combined_score"]
        if abs(sc) < V1_THRESHOLD: continue
        direction = cv["direction"]
        if direction == "NEUTRAL": continue

        entry = float(c[bar_i]); is_long = direction == "BULLISH"
        sl_p  = entry - V1_SL_MULT*atr if is_long else entry + V1_SL_MULT*atr
        tp_p  = entry + V1_TP_MULT*atr if is_long else entry - V1_TP_MULT*atr

        pnl, reason, eb = _exit_scan(o, h, l, c, bar_i, is_long, entry, sl_p, tp_p,
                                      MAX_HOLD_V1, n, SLIPPAGE)
        trades.append({"entry_bar":bar_i,"exit_bar":eb,"direction":direction,
                        "entry":entry,"sl":sl_p,"tp":tp_p,"exit_reason":reason,
                        "pnl_pct":pnl,"cv_score":sc,"atr":atr,"version":"v1"})
        next_ok = eb + 1
    return trades


def simulate_v2(signal_rows, o, h, l, c):
    """
    V2: regime gate + volatility gate + dynamic TP/SL + next-bar-open entry
        + no-gap-chase filter + higher threshold.
    """
    trades = []; next_ok = 0; n = len(c)
    for bar_i, scores, wy_phase, atr, regime_ok, vol_ok in signal_rows:
        if bar_i < next_ok: continue
        if not regime_ok or not vol_ok: continue

        cv = cis.conviction_fusion_v2(scores, wy_phase)
        sc = cv["combined_score"]
        if abs(sc) < V2_THRESHOLD: continue
        direction = cv["direction"]
        if direction == "NEUTRAL": continue

        # Entry at NEXT BAR OPEN (realistic)
        entry_bar = bar_i + 1
        if entry_bar >= n: continue
        entry = float(o[entry_bar])
        is_long = direction == "BULLISH"

        # No-gap-chase filter: if price already moved >0.5% in signal direction, skip
        close_signal = float(c[bar_i])
        gap_pct = (entry - close_signal) / close_signal if close_signal else 0.0
        if is_long  and gap_pct >  0.005: continue   # gapped up — chasing
        if not is_long and gap_pct < -0.005: continue  # gapped down for short — chasing

        # Dynamic SL/TP based on conviction strength
        abs_sc = abs(sc)
        sl_m, tp_m = V2_TP_TABLE[-1][1], V2_TP_TABLE[-1][2]  # defaults
        for thresh, sm, tm in V2_TP_TABLE:
            if abs_sc >= thresh:
                sl_m, tp_m = sm, tm; break

        sl_p = entry - sl_m*atr if is_long else entry + sl_m*atr
        tp_p = entry + tp_m*atr if is_long else entry - tp_m*atr

        pnl, reason, eb = _exit_scan(o, h, l, c, entry_bar, is_long, entry, sl_p, tp_p,
                                      MAX_HOLD_V2, n, SLIPPAGE)
        trades.append({"entry_bar":entry_bar,"exit_bar":eb,"direction":direction,
                        "entry":entry,"sl":sl_p,"tp":tp_p,"exit_reason":reason,
                        "pnl_pct":pnl,"cv_score":sc,"atr":atr,"sl_m":sl_m,"tp_m":tp_m,
                        "version":"v2"})
        next_ok = eb + 1
    return trades


def _exit_scan(o, h, l, c, entry_bar, is_long, entry, sl_p, tp_p, max_hold, n, slip):
    """Scan forward bars for SL/TP hit or time exit. Returns (pnl_pct, reason, exit_bar)."""
    for j in range(entry_bar + 1, min(entry_bar + max_hold + 1, n)):
        if is_long:
            if float(l[j]) <= sl_p:
                return (sl_p - entry)/entry - slip, "SL", j
            if float(h[j]) >= tp_p:
                return (tp_p - entry)/entry - slip, "TP", j
        else:
            if float(h[j]) >= sl_p:
                return (entry - sl_p)/entry - slip, "SL", j
            if float(l[j]) <= tp_p:
                return (entry - tp_p)/entry - slip, "TP", j
    # Time exit
    eb = min(entry_bar + max_hold, n - 1)
    ep = float(c[eb])
    pnl = ((ep - entry)/entry if is_long else (entry - ep)/entry) - slip
    return pnl, "TIME", eb

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades, timestamps=None):
    if not trades:
        return {"n":0,"win_rate":0,"pf":0,"cagr":None,"max_dd":0,"sharpe":0,
                "total_pct":0,"avg_win":0,"avg_loss":0,"equity":[1.0],
                "tp_pct":0,"sl_pct":0,"time_pct":0,"trades":[]}
    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    loss = [p for p in pnls if p <= 0]
    n    = len(pnls)
    pf   = sum(wins)/abs(sum(loss)) if loss and sum(loss)!=0 else float("inf")

    # Compound equity (1% portfolio risk each trade, normalized to ATR-based risk)
    eq = [1.0]
    for p in pnls:
        eq.append(eq[-1] * (1 + p * RISK_PER_TRADE / 0.015))

    # CAGR
    cagr = None
    if timestamps is not None and trades and len(eq) > 1:
        t0 = int(timestamps[trades[0]["entry_bar"]]) / 1000
        t1 = int(timestamps[trades[-1]["exit_bar"]]) / 1000
        yrs = max((t1 - t0) / (365.25 * 86400), 0.01)
        if eq[-1] > 0:
            cagr = (eq[-1] ** (1.0 / yrs) - 1.0) * 100.0

    # Max DD
    peak = 1.0; mdd = 0.0
    for e in eq:
        if e > peak: peak = e
        mdd = max(mdd, (peak - e) / peak)

    # Sharpe
    sharpe = 0.0
    if len(pnls) >= 2 and np.std(pnls) > 0:
        mean_p = float(np.mean(pnls)); std_p = float(np.std(pnls, ddof=1))
        tpy = 365.0 / MAX_HOLD_V2
        if cagr is not None and timestamps is not None and trades:
            t0 = int(timestamps[trades[0]["entry_bar"]]) / 1000
            t1 = int(timestamps[trades[-1]["exit_bar"]]) / 1000
            yrs = max((t1 - t0) / (365.25 * 86400), 0.01)
            tpy = n / yrs
        sharpe = mean_p / std_p * math.sqrt(tpy) if std_p > 0 else 0.0

    by_r = {}
    for t in trades: by_r.setdefault(t["exit_reason"],[]).append(1)
    return {
        "n": n, "win_rate": len(wins)/n,
        "avg_win": float(np.mean(wins))*100 if wins else 0,
        "avg_loss": abs(float(np.mean(loss)))*100 if loss else 0,
        "pf": pf, "total_pct": sum(pnls)*100, "cagr": cagr,
        "max_dd": mdd*100, "sharpe": sharpe, "equity": eq,
        "tp_pct":  len(by_r.get("TP",[])) / n * 100,
        "sl_pct":  len(by_r.get("SL",[])) / n * 100,
        "time_pct":len(by_r.get("TIME",[])) / n * 100,
        "trades": trades,
    }


def fw_accuracy(signal_rows, c, n_total):
    """Per-framework 5-day forward return accuracy."""
    fw = {name: {"bull":[], "bear":[]} for name in FRAMEWORK_NAMES}
    for bar_i, scores, _, _, _, _ in signal_rows:
        if bar_i + FWD_DAYS >= n_total: continue
        ret = float(c[bar_i + FWD_DAYS] / c[bar_i] - 1.0)
        for name, sc in scores:
            if name not in fw: continue
            if sc > 0: fw[name]["bull"].append(ret)
            elif sc < 0: fw[name]["bear"].append(-ret)
    rows = []
    for name in FRAMEWORK_NAMES:
        d = fw[name]; combined = d["bull"] + d["bear"]
        if not combined: rows.append({"name":name,"n":0,"acc":None,"ev":None}); continue
        rows.append({
            "name": name, "n": len(combined),
            "acc": sum(1 for r in combined if r > 0) / len(combined),
            "ev":  float(np.mean(combined)),
            "n_bull": len(d["bull"]), "n_bear": len(d["bear"]),
        })
    return sorted(rows, key=lambda x: -(x["ev"] or -999))

# ─────────────────────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body{margin:0;padding:16px;background:#0d1117;color:#c9d1d9;font-family:'Courier New',monospace;font-size:13px}
h1{color:#58a6ff;border-bottom:1px solid #21262d;padding-bottom:8px;font-size:18px}
h2{color:#8b949e;font-size:14px;margin-top:28px;margin-bottom:6px;border-left:3px solid #58a6ff;padding-left:8px}
h3{color:#8b949e;font-size:12px;margin:10px 0 4px}
table{border-collapse:collapse;width:100%;margin-bottom:14px}
th{background:#161b22;padding:6px 10px;color:#58a6ff;text-align:left;border-bottom:2px solid #21262d;font-size:11px}
td{padding:4px 10px;border-bottom:1px solid #21262d;font-size:12px}
tr:hover td{background:#161b22}
.pos{color:#22c55e}.neg{color:#ef4444}.neu{color:#8b949e}.warn{color:#f59e0b}
.card{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:12px;margin-bottom:14px;display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card-header{grid-column:1/-1;display:flex;align-items:baseline;gap:12px;margin-bottom:4px}
.stat{text-align:center;padding:6px 10px;background:#0d1117;border:1px solid #21262d;border-radius:4px}
.stat-l{font-size:10px;color:#8b949e}.stat-v{font-size:15px;font-weight:bold}
.grp{display:flex;flex-wrap:wrap;gap:8px}
.badge{padding:1px 7px;border-radius:12px;font-size:11px}
.b-bull{background:#14532d;color:#22c55e}.b-bear{background:#450a0a;color:#ef4444}
.b-neu{background:#1e293b;color:#8b949e}
.v2tag{background:#1e3a5f;color:#58a6ff;padding:1px 6px;border-radius:4px;font-size:10px;font-weight:bold}
.v1tag{background:#2d1b00;color:#f59e0b;padding:1px 6px;border-radius:4px;font-size:10px}
"""

def _c(v, suf="", fmt=".2f", pos_ok=True):
    if v is None: return '<span class="neu">—</span>'
    try:
        f = float(v)
        cls = "neu" if not pos_ok else ("pos" if f >= 0 else "neg")
        return f'<span class="{cls}">{f:{fmt}}{suf}</span>'
    except Exception: return str(v)

def _grade(pf):
    if not pf or pf == 0: return '<span class="neg">F</span>'
    if pf >= 2.0:  return '<span class="pos"><b>A+</b></span>'
    if pf >= 1.6:  return '<span class="pos">A</span>'
    if pf >= 1.3:  return '<span class="pos">B+</span>'
    if pf >= 1.1:  return '<span style="color:#86efac">B</span>'
    if pf >= 0.9:  return '<span class="neu">C</span>'
    return '<span class="neg">D</span>'

def _svg(eq, trades, w=340, h=90):
    n = len(eq)
    if n < 2:
        return f'<svg width="{w}" height="{h}"><text x="50%" y="50%" text-anchor="middle" fill="#444" font-size="11">No trades</text></svg>'
    mn = min(eq); mx = max(eq); sp = max(mx-mn, 1e-6)
    px, py = 8, 6; W = w-2*px; H = h-2*py
    pts = " ".join(f"{px+i/(n-1)*W:.1f},{py+H-(eq[i]-mn)/sp*H:.1f}" for i in range(n))
    base = py + H - (1.0-mn)/sp*H
    final_pct = (eq[-1]-1)*100; col = "#22c55e" if final_pct >= 0 else "#ef4444"
    return (f'<svg width="{w}" height="{h}" style="background:#0d1117;border-radius:3px">'
            f'<line x1="{px}" y1="{base:.1f}" x2="{px+W}" y2="{base:.1f}" stroke="#2d333b" stroke-width="1" stroke-dasharray="3,2"/>'
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8" stroke-linejoin="round"/>'
            f'<text x="{px+W-2}" y="{py+13}" fill="{col}" text-anchor="end" font-size="10">{final_pct:+.1f}%</text>'
            f'<text x="{px+2}" y="{py+13}" fill="#444" font-size="10">{len(trades)}T</text></svg>')

def _stat(label, val):
    return f'<div class="stat"><div class="stat-l">{label}</div><div class="stat-v">{val}</div></div>'

def _mrow(label, m1, m2, v1_trades, v2_trades):
    """Side-by-side metric row for v1 vs v2."""
    n1, n2 = m1["n"], m2["n"]
    wr1, wr2 = m1["win_rate"]*100, m2["win_rate"]*100
    pf1, pf2 = m1["pf"], m2["pf"]
    cagr1, cagr2 = m1["cagr"], m2["cagr"]
    dd1, dd2 = m1["max_dd"], m2["max_dd"]
    sh1, sh2 = m1["sharpe"], m2["sharpe"]

    def _imp(a, b, higher_better=True):
        if a is None or b is None: return ""
        delta = b - a
        if abs(delta) < 0.01: return ""
        good = delta > 0 if higher_better else delta < 0
        cls = "pos" if good else "neg"
        sym = "▲" if delta > 0 else "▼"
        return f' <span class="{cls}" style="font-size:10px">{sym}{abs(delta):.1f}</span>'

    def _pf_fmt(pf):
        if pf == float("inf"): return "∞"
        if pf is None or pf == 0: return "—"
        return f"{pf:.2f}"

    rows = f"""
<tr><td><b>{label}</b></td>
<td><span class="v1tag">V1</span></td>
<td>{n1}</td>
<td class="{'pos' if wr1>=50 else 'neg'}">{wr1:.0f}%</td>
<td class="{'pos' if pf1>=1.1 else ('neg' if pf1<0.9 else 'neu')}">{_pf_fmt(pf1)}</td>
<td>{_c(cagr1,'%','+.1f')}</td>
<td>{_c(-dd1,'%','.1f')}</td>
<td>{_c(sh1,fmt='.2f')}</td>
</tr>
<tr><td></td>
<td><span class="v2tag">V2</span></td>
<td>{n2}{_imp(n1,n2,False)}</td>
<td class="{'pos' if wr2>=50 else 'neg'}">{wr2:.0f}%{_imp(wr1,wr2)}</td>
<td class="{'pos' if pf2>=1.1 else ('neg' if pf2<0.9 else 'neu')}">{_pf_fmt(pf2)}{_imp(pf1,pf2)}</td>
<td>{_c(cagr2,'%','+.1f')}{_imp(cagr1,cagr2) if cagr1 is not None and cagr2 is not None else ''}</td>
<td>{_c(-dd2,'%','.1f')}{_imp(-dd1 if dd1 else None,-dd2 if dd2 else None)}</td>
<td>{_c(sh2,fmt='.2f')}{_imp(sh1,sh2)}</td>
</tr>
<tr><td colspan="8" style="padding:0;height:1px;background:#21262d"></td></tr>"""
    return rows


def _symbol_card(sym, m1, m2, ts):
    eq1, eq2 = m1["equity"], m2["equity"]
    t1, t2   = m1["trades"], m2["trades"]
    pf1, pf2 = m1["pf"], m2["pf"]

    stats_v1 = "".join(_stat(l,v) for l,v in [
        ("Trades",m1["n"]),("Win%",f'{m1["win_rate"]*100:.0f}%'),
        ("PF",f'{pf1:.2f}' if pf1!=float("inf") else "∞"),
        ("CAGR",f'{m1["cagr"]:.1f}%' if m1["cagr"] else "—"),
        ("MaxDD",f'{m1["max_dd"]:.1f}%'),("Sharpe",f'{m1["sharpe"]:.2f}'),
    ])
    stats_v2 = "".join(_stat(l,v) for l,v in [
        ("Trades",m2["n"]),("Win%",f'{m2["win_rate"]*100:.0f}%'),
        ("PF",f'{pf2:.2f}' if pf2!=float("inf") else "∞"),
        ("CAGR",f'{m2["cagr"]:.1f}%' if m2["cagr"] else "—"),
        ("MaxDD",f'{m2["max_dd"]:.1f}%'),("Sharpe",f'{m2["sharpe"]:.2f}'),
    ])
    return f"""
<div class="card">
  <div class="card-header">
    <b>{sym}</b> {_grade(pf1)} V1 → {_grade(pf2)} V2
  </div>
  <div>
    <h3><span class="v1tag">V1</span> Original</h3>
    <div class="grp">{stats_v1}</div>
    {_svg(eq1, t1)}
  </div>
  <div>
    <h3><span class="v2tag">V2</span> Optimised</h3>
    <div class="grp">{stats_v2}</div>
    {_svg(eq2, t2)}
  </div>
</div>"""


def _fw_table(fw_rows):
    rows = ""
    for r in fw_rows:
        name = r["name"]; n = r["n"]
        excl = name in cis.EXCLUDED_FRAMEWORKS_V2
        label = f'<s style="color:#555">{name}</s> <span style="color:#ef4444;font-size:10px">EXCLUDED V2</span>' if excl else name
        if n == 0:
            rows += f'<tr><td>{label}</td>' + '<td class="neu">—</td>'*5 + '</tr>'; continue
        acc = r["acc"] or 0; ev = r["ev"] or 0
        new_w = cis.FRAMEWORK_WEIGHTS_V2.get(name, "EXC" if excl else "1.0")
        a_cls = "pos" if acc >= 0.52 else ("neg" if acc < 0.48 else "neu")
        e_cls = "pos" if ev > 0.001 else ("neg" if ev < -0.001 else "neu")
        rows += (f'<tr>'
                 f'<td>{label}</td>'
                 f'<td>{n}</td>'
                 f'<td>{r.get("n_bull",0)}B / {r.get("n_bear",0)}S</td>'
                 f'<td class="{a_cls}">{acc*100:.1f}%</td>'
                 f'<td class="{e_cls}">{ev*100:+.3f}%</td>'
                 f'<td style="color:#8b949e">{new_w}</td>'
                 f'</tr>')
    return (
        '<table><tr><th>Framework</th><th>Signals</th><th>Bull/Bear</th>'
        '<th>5d Accuracy</th><th>5d Exp.Value</th><th>V2 Weight</th></tr>'
        f'{rows}</table>')


def build_report(all_res, fw_rows, interval, start_d, end_d):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Comparison table
    cmp_rows = ""
    for sym, res in all_res.items():
        m1 = res["m1"]; m2 = res["m2"]
        cmp_rows += _mrow(sym, m1, m2, m1["trades"], m2["trades"])

    # Symbol cards
    cards = "".join(_symbol_card(sym, res["m1"], res["m2"], res["ts"])
                    for sym, res in all_res.items())

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Crypto India Strategies — 5yr Backtest V1 vs V2</title>
<style>{_CSS}</style></head><body>
<h1>CRYPTO INDIA STRATEGIES — 5-YEAR BACKTEST  |  V1 vs V2</h1>
<p style="color:#555;font-size:11px">
  {now} &nbsp;|&nbsp; Interval: {interval} &nbsp;|&nbsp; Data: {start_d} → {end_d}<br>
  V1: 14 frameworks, threshold 0.4, SL 1.5×ATR / TP 2.0×ATR, entry at close<br>
  V2: 11 frameworks (3 excluded), regime gate, vol gate, dynamic SL/TP (up to 4×ATR),
      entry at next-bar open, no-gap-chase, threshold 0.5
</p>

<h2>V1 VS V2 — ALL SYMBOLS</h2>
<table>
<tr><th>Symbol</th><th>Ver</th><th>Trades</th><th>Win%</th><th>Profit Factor</th>
<th>CAGR</th><th>Max DD</th><th>Sharpe</th></tr>
{cmp_rows}
</table>

<h2>FRAMEWORK SIGNAL ACCURACY (5-day forward return analysis)</h2>
<p style="color:#555;font-size:11px">
  Frameworks marked EXCLUDED V2 had negative expected value in 5yr backtest
  and are removed from V2 conviction scoring.<br>
  V2 Weight = new weighting coefficient (original was all ~1.0).
</p>
{_fw_table(fw_rows)}

<h2>PER-SYMBOL EQUITY CURVES</h2>
{cards}

<div style="margin-top:24px;padding:10px;background:#161b22;border-radius:6px;font-size:11px;color:#555">
  DISCLAIMER: Backtest results do not guarantee future performance. Crypto = high volatility.
  All trades use ATR-based SL/TP with 1% portfolio risk. No live execution implied.
</div>
</body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols",  nargs="+", default=SYMBOLS)
    ap.add_argument("--interval", default="1d", choices=["1d","4h"])
    ap.add_argument("--years",    type=int, default=BACKTEST_YEARS)
    ap.add_argument("--version",  default="both", choices=["v1","v2","both"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--report",   default=REPORT_PATH)
    args = ap.parse_args()

    if args.no_cache:
        import shutil
        if os.path.exists(CACHE_DIR): shutil.rmtree(CACHE_DIR); print("[cache] cleared")

    all_res = {}; start_d = ""; end_d = ""
    all_fw_rows = {name: {"bull":[],"bear":[]} for name in FRAMEWORK_NAMES}
    t_global = time.time()

    for symbol in args.symbols:
        print(f"\n{'='*55}\n  {symbol}\n{'='*55}")
        try:
            klines = fetch_full_history(symbol, args.interval, args.years)
        except Exception as e:
            print(f"  [ERR] {e}"); continue
        if len(klines) < WINDOW + MAX_HOLD_V2 + 50:
            print(f"  [SKIP] only {len(klines)} bars"); continue

        ts, o, h, l, c, v = to_arrays(klines)
        if not start_d:
            start_d = datetime.fromtimestamp(ts[0]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        end_d = datetime.fromtimestamp(ts[-1]/1000, tz=timezone.utc).strftime("%Y-%m-%d")

        print(f"  {symbol}: {len(c)} bars, computing signals...", flush=True)
        signal_rows = build_signal_rows(o, h, l, c, v, WINDOW, STEP)

        # Framework accuracy
        for row in fw_accuracy(signal_rows, c, len(c)):
            if row["n"] == 0: continue
            fw_name = row["name"]
            if fw_name in all_fw_rows:
                all_fw_rows[fw_name]["bull"] += [row["ev"]] * row.get("n_bull",0)
                all_fw_rows[fw_name]["bear"] += [row["ev"]] * row.get("n_bear",0)

        # Simulate
        t1_trades = simulate_v1(signal_rows, o, h, l, c) if args.version in ("v1","both") else []
        t2_trades = simulate_v2(signal_rows, o, h, l, c) if args.version in ("v2","both") else []

        m1 = metrics(t1_trades, ts)
        m2 = metrics(t2_trades, ts)
        all_res[symbol] = {"m1": m1, "m2": m2, "ts": ts}

        # Console
        print(f"\n  {'VER':4} {'TRADES':7} {'WIN%':7} {'PF':6} {'CAGR':8} {'DD':7} {'SHARPE':7}")
        print(f"  {'-'*48}")
        for ver, m in [("V1", m1), ("V2", m2)]:
            if m["n"] == 0: print(f"  {ver:4} {'—':>7}"); continue
            cagr_s = f'{m["cagr"]:+.1f}%' if m["cagr"] is not None else "—"
            pf_s   = f'{m["pf"]:.2f}' if m["pf"] != float("inf") else "∞"
            flag   = "★" if m["pf"] >= 1.3 and (m["cagr"] or 0) > 5 else " "
            print(f"{flag} {ver:4} {m['n']:7d} {m['win_rate']*100:6.0f}% {pf_s:6} {cagr_s:>8} {m['max_dd']:5.1f}% {m['sharpe']:7.2f}")

    # Aggregate fw stats
    merged = []
    for name in FRAMEWORK_NAMES:
        d = all_fw_rows[name]
        combined = d["bull"] + d["bear"]
        if not combined: merged.append({"name":name,"n":0,"acc":None,"ev":None}); continue
        merged.append({"name":name,"n":len(combined),
                       "acc": sum(1 for x in combined if x > 0)/len(combined),
                       "ev":  float(np.mean(combined)),
                       "n_bull":len(d["bull"]), "n_bear":len(d["bear"])})
    merged.sort(key=lambda x: -(x["ev"] or -999))

    # Overall console summary
    print(f"\n{'='*72}")
    print(f"  5-YEAR SUMMARY  ({time.time()-t_global:.0f}s total)")
    print(f"{'='*72}")
    print(f"  {'SYM':10} {'V1-PF':8} {'V2-PF':8} {'V1-CAGR':9} {'V2-CAGR':9} {'DELTA-WR':9}")
    print(f"  {'-'*58}")
    for sym, res in all_res.items():
        m1, m2 = res["m1"], res["m2"]
        pf1  = f'{m1["pf"]:.2f}' if m1["n"] else "—"
        pf2  = f'{m2["pf"]:.2f}' if m2["n"] else "—"
        cg1  = f'{m1["cagr"]:+.1f}%' if m1["cagr"] else "—"
        cg2  = f'{m2["cagr"]:+.1f}%' if m2["cagr"] else "—"
        dwr  = (m2["win_rate"]-m1["win_rate"])*100 if m1["n"] and m2["n"] else 0
        flag = "★" if m2["pf"] >= 1.3 else " "
        print(f"{flag} {sym:10} {pf1:>8} {pf2:>8} {cg1:>9} {cg2:>9} {dwr:+7.1f}pp")

    print(f"\n  TOP FRAMEWORKS (positive EV):")
    for fw in [r for r in merged if (r["ev"] or 0) > 0 and r["n"] > 0][:7]:
        excl = "EXCLUDED" if fw["name"] in cis.EXCLUDED_FRAMEWORKS_V2 else f"w={cis.FRAMEWORK_WEIGHTS_V2.get(fw['name'],'?')}"
        print(f"    {fw['name']:<28} acc={fw['acc']*100:.1f}%  EV={fw['ev']*100:+.3f}%  [{excl}]")

    # HTML report
    html = build_report(all_res, merged, args.interval, start_d, end_d)
    with open(args.report, "w", encoding="utf-8") as f: f.write(html)
    print(f"\n  Report: {os.path.abspath(args.report)}")


if __name__ == "__main__":
    main()
