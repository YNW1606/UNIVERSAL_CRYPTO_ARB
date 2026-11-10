"""
12-Factor Quant Alpha Model — Crypto (Binance)
===============================================
Factors: Momentum(1h/4h/1d), Reversal(7d), Carry, Volatility, Hurst,
         Liquidity, Volume-Momentum, OI-Momentum, Technical, Carry-Term
Portfolio: Ledoit-Wolf shrinkage covariance + max-Sharpe SLSQP optimization
"""

import numpy as np
import requests
import time
import os
import json
from datetime import datetime, timezone
from scipy.optimize import minimize

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "ADAUSDT"]
CACHE_DIR = "factor_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

FACTOR_NAMES = [
    "MOMENTUM_1H", "MOMENTUM_4H", "MOMENTUM_1D", "REVERSAL_7D", "CARRY",
    "VOLATILITY", "TREND_QUALITY", "LIQUIDITY", "VOLUME_MOMENTUM",
    "OI_MOMENTUM", "TECHNICAL", "CARRY_TERM"
]


# ── DATA LAYER ────────────────────────────────────────────────────────────────

def fetch_klines(symbol, interval, limit=500):
    """Fetch klines with 55-min file cache. Returns raw list."""
    cache_key = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{limit}.json")
    if os.path.exists(cache_key):
        if time.time() - os.path.getmtime(cache_key) < 3300:
            with open(cache_key) as f:
                return json.load(f)
    r = requests.get(
        f"{BINANCE_SPOT}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10
    )
    data = r.json()
    with open(cache_key, "w") as f:
        json.dump(data, f)
    return data


def klines_to_arrays(klines):
    """Returns o, h, l, c, v as numpy float64 arrays."""
    o = np.array([float(k[1]) for k in klines])
    h = np.array([float(k[2]) for k in klines])
    l = np.array([float(k[3]) for k in klines])
    c = np.array([float(k[4]) for k in klines])
    v = np.array([float(k[5]) for k in klines])
    return o, h, l, c, v


def fetch_funding_history(symbol, limit=20):
    """Returns list of dicts with fundingRate key."""
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": limit},
            timeout=5
        )
        return r.json()
    except Exception:
        return []


def fetch_oi_history(symbol, period="1h", limit=48):
    """Returns list of dicts with sumOpenInterest key."""
    try:
        r = requests.get(
            f"{BINANCE_FAPI}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": period, "limit": limit},
            timeout=5
        )
        return r.json()
    except Exception:
        return []


def fetch_orderbook_spread(symbol):
    """Returns bid-ask spread in basis points."""
    try:
        r = requests.get(
            f"{BINANCE_SPOT}/api/v3/ticker/bookTicker",
            params={"symbol": symbol},
            timeout=5
        )
        d = r.json()
        bid = float(d["bidPrice"])
        ask = float(d["askPrice"])
        return (ask - bid) / bid * 10000 if bid > 0 else 1.0
    except Exception:
        return 1.0


# ── MATH HELPERS ──────────────────────────────────────────────────────────────

def _ema(values, period):
    """Exponential moving average."""
    result = np.empty(len(values))
    alpha = 2.0 / (period + 1)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1]
    return result


def _atr(h, l, c, period=14):
    """Average True Range via EMA smoothing."""
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )
    atr = np.empty(len(tr))
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr


def compute_hurst(prices, min_lag=2, max_lag=20):
    """
    Hurst exponent via R/S analysis.
    H > 0.5 → trending  |  H < 0.5 → mean-reverting  |  H ≈ 0.5 → random walk
    """
    prices = np.array(prices, dtype=float)
    if len(prices) < max_lag * 2:
        return 0.5

    tau_pairs = []
    for lag in range(min_lag, min(max_lag, len(prices) // 4)):
        chunks = len(prices) // lag
        rs_vals = []
        for i in range(chunks):
            chunk = prices[i * lag:(i + 1) * lag]
            if len(chunk) < 2:
                continue
            mean_c = np.mean(chunk)
            dev = np.cumsum(chunk - mean_c)
            R = np.max(dev) - np.min(dev)
            S = np.std(chunk, ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            tau_pairs.append((lag, np.mean(rs_vals)))

    if len(tau_pairs) < 3:
        return 0.5

    log_lags = np.log([t[0] for t in tau_pairs])
    log_rs   = np.log([t[1] for t in tau_pairs])
    X = np.column_stack([np.ones(len(log_lags)), log_lags])
    coef, _, _, _ = np.linalg.lstsq(X, log_rs, rcond=None)
    return float(np.clip(coef[1], 0.0, 1.0))


# ── 12 FACTOR COMPUTATION ─────────────────────────────────────────────────────

def compute_factors_for_symbol(symbol):
    """
    Compute all 12 factors for one symbol from live Binance data.
    Returns dict: factor_name → float.  Zero on any error (never raises).
    """
    factors = {f: 0.0 for f in FACTOR_NAMES}

    try:
        klines_1h = fetch_klines(symbol, "1h",  limit=200)
        klines_4h = fetch_klines(symbol, "4h",  limit=100)
        klines_1d = fetch_klines(symbol, "1d",  limit=50)

        _, h1, l1, c1, v1   = klines_to_arrays(klines_1h)
        _, h4, l4, c4, v4   = klines_to_arrays(klines_4h)
        _, h1d, l1d, c1d, _ = klines_to_arrays(klines_1d)

        log_c1  = np.log(c1)
        log_c4  = np.log(c4)
        log_c1d = np.log(c1d)

        # F1: fast momentum
        if len(c1) >= 2:
            factors["MOMENTUM_1H"] = float(log_c1[-1] - log_c1[-2])

        # F2: medium momentum
        if len(c4) >= 2:
            factors["MOMENTUM_4H"] = float(log_c4[-1] - log_c4[-2])

        # F3: 24h momentum (via 1h bars)
        if len(c1) >= 25:
            factors["MOMENTUM_1D"] = float(log_c1[-1] - log_c1[-25])

        # F4: 7d reversal (negative = overcrowded momentum fades)
        if len(c1d) >= 8:
            factors["REVERSAL_7D"] = -float(log_c1d[-1] - log_c1d[-8])

        # F5: carry = -funding_rate (negative funding = positive carry for longs)
        fund_hist = fetch_funding_history(symbol, limit=10)
        if fund_hist and isinstance(fund_hist, list) and len(fund_hist) > 0:
            last_rate = fund_hist[-1]
            if isinstance(last_rate, dict):
                current_funding = float(last_rate.get("fundingRate", 0))
                factors["CARRY"] = -current_funding

        # F6: inverse volatility (lower vol outperforms in carry regimes)
        if len(c1) >= 25:
            rets_24h = np.diff(log_c1[-26:])
            realized_vol = float(np.std(rets_24h) * np.sqrt(24))
            factors["VOLATILITY"] = -realized_vol

        # F7: trend quality via Hurst exponent
        if len(c1) >= 50:
            h = compute_hurst(c1[-100:])
            factors["TREND_QUALITY"] = float(h - 0.5)

        # F8: liquidity = -spread_bps (tighter = more liquid = preferred)
        spread = fetch_orderbook_spread(symbol)
        factors["LIQUIDITY"] = -float(spread)

        # F9: volume momentum = log(current vol / 20h avg vol)
        if len(v1) >= 20:
            avg_vol = float(np.mean(v1[-20:]))
            cur_vol = float(v1[-1])
            factors["VOLUME_MOMENTUM"] = float(np.log(cur_vol / avg_vol)) if avg_vol > 0 else 0.0

        # F10: OI momentum (1h OI change %)
        oi_hist = fetch_oi_history(symbol, period="1h", limit=48)
        if isinstance(oi_hist, list) and len(oi_hist) >= 2:
            try:
                oi_new = float(oi_hist[-1].get("sumOpenInterest", 0))
                oi_old = float(oi_hist[-2].get("sumOpenInterest", 1))
                factors["OI_MOMENTUM"] = float((oi_new - oi_old) / oi_old) if oi_old > 0 else 0.0
            except Exception:
                pass

        # F11: technical = (price - EMA20) / ATR20
        if len(c1) >= 25:
            ema20 = float(_ema(c1, 20)[-1])
            atr_arr = _atr(h1, l1, c1, 20)
            atr20 = float(atr_arr[-1]) if len(atr_arr) > 0 else 1.0
            factors["TECHNICAL"] = float((c1[-1] - ema20) / atr20) if atr20 > 0 else 0.0

        # F12: carry term (slope of recent funding rate history × 1000 for scale)
        if isinstance(fund_hist, list) and len(fund_hist) >= 5:
            try:
                rates = [float(f.get("fundingRate", 0)) for f in fund_hist[-8:]
                         if isinstance(f, dict)]
                if len(rates) >= 3:
                    idx = np.arange(len(rates), dtype=float)
                    X_l = np.column_stack([np.ones(len(idx)), idx])
                    coef, _, _, _ = np.linalg.lstsq(X_l, rates, rcond=None)
                    factors["CARRY_TERM"] = float(coef[1]) * 1000
            except Exception:
                pass

    except Exception:
        pass  # return zeros on any outer error

    return factors


# ── PORTFOLIO MATH ────────────────────────────────────────────────────────────

def ledoit_wolf_cov(returns_matrix):
    """
    Analytical Ledoit-Wolf shrinkage toward scaled identity.
    returns_matrix: ndarray (n_time, n_assets).
    """
    T, p = returns_matrix.shape
    S = np.cov(returns_matrix, rowvar=False)  # sample covariance

    # Shrinkage target: μ·I where μ = trace(S)/p
    mu = np.trace(S) / p
    target = mu * np.eye(p)

    # Oracle shrinkage intensity
    delta_sq = float(np.sum((S - target) ** 2))

    # Estimate beta² (sum of squared diffs from sample mean outer products)
    beta_sq = 0.0
    for i in range(T):
        xi = returns_matrix[i, :]
        outer = np.outer(xi, xi)
        beta_sq += float(np.sum((outer - S) ** 2))
    beta_sq /= T ** 2

    alpha = float(np.clip(beta_sq / (delta_sq + 1e-20), 0.0, 1.0))
    return (1.0 - alpha) * S + alpha * target


def max_sharpe_weights(expected_returns, cov_matrix, min_w=-0.25, max_w=0.25):
    """
    Maximum Sharpe portfolio via SLSQP.
    Returns weights array summing to 1, each in [min_w, max_w].
    Falls back to equal-weight on failure.
    """
    n = len(expected_returns)
    mu = np.array(expected_returns, dtype=float)
    sigma = np.array(cov_matrix, dtype=float)
    w0 = np.ones(n) / n

    def neg_sharpe(w):
        ret  = float(w @ mu)
        var  = float(w @ sigma @ w)
        vol  = np.sqrt(max(var, 1e-20))
        return -ret / vol

    def neg_sharpe_jac(w):
        ret  = float(w @ mu)
        var  = float(w @ sigma @ w)
        vol  = np.sqrt(max(var, 1e-20))
        # d(-Sharpe)/dw = -(mu·vol - ret·Σw/vol) / var
        return -(mu * vol - ret * (sigma @ w) / vol) / var

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_w, max_w)] * n

    result = minimize(
        neg_sharpe, w0, jac=neg_sharpe_jac,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 500}
    )
    return result.x if result.success else w0


def compute_portfolio(symbols, expected_returns, cov_matrix, max_position=0.25):
    """
    Compute max-Sharpe portfolio given expected_returns and cov_matrix.
    Returns (weights_dict, risk_metrics_dict).
    """
    n = len(symbols)
    if n == 0:
        return {}, {}

    cov = np.clip(cov_matrix, -1e6, 1e6)
    weights = max_sharpe_weights(expected_returns, cov,
                                 min_w=-max_position, max_w=max_position)

    # Annualise assuming hourly bars (8760 h/yr)
    ann = 24 * 365
    port_ret = float(weights @ expected_returns) * ann
    port_var = float(weights @ cov @ weights) * ann
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe   = port_ret / port_vol if port_vol > 0 else 0.0

    return (
        {sym: round(float(w), 4) for sym, w in zip(symbols, weights)},
        {
            "expected_return_ann_%": round(port_ret * 100, 2),
            "volatility_ann_%":      round(port_vol * 100, 2),
            "sharpe_ratio":          round(sharpe, 3),
            "net_exposure":          round(float(np.sum(weights)), 4),
            "gross_exposure":        round(float(np.sum(np.abs(weights))), 4),
            "max_long":              round(float(np.max(np.maximum(weights, 0))), 4),
            "max_short":             round(float(np.min(np.minimum(weights, 0))), 4),
        }
    )


# ── IC CALCULATION ────────────────────────────────────────────────────────────

def factor_ic(X, y):
    """
    Per-factor Spearman IC vs forward-return proxy y.
    IC > 0.05 → edge.  Returns {factor_name: ic_value}.
    """
    n = len(y)
    if n < 4:
        return {fn: 0.0 for fn in FACTOR_NAMES}

    y_rank = np.argsort(np.argsort(y)).astype(float)
    ics = {}
    for i, fn in enumerate(FACTOR_NAMES):
        x_rank = np.argsort(np.argsort(X[:, i])).astype(float)
        d = x_rank - y_rank
        ic = 1.0 - 6.0 * float(np.sum(d ** 2)) / (n * (n ** 2 - 1))
        ics[fn] = round(ic, 4)
    return ics


# ── MAIN SCAN ─────────────────────────────────────────────────────────────────

def run_factor_scan(symbols=None):
    if symbols is None:
        symbols = SYMBOLS

    print(f"Computing 12 factors for {len(symbols)} symbols...")

    X_rows = []
    factor_dicts = {}

    for sym in symbols:
        print(f"  {sym}...", end="", flush=True)
        try:
            fd = compute_factors_for_symbol(sym)
            row = [fd.get(fn, 0.0) for fn in FACTOR_NAMES]
            X_rows.append(row)
            factor_dicts[sym] = fd
            print(" OK")
        except Exception as e:
            X_rows.append([0.0] * len(FACTOR_NAMES))
            factor_dicts[sym] = {fn: 0.0 for fn in FACTOR_NAMES}
            print(f" error: {e}")

    X_arr = np.array(X_rows)

    # Build 1h return history for covariance estimation
    print("\nFetching return history for Ledoit-Wolf covariance...")
    ret_cols = []
    for sym in symbols:
        try:
            klines = fetch_klines(sym, "1h", limit=200)
            _, _, _, c, _ = klines_to_arrays(klines)
            ret_cols.append(np.diff(np.log(c))[-100:])
        except Exception:
            ret_cols.append(np.zeros(100))

    min_len = min(len(r) for r in ret_cols)
    returns_matrix = np.column_stack([r[-min_len:] for r in ret_cols])

    cov = ledoit_wolf_cov(returns_matrix)

    # Expected return proxy: blend of momentum + carry + trend quality
    expected_rets = np.array([
        factor_dicts[sym].get("MOMENTUM_4H", 0.0) * 2.0
        + factor_dicts[sym].get("MOMENTUM_1D", 0.0) * 1.0
        + factor_dicts[sym].get("CARRY", 0.0) * 100.0
        + factor_dicts[sym].get("TREND_QUALITY", 0.0) * 0.01
        for sym in symbols
    ])

    weights, risk_metrics = compute_portfolio(symbols, expected_rets, cov)
    ics = factor_ic(X_arr, expected_rets)

    # ── DISPLAY ───────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    sep = "═" * 74
    print(f"\n{sep}")
    print(f"  12-FACTOR QUANT MODEL  |  {ts}")
    print(sep)

    # Factor exposure table
    print("\n── FACTOR EXPOSURES ──────────────────────────────────────────────────")
    header = f"{'Factor':20s}" + "".join(f"{s[:7]:>8s}" for s in symbols)
    print(header)
    print("─" * len(header))
    for fn in FACTOR_NAMES:
        vals = [factor_dicts[s].get(fn, 0.0) for s in symbols]
        ic   = ics.get(fn, 0.0)
        star = "★" if abs(ic) > 0.05 else " "
        row  = f"{fn:20s}" + "".join(f"{v:+8.4f}" for v in vals)
        row += f"  IC={ic:+.3f}{star}"
        print(row)

    # Optimal portfolio
    print("\n── OPTIMAL PORTFOLIO (Max Sharpe) ────────────────────────────────────")
    print(f"{'Symbol':12s}  {'Weight':>8s}  {'Direction':>10s}  {'Exp Ret %/4h':>14s}")
    scored = [(sym, weights.get(sym, 0.0), expected_rets[i])
              for i, sym in enumerate(symbols)]
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    for sym, w, er in scored:
        if abs(w) < 0.0005:
            continue
        direction = "LONG " if w > 0 else "SHORT"
        print(f"  {sym:12s}  {w:+8.4f}  {direction:>10s}  {er*100:+10.4f}")

    # Risk metrics
    print("\n── PORTFOLIO RISK METRICS ────────────────────────────────────────────")
    for k, v in risk_metrics.items():
        print(f"  {k:35s}  {v}")

    # Signal summary
    print("\n── TOP SIGNALS BY CONVICTION ─────────────────────────────────────────")
    for sym, w, er in scored:
        icon = "⚡" if abs(w) > 0.15 else "▶" if abs(w) > 0.05 else "○"
        direction = "LONG" if w > 0 else "SHORT" if w < 0 else "NEUTRAL"
        print(f"  {icon} {sym:12s}  weight={w:+.4f}  exp={er*100:+.4f}%/4h  → {direction}")

    print(sep + "\n")

    return {
        "factors": factor_dicts,
        "weights": weights,
        "risk_metrics": risk_metrics,
        "factor_ics": ics,
        "expected_returns": {sym: float(expected_rets[i])
                             for i, sym in enumerate(symbols)},
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="12-Factor Quant Alpha Model")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--watch", type=int, help="Re-scan interval (seconds)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    syms = args.symbols or SYMBOLS

    def once():
        result = run_factor_scan(syms)
        if args.json:
            print(json.dumps(result, indent=2, default=str))

    if args.watch:
        while True:
            once()
            print(f"[Next scan in {args.watch}s... Ctrl+C to stop]")
            time.sleep(args.watch)
    else:
        once()


if __name__ == "__main__":
    main()
