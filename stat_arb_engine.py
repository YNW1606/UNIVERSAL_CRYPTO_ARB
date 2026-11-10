"""
Statistical Arbitrage Engine — Crypto Pairs on Binance
Layer 2 of the Global Quant Fund System

Signals:
  - Engle-Granger cointegration testing (pure numpy, no statsmodels)
  - Kalman filter time-varying hedge ratio
  - Rolling z-score spread signals (entry >2.5, exit <0.5, stop >4.0)
  - BTC beta factor extraction (residual alpha per symbol)
  - Half-life of mean reversion filter

Run:
  python stat_arb_engine.py                     # scan all 8 symbols
  python stat_arb_engine.py --interval 4h       # use 4h bars
  python stat_arb_engine.py --watch 300         # rescan every 5 min
  python stat_arb_engine.py --backtest          # backtest all pairs
"""

import numpy as np
import pandas as pd
import requests
import json
import time
import os
import argparse
from datetime import datetime, timezone
from itertools import combinations

BINANCE_SPOT = "https://api.binance.com"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "ADAUSDT"]
CACHE_DIR = "stat_arb_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# SECTION 1: DATA LAYER
# ══════════════════════════════════════════════════════════

def fetch_klines_df(symbol, interval="1h", limit=500):
    """Fetch OHLCV as DataFrame. Cache to CSV for 55 min."""
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}_{limit}.csv")
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < 3300:
            try:
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if "log_close" in df.columns and len(df) > 10:
                    return df
            except Exception:
                pass

    r = requests.get(
        f"{BINANCE_SPOT}/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15
    )
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms")
    df.set_index("open_time", inplace=True)
    df["log_close"] = np.log(df["close"])
    df.to_csv(cache_file)
    return df


def load_all_prices(symbols, interval="1h", limit=500):
    """Return DataFrame of log-close prices: index=time, cols=symbols."""
    dfs = {}
    for sym in symbols:
        try:
            df = fetch_klines_df(sym, interval, limit)
            dfs[sym] = df["log_close"]
            print(f"  ✓ {sym:12s}  {len(df)} bars")
        except Exception as e:
            print(f"  ✗ {sym}: {e}")
    combined = pd.DataFrame(dfs)
    combined.dropna(inplace=True)
    return combined


# ══════════════════════════════════════════════════════════
# SECTION 2: ADF TEST (pure numpy, no statsmodels)
# ══════════════════════════════════════════════════════════

def adf_test(series):
    """
    Augmented Dickey-Fuller test, pure numpy.
    Tests H0: series has a unit root (non-stationary).
    Rejects H0 if ADF stat < critical value.

    Returns: (adf_stat, p_value_approx, is_stationary)
    Critical values (MacKinnon 1994, constant-only model):
      1% = -3.43,  5% = -2.86,  10% = -2.57
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 20:
        return 0.0, 1.0, False

    ds = np.diff(s)       # first differences
    s_lag = s[:-1]        # lagged level

    X = np.column_stack([np.ones(len(s_lag)), s_lag])
    coef, _, _, _ = np.linalg.lstsq(X, ds, rcond=None)
    beta = coef[1]

    residuals = ds - X @ coef
    sigma2 = np.sum(residuals ** 2) / max(n - 3, 1)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(max(sigma2 * XtX_inv[1, 1], 1e-30))
    except np.linalg.LinAlgError:
        return 0.0, 1.0, False

    adf_stat = beta / se_beta

    if adf_stat < -3.43:
        p_approx = 0.01
    elif adf_stat < -2.86:
        p_approx = 0.05
    elif adf_stat < -2.57:
        p_approx = 0.10
    else:
        p_approx = 0.50

    return float(adf_stat), float(p_approx), p_approx < 0.05


# ══════════════════════════════════════════════════════════
# SECTION 3: COINTEGRATION TESTING (Engle-Granger 2-step)
# ══════════════════════════════════════════════════════════

def engle_granger_coint(s1_log, s2_log):
    """
    Test cointegration between two log-price series.
    Step 1: OLS  s1 = alpha + beta*s2 + residuals
    Step 2: ADF test on residuals

    Returns: (adf_stat, p_value, beta_ols, alpha_ols)
    """
    s1 = np.asarray(s1_log, dtype=float)
    s2 = np.asarray(s2_log, dtype=float)

    if len(s1) != len(s2) or len(s1) < 60:
        return 0.0, 1.0, 1.0, 0.0

    X = np.column_stack([np.ones(len(s2)), s2])
    coef, _, _, _ = np.linalg.lstsq(X, s1, rcond=None)
    alpha_ols, beta_ols = coef[0], coef[1]

    residuals = s1 - (alpha_ols + beta_ols * s2)
    adf_stat, p_value, _ = adf_test(residuals)

    return adf_stat, p_value, float(beta_ols), float(alpha_ols)


def compute_half_life(spread_series):
    """
    AR(1) mean-reversion half-life.
    Δspread = λ * spread_lag + ε
    half_life = -log(2) / log(1 + λ)
    Returns inf if no mean reversion.
    """
    spread = np.asarray(spread_series, dtype=float)
    spread = spread[~np.isnan(spread)]
    if len(spread) < 20:
        return float("inf")

    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)

    X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
    coef, _, _, _ = np.linalg.lstsq(X, spread_diff, rcond=None)
    lam = coef[1]

    if lam >= 0:
        return float("inf")
    base = 1 + lam
    if base <= 0:
        return float("inf")
    return float(-np.log(2) / np.log(base))


def find_all_pairs(prices_df, p_threshold=0.10):
    """
    Test all symbol pairs for cointegration.
    Returns list of cointegrated pairs sorted by p-value,
    filtered to practical half-lives (2h < HL < 120h).
    """
    pairs = []
    syms = prices_df.columns.tolist()

    for s1_sym, s2_sym in combinations(syms, 2):
        try:
            s1 = prices_df[s1_sym].values
            s2 = prices_df[s2_sym].values
            adf, p, beta, alpha = engle_granger_coint(s1, s2)

            if p < p_threshold:
                raw_spread = s1 - beta * s2 - alpha
                hl = compute_half_life(raw_spread)

                # Filter: only trade if half-life is actionable
                if hl == float("inf") or hl < 1.0 or hl > 200:
                    continue

                pairs.append({
                    "s1": s1_sym,
                    "s2": s2_sym,
                    "p_value": round(p, 4),
                    "adf_stat": round(adf, 3),
                    "beta": round(beta, 5),
                    "alpha": round(alpha, 6),
                    "half_life_h": round(hl, 1),
                })
        except Exception:
            pass

    return sorted(pairs, key=lambda x: x["p_value"])


# ══════════════════════════════════════════════════════════
# SECTION 4: KALMAN FILTER HEDGE RATIO
# ══════════════════════════════════════════════════════════

def kalman_hedge_ratio(s1_log, s2_log, delta=1e-4):
    """
    Time-varying hedge ratio via Kalman filter (pure numpy).
    State vector = [beta, alpha]
    Observation:  s1_t = beta_t * s2_t + alpha_t + noise

    Returns: (betas, alphas) as 1-D arrays of length n.
    """
    s1 = np.asarray(s1_log, dtype=float)
    s2 = np.asarray(s2_log, dtype=float)
    n = len(s1)

    Q = (delta / (1 - delta)) * np.eye(2)   # process noise covariance
    R = 1e-3                                  # observation noise variance

    P = np.eye(2) * 10.0                      # initial uncertainty (wide)
    state = np.array([1.0, 0.0])              # initial [beta, alpha]

    betas = np.empty(n)
    alphas = np.empty(n)

    for t in range(n):
        obs_mat = np.array([s2[t], 1.0])      # observation matrix row

        # --- Predict ---
        # state stays (random walk model)
        P = P + Q

        # --- Update ---
        pred_obs = float(obs_mat @ state)
        innovation = s1[t] - pred_obs
        S = float(obs_mat @ P @ obs_mat) + R  # innovation covariance
        K = (P @ obs_mat) / S                  # Kalman gain

        state = state + K * innovation
        P = (np.eye(2) - np.outer(K, obs_mat)) @ P

        betas[t] = state[0]
        alphas[t] = state[1]

    return betas, alphas


def compute_zscore(spread, window=60):
    """Rolling z-score of spread with given lookback window."""
    spread = np.asarray(spread, dtype=float)
    n = len(spread)
    zscores = np.full(n, np.nan)

    for i in range(window, n):
        window_data = spread[i - window:i]
        mu = np.mean(window_data)
        sigma = np.std(window_data, ddof=1)
        if sigma > 1e-12:
            zscores[i] = (spread[i] - mu) / sigma
        else:
            zscores[i] = 0.0

    return zscores


# ══════════════════════════════════════════════════════════
# SECTION 5: SIGNAL GENERATION
# ══════════════════════════════════════════════════════════

def generate_pair_signal(prices_df, pair_info, z_entry=2.5, z_exit=0.5, z_stop=4.0):
    """
    Compute current live signal for a cointegrated pair.
    Returns dict with signal, z_score, recommended action, spread context.
    """
    s1_sym = pair_info["s1"]
    s2_sym = pair_info["s2"]

    s1 = prices_df[s1_sym].values
    s2 = prices_df[s2_sym].values

    # Kalman hedge ratio (time-varying)
    betas, alphas = kalman_hedge_ratio(s1, s2)

    # Spread
    spread = s1 - betas * s2 - alphas

    # Z-score
    zscores = compute_zscore(spread, window=60)

    current_z = float(zscores[-1]) if not np.isnan(zscores[-1]) else None
    current_beta = float(betas[-1])

    # Half-life on recent data
    hl = compute_half_life(spread[-120:] if len(spread) >= 120 else spread)

    # Spread percentile
    valid_spread = spread[~np.isnan(spread)]
    spread_pct = float(np.mean(valid_spread < spread[-1]) * 100) if len(valid_spread) > 0 else 50.0

    # Trading signal
    if current_z is None:
        action, signal = "WAIT_DATA", 0
    elif current_z > z_stop:
        action = f"STOP_SPREAD_HIGH — {s1_sym} extremely expensive vs {s2_sym}"
        signal = -2  # emergency exit
    elif current_z < -z_stop:
        action = f"STOP_SPREAD_LOW — {s1_sym} extremely cheap vs {s2_sym}"
        signal = 2
    elif current_z > z_entry:
        action = f"SHORT_{s1_sym}  LONG_{s2_sym}  (spread z={current_z:+.2f} → mean)"
        signal = -1
    elif current_z < -z_entry:
        action = f"LONG_{s1_sym}  SHORT_{s2_sym}  (spread z={current_z:+.2f} → mean)"
        signal = 1
    elif abs(current_z) < z_exit:
        action = "EXIT — spread has mean-reverted"
        signal = 0
    else:
        action = "HOLD — spread within neutral zone"
        signal = 0

    return {
        "pair": f"{s1_sym}/{s2_sym}",
        "z_score": round(current_z, 3) if current_z is not None else None,
        "hedge_ratio": round(current_beta, 4),
        "signal": signal,
        "action": action,
        "half_life_h": round(hl, 1) if hl != float("inf") else "inf",
        "p_value": pair_info["p_value"],
        "spread_percentile": round(spread_pct, 1),
        "spread_last": round(float(spread[-1]), 6),
    }


# ══════════════════════════════════════════════════════════
# SECTION 6: BTC BETA FACTOR EXTRACTION
# ══════════════════════════════════════════════════════════

def btc_beta_signals(prices_df, window=48):
    """
    For each symbol, extract BTC beta and compute residual (pure alpha) signal.
    Residual z-score > 1.5 → SHORT (over-performed BTC, revert)
    Residual z-score < -1.5 → LONG (under-performed BTC, mean-revert)
    """
    if "BTCUSDT" not in prices_df.columns:
        return {}

    # Log returns
    returns = prices_df.diff().dropna()
    btc_ret = returns["BTCUSDT"].values

    results = {}

    for sym in prices_df.columns:
        if sym == "BTCUSDT":
            continue
        try:
            sym_ret = returns[sym].values
            # Align lengths
            min_len = min(len(btc_ret), len(sym_ret))
            b = btc_ret[-min(window, min_len):]
            s = sym_ret[-min(window, min_len):]

            if len(b) < 10:
                continue

            # OLS: sym_ret = alpha + beta * btc_ret
            X = np.column_stack([np.ones(len(b)), b])
            coef, _, _, _ = np.linalg.lstsq(X, s, rcond=None)
            alpha_daily, beta = float(coef[0]), float(coef[1])

            # Residuals over full history
            b_full = btc_ret[-min(200, min_len):]
            s_full = sym_ret[-min(200, min_len):]
            X_full = np.column_stack([np.ones(len(b_full)), b_full])
            resids = s_full - X_full @ np.array([alpha_daily, beta])

            # Z-score of latest residual
            resid_z = (resids[-1] - np.mean(resids)) / (np.std(resids, ddof=1) + 1e-12)

            # R² for quality
            ss_res = np.sum(resids**2)
            ss_tot = np.sum((s_full - np.mean(s_full))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            direction = "LONG" if resid_z < -1.5 else "SHORT" if resid_z > 1.5 else "NEUTRAL"
            strength = abs(resid_z)

            results[sym] = {
                "beta_to_btc": round(beta, 3),
                "alpha_bps": round(alpha_daily * 10000, 2),
                "residual_z": round(float(resid_z), 3),
                "r_squared": round(float(r2), 3),
                "trade_direction": direction,
                "signal_strength": round(float(strength), 2),
            }
        except Exception:
            pass

    return results


# ══════════════════════════════════════════════════════════
# SECTION 7: BACKTEST
# ══════════════════════════════════════════════════════════

def backtest_pair(prices_df, pair_info, z_entry=2.5, z_exit=0.5, z_stop=4.0):
    """
    Simplified backtest for one cointegrated pair.
    PnL denominated in z-score units (convergence = profit).
    """
    s1_sym, s2_sym = pair_info["s1"], pair_info["s2"]

    try:
        s1 = prices_df[s1_sym].values
        s2 = prices_df[s2_sym].values

        betas, alphas = kalman_hedge_ratio(s1, s2)
        spread = s1 - betas * s2 - alphas
        zscores = compute_zscore(spread, window=60)

        trades = []
        position = 0
        entry_z = 0.0
        entry_bar = 0

        for i in range(len(zscores)):
            if np.isnan(zscores[i]):
                continue
            z = float(zscores[i])

            if position == 0:
                if z > z_entry:
                    position = -1
                    entry_z = z
                    entry_bar = i
                elif z < -z_entry:
                    position = 1
                    entry_z = z
                    entry_bar = i
            else:
                should_exit = (
                    abs(z) < z_exit
                    or (position == 1 and z > z_stop)
                    or (position == -1 and z < -z_stop)
                )
                if should_exit:
                    # z convergence → profit: position * (entry_z - z)
                    pnl = position * (entry_z - z)
                    trades.append({
                        "entry_bar": entry_bar,
                        "exit_bar": i,
                        "hold_bars": i - entry_bar,
                        "entry_z": round(entry_z, 3),
                        "exit_z": round(z, 3),
                        "pnl_z": round(pnl, 4),
                    })
                    position = 0

        if not trades:
            return {
                "pair": f"{s1_sym}/{s2_sym}", "n_trades": 0,
                "win_rate": 0.0, "sharpe": 0.0,
                "avg_pnl_z": 0.0, "total_pnl_z": 0.0, "max_dd_z": 0.0
            }

        pnls = np.array([t["pnl_z"] for t in trades])
        equity_curve = np.cumsum(pnls)
        running_max = np.maximum.accumulate(equity_curve)
        max_dd = float(np.min(equity_curve - running_max))

        wins = int(np.sum(pnls > 0))
        win_rate = wins / len(trades)
        avg_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1))
        sharpe = avg_pnl / (std_pnl + 1e-10) * np.sqrt(252)

        return {
            "pair": f"{s1_sym}/{s2_sym}",
            "n_trades": len(trades),
            "win_rate": round(win_rate, 3),
            "sharpe": round(float(sharpe), 2),
            "avg_pnl_z": round(avg_pnl, 4),
            "total_pnl_z": round(float(np.sum(pnls)), 3),
            "max_dd_z": round(max_dd, 3),
            "avg_hold_bars": round(float(np.mean([t["hold_bars"] for t in trades])), 1),
        }

    except Exception as e:
        return {"pair": f"{s1_sym}/{s2_sym}", "n_trades": 0, "error": str(e)}


# ══════════════════════════════════════════════════════════
# SECTION 8: PORTFOLIO NEUTRAL SIZING
# ══════════════════════════════════════════════════════════

def compute_neutral_sizing(pair_signal, portfolio_usd=100_000, max_pair_pct=0.10):
    """
    Dollar-neutral sizing for a pair trade.
    Long $X in cheap leg + Short $X * beta in expensive leg.
    Returns: {long_sym, long_usd, short_sym, short_usd, net_exposure}
    """
    if pair_signal["signal"] == 0:
        return None

    max_usd = portfolio_usd * max_pair_pct
    beta = abs(pair_signal["hedge_ratio"])
    s1_sym, s2_sym = pair_signal["pair"].split("/")

    # z-score based position scaling: larger z → larger position (capped)
    z_abs = min(abs(pair_signal["z_score"] or 0), 4.0)
    scale = min(z_abs / 4.0, 1.0)
    base_usd = max_usd * scale

    if pair_signal["signal"] == 1:  # LONG s1, SHORT s2
        long_sym, short_sym = s1_sym, s2_sym
        long_usd = base_usd
        short_usd = base_usd * beta
    else:  # SHORT s1, LONG s2
        long_sym, short_sym = s2_sym, s1_sym
        long_usd = base_usd * beta
        short_usd = base_usd

    net_exposure = long_usd - short_usd

    return {
        "long_sym": long_sym,
        "long_usd": round(long_usd, 2),
        "short_sym": short_sym,
        "short_usd": round(short_usd, 2),
        "net_exposure_usd": round(net_exposure, 2),
        "hedge_ratio": round(beta, 4),
    }


# ══════════════════════════════════════════════════════════
# SECTION 9: MAIN SCANNER
# ══════════════════════════════════════════════════════════

def run_stat_arb_scan(symbols=None, interval="1h", limit=500, run_backtest=False):
    """Full scan: load data → find pairs → generate signals → (optionally backtest)."""
    symbols = symbols or SYMBOLS

    print("\n" + "═" * 65)
    print(f"  STATISTICAL ARBITRAGE ENGINE  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("═" * 65)

    print(f"\nFetching {len(symbols)} symbols  ({interval} bars × {limit})...")
    prices_df = load_all_prices(symbols, interval, limit)

    if len(prices_df) < 60:
        print("  ERROR: Not enough data loaded")
        return

    print(f"\n  Aligned: {len(prices_df)} bars  "
          f"({prices_df.index[0].strftime('%Y-%m-%d')} → {prices_df.index[-1].strftime('%Y-%m-%d')})")

    n_pairs = len(symbols) * (len(symbols) - 1) // 2
    print(f"\nTesting {n_pairs} pairs for cointegration...")

    pairs = find_all_pairs(prices_df, p_threshold=0.10)
    n_found = len(pairs)

    if n_found == 0:
        print("  No cointegrated pairs found  (market may be in regime transition)")
    else:
        print(f"  Found {n_found} cointegrated pairs  (p < 0.10, practical HL)\n")

    # ── Active signals ──────────────────────────────────────
    active = []
    for pair in pairs[:15]:
        try:
            sig = generate_pair_signal(prices_df, pair)
            if run_backtest:
                sig["backtest"] = backtest_pair(prices_df, pair)
            if sig["signal"] != 0:
                active.append(sig)
        except Exception as e:
            pass

    if active:
        print("⚡  ACTIVE STAT-ARB SIGNALS:")
        print("─" * 65)
        for sig in sorted(active, key=lambda x: abs(x.get("z_score") or 0), reverse=True):
            z = sig["z_score"]
            beta = sig["hedge_ratio"]
            hl = sig["half_life_h"]
            p = sig["p_value"]
            icon = "★" if abs(z or 0) > 3.0 else "▶"
            print(f"\n{icon}  {sig['pair']:<22s}  z={z:+.3f}  β={beta:.4f}  "
                  f"HL={hl}h  p={p:.4f}")
            print(f"   ACTION : {sig['action']}")
            print(f"   Spread % of dist: {sig['spread_percentile']:.1f}th percentile")

            # Sizing
            sizing = compute_neutral_sizing(sig)
            if sizing:
                print(f"   SIZING  : LONG {sizing['long_sym']} ${sizing['long_usd']:,.0f}  "
                      f"SHORT {sizing['short_sym']} ${sizing['short_usd']:,.0f}  "
                      f"(net ${sizing['net_exposure_usd']:+,.0f})")

            if run_backtest and "backtest" in sig:
                bt = sig["backtest"]
                print(f"   BACKTEST: {bt['n_trades']} trades  "
                      f"WR={bt['win_rate']:.0%}  "
                      f"Sharpe={bt['sharpe']:.2f}  "
                      f"Avg PnL={bt['avg_pnl_z']:+.3f}z  "
                      f"MaxDD={bt['max_dd_z']:.3f}z")
    else:
        print("○  No actionable stat-arb signals right now (all pairs within z < 2.5)")

    # ── All pairs summary ────────────────────────────────────
    if pairs:
        print(f"\n{'─'*65}")
        print(f"  ALL COINTEGRATED PAIRS  (top {min(10, len(pairs))}):")
        print(f"  {'Pair':<22} {'z':>7} {'β':>8} {'HL':>8} {'p':>7}  Signal")
        print(f"  {'─'*22} {'─'*7} {'─'*8} {'─'*8} {'─'*7}  ──────")
        for pair in pairs[:10]:
            try:
                sig = generate_pair_signal(prices_df, pair)
                z = sig["z_score"]
                z_str = f"{z:+.3f}" if z is not None else "  N/A "
                hl = sig["half_life_h"]
                icon = "⚡" if sig["signal"] != 0 else "○"
                print(f"  {sig['pair']:<22} {z_str:>7} {sig['hedge_ratio']:>8.4f} "
                      f"{str(hl):>7}h {pair['p_value']:>7.4f}  {icon} {sig['action'][:30]}")
            except Exception:
                pass

    # ── BTC beta residual signals ────────────────────────────
    print(f"\n{'─'*65}")
    print("  BTC BETA RESIDUAL SIGNALS (alpha per coin):")
    print(f"  {'Symbol':<12} {'β':>7} {'α(bps)':>8} {'Resid_z':>9} {'R²':>6}  Direction")
    print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*9} {'─'*6}  ─────────")

    beta_sigs = btc_beta_signals(prices_df)
    for sym, sig in sorted(beta_sigs.items(),
                           key=lambda x: abs(x[1]["residual_z"]), reverse=True):
        icon = "⚡" if sig["trade_direction"] != "NEUTRAL" else "○"
        print(f"  {sym:<12} {sig['beta_to_btc']:>7.3f} {sig['alpha_bps']:>8.2f} "
              f"{sig['residual_z']:>9.3f} {sig['r_squared']:>6.3f}  "
              f"{icon} {sig['trade_direction']}")

    print("\n" + "═" * 65)


# ══════════════════════════════════════════════════════════
# SECTION 10: IMPORTABLE API
# ══════════════════════════════════════════════════════════

def get_pair_signals(symbols=None, interval="1h", limit=300):
    """
    Importable: returns dict of active pair signals and BTC beta signals.
    Used by global_quant_system.py for integration.
    """
    symbols = symbols or SYMBOLS
    try:
        prices_df = load_all_prices(symbols, interval, limit)
        if len(prices_df) < 60:
            return {"pairs": [], "btc_beta": {}}

        pairs = find_all_pairs(prices_df, p_threshold=0.10)
        active_signals = []
        for pair in pairs[:10]:
            try:
                sig = generate_pair_signal(prices_df, pair)
                active_signals.append(sig)
            except Exception:
                pass

        btc_beta = btc_beta_signals(prices_df)
        return {"pairs": active_signals, "btc_beta": btc_beta}
    except Exception as e:
        return {"pairs": [], "btc_beta": {}, "error": str(e)}


# ══════════════════════════════════════════════════════════
# SECTION 11: MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage Engine — Crypto Pairs on Binance"
    )
    parser.add_argument("--interval", default="1h",
                        choices=["15m", "1h", "4h", "1d"],
                        help="Kline interval (default: 1h)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of bars per symbol (default: 500)")
    parser.add_argument("--watch", type=int, default=0,
                        help="Watch mode: re-scan every N seconds")
    parser.add_argument("--z-entry", type=float, default=2.5,
                        help="Z-score threshold to enter trade (default: 2.5)")
    parser.add_argument("--backtest", action="store_true",
                        help="Include backtests for each found pair")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override default symbol list")
    args = parser.parse_args()

    syms = args.symbols or SYMBOLS

    def once():
        run_stat_arb_scan(
            symbols=syms,
            interval=args.interval,
            limit=args.limit,
            run_backtest=args.backtest,
        )

    if args.watch > 0:
        try:
            while True:
                once()
                print(f"\n[Next scan in {args.watch}s... Ctrl+C to stop]\n")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        once()


if __name__ == "__main__":
    main()
