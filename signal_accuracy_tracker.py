"""
Signal Accuracy Tracker
Logs every signal from global_quant_system, checks outcomes after 4h,
and builds a running win-rate / IC / profit-factor report.

Usage:
  python signal_accuracy_tracker.py --once       # one cycle
  python signal_accuracy_tracker.py --watch 1800 # every 30 min
  python signal_accuracy_tracker.py --report     # report only
  python signal_accuracy_tracker.py --clear      # reset DB
"""

import sqlite3, os, sys, time, json, requests, argparse
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_db")
DB_PATH = os.path.join(DB_DIR, "signals.db")
BINANCE = "https://api.binance.com"

WIN_THRESHOLD   = 0.003  # 0.3% net of fees to call it a WIN
OUTCOME_AFTER_H = 4      # hours after signal → check price
MIN_SCORE       = 15     # log all signals with |unified_score| >= this (was 35)
ACTIONABLE_MIN  = 35     # "actionable" signals shown prominently in reports

os.makedirs(DB_DIR, exist_ok=True)


# ── DATABASE ─────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS signals (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp     TEXT NOT NULL,
        symbol        TEXT NOT NULL,
        direction     TEXT NOT NULL,
        unified_score INTEGER NOT NULL,
        master_score  INTEGER,
        hft_score     INTEGER,
        factor_weight REAL,
        stat_arb_z    REAL,
        entry_price   REAL,
        outcome       TEXT DEFAULT 'PENDING',
        exit_price    REAL,
        return_pct    REAL,
        checked_at    TEXT
    );
    CREATE TABLE IF NOT EXISTS accuracy_summary (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        computed_at      TEXT,
        total_signals    INTEGER,
        win_rate         REAL,
        avg_return_pct   REAL,
        profit_factor    REAL,
        tier_perfect_wr  REAL,
        tier_high_wr     REAL,
        tier_moderate_wr REAL,
        master_ic        REAL,
        hft_ic           REAL,
        factor_ic        REAL,
        best_symbol      TEXT,
        best_symbol_wr   REAL
    );
    """)
    conn.commit()
    conn.close()


# ── SIGNAL LOGGING ────────────────────────────────────────────────────────────

def get_current_price(symbol):
    try:
        from data_utils import fetch_ticker_price
        px = fetch_ticker_price(symbol)
        if px and px > 0:
            return float(px)
    except Exception:
        pass
    try:
        r = requests.get(f"{BINANCE}/api/v3/ticker/price",
                         params={"symbol": symbol}, timeout=5)
        return float(r.json()["price"])
    except Exception:
        return None

def log_signals(scan_results):
    """
    scan_results: list of fused signal dicts from global_quant_system.run_global_scan()
    Logs every signal with |unified_score| >= MIN_SCORE, deduplicates within same 30-min window.
    """
    conn = get_conn()
    now  = datetime.now(timezone.utc).isoformat()
    window_start = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    logged = 0

    for r in scan_results:
        score = r.get("unified_score", 0)
        if abs(score) < MIN_SCORE:
            continue

        sym       = r.get("symbol", "")
        direction = r.get("direction", "NEUTRAL")
        if direction == "NEUTRAL":
            continue

        # Dedup: skip if same symbol+direction already logged in last 30 min
        existing = conn.execute(
            "SELECT id FROM signals WHERE symbol=? AND direction=? AND timestamp>?",
            (sym, direction, window_start)
        ).fetchone()
        if existing:
            continue

        entry_price = get_current_price(sym)

        # Extract component scores safely
        comps        = r.get("components", {})
        master_score = int(comps.get("master", 0))
        hft_score    = int(comps.get("hft", 0))
        factor_w     = float(r.get("factor_weight") or 0)
        sa           = r.get("stat_arb") or {}
        stat_arb_z   = float(sa.get("residual_z", 0)) if sa else 0.0

        conn.execute("""
            INSERT INTO signals
              (timestamp, symbol, direction, unified_score,
               master_score, hft_score, factor_weight, stat_arb_z, entry_price)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (now, sym, direction, score,
              master_score, hft_score, factor_w, stat_arb_z, entry_price))
        logged += 1

    conn.commit()
    conn.close()
    return logged


# ── OUTCOME CHECKING ──────────────────────────────────────────────────────────

def get_price_n_hours_ago(symbol, hours):
    """Return the close price approximately `hours` ago from 1h klines."""
    try:
        from data_utils import fetch_klines
        limit = hours + 3
        klines = fetch_klines(symbol, "1h", limit)
        if len(klines) < hours + 1:
            return None
        idx = -(hours + 1)
        return float(klines[idx][4])   # close price
    except Exception:
        pass
    try:
        limit = hours + 3
        r = requests.get(f"{BINANCE}/api/v3/klines",
                         params={"symbol": symbol, "interval": "1h", "limit": limit},
                         timeout=8)
        klines = r.json()
        if len(klines) < hours + 1:
            return None
        idx = -(hours + 1)
        return float(klines[idx][4])
    except Exception:
        return None

def check_pending_outcomes():
    """
    For all PENDING signals older than OUTCOME_AFTER_H hours,
    fetch the price AT signal_time + OUTCOME_AFTER_H (not current price).
    Uses klines to look back the correct number of hours.
    """
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=OUTCOME_AFTER_H)).isoformat()
    pending = conn.execute(
        "SELECT * FROM signals WHERE outcome='PENDING' AND timestamp<=?",
        (cutoff,)
    ).fetchall()

    resolved = 0
    now_utc = datetime.now(timezone.utc)
    for row in pending:
        sym         = row["symbol"]
        direction   = row["direction"]
        entry_price = row["entry_price"]
        if entry_price is None or entry_price == 0:
            continue

        # Determine how many hours ago the target exit time was.
        # signal logged at T → exit price should be AT T + OUTCOME_AFTER_H
        # from now that is (now - T) - OUTCOME_AFTER_H hours ago
        try:
            ts_str = row["timestamp"]
            # Handle both naive ISO strings and timezone-aware ones
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            if "+" in ts_str[10:] or ts_str.endswith("UTC"):
                signal_time = datetime.fromisoformat(ts_str.replace(" UTC", "+00:00"))
            else:
                signal_time = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
            hours_since = (now_utc - signal_time).total_seconds() / 3600
        except Exception:
            hours_since = OUTCOME_AFTER_H + 0.5  # fallback: treat as just resolved

        # hours_back = how many hours ago was T + OUTCOME_AFTER_H
        hours_back = max(0, int(round(hours_since - OUTCOME_AFTER_H)))

        if hours_back <= 1:
            # Signal just became resolvable: use current price as good proxy
            exit_price = get_current_price(sym)
        else:
            # Use historical kline at the correct point in time
            exit_price = get_price_n_hours_ago(sym, hours_back)
            if exit_price is None:
                exit_price = get_current_price(sym)

        if exit_price is None or exit_price == 0:
            continue

        raw_ret = (exit_price - entry_price) / entry_price
        if direction == "SHORT":
            raw_ret = -raw_ret

        outcome    = "WIN" if raw_ret >= WIN_THRESHOLD else "LOSS"
        return_pct = round(raw_ret * 100, 4)
        checked_at = now_utc.isoformat()

        conn.execute("""
            UPDATE signals SET outcome=?, exit_price=?, return_pct=?, checked_at=?
            WHERE id=?
        """, (outcome, exit_price, return_pct, checked_at, row["id"]))
        resolved += 1

    conn.commit()
    conn.close()
    return resolved


# ── ACCURACY REPORT ───────────────────────────────────────────────────────────

def _spearman_ic(scores, returns):
    """Spearman rank correlation between scores list and returns list."""
    n = len(scores)
    if n < 4:
        return 0.0
    import numpy as np
    s_rank = _ranks(scores)
    r_rank = _ranks(returns)
    d = [s_rank[i] - r_rank[i] for i in range(n)]
    rho = 1 - 6 * sum(x**2 for x in d) / (n * (n**2 - 1))
    return round(float(rho), 4)

def _ranks(vals):
    sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0] * len(vals)
    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank
    return ranks

def _tier_stats(rows, low, high):
    """Win rate and avg return for signals with |score| in [low, high)."""
    subset = [r for r in rows if low <= abs(r["unified_score"]) < high]
    resolved = [r for r in subset if r["outcome"] != "PENDING"]
    if not resolved:
        return 0, 0.0, 0.0
    wins = sum(1 for r in resolved if r["outcome"] == "WIN")
    wr   = wins / len(resolved)
    avg_ret = sum(r["return_pct"] or 0 for r in resolved) / len(resolved)
    return len(subset), round(wr, 4), round(avg_ret, 4)

def compute_accuracy_report():
    conn  = get_conn()
    rows  = conn.execute("SELECT * FROM signals ORDER BY timestamp").fetchall()
    conn.close()

    total    = len(rows)
    resolved = [r for r in rows if r["outcome"] != "PENDING"]
    pending  = total - len(resolved)
    wins     = [r for r in resolved if r["outcome"] == "WIN"]

    win_rate  = len(wins) / len(resolved) if resolved else 0.0
    returns   = [r["return_pct"] or 0 for r in resolved]
    avg_ret   = sum(returns) / len(returns) if returns else 0.0

    # Profit factor
    gains  = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    pf     = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
    if pf == float("inf"):
        pf_str = "∞"
    else:
        pf_str = f"{pf:.2f}"

    # Tier breakdown (now includes MONITORING 15-34)
    n_perf, wr_perf, ret_perf = _tier_stats(rows, 65, 101)
    n_high, wr_high, ret_high = _tier_stats(rows, 50, 65)
    n_mod,  wr_mod,  ret_mod  = _tier_stats(rows, 35, 50)
    n_mon,  wr_mon,  ret_mon  = _tier_stats(rows, 15, 35)

    # Engine ICs (using resolved signals only)
    ic_master = ic_hft = ic_factor = 0.0
    if len(resolved) >= 4:
        rets_list = [r["return_pct"] or 0 for r in resolved]
        ic_master = _spearman_ic([r["master_score"] or 0 for r in resolved], rets_list)
        ic_hft    = _spearman_ic([r["hft_score"] or 0 for r in resolved], rets_list)
        ic_factor = _spearman_ic([r["factor_weight"] or 0 for r in resolved], rets_list)

    # Per-symbol stats
    syms = {}
    for r in resolved:
        s = r["symbol"]
        if s not in syms:
            syms[s] = {"n": 0, "wins": 0}
        syms[s]["n"] += 1
        if r["outcome"] == "WIN":
            syms[s]["wins"] += 1
    sym_wrs = {s: syms[s]["wins"] / syms[s]["n"] for s in syms if syms[s]["n"] > 0}
    best_sym    = max(sym_wrs, key=sym_wrs.get) if sym_wrs else "N/A"
    best_sym_wr = sym_wrs.get(best_sym, 0.0)

    # Timestamps
    first_ts = rows[0]["timestamp"][:16] if rows else "N/A"
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Last 10 signals
    last10 = list(rows)[-10:]

    # Save summary snapshot
    conn2 = get_conn()
    conn2.execute("""
        INSERT INTO accuracy_summary
          (computed_at, total_signals, win_rate, avg_return_pct, profit_factor,
           tier_perfect_wr, tier_high_wr, tier_moderate_wr,
           master_ic, hft_ic, factor_ic, best_symbol, best_symbol_wr)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (now_str, total, win_rate, avg_ret,
          pf if pf != float("inf") else 9999,
          wr_perf, wr_high, wr_mod,
          ic_master, ic_hft, ic_factor,
          best_sym, best_sym_wr))
    conn2.commit()
    conn2.close()

    # ── Print ────────────────────────────────────────────────────────────────
    W = 58
    print("\n" + "═" * W)
    print(f"  SIGNAL ACCURACY REPORT  |  {now_str}")
    print(f"  Running since: {first_ts}  |  {total} signals total")
    print("═" * W)

    print(f"\n  Total signals logged: {total}")
    print(f"  Outcomes resolved:    {len(resolved)}  ({len(resolved)/total*100:.0f}%)" if total else "  Outcomes resolved:    0")
    print(f"  Still pending:        {pending}")

    if resolved:
        print(f"\n  OVERALL PERFORMANCE:")
        print(f"  Win Rate:        {win_rate*100:.1f}%  ({len(wins)}/{len(resolved)})")
        print(f"  Avg Return:     {avg_ret:+.3f}%  per signal")
        print(f"  Profit Factor:   {pf_str}")

        print(f"\n  BY SCORE TIER:")
        def tier_line(label, n, wr, ret):
            if n == 0:
                return f"  {label}  no signals yet"
            return (f"  {label}  {n:3d} signals, "
                    f"{wr*100:5.1f}% WR,  {ret:+.3f}% avg")
        print(tier_line("★★★ PERFECT    (≥65): ", n_perf, wr_perf, ret_perf))
        print(tier_line("★★  HIGH       (50-64):", n_high, wr_high, ret_high))
        print(tier_line("★   MODERATE   (35-49):", n_mod,  wr_mod,  ret_mod))
        print(tier_line("○   MONITORING (15-34):", n_mon,  wr_mon,  ret_mon))

        print(f"\n  BY ENGINE CONTRIBUTION (IC = Spearman rank corr with actual returns):")
        def ic_label(v):
            if abs(v) > 0.15: return "strong"
            if abs(v) > 0.08: return "good"
            if abs(v) > 0.04: return "moderate"
            return "marginal"
        print(f"  Master engine IC:   {ic_master:+.3f}  ({ic_label(ic_master)})")
        print(f"  HFT engine IC:      {ic_hft:+.3f}  ({ic_label(ic_hft)})")
        print(f"  Factor model IC:    {ic_factor:+.3f}  ({ic_label(ic_factor)})")

        if sym_wrs:
            print(f"\n  BY SYMBOL (resolved only):")
            for s in sorted(sym_wrs, key=sym_wrs.get, reverse=True)[:5]:
                n  = syms[s]["n"]
                wr = sym_wrs[s]
                tag = " ← best" if s == best_sym else ""
                print(f"  {s:12s}  {n:3d} signals, {wr*100:.1f}% WR{tag}")

    if last10:
        print(f"\n  LAST {len(last10)} SIGNALS:")
        for r in reversed(last10):
            ts   = r["timestamp"][:16]
            sym  = r["symbol"]
            d    = r["direction"]
            sc   = r["unified_score"]
            oc   = r["outcome"]
            ret  = f"{r['return_pct']:+.2f}%" if r["return_pct"] is not None else ""
            line = f"  {ts}  {sym:12s} {d:5s} {sc:+4d}  → {oc}"
            if ret:
                line += f"  {ret}"
            print(line)

    print("═" * W + "\n")

    return {
        "total": total, "resolved": len(resolved), "pending": pending,
        "win_rate": win_rate, "avg_return": avg_ret, "profit_factor": pf,
        "best_symbol": best_sym,
    }


# ── JSON EXPORT ───────────────────────────────────────────────────────────────

def save_scan_json(scan_results, extra):
    """Save latest_scan.json for the dashboard generator."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    active = []
    watchlist = []  # all symbols sorted by |score|, even below threshold
    for r in scan_results:
        sc   = r.get("unified_score", 0)
        appr = r.get("approval") or {}
        comp = r.get("components") or {}
        entry = {
            "symbol":        r.get("symbol"),
            "direction":     r.get("direction"),
            "unified_score": sc,
            "tier":          r.get("tier", ""),
            "components":    {k: round(float(v), 1) for k, v in comp.items()},
            "approved":      bool(appr.get("approved")),
            "size_usd":      round(float(appr.get("size_usd") or 0), 0),
            "size_pct":      round(float(appr.get("size_pct") or 0), 2),
        }
        watchlist.append(entry)
        if abs(sc) >= ACTIONABLE_MIN and r.get("direction") != "NEUTRAL":
            active.append(entry)

    carry_raw = extra.get("carry_ops") or []
    carry = []
    for c in carry_raw:
        carry.append({
            "symbol":    c.get("symbol"),
            "ann_pct":   round(float(c.get("ann_pct") or 0), 2),
            "basis_pct": round(float(c.get("basis_pct") or 0), 4),
            "action":    c.get("action"),
        })

    pairs = []
    for p in (extra.get("pair_sigs") or []):
        z = p.get("z_score")
        pv = p.get("p_value")
        pairs.append({
            "pair":        p.get("pair"),
            "z_score":     round(float(z), 3) if z is not None else None,
            "action":      p.get("action", "HOLD"),
            "half_life_h": p.get("half_life_h"),
            "p_value":     round(float(pv), 4) if pv is not None else None,
            "hedge_ratio": round(float(p.get("hedge_ratio") or 0), 3),
        })

    es = extra.get("engine_status") or {}
    engines = {k: (v.startswith("✓")) for k, v in es.items()}

    data = {
        "timestamp":      now,
        "engines_online": int(extra.get("engines_online") or 0),
        "engines":        engines,
        "active_signals": active,
        "watchlist":      watchlist,   # all symbols regardless of threshold
        "carry_arb":      carry,
        "stat_arb_pairs": pairs,
        "risk":           extra.get("risk_summary") or {},
    }

    json_path = os.path.join(DB_DIR, "latest_scan.json")
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  Saved latest_scan.json ({len(active)} active, {len(watchlist)} watchlist)")
    except Exception as e:
        print(f"  JSON save error: {e}")


# ── MAIN CYCLE ────────────────────────────────────────────────────────────────

def run_cycle(portfolio_size=100_000.0, verbose=False):
    print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] Starting cycle...")

    # 1. Resolve pending outcomes
    resolved = check_pending_outcomes()
    if resolved:
        print(f"  Resolved {resolved} pending signal(s).")

    # 2. Run global scan
    print("  Running global scan (all 5 engines)...")
    scan_results = []
    extra = {}
    try:
        import global_quant_system as gqs
        scan_results, extra = gqs.run_global_scan(portfolio_size=portfolio_size)
    except Exception as e:
        print(f"  Global scan error: {e}")

    # 3. Log new signals
    logged = log_signals(scan_results)
    print(f"  Logged {logged} new signal(s) (|score| >= {MIN_SCORE}).")

    # 4. Save JSON for dashboard
    save_scan_json(scan_results, extra)

    # 5. Accuracy report
    compute_accuracy_report()


def main():
    parser = argparse.ArgumentParser(description="Signal Accuracy Tracker")
    parser.add_argument("--once",      action="store_true", help="Run one cycle then exit")
    parser.add_argument("--watch",     type=int, metavar="SECONDS", help="Repeat every N seconds")
    parser.add_argument("--report",    action="store_true", help="Show report only, no scan")
    parser.add_argument("--clear",     action="store_true", help="Delete the signals database and start fresh")
    parser.add_argument("--portfolio", type=float, default=100_000.0, help="Portfolio size USD")
    args = parser.parse_args()

    init_db()

    if args.clear:
        conn = get_conn()
        conn.execute("DELETE FROM signals")
        conn.execute("DELETE FROM accuracy_summary")
        conn.commit()
        conn.close()
        print("Database cleared.")
        return

    if args.report:
        compute_accuracy_report()
        return

    if args.once:
        run_cycle(args.portfolio)
        return

    if args.watch:
        print(f"Watch mode: scanning every {args.watch}s. Ctrl+C to stop.")
        while True:
            try:
                run_cycle(args.portfolio)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"  Cycle error: {e}")
            print(f"  Sleeping {args.watch}s...")
            time.sleep(args.watch)
        return

    # Default: one cycle
    run_cycle(args.portfolio)


if __name__ == "__main__":
    main()
