"""
Perpetual Arbitrage Paper Trader
=================================
Strategies:
  CARRY_ARB    — Long spot + Short perp; collect positive funding (earns when longs pay shorts)
  NEG_CARRY    — Long perp + Short spot; collect negative funding (earns when shorts pay longs)
  SPOT_PREMIUM — Short perp + Long spot when perp trades at premium (>0.25%); fade on convergence
  SPOT_DISCOUNT— Long perp + Short spot when perp trades at discount (<-0.25%); fade on convergence

All trades are PAPER ONLY — no real money. Realistic simulation:
  - Taker fees: 0.04% per leg × 2 legs × open+close = 0.16% round-trip
  - Funding applied every 8 h (Bybit schedule: 00:00 / 08:00 / 16:00 UTC)
  - P&L split into: funding income, basis change, fees

Usage:
  python perp_arb_paper.py --once    # one full cycle
  python perp_arb_paper.py --watch 300
  python perp_arb_paper.py --report  # report only, no scan
  python perp_arb_paper.py --reset   # clear DB
"""

import sqlite3, os, sys, json, time, argparse, uuid
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    fetch_ticker_price, fetch_perp_price, fetch_funding_rate,
    fetch_funding_history,
)

# ── SYMBOLS ──────────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    "DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","NEARUSDT",
    "LINKUSDT","UNIUSDT","LTCUSDT","TRXUSDT","ATOMUSDT",
    "SUIUSDT","APTUSDT","ARBUSDT","OPUSDT",
    # Note: MATICUSDT is delisted on Bybit perps (renamed POL) — excluded
]

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
CARRY_MIN_RATE      = 0.0001    # 0.01%/8h = ~11%/yr — min to open carry arb
CARRY_EXIT_RATE     = 0.00004   # 0.004%/8h — close when funding drops here
NEG_CARRY_MIN_RATE  = -0.0001   # negative funding threshold for neg-carry
BASIS_MIN_PCT       = 0.25      # 0.25% basis premium/discount to enter
BASIS_EXIT_PCT      = 0.05      # close when basis converges to this
BASIS_STOP_PCT      = 0.60      # stop-loss if basis blows out this much more
CARRY_MAX_DAYS      = 30        # max holding for carry arb (days)
BASIS_MAX_HOURS     = 72        # max holding for basis arb (hours)
FEE_PER_LEG         = 0.0004    # 0.04% taker fee per leg
ROUND_TRIP_FEES     = FEE_PER_LEG * 4  # open+close, 2 legs each = 0.16%
STARTING_BALANCE    = 1_000_000.0       # paper account opening balance ($1M)
MAX_OPEN_TRADES     = 15                # max simultaneous paper trades
POSITION_USD        = STARTING_BALANCE / MAX_OPEN_TRADES  # ~$66,667 per leg
FUNDING_INTERVAL_H  = 8                 # hours between funding payments

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_db")
DB_PATH = os.path.join(DB_DIR, "perp_arb.db")
os.makedirs(DB_DIR, exist_ok=True)


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = _conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
        id               TEXT PRIMARY KEY,
        strategy         TEXT NOT NULL,
        symbol           TEXT NOT NULL,
        direction        TEXT NOT NULL,   -- LONG_SPOT_SHORT_PERP or LONG_PERP_SHORT_SPOT
        status           TEXT DEFAULT 'OPEN',
        opened_at        TEXT NOT NULL,
        closed_at        TEXT,

        spot_entry       REAL,
        perp_entry       REAL,
        basis_entry_pct  REAL,
        funding_at_open  REAL,
        ann_yield_at_open REAL,

        size_usd         REAL DEFAULT 1000,
        fees_total       REAL DEFAULT 0,

        funding_total    REAL DEFAULT 0,
        last_funding_at  TEXT,
        funding_periods  INTEGER DEFAULT 0,

        spot_now         REAL,
        perp_now         REAL,
        basis_now_pct    REAL,
        unrealized_pnl   REAL DEFAULT 0,

        spot_exit        REAL,
        perp_exit        REAL,
        realized_pnl     REAL,
        exit_reason      TEXT
    );

    CREATE TABLE IF NOT EXISTS funding_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_id    TEXT REFERENCES trades(id),
        logged_at   TEXT,
        funding_rate REAL,
        payment_usd  REAL,
        direction   TEXT
    );
    """)
    c.commit()
    c.close()


# ── MARKET SCAN ───────────────────────────────────────────────────────────────

def scan_opportunities():
    """Return list of arb opportunities sorted by expected yield."""
    opps = []
    for sym in SYMBOLS:
        try:
            spot = fetch_ticker_price(sym)
            if not spot or spot <= 0:
                continue
            fund_data = fetch_funding_rate(sym)
            if not fund_data:
                continue
            fr   = float(fund_data.get("lastFundingRate") or 0)
            perp = float(fund_data.get("markPrice") or 0) or fetch_perp_price(sym)
            if not perp or perp <= 0:
                continue

            basis_pct = (perp - spot) / spot * 100
            ann_pct   = fr * 3 * 365 * 100  # 3 payments/day × 365

            # Strategy 1: Carry arb (positive funding)
            if fr >= CARRY_MIN_RATE:
                opps.append({
                    "strategy":      "CARRY_ARB",
                    "symbol":        sym,
                    "direction":     "LONG_SPOT_SHORT_PERP",
                    "funding_rate":  fr,
                    "ann_pct":       ann_pct,
                    "basis_pct":     basis_pct,
                    "spot":          spot,
                    "perp":          perp,
                    "score":         ann_pct,
                })

            # Strategy 2: Negative carry (negative funding)
            if fr <= NEG_CARRY_MIN_RATE:
                opps.append({
                    "strategy":      "NEG_CARRY",
                    "symbol":        sym,
                    "direction":     "LONG_PERP_SHORT_SPOT",
                    "funding_rate":  fr,
                    "ann_pct":       abs(ann_pct),
                    "basis_pct":     basis_pct,
                    "spot":          spot,
                    "perp":          perp,
                    "score":         abs(ann_pct),
                })

            # Strategy 3: Spot premium (perp > spot significantly)
            if basis_pct >= BASIS_MIN_PCT:
                opps.append({
                    "strategy":      "SPOT_PREMIUM",
                    "symbol":        sym,
                    "direction":     "LONG_SPOT_SHORT_PERP",
                    "funding_rate":  fr,
                    "ann_pct":       ann_pct,
                    "basis_pct":     basis_pct,
                    "spot":          spot,
                    "perp":          perp,
                    "score":         basis_pct * 10,  # basis convergence is the edge
                })

            # Strategy 4: Spot discount (perp < spot significantly)
            if basis_pct <= -BASIS_MIN_PCT:
                opps.append({
                    "strategy":      "SPOT_DISCOUNT",
                    "symbol":        sym,
                    "direction":     "LONG_PERP_SHORT_SPOT",
                    "funding_rate":  fr,
                    "ann_pct":       ann_pct,
                    "basis_pct":     basis_pct,
                    "spot":          spot,
                    "perp":          perp,
                    "score":         abs(basis_pct) * 10,
                })

        except Exception as e:
            pass

    opps.sort(key=lambda x: x["score"], reverse=True)
    return opps


# ── OPEN PAPER TRADE ──────────────────────────────────────────────────────────

def open_trade(opp):
    """Insert a new paper trade into the DB. Skip if one already open for (symbol, strategy)."""
    c = _conn()
    existing = c.execute(
        "SELECT id FROM trades WHERE symbol=? AND strategy=? AND status='OPEN'",
        (opp["symbol"], opp["strategy"])
    ).fetchone()
    if existing:
        c.close()
        return None

    now   = datetime.now(timezone.utc).isoformat()
    tid   = f"{opp['strategy'][:4]}_{opp['symbol']}_{datetime.now(timezone.utc).strftime('%m%d_%H%M')}"
    fees  = POSITION_USD * FEE_PER_LEG * 2  # open: 2 legs
    fr    = opp["funding_rate"]
    ann   = opp["ann_pct"]

    c.execute("""
        INSERT INTO trades (
            id, strategy, symbol, direction, opened_at,
            spot_entry, perp_entry, basis_entry_pct,
            funding_at_open, ann_yield_at_open,
            size_usd, fees_total,
            last_funding_at, spot_now, perp_now, basis_now_pct
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        tid, opp["strategy"], opp["symbol"], opp["direction"], now,
        opp["spot"], opp["perp"], opp["basis_pct"],
        fr, ann,
        POSITION_USD, fees,
        now, opp["spot"], opp["perp"], opp["basis_pct"],
    ))
    c.commit()
    c.close()
    return tid


# ── FUNDING APPLICATION ───────────────────────────────────────────────────────

def _next_funding_time(last_funding_at_iso):
    """Returns True if 8+ hours have passed since last funding application."""
    try:
        ts = last_funding_at_iso
        if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
        last = datetime.fromisoformat(ts)
        return (datetime.now(timezone.utc) - last).total_seconds() >= FUNDING_INTERVAL_H * 3600
    except Exception:
        return True


def apply_funding(trade):
    """Check if funding should be applied; if so, calculate and persist it."""
    if not _next_funding_time(trade["last_funding_at"] or trade["opened_at"]):
        return 0.0

    sym = trade["symbol"]
    try:
        fund_data = fetch_funding_rate(sym)
        fr = float(fund_data.get("lastFundingRate") or 0)
    except Exception:
        return 0.0

    # For LONG_SPOT_SHORT_PERP: we're short perp; if fr>0, shorts receive payment
    # For LONG_PERP_SHORT_SPOT: we're long perp; if fr>0, longs pay, so negative for us
    direction   = trade["direction"]
    size_usd    = float(trade["size_usd"])
    if direction == "LONG_SPOT_SHORT_PERP":
        payment = fr * size_usd          # positive fr → we receive
    else:
        payment = -fr * size_usd         # positive fr → we pay

    now = datetime.now(timezone.utc).isoformat()
    c   = _conn()
    c.execute("""
        UPDATE trades SET
            funding_total   = funding_total + ?,
            funding_periods = funding_periods + 1,
            last_funding_at = ?
        WHERE id=?
    """, (payment, now, trade["id"]))
    c.execute("""
        INSERT INTO funding_log (trade_id, logged_at, funding_rate, payment_usd, direction)
        VALUES (?,?,?,?,?)
    """, (trade["id"], now, fr, payment, "RECEIVE" if payment >= 0 else "PAY"))
    c.commit()
    c.close()
    return payment


# ── UPDATE OPEN POSITIONS ─────────────────────────────────────────────────────

def _basis_pnl(trade, spot_now, perp_now):
    """
    P&L from basis movement.
    LONG_SPOT_SHORT_PERP: gain if spot rises vs perp (we're long spot, short perp)
    LONG_PERP_SHORT_SPOT: gain if perp rises vs spot
    """
    se  = float(trade["spot_entry"])
    pe  = float(trade["perp_entry"])
    sz  = float(trade["size_usd"])
    if se == 0 or pe == 0:
        return 0.0
    spot_ret = (spot_now - se) / se
    perp_ret = (perp_now - pe) / pe
    if trade["direction"] == "LONG_SPOT_SHORT_PERP":
        return sz * (spot_ret - perp_ret)
    else:
        return sz * (perp_ret - spot_ret)


def _should_close(trade, spot_now, perp_now, fr):
    """Return exit_reason string if trade should be closed, else None."""
    strategy   = trade["strategy"]
    opened_iso = trade["opened_at"]
    try:
        ts = opened_iso
        if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
        opened_dt = datetime.fromisoformat(ts)
    except Exception:
        opened_dt = datetime.now(timezone.utc)

    hours_open = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 3600
    basis_now  = (perp_now - spot_now) / spot_now * 100 if spot_now > 0 else 0

    if strategy == "CARRY_ARB":
        if fr <= CARRY_EXIT_RATE:
            return "FUNDING_DROPPED"
        if hours_open >= CARRY_MAX_DAYS * 24:
            return "MAX_DURATION"
        # Stop-loss on basis blowout
        basis_entry = float(trade["basis_entry_pct"])
        if (basis_now - basis_entry) > BASIS_STOP_PCT:  # basis moved against long-spot
            return "STOP_LOSS_BASIS"

    elif strategy == "NEG_CARRY":
        if fr >= -CARRY_EXIT_RATE:
            return "FUNDING_NORMALIZED"
        if hours_open >= CARRY_MAX_DAYS * 24:
            return "MAX_DURATION"
        basis_entry = float(trade["basis_entry_pct"])
        if (basis_entry - basis_now) > BASIS_STOP_PCT:
            return "STOP_LOSS_BASIS"

    elif strategy == "SPOT_PREMIUM":
        if basis_now <= BASIS_EXIT_PCT:
            return "BASIS_CONVERGED"
        if hours_open >= BASIS_MAX_HOURS:
            return "MAX_DURATION"
        basis_entry = float(trade["basis_entry_pct"])
        if basis_now > basis_entry + BASIS_STOP_PCT:
            return "STOP_LOSS_BASIS"

    elif strategy == "SPOT_DISCOUNT":
        if basis_now >= -BASIS_EXIT_PCT:
            return "BASIS_CONVERGED"
        if hours_open >= BASIS_MAX_HOURS:
            return "MAX_DURATION"
        basis_entry = float(trade["basis_entry_pct"])
        if basis_now < basis_entry - BASIS_STOP_PCT:
            return "STOP_LOSS_BASIS"

    return None


def close_trade(trade_id, spot_exit, perp_exit, reason):
    c = _conn()
    trade = c.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    if not trade:
        c.close()
        return

    close_fees   = float(trade["size_usd"]) * FEE_PER_LEG * 2
    total_fees   = float(trade["fees_total"]) + close_fees
    basis_pnl    = _basis_pnl(trade, spot_exit, perp_exit)
    funding_pnl  = float(trade["funding_total"])
    realized_pnl = basis_pnl + funding_pnl - total_fees

    basis_now    = (perp_exit - spot_exit) / spot_exit * 100 if spot_exit > 0 else 0
    now          = datetime.now(timezone.utc).isoformat()

    c.execute("""
        UPDATE trades SET
            status        = 'CLOSED',
            closed_at     = ?,
            spot_exit     = ?,
            perp_exit     = ?,
            basis_now_pct = ?,
            fees_total    = ?,
            realized_pnl  = ?,
            exit_reason   = ?
        WHERE id=?
    """, (now, spot_exit, perp_exit, basis_now,
          total_fees, realized_pnl, reason, trade_id))
    c.commit()
    c.close()


def update_positions():
    """Refresh prices, apply funding, check exits for all open trades."""
    c = _conn()
    open_trades = c.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()
    c.close()

    opened = closed = funding_applied = 0
    for trade in open_trades:
        sym = trade["symbol"]
        try:
            spot_now  = fetch_ticker_price(sym)
            fund_data = fetch_funding_rate(sym)
            fr        = float(fund_data.get("lastFundingRate") or 0)
            perp_now  = float(fund_data.get("markPrice") or 0) or fetch_perp_price(sym)
        except Exception:
            continue

        if not spot_now or not perp_now:
            continue

        # Apply funding if interval elapsed
        payment = apply_funding(trade)
        if payment != 0.0:
            funding_applied += 1
            # Reload trade row with updated funding_total
            c2 = _conn()
            trade = c2.execute("SELECT * FROM trades WHERE id=?", (trade["id"],)).fetchone()
            c2.close()

        basis_now  = (perp_now - spot_now) / spot_now * 100
        basis_pnl  = _basis_pnl(trade, spot_now, perp_now)
        funding_pnl = float(trade["funding_total"])
        fees_so_far = float(trade["fees_total"])
        unrealized  = basis_pnl + funding_pnl - fees_so_far

        # Update live prices + unrealized
        c3 = _conn()
        c3.execute("""
            UPDATE trades SET spot_now=?, perp_now=?, basis_now_pct=?, unrealized_pnl=?
            WHERE id=?
        """, (spot_now, perp_now, basis_now, unrealized, trade["id"]))
        c3.commit()
        c3.close()

        # Check exit
        exit_reason = _should_close(trade, spot_now, perp_now, fr)
        if exit_reason:
            close_trade(trade["id"], spot_now, perp_now, exit_reason)
            closed += 1

    return {"funding_applied": funding_applied, "closed": closed}


# ── OPEN NEW TRADES ───────────────────────────────────────────────────────────

def open_new_trades():
    """Scan for opportunities and open paper trades up to MAX_OPEN_TRADES."""
    c    = _conn()
    n_open = c.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()[0]
    c.close()

    if n_open >= MAX_OPEN_TRADES:
        return []

    opps   = scan_opportunities()
    opened = []
    for opp in opps:
        if n_open >= MAX_OPEN_TRADES:
            break
        tid = open_trade(opp)
        if tid:
            opened.append({"id": tid, "strategy": opp["strategy"],
                           "symbol": opp["symbol"], "score": opp.get("score", 0)})
            n_open += 1
    return opened


# ── REPORT ────────────────────────────────────────────────────────────────────

def compute_report(verbose=True):
    c    = _conn()
    all_ = c.execute("SELECT * FROM trades ORDER BY opened_at DESC").fetchall()
    c.close()

    all_  = [dict(r) for r in all_]
    open_ = [r for r in all_ if r["status"] == "OPEN"]
    closed = [r for r in all_ if r["status"] == "CLOSED"]

    wins  = [r for r in closed if (r["realized_pnl"] or 0) > 0]
    loss  = [r for r in closed if (r["realized_pnl"] or 0) <= 0]
    total_pnl    = sum(r["realized_pnl"] or 0 for r in closed)
    total_unreal = sum(r["unrealized_pnl"] or 0 for r in open_)

    # Account balance tracking
    capital_deployed = sum(float(r["size_usd"] or 0) * 2 for r in open_)  # 2 legs
    account_equity   = STARTING_BALANCE + total_pnl + total_unreal
    available_cash   = STARTING_BALANCE + total_pnl - capital_deployed
    return_pct       = (account_equity - STARTING_BALANCE) / STARTING_BALANCE * 100

    win_rate  = len(wins) / len(closed) * 100 if closed else 0.0
    gains     = sum(r["realized_pnl"] for r in wins)
    losses    = abs(sum(r["realized_pnl"] for r in loss))
    pf        = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)

    # By strategy
    strats = {}
    for r in closed:
        s = r["strategy"]
        if s not in strats:
            strats[s] = {"n": 0, "wins": 0, "pnl": 0.0, "funding": 0.0, "fees": 0.0}
        strats[s]["n"]   += 1
        strats[s]["pnl"] += r["realized_pnl"] or 0
        strats[s]["fees"]+= r["fees_total"] or 0
        strats[s]["funding"] += r["funding_total"] or 0
        if (r["realized_pnl"] or 0) > 0:
            strats[s]["wins"] += 1

    # Exit reason breakdown
    exit_counts = {}
    for r in closed:
        reason = r["exit_reason"] or "UNKNOWN"
        exit_counts[reason] = exit_counts.get(reason, 0) + 1

    if verbose:
        W  = 66
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        eq_cls = "▲" if account_equity >= STARTING_BALANCE else "▼"
        print("\n" + "═" * W)
        print(f"  PERP ARB PAPER TRADER  |  {ts}")
        print("═" * W)
        print(f"  Opening Balance:  ${STARTING_BALANCE:>14,.2f}")
        print(f"  Account Equity:   ${account_equity:>14,.2f}  {eq_cls}  ({return_pct:+.4f}%)")
        print(f"  Realized P&L:     ${total_pnl:>+14,.2f}")
        print(f"  Unrealized P&L:   ${total_unreal:>+14,.2f}  ({len(open_)} open positions)")
        print(f"  Capital Deployed: ${capital_deployed:>14,.2f}  ({capital_deployed/STARTING_BALANCE*100:.1f}% of balance)")
        print(f"  Available Cash:   ${available_cash:>14,.2f}")
        print(f"  Trades:           {len(all_):3d} total  ({len(open_)} open, {len(closed)} closed)")

        if closed:
            pf_str = f"{pf:.2f}×" if pf != float("inf") else "∞"
            print(f"\n  CLOSED PERFORMANCE  ({len(closed)} trades)")
            print(f"  Win Rate:      {win_rate:.1f}%  ({len(wins)} wins / {len(closed)})")
            print(f"  Profit Factor: {pf_str}")
            avg_win  = gains / len(wins)  if wins  else 0
            avg_loss = losses / len(loss) if loss  else 0
            print(f"  Avg Win:      ${avg_win:+.2f}   Avg Loss: ${avg_loss:+.2f}")

            print(f"\n  BY STRATEGY:")
            for s, d in sorted(strats.items(), key=lambda x: x[1]["pnl"], reverse=True):
                wr = d["wins"] / d["n"] * 100 if d["n"] else 0
                print(f"  {s:<18s}  {d['n']:2d} trades  WR {wr:.0f}%  "
                      f"P&L ${d['pnl']:+.2f}  "
                      f"(fund ${d['funding']:+.2f}  fees ${d['fees']:.2f})")

            print(f"\n  EXIT REASONS:")
            for reason, cnt in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason:<24s}  {cnt}")

        if open_:
            print(f"\n  OPEN POSITIONS  ({len(open_)})")
            print(f"  {'ID':<26} {'Strategy':<14} {'Dir':<24} {'Unreal':>8} {'Fund':>8} {'Age':>5}")
            print(f"  {'─'*26} {'─'*14} {'─'*24} {'─'*8} {'─'*8} {'─'*5}")
            for r in sorted(open_, key=lambda x: x["unrealized_pnl"] or 0, reverse=True)[:15]:
                try:
                    ts_str = r["opened_at"]
                    if ts_str.endswith("Z"): ts_str = ts_str[:-1] + "+00:00"
                    age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(ts_str)).total_seconds() / 3600
                except Exception:
                    age_h = 0
                print(f"  {r['id']:<26} {r['strategy']:<14} {r['direction']:<24} "
                      f"${r['unrealized_pnl']:+6.2f}  ${r['funding_total']:+6.2f}  {age_h:.0f}h")

        if closed:
            print(f"\n  LAST 10 CLOSED TRADES:")
            print(f"  {'Symbol':<12} {'Strategy':<14} {'P&L':>8} {'Fund':>7} {'Fees':>6} {'Exit Reason'}")
            print(f"  {'─'*12} {'─'*14} {'─'*8} {'─'*7} {'─'*6} {'─'*20}")
            for r in sorted(closed, key=lambda x: x["closed_at"] or "", reverse=True)[:10]:
                print(f"  {r['symbol']:<12} {r['strategy']:<14} "
                      f"${r['realized_pnl']:+6.2f}  "
                      f"${r['funding_total']:+5.2f}  "
                      f"${r['fees_total']:.2f}  "
                      f"{r['exit_reason'] or 'N/A'}")

        print("═" * W + "\n")

    return {
        # Account summary
        "starting_balance":  STARTING_BALANCE,
        "account_equity":    round(account_equity, 2),
        "return_pct":        round(return_pct, 4),
        "capital_deployed":  round(capital_deployed, 2),
        "available_cash":    round(available_cash, 2),
        # Trade stats
        "total_trades":   len(all_),
        "open_trades":    len(open_),
        "closed_trades":  len(closed),
        "win_rate":       round(win_rate, 1),
        "profit_factor":  round(pf, 2) if pf != float("inf") else 999.0,
        "realized_pnl":   round(total_pnl, 2),
        "unrealized_pnl": round(total_unreal, 2),
        "wins":           len(wins),
        "losses":         len(loss),
        "by_strategy":    {s: {"n": d["n"], "win_rate": round(d["wins"]/d["n"]*100,1) if d["n"] else 0,
                               "pnl": round(d["pnl"],2)} for s,d in strats.items()},
        "exit_reasons":   exit_counts,
        "open_positions": [
            {
                "id":          r["id"],
                "strategy":    r["strategy"],
                "symbol":      r["symbol"],
                "direction":   r["direction"],
                "opened_at":   r["opened_at"],
                "funding_at_open": r["funding_at_open"],
                "ann_yield":   r["ann_yield_at_open"],
                "spot_entry":  r["spot_entry"],
                "perp_entry":  r["perp_entry"],
                "basis_entry": r["basis_entry_pct"],
                "spot_now":    r["spot_now"],
                "perp_now":    r["perp_now"],
                "basis_now":   r["basis_now_pct"],
                "funding_earned": round(float(r["funding_total"] or 0), 4),
                "unrealized_pnl": round(float(r["unrealized_pnl"] or 0), 4),
                "funding_periods": r["funding_periods"],
            }
            for r in sorted(open_, key=lambda x: x["unrealized_pnl"] or 0, reverse=True)
        ],
        "recent_closed": [
            {
                "id":          r["id"],
                "strategy":    r["strategy"],
                "symbol":      r["symbol"],
                "direction":   r["direction"],
                "realized_pnl": round(float(r["realized_pnl"] or 0), 4),
                "funding_earned": round(float(r["funding_total"] or 0), 4),
                "fees":        round(float(r["fees_total"] or 0), 4),
                "exit_reason": r["exit_reason"],
                "opened_at":   r["opened_at"],
                "closed_at":   r["closed_at"],
            }
            for r in sorted(closed, key=lambda x: x["closed_at"] or "", reverse=True)[:20]
        ],
    }


# ── SAVE JSON ─────────────────────────────────────────────────────────────────

def save_json(report):
    path = os.path.join(DB_DIR, "perp_arb.json")
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    except Exception as e:
        print(f"  JSON save error: {e}")


# ── FULL CYCLE ────────────────────────────────────────────────────────────────

def run_cycle(verbose=True):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n[{ts}] Perp Arb Cycle starting...")

    # 1. Update existing positions (funding + exit checks)
    upd = update_positions()
    print(f"  Updated positions — funding applied: {upd['funding_applied']}, "
          f"closed: {upd['closed']}")

    # 2. Scan and open new trades
    opened = open_new_trades()
    if opened:
        print(f"  Opened {len(opened)} new trade(s):")
        for t in opened:
            print(f"    {t['id']}  {t['strategy']}")
    else:
        c = _conn()
        n_open = c.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()[0]
        c.close()
        print(f"  No new trades opened  ({n_open}/{MAX_OPEN_TRADES} slots used)")

    # 3. Report + save JSON
    report = compute_report(verbose=verbose)
    save_json(report)
    return report


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once",   action="store_true", help="One cycle then exit")
    parser.add_argument("--watch",  type=int,            help="Repeat every N seconds")
    parser.add_argument("--report", action="store_true", help="Report only")
    parser.add_argument("--reset",  action="store_true", help="Delete all paper trades")
    parser.add_argument("--scan",   action="store_true", help="Show opportunities only, don't trade")
    args = parser.parse_args()

    init_db()

    if args.reset:
        c = _conn()
        c.execute("DELETE FROM trades")
        c.execute("DELETE FROM funding_log")
        c.commit()
        c.close()
        print("All paper trades cleared.")
        return

    if args.report:
        compute_report(verbose=True)
        return

    if args.scan:
        opps = scan_opportunities()
        print(f"\n{len(opps)} opportunities found:\n")
        print(f"  {'Symbol':<12} {'Strategy':<14} {'Direction':<24} {'Fund%':>8} {'Ann%':>8} {'Basis%':>8}")
        print("  " + "─" * 76)
        for o in opps:
            print(f"  {o['symbol']:<12} {o['strategy']:<14} {o['direction']:<24} "
                  f"{o['funding_rate']*100:>+7.4f}  {o['ann_pct']:>+7.1f}%  {o['basis_pct']:>+7.3f}%")
        return

    if args.watch:
        print(f"Watch mode: cycling every {args.watch}s. Ctrl+C to stop.")
        while True:
            try:
                run_cycle()
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"  Cycle error: {e}")
            time.sleep(args.watch)
        return

    # Default / --once
    run_cycle()


if __name__ == "__main__":
    main()
