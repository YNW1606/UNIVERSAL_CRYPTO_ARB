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
    fetch_funding_history, fetch_all_funding_rates,
)

# ── SYMBOLS ──────────────────────────────────────────────────────────────────
# Large-cap anchors
SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT",
    # Mid-cap momentum
    "DOGEUSDT","ADAUSDT","AVAXUSDT","DOTUSDT","NEARUSDT",
    # DeFi / ecosystem
    "LINKUSDT","UNIUSDT","LTCUSDT","TRXUSDT","ATOMUSDT",
    # High-beta / high-funding altcoins
    "SUIUSDT","APTUSDT","ARBUSDT","OPUSDT","INJUSDT",
    "SEIUSDT","TIAUSDT","WLDUSDT","JUPUSDT","PYTHUSDT",
    "ORDIUSDT","STXUSDT","RUNEUSDT","RENDERUSDT","FETUSDT",
    # Note: MATICUSDT is delisted on Bybit perps (renamed POL) — excluded
]

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
CARRY_MIN_RATE      = 0.000137  # 0.0137%/8h = ~15%/yr minimum — quality bar
CARRY_EXIT_RATE     = 0.00005
NEG_CARRY_MIN_RATE  = -0.000137
BASIS_MIN_PCT       = 0.20
BASIS_EXIT_PCT      = 0.04
BASIS_STOP_PCT      = 0.60
CARRY_MAX_DAYS      = 30
BASIS_MAX_HOURS     = 72
FEE_PER_LEG         = 0.0004
ROUND_TRIP_FEES     = FEE_PER_LEG * 4
STARTING_BALANCE    = 1_000_000.0
MAX_OPEN_TRADES     = 15        # 15 best pairs — fills all $1M efficiently
MIN_POSITION_USD    = 30_000
MAX_POSITION_USD    = 250_000
POSITION_USD        = STARTING_BALANCE / MAX_OPEN_TRADES  # $66.7k flat fallback
FUNDING_INTERVAL_H  = 8
MIN_VOLUME_24H      = 5_000_000
MAX_ANN_RATE        = 500.0

# ── PER-POSITION SMART EXIT ───────────────────────────────────────────────────
PROFIT_TARGET_PCT   = 0.030   # +3.0% per position
STOP_LOSS_PCT       = 0.012   # -1.2% per position (tighter)
TRAIL_ACTIVATE_PCT  = 0.015   # trailing activates at +1.5%
TRAIL_DROP_PCT      = 0.007   # trails 0.7% below peak
FUNDING_ROI_MULT    = 8.0     # close when funding > 8× open fees
FUNDING_DECAY_PCT   = 0.50    # close if rate < 50% of entry rate

# ── PORTFOLIO-LEVEL PROTECTION (the key layer that was missing) ────────────────
PORT_TP_USD         = 4500.0  # lock ALL profit when portfolio hits +$4,500
PORT_TRAIL_DROP_PCT = 0.30    # once peak > $1k, never give back more than 30%
                               # e.g. peak $5k → close all if drops to $3,500
PORT_SL_USD         = -2500.0 # close ALL if portfolio loses $2,500 total
PORT_REOPEN_MINS    = 20      # wait 20 min after booking before reopening

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
        peak_pnl         REAL DEFAULT 0,   -- highest unrealized P&L ever seen (for trailing stop)

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

    CREATE TABLE IF NOT EXISTS portfolio_meta (
        id               INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
        peak_total_pnl   REAL    DEFAULT 0,   -- highest portfolio unrealized P&L ever seen
        total_booked_pnl REAL    DEFAULT 0,   -- cumulative locked profit from all book events
        book_count       INTEGER DEFAULT 0,   -- how many times we've booked profit
        last_booked_at   TEXT,                -- ISO timestamp of last book event
        last_book_reason TEXT                 -- e.g. PORT_TP / PORT_TRAIL / PORT_SL
    );
    INSERT OR IGNORE INTO portfolio_meta (id) VALUES (1);
    """)
    # Migrate: add peak_pnl to trades if missing
    cols = [r[1] for r in c.execute("PRAGMA table_info(trades)").fetchall()]
    if cols and "peak_pnl" not in cols:
        c.execute("ALTER TABLE trades ADD COLUMN peak_pnl REAL DEFAULT 0")
    c.commit()
    c.close()


# ── MARKET SCAN ───────────────────────────────────────────────────────────────

def scan_opportunities():
    """
    Fetch ALL Bybit linear perps at once (single API call), score each one,
    return opportunities sorted by yield desc.
    Falls back to per-symbol fetch for the SYMBOLS watchlist if bulk fails.
    """
    all_tickers = fetch_all_funding_rates(min_volume_usd=MIN_VOLUME_24H)

    # Fallback: scan fixed symbol list one-by-one
    if not all_tickers:
        all_tickers = []
        for sym in SYMBOLS:
            try:
                fd = fetch_funding_rate(sym)
                if fd:
                    all_tickers.append({
                        "symbol":      sym,
                        "fundingRate": float(fd.get("lastFundingRate") or 0),
                        "markPrice":   float(fd.get("markPrice") or 0),
                        "ann_pct":     float(fd.get("lastFundingRate") or 0) * 3 * 365 * 100,
                        "volume24h":   MIN_VOLUME_24H,
                    })
            except Exception:
                pass

    opps = []
    for t in all_tickers:
        sym  = t["symbol"]
        fr   = float(t.get("fundingRate") or t.get("lastFundingRate") or 0)
        perp = float(t.get("markPrice") or 0)
        vol  = float(t.get("volume24h") or 0)
        ann_pct = fr * 3 * 365 * 100

        if not perp or perp <= 0:
            continue

        # Quality filter: skip tokens with extreme/unstable rates or low liquidity
        if abs(ann_pct) > MAX_ANN_RATE:
            continue
        if vol < MIN_VOLUME_24H:
            continue

        # Fetch spot price
        try:
            spot = fetch_ticker_price(sym)
        except Exception:
            spot = perp
        if not spot or spot <= 0:
            spot = perp

        basis_pct = (perp - spot) / spot * 100

        if fr >= CARRY_MIN_RATE:
            opps.append({
                "strategy":     "CARRY_ARB",
                "symbol":       sym,
                "direction":    "LONG_SPOT_SHORT_PERP",
                "funding_rate": fr,
                "ann_pct":      ann_pct,
                "basis_pct":    basis_pct,
                "spot":         spot,
                "perp":         perp,
                "volume24h":    vol,
                "score":        ann_pct,
            })
        elif fr <= NEG_CARRY_MIN_RATE:
            opps.append({
                "strategy":     "NEG_CARRY",
                "symbol":       sym,
                "direction":    "LONG_PERP_SHORT_SPOT",
                "funding_rate": fr,
                "ann_pct":      abs(ann_pct),
                "basis_pct":    basis_pct,
                "spot":         spot,
                "perp":         perp,
                "volume24h":    vol,
                "score":        abs(ann_pct),
            })

        if basis_pct >= BASIS_MIN_PCT:
            opps.append({
                "strategy":     "SPOT_PREMIUM",
                "symbol":       sym,
                "direction":    "LONG_SPOT_SHORT_PERP",
                "funding_rate": fr,
                "ann_pct":      ann_pct,
                "basis_pct":    basis_pct,
                "spot":         spot,
                "perp":         perp,
                "volume24h":    vol,
                "score":        basis_pct * 10,
            })
        elif basis_pct <= -BASIS_MIN_PCT:
            opps.append({
                "strategy":     "SPOT_DISCOUNT",
                "symbol":       sym,
                "direction":    "LONG_PERP_SHORT_SPOT",
                "funding_rate": fr,
                "ann_pct":      ann_pct,
                "basis_pct":    basis_pct,
                "spot":         spot,
                "perp":         perp,
                "volume24h":    vol,
                "score":        abs(basis_pct) * 10,
            })

    opps.sort(key=lambda x: x["score"], reverse=True)
    return opps


def _yield_weighted_size(opps, n_slots, budget):
    """
    Allocate budget across n_slots using sqrt(yield) weighting.
    Hard cap: total allocated never exceeds budget.
    """
    if not opps or budget <= 0:
        return {}
    picks = opps[:n_slots]
    yields  = [max(abs(o["ann_pct"]), 1.0) for o in picks]
    weights = [y ** 0.5 for y in yields]
    total_w = sum(weights)

    # First pass: proportional with per-position min/max caps
    raw = [budget * w / total_w for w in weights]
    capped = [max(MIN_POSITION_USD, min(MAX_POSITION_USD, r)) for r in raw]

    # Second pass: if capped total exceeds budget, scale all down proportionally
    total = sum(capped)
    if total > budget:
        scale = budget / total
        capped = [s * scale for s in capped]

    return {(o["symbol"], o["strategy"]): round(s, 2)
            for o, s in zip(picks, capped)}


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

    now      = datetime.now(timezone.utc).isoformat()
    tid      = f"{opp['strategy'][:4]}_{opp['symbol']}_{datetime.now(timezone.utc).strftime('%m%d_%H%M')}"
    size_usd = float(opp.get("_size_usd") or POSITION_USD)
    fees     = size_usd * FEE_PER_LEG * 2  # open: 2 legs
    fr       = opp["funding_rate"]
    ann      = opp["ann_pct"]

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
        size_usd, fees,
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
    """
    Smart exit logic — checks in priority order:
      1. Hard stop loss      (-1.5% of position)
      2. Profit target       (+3.0% of position)
      3. Trailing stop       (activates at +1.5%, drops 0.8% below peak)
      4. Funding ROI met     (funding > 8× open fees → redeploy to fresh opp)
      5. Funding dropped     (rate below exit threshold)
      6. Funding decay       (rate < 50% of entry rate)
      7. Basis stop loss     (basis blowout)
      8. Max duration        (30d carry / 72h basis)
    """
    strategy    = trade["strategy"]
    size_usd    = float(trade["size_usd"] or POSITION_USD)
    unrealized  = float(trade["unrealized_pnl"] or 0)
    peak_pnl    = float(trade["peak_pnl"] or 0)
    funding_tot = float(trade["funding_total"] or 0)
    fees_open   = float(trade["fees_total"] or 0)   # fees charged at open (half of RT)
    fr_entry    = float(trade["funding_at_open"] or 0)

    try:
        ts = trade["opened_at"]
        if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
        hours_open = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
    except Exception:
        hours_open = 0

    basis_now   = (perp_now - spot_now) / spot_now * 100 if spot_now > 0 else 0
    basis_entry = float(trade["basis_entry_pct"] or 0)

    # ── 1. HARD STOP LOSS: total P&L < -1.5% of position ────────────────────
    stop_usd = -size_usd * STOP_LOSS_PCT
    if unrealized < stop_usd:
        return f"STOP_LOSS_PNL"            # e.g. -$1,500 on $100k position

    # ── 2. PROFIT TARGET: total P&L > +3.0% of position ─────────────────────
    target_usd = size_usd * PROFIT_TARGET_PCT
    if unrealized > target_usd:
        return "PROFIT_TARGET"             # e.g. +$3,000 on $100k position

    # ── 3. TRAILING STOP ─────────────────────────────────────────────────────
    # Activates once up 1.5%, then closes if drops 0.8% below the peak
    trail_activate = size_usd * TRAIL_ACTIVATE_PCT
    trail_drop     = size_usd * TRAIL_DROP_PCT
    if peak_pnl >= trail_activate and unrealized < (peak_pnl - trail_drop):
        return "TRAILING_STOP"             # peak was $2k, dropped $800 → exit

    # ── 4. FUNDING ROI MET: funding income > 8× open fees ───────────────────
    if fees_open > 0 and funding_tot >= fees_open * FUNDING_ROI_MULT:
        return "FUNDING_ROI_MET"           # paid for itself 8× — redeploy

    # ── 5–8. Strategy-specific exits ─────────────────────────────────────────
    if strategy in ("CARRY_ARB", "NEG_CARRY"):
        # Funding dropped below exit threshold
        if strategy == "CARRY_ARB" and fr <= CARRY_EXIT_RATE:
            return "FUNDING_DROPPED"
        if strategy == "NEG_CARRY" and fr >= -CARRY_EXIT_RATE:
            return "FUNDING_NORMALIZED"

        # Funding decay: rate fell >50% from entry (edge is fading)
        if fr_entry != 0 and abs(fr) < abs(fr_entry) * FUNDING_DECAY_PCT:
            return "FUNDING_DECAY"

        # Basis blowout stop (secondary safety net)
        if strategy == "CARRY_ARB" and (basis_now - basis_entry) > BASIS_STOP_PCT:
            return "STOP_LOSS_BASIS"
        if strategy == "NEG_CARRY" and (basis_entry - basis_now) > BASIS_STOP_PCT:
            return "STOP_LOSS_BASIS"

        if hours_open >= CARRY_MAX_DAYS * 24:
            return "MAX_DURATION"

    elif strategy == "SPOT_PREMIUM":
        if basis_now <= BASIS_EXIT_PCT:
            return "BASIS_CONVERGED"
        if basis_now > basis_entry + BASIS_STOP_PCT:
            return "STOP_LOSS_BASIS"
        if hours_open >= BASIS_MAX_HOURS:
            return "MAX_DURATION"

    elif strategy == "SPOT_DISCOUNT":
        if basis_now >= -BASIS_EXIT_PCT:
            return "BASIS_CONVERGED"
        if basis_now < basis_entry - BASIS_STOP_PCT:
            return "STOP_LOSS_BASIS"
        if hours_open >= BASIS_MAX_HOURS:
            return "MAX_DURATION"

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

        # Track peak P&L for trailing stop
        new_peak = max(float(trade["peak_pnl"] or 0), unrealized)

        # Update live prices + unrealized + peak
        c3 = _conn()
        c3.execute("""
            UPDATE trades SET spot_now=?, perp_now=?, basis_now_pct=?,
                              unrealized_pnl=?, peak_pnl=?
            WHERE id=?
        """, (spot_now, perp_now, basis_now, unrealized, new_peak, trade["id"]))
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
    """Scan ALL Bybit perps, pick best yield-weighted opportunities, open up to MAX_OPEN_TRADES."""
    c = _conn()
    rows   = c.execute("SELECT size_usd FROM trades WHERE status='OPEN'").fetchall()
    c.close()
    n_open = len(rows)
    already_deployed = sum(float(r[0] or 0) for r in rows)

    if n_open >= MAX_OPEN_TRADES:
        return []

    slots_free       = MAX_OPEN_TRADES - n_open
    available_budget = max(0.0, STARTING_BALANCE - already_deployed)

    if available_budget < MIN_POSITION_USD:
        return []  # not enough cash left to open even a minimum slot

    opps = scan_opportunities()

    # Size positions only against REMAINING budget and REMAINING slot count
    sizes = _yield_weighted_size(opps, slots_free, available_budget)

    opened = []
    for opp in opps:
        if len(opened) >= slots_free:
            break
        # Look up yield-weighted size; fall back to flat
        size = sizes.get((opp["symbol"], opp["strategy"]), POSITION_USD)
        opp["_size_usd"] = size
        tid = open_trade(opp)
        if tid:
            opened.append({
                "id":       tid,
                "strategy": opp["strategy"],
                "symbol":   opp["symbol"],
                "score":    opp.get("score", 0),
                "size_usd": size,
                "ann_pct":  opp.get("ann_pct", 0),
            })
    return opened


# ── REPORT ────────────────────────────────────────────────────────────────────

def compute_report(verbose=True):
    c    = _conn()
    all_ = c.execute("SELECT * FROM trades ORDER BY opened_at DESC").fetchall()
    c.close()

    all_  = [dict(r) for r in all_]
    open_ = [r for r in all_ if r["status"] == "OPEN"]
    closed = [r for r in all_ if r["status"] == "CLOSED"]

    # Portfolio meta (for display)
    meta           = _get_meta()
    total_booked   = float(meta.get("total_booked_pnl") or 0)
    port_peak      = float(meta.get("peak_total_pnl") or 0)
    book_count     = int(meta.get("book_count") or 0)
    last_book_rsn  = meta.get("last_book_reason") or "-"

    wins  = [r for r in closed if (r["realized_pnl"] or 0) > 0]
    loss  = [r for r in closed if (r["realized_pnl"] or 0) <= 0]
    total_pnl    = sum(r["realized_pnl"] or 0 for r in closed)
    total_unreal = sum(r["unrealized_pnl"] or 0 for r in open_)

    capital_deployed = sum(float(r["size_usd"] or 0) for r in open_)
    # True equity includes ALL realized (closed trades + booked profit events)
    account_equity   = STARTING_BALANCE + total_booked + total_pnl + total_unreal
    available_cash   = STARTING_BALANCE + total_booked + total_pnl - capital_deployed
    return_pct       = (account_equity - STARTING_BALANCE) / STARTING_BALANCE * 100

    # Portfolio trailing stop floor (live display)
    trail_floor = port_peak * (1 - PORT_TRAIL_DROP_PCT) if port_peak >= 1000 else None

    # Projected daily funding income from open positions
    proj_daily_funding = 0.0
    for r in open_:
        fr = float(r["funding_at_open"] or 0)
        sz = float(r["size_usd"] or 0)
        proj_daily_funding += abs(fr) * 3 * sz  # 3 payments/day × size

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
        print(f"  Booked Profit:    ${total_booked:>+14,.2f}  ({book_count} book events)")
        print(f"  Realized P&L:     ${total_pnl:>+14,.2f}")
        print(f"  Unrealized P&L:   ${total_unreal:>+14,.2f}  ({len(open_)} open positions)")
        print(f"  Port Peak P&L:    ${port_peak:>+14,.2f}  (trail floor: ${trail_floor:,.2f})" if trail_floor else f"  Port Peak P&L:    ${port_peak:>+14,.2f}  (trail activates at $1,000)")
        print(f"  Capital Deployed: ${capital_deployed:>14,.2f}  ({capital_deployed/STARTING_BALANCE*100:.1f}% of balance)")
        print(f"  Available Cash:   ${available_cash:>14,.2f}")
        print(f"  Trades:           {len(all_):3d} total  ({len(open_)} open, {len(closed)} closed)")
        if proj_daily_funding > 0:
            print(f"\n  PROJECTED INCOME (from open carry positions):")
            print(f"  Per 8h funding:  ${proj_daily_funding/3:>10,.2f}")
            print(f"  Daily:           ${proj_daily_funding:>10,.2f}")
            print(f"  Monthly (30d):   ${proj_daily_funding*30:>10,.2f}")
            print(f"  Annual:          ${proj_daily_funding*365:>10,.2f}  ({proj_daily_funding*365/STARTING_BALANCE*100:.2f}%/yr)")

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
        "starting_balance":     STARTING_BALANCE,
        "account_equity":       round(account_equity, 2),
        "return_pct":           round(return_pct, 4),
        "capital_deployed":     round(capital_deployed, 2),
        "available_cash":       round(available_cash, 2),
        "proj_daily_funding":   round(proj_daily_funding, 2),
        "proj_monthly_funding": round(proj_daily_funding * 30, 2),
        "proj_annual_funding":  round(proj_daily_funding * 365, 2),
        "proj_annual_pct":      round(proj_daily_funding * 365 / STARTING_BALANCE * 100, 2),
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
        # Portfolio protection
        "total_booked_pnl":  round(total_booked, 2),
        "book_count":        book_count,
        "port_peak":         round(port_peak, 2),
        "trail_floor":       round(trail_floor, 2) if trail_floor is not None else None,
        "last_book_reason":  last_book_rsn,
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
                "funding_earned":  round(float(r["funding_total"] or 0), 4),
                "unrealized_pnl":  round(float(r["unrealized_pnl"] or 0), 4),
                "peak_pnl":        round(float(r["peak_pnl"] or 0), 4),
                "funding_periods": r["funding_periods"],
                "fees_total":      round(float(r["fees_total"] or 0), 4),
                "size_usd":        round(float(r["size_usd"] or 0), 2),
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


# ── PORTFOLIO META ────────────────────────────────────────────────────────────

def _get_meta():
    c = _conn()
    row = c.execute("SELECT * FROM portfolio_meta WHERE id=1").fetchone()
    c.close()
    return dict(row) if row else {}

def _update_meta(**kwargs):
    c = _conn()
    sets = ", ".join(f"{k}=?" for k in kwargs)
    c.execute(f"UPDATE portfolio_meta SET {sets} WHERE id=1", list(kwargs.values()))
    c.commit()
    c.close()

def _book_all_positions(reason):
    """Close every open position and lock the profit. Returns total realized from this book."""
    c = _conn()
    open_trades = c.execute("SELECT * FROM trades WHERE status='OPEN'").fetchall()
    c.close()
    total_booked = 0.0
    for t in open_trades:
        sym = t["symbol"]
        try:
            spot_now = fetch_ticker_price(sym)
            fd       = fetch_funding_rate(sym)
            perp_now = float(fd.get("markPrice") or 0) or fetch_perp_price(sym)
        except Exception:
            spot_now = float(t["spot_now"] or t["spot_entry"] or 0)
            perp_now = float(t["perp_now"] or t["perp_entry"] or 0)
        if spot_now and perp_now:
            close_trade(t["id"], spot_now, perp_now, reason)
            total_booked += float(t["unrealized_pnl"] or 0)
    return total_booked

def _can_reopen():
    """Return True if enough time has passed since the last book event."""
    meta = _get_meta()
    last = meta.get("last_booked_at")
    if not last:
        return True
    try:
        if last.endswith("Z"): last = last[:-1] + "+00:00"
        mins_ago = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
        return mins_ago >= PORT_REOPEN_MINS
    except Exception:
        return True

def check_portfolio_protection():
    """
    Portfolio-level profit lock + stop loss. Runs every cycle.
    Priority:
      1. Portfolio stop loss  (total < -$2,500)
      2. Portfolio TP         (total > +$4,500)
      3. Portfolio trailing   (dropped >30% from peak)
    Returns reason if positions were closed, else None.
    """
    c = _conn()
    open_ = c.execute("SELECT unrealized_pnl FROM trades WHERE status='OPEN'").fetchall()
    c.close()
    if not open_:
        return None

    total_unreal = sum(float(r["unrealized_pnl"] or 0) for r in open_)
    meta         = _get_meta()
    peak         = float(meta.get("peak_total_pnl") or 0)
    new_peak     = max(peak, total_unreal)

    # Update peak in DB
    if new_peak > peak:
        _update_meta(peak_total_pnl=new_peak)

    # 1. Portfolio hard stop loss
    if total_unreal <= PORT_SL_USD:
        booked = _book_all_positions("PORT_STOP_LOSS")
        total_booked = float(meta.get("total_booked_pnl") or 0) + booked
        _update_meta(
            total_booked_pnl=total_booked,
            book_count=int(meta.get("book_count") or 0) + 1,
            last_booked_at=datetime.now(timezone.utc).isoformat(),
            last_book_reason="PORT_STOP_LOSS",
            peak_total_pnl=0.0,
        )
        print(f"  ⛔ PORTFOLIO STOP LOSS: total P&L ${total_unreal:+,.2f} → closing all")
        return "PORT_STOP_LOSS"

    # 2. Portfolio take profit
    if total_unreal >= PORT_TP_USD:
        booked = _book_all_positions("PORT_TAKE_PROFIT")
        total_booked = float(meta.get("total_booked_pnl") or 0) + booked
        _update_meta(
            total_booked_pnl=total_booked,
            book_count=int(meta.get("book_count") or 0) + 1,
            last_booked_at=datetime.now(timezone.utc).isoformat(),
            last_book_reason="PORT_TAKE_PROFIT",
            peak_total_pnl=0.0,
        )
        print(f"  ✅ PORTFOLIO TP HIT: locked ${booked:+,.2f} (total booked: ${total_booked:+,.2f})")
        return "PORT_TAKE_PROFIT"

    # 3. Portfolio trailing stop (activates once peak > $1,000)
    if new_peak >= 1000:
        trail_floor = new_peak * (1 - PORT_TRAIL_DROP_PCT)
        if total_unreal <= trail_floor:
            booked = _book_all_positions("PORT_TRAILING_STOP")
            total_booked = float(meta.get("total_booked_pnl") or 0) + booked
            _update_meta(
                total_booked_pnl=total_booked,
                book_count=int(meta.get("book_count") or 0) + 1,
                last_booked_at=datetime.now(timezone.utc).isoformat(),
                last_book_reason="PORT_TRAILING_STOP",
                peak_total_pnl=0.0,
            )
            print(f"  🔒 PORTFOLIO TRAIL STOP: peak was ${new_peak:+,.2f}, dropped to ${total_unreal:+,.2f} → locked ${booked:+,.2f}")
            return "PORT_TRAILING_STOP"

    return None


# ── FULL CYCLE ────────────────────────────────────────────────────────────────

def run_cycle(verbose=True):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print(f"\n[{ts}] Perp Arb Cycle starting...")

    # 1. Update existing positions (funding + per-position exits)
    upd = update_positions()
    print(f"  Updated positions — funding applied: {upd['funding_applied']}, "
          f"closed: {upd['closed']}")

    # 2. Portfolio-level protection (TP / trailing / stop-loss)
    port_action = check_portfolio_protection()
    if port_action:
        print(f"  Portfolio action: {port_action}")

    # 3. Open new trades — but only if no cooldown after a book event
    if _can_reopen():
        opened = open_new_trades()
        if opened:
            print(f"  Opened {len(opened)} new trade(s):")
            for t in opened:
                print(f"    {t['id']}  {t['strategy']}  ann={t.get('ann_pct',0):.1f}%  ${t.get('size_usd',0):,.0f}")
        else:
            c = _conn()
            n_open = c.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()[0]
            c.close()
            print(f"  No new trades opened  ({n_open}/{MAX_OPEN_TRADES} slots used)")
    else:
        meta = _get_meta()
        last = meta.get("last_booked_at","")
        try:
            if last.endswith("Z"): last = last[:-1] + "+00:00"
            mins = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() / 60
        except Exception:
            mins = 0
        print(f"  Cooldown after {meta.get('last_book_reason','book')} — {PORT_REOPEN_MINS-int(mins):.0f} min until reopen")

    # 4. Report + save JSON
    report = compute_report(verbose=verbose)
    save_json(report)
    return report


# ── MAIN ─────────────────────────────────────────────────────────────────────

LOCK_PATH = os.path.join(DB_DIR, "perp_arb.lock")


def _acquire_lock():
    """Return True if lock acquired, False if a fresh lock exists (< 10 min old)."""
    if os.path.exists(LOCK_PATH):
        age = time.time() - os.path.getmtime(LOCK_PATH)
        if age < 600:   # 10-minute stale timeout
            return False
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))
    return True


def _release_lock():
    try:
        os.remove(LOCK_PATH)
    except Exception:
        pass


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
        c.execute("UPDATE portfolio_meta SET peak_total_pnl=0, total_booked_pnl=0, book_count=0, last_booked_at=NULL, last_book_reason=NULL WHERE id=1")
        c.commit()
        c.close()
        # Also clear the JSON snapshot
        json_path = os.path.join(DB_DIR, "perp_arb.json")
        if os.path.exists(json_path):
            os.remove(json_path)
        _release_lock()
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
                if _acquire_lock():
                    run_cycle()
                    _release_lock()
                else:
                    print("  Another instance running — skipping cycle")
            except KeyboardInterrupt:
                _release_lock()
                print("\nStopped.")
                break
            except Exception as e:
                _release_lock()
                print(f"  Cycle error: {e}")
            time.sleep(args.watch)
        return

    # Default / --once — guard against concurrent GitHub Actions runs
    if not _acquire_lock():
        print("Another perp_arb instance is running — exiting to prevent duplicate positions.")
        sys.exit(0)
    try:
        run_cycle()
    finally:
        _release_lock()


if __name__ == "__main__":
    main()
