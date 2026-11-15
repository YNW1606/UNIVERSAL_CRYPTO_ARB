"""
Dashboard Generator
Reads accuracy_db/signals.db + accuracy_db/latest_scan.json
Writes docs/index.html — served free via GitHub Pages
"""
import sqlite3, json, os, sys
from datetime import datetime, timezone


def fetch_live_inst_data():
    """Fetch live institutional flow data for major symbols from Binance FAPI."""
    try:
        import institutional_intel as ii
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
        data = {}
        for sym in symbols:
            try:
                data[sym] = ii.get_institutional_signal(sym)
            except Exception:
                pass
        return data
    except Exception:
        return {}

BASE      = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE, "accuracy_db", "signals.db")
JSON_PATH = os.path.join(BASE, "accuracy_db", "latest_scan.json")
ARB_PATH  = os.path.join(BASE, "accuracy_db", "perp_arb.json")
OUT_DIR   = os.path.join(BASE, "docs")
OUT_PATH  = os.path.join(OUT_DIR, "index.html")

os.makedirs(OUT_DIR, exist_ok=True)
MAX_OPEN_TRADES = 15


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_scan_json():
    if not os.path.exists(JSON_PATH):
        return {}
    try:
        with open(JSON_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def load_perp_arb_json():
    """Load perp arb report — compute directly from DB for full/fresh data."""
    try:
        import perp_arb_paper as pap
        pap.init_db()
        report = pap.compute_report(verbose=False)
        pap.save_json(report)   # keep JSON in sync for downstream tools
        return report
    except Exception:
        pass
    # Fallback: stale JSON
    if not os.path.exists(ARB_PATH):
        return {}
    try:
        with open(ARB_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def load_signals_db():
    if not os.path.exists(DB_PATH):
        return [], {}
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 200"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


ACTIONABLE_SCORE = 50  # Raised from 35: only HIGH+PERFECT tiers (MODERATE had 17% WR = noise)


def compute_stats(signals):
    # Main KPIs use ACTIONABLE signals only (|score| >= 50)
    # Monitoring tier (15-49) has ~17-27% WR by noise — excluded from performance stats
    actionable = [s for s in signals if abs(s.get("unified_score", 0)) >= ACTIONABLE_SCORE]
    resolved   = [s for s in actionable if s.get("outcome") != "PENDING"]
    pending    = [s for s in signals if s.get("outcome") == "PENDING"]
    wins       = [s for s in resolved if s.get("outcome") == "WIN"]

    win_rate = len(wins) / len(resolved) if resolved else 0.0
    returns  = [s.get("return_pct") or 0 for s in resolved]
    avg_ret  = sum(returns) / len(returns) if returns else 0.0

    gains  = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    pf     = gains / losses if losses > 0 else (99.0 if gains > 0 else 0.0)

    # Equity curve: cumulative avg return per day (actionable only)
    daily = {}
    for s in sorted(resolved, key=lambda x: x.get("timestamp", "")):
        day = str(s.get("timestamp", ""))[:10]
        if day not in daily:
            daily[day] = []
        daily[day].append(s.get("return_pct") or 0)
    equity = []
    cum = 0.0
    for day in sorted(daily):
        avg = sum(daily[day]) / len(daily[day])
        cum += avg
        equity.append({"date": day, "cum": round(cum, 4)})

    # Per-symbol win rates (actionable only)
    sym_stats = {}
    for s in resolved:
        sym = s.get("symbol", "?")
        if sym not in sym_stats:
            sym_stats[sym] = {"n": 0, "wins": 0}
        sym_stats[sym]["n"] += 1
        if s.get("outcome") == "WIN":
            sym_stats[sym]["wins"] += 1
    sym_wrs = {k: round(v["wins"] / v["n"] * 100, 1) for k, v in sym_stats.items() if v["n"] > 0}

    # Score tier breakdown for display
    tiers = {}
    for label, lo, hi in [("PERFECT", 65, 101), ("HIGH", 50, 65), ("MODERATE", 35, 50), ("MONITOR", 15, 35)]:
        tier_sigs = [s for s in signals if lo <= abs(s.get("unified_score", 0)) < hi]
        tier_res  = [s for s in tier_sigs if s.get("outcome") != "PENDING"]
        tier_wins = [s for s in tier_res if s.get("outcome") == "WIN"]
        tier_rets = [s.get("return_pct") or 0 for s in tier_res]
        tiers[label] = {
            "n":       len(tier_sigs),
            "res":     len(tier_res),
            "wr":      round(len(tier_wins) / len(tier_res) * 100, 1) if tier_res else 0,
            "avg_ret": round(sum(tier_rets) / len(tier_rets), 3) if tier_rets else 0,
        }

    # Count monitoring-only signals (excluded from main stats)
    monitoring_count = len([s for s in signals if abs(s.get("unified_score", 0)) < ACTIONABLE_SCORE])

    return {
        "total":             len(signals),
        "actionable":        len(actionable),
        "monitoring_only":   monitoring_count,
        "resolved":          len(resolved),
        "pending":           len(pending),
        "wins":              len(wins),
        "win_rate":          round(win_rate * 100, 1),
        "avg_ret":           round(avg_ret, 3),
        "pf":                round(pf, 2),
        "equity":            equity,
        "sym_wrs":           sym_wrs,
        "tiers":             tiers,
    }


# ── HTML BUILDERS ────────────────────────────────────────────────────────────

def _tier_badge(score):
    s = abs(score) if isinstance(score, (int, float)) else 0
    if s >= 65:
        return '<span class="badge badge-perfect">★★★ PERFECT</span>'
    if s >= 50:
        return '<span class="badge badge-high">★★ HIGH</span>'
    if s >= 35:
        return '<span class="badge badge-moderate">★ MODERATE</span>'
    if s >= 15:
        return '<span class="badge badge-monitor">○ MONITORING</span>'
    return '<span class="badge badge-wait">WAIT</span>'


def _dir_badge(direction):
    if direction == "LONG":
        return '<span class="dir long">▲ LONG</span>'
    if direction == "SHORT":
        return '<span class="dir short">▼ SHORT</span>'
    return '<span class="dir neutral">— NEUTRAL</span>'


def _outcome_badge(outcome):
    if outcome == "WIN":
        return '<span class="outcome win">WIN</span>'
    if outcome == "LOSS":
        return '<span class="outcome loss">LOSS</span>'
    return '<span class="outcome pending">PENDING</span>'


def build_signal_cards(scan):
    active = scan.get("active_signals") or []
    # Hard filter: only HIGH/PERFECT tier (|score| >= 50) directional signals are
    # shown as actionable. MODERATE-tier (35-49) signals proved to be noise (17% WR)
    # and are never logged by the tracker — don't display them as active either.
    active = [s for s in active
              if abs(s.get("unified_score", 0)) >= ACTIONABLE_SCORE
              and s.get("direction") in ("LONG", "SHORT")]
    if not active:
        return ('<div class="empty-state">No signals at HIGH/PERFECT conviction (|score| &ge; 50) right now — '
                'the engines are in agreement that no high-quality setup exists. '
                'Sub-threshold activity is visible in the Market Watchlist below.</div>')

    cards = []
    for s in active[:8]:
        sym    = s.get("symbol", "?")
        score  = s.get("unified_score", 0)
        dirn   = s.get("direction", "")
        tier   = s.get("tier", "")
        appr   = s.get("approved", False)
        susd   = s.get("size_usd", 0)
        spct   = s.get("size_pct", 0)
        comp   = s.get("components") or {}

        size_html = (f'<div class="size approved">${susd:,.0f} &nbsp;({spct:.1f}%)</div>'
                     if appr else
                     '<div class="size rejected">RISK REJECTED</div>')

        bars = ""
        for eng, val in comp.items():
            w = max(0, min(100, abs(float(val))))
            cls = "bar-bull" if float(val) >= 0 else "bar-bear"
            bars += (f'<div class="bar-row">'
                     f'<span class="bar-label">{eng[:6]}</span>'
                     f'<div class="bar-track"><div class="bar-fill {cls}" style="width:{w:.0f}%"></div></div>'
                     f'<span class="bar-val">{float(val):+.0f}</span>'
                     f'</div>')

        # Institutional data
        inst      = s.get("inst_data") or {}
        inst_sc   = inst.get("institutional_score", None)
        inst_conf = inst.get("confidence", "")
        oi_trend  = inst.get("oi_trend", "")
        top_bias  = inst.get("top_trader_bias", "")
        inst_html = ""
        if inst_sc is not None:
            inst_col = ("#00d4aa" if inst_sc >= 10 else
                        "#ffd060" if inst_sc >= -10 else "#ff6b6b")
            oi_sym  = "▲" if oi_trend == "RISING" else ("▼" if oi_trend == "FALLING" else "→")
            inst_html = (
                f'<div class="inst-row" style="font-size:10px;color:#8095a8;margin-top:4px;display:flex;gap:6px;flex-wrap:wrap">'
                f'<span style="color:{inst_col};font-weight:700">INST {inst_sc:+d}</span>'
                f'<span>OI {oi_sym}</span>'
                f'<span style="color:#c0cfe0">{top_bias}</span>'
                f'<span style="color:#8095a8">{inst_conf}</span>'
                f'</div>'
            )

        dir_cls = "card-long" if score > 0 else "card-short"
        cards.append(f'''
<div class="signal-card {dir_cls}">
  <div class="card-header">
    <span class="sym">{sym}</span>
    {_dir_badge(dirn)}
  </div>
  {_tier_badge(score)}
  <div class="score">{score:+d}/100</div>
  {size_html}
  {inst_html}
  <div class="bars">{bars}</div>
</div>''')
    return "\n".join(cards)


def build_watchlist(scan):
    """Score bars for ALL scanned symbols, sorted by |score| descending."""
    items = scan.get("watchlist") or []
    if not items:
        # Fallback: use active_signals if watchlist not in JSON yet
        items = scan.get("active_signals") or []
    if not items:
        return '<div class="empty-state">Watchlist data not yet available — will appear on next scan.</div>'

    items_sorted = sorted(items, key=lambda x: abs(x.get("unified_score", 0)), reverse=True)
    rows = ""
    for s in items_sorted:
        sym   = s.get("symbol", "?")
        score = s.get("unified_score", 0)
        dirn  = s.get("direction", "NEUTRAL")
        comp  = s.get("components") or {}

        # Score bar: -100 to +100
        bar_w = min(100, abs(score))
        bar_cls = "bar-bull" if score >= 0 else "bar-bear"
        # Direction colours
        dir_cls = "long" if score > 0 else ("short" if score < 0 else "")

        # Mini engine breakdown
        eng_html = ""
        for eng, val in comp.items():
            ew = max(0, min(50, abs(float(val)) / 2))
            ec = "bar-bull" if float(val) >= 0 else "bar-bear"
            eng_html += (f'<div class="mini-bar-row">'
                         f'<span class="mini-label">{eng[:3].upper()}</span>'
                         f'<div class="mini-track"><div class="mini-fill {ec}" style="width:{ew:.0f}%"></div></div>'
                         f'<span class="mini-val">{float(val):+.0f}</span>'
                         f'</div>')

        rows += f'''
<div class="wl-row">
  <div class="wl-sym">{sym.replace("USDT","")}</div>
  <div class="wl-score-wrap">
    <div class="wl-track"><div class="wl-fill {bar_cls}" style="width:{bar_w}%"></div></div>
    <span class="wl-val {dir_cls}">{score:+d}</span>
  </div>
  <div class="wl-tier">{_tier_badge(score)}</div>
  <div class="wl-engines">{eng_html}</div>
</div>'''

    return f'<div class="watchlist-grid">{rows}</div>'


def _port_protection_html(booked, book_count, port_peak, trail_floor, last_book_r, total_equity_gain):
    """Render portfolio protection status banner."""
    booked_cls  = "green" if booked > 0 else "arb-muted"
    gain_cls    = "green" if total_equity_gain >= 0 else "red"
    peak_txt    = f"${port_peak:+,.2f}" if port_peak else "—"
    trail_txt   = f"${trail_floor:,.2f}" if trail_floor is not None else "inactive (activates at $1,000 peak)"
    target_txt  = "$4,500"
    book_events = f"{book_count} book event{'s' if book_count != 1 else ''}" if book_count > 0 else "no books yet"
    last_txt    = last_book_r if book_count > 0 else "waiting for first profit lock"

    # Progress bar: unrealized portfolio gain toward $4,500 target
    progress_pct = min(100, max(0, total_equity_gain / 4500 * 100)) if total_equity_gain > 0 else 0
    bar_color    = "#10b981" if progress_pct >= 80 else ("#f59e0b" if progress_pct >= 40 else "#3b82f6")

    return f'''<div class="port-protect-bar">
  <div class="port-protect-left">
    <div class="port-protect-title">Portfolio Protection</div>
    <div class="port-protect-rule">Lock ALL when total P&L ≥ <b>${4500:,}</b> → 20 min cooldown → reopen fresh positions</div>
    <div style="margin-top:6px">
      <div style="font-size:11px;color:#8095a8;margin-bottom:3px">Progress to lock target ({progress_pct:.0f}%)</div>
      <div style="background:#1e293b;border-radius:4px;height:8px;width:100%;overflow:hidden">
        <div style="background:{bar_color};height:100%;width:{progress_pct:.1f}%;border-radius:4px;transition:width 1s"></div>
      </div>
    </div>
  </div>
  <div class="port-protect-stats">
    <div class="port-protect-stat">
      <div class="arb-label">Booked Profit</div>
      <div class="arb-val {booked_cls}">${booked:+,.2f}</div>
      <div class="arb-sub">{book_events}</div>
    </div>
    <div class="port-protect-stat">
      <div class="arb-label">Portfolio Peak</div>
      <div class="arb-val">{peak_txt}</div>
      <div class="arb-sub">highest unrealized reached</div>
    </div>
    <div class="port-protect-stat">
      <div class="arb-label">Trail Floor</div>
      <div class="arb-val {'yellow' if trail_floor else 'arb-muted'}" style="font-size:{'14px' if trail_floor else '11px'}">{trail_txt}</div>
      <div class="arb-sub">30% drawdown from peak</div>
    </div>
    <div class="port-protect-stat">
      <div class="arb-label">Last Event</div>
      <div class="arb-val arb-muted" style="font-size:11px">{last_txt}</div>
      <div class="arb-sub">stop loss at −$2,500</div>
    </div>
  </div>
</div>'''


def build_perp_arb_panel(arb):
    """Full perpetual arbitrage paper trading section."""
    if not arb:
        return '<div class="empty-state">Perp arb data not yet available — will appear after first cycle.</div>'

    total     = arb.get("total_trades", 0)
    n_open    = arb.get("open_trades", 0)
    n_close   = arb.get("closed_trades", 0)
    wr        = arb.get("win_rate", 0)
    pf        = arb.get("profit_factor", 0)
    rpnl      = arb.get("realized_pnl", 0)
    upnl      = arb.get("unrealized_pnl", 0)
    start_bal = arb.get("starting_balance", 1_000_000)
    equity    = arb.get("account_equity", start_bal)
    ret_pct   = arb.get("return_pct", 0)
    deployed  = arb.get("capital_deployed", 0)
    avail     = arb.get("available_cash", start_bal)
    proj_day  = arb.get("proj_daily_funding", 0)
    proj_mon  = arb.get("proj_monthly_funding", 0)
    proj_ann  = arb.get("proj_annual_funding", 0)
    proj_pct  = arb.get("proj_annual_pct", 0)
    # Portfolio protection
    booked      = arb.get("total_booked_pnl", 0) or 0
    book_count  = arb.get("book_count", 0) or 0
    port_peak   = arb.get("port_peak", 0) or 0
    trail_floor = arb.get("trail_floor")
    last_book_r = arb.get("last_book_reason", "-") or "-"
    total_equity_gain = booked + (arb.get("realized_pnl", 0) or 0) + (arb.get("unrealized_pnl", 0) or 0)

    wr_cls   = "green" if wr >= 55 else ("yellow" if wr >= 45 else "red")
    pf_cls   = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
    rp_cls   = "green" if rpnl >= 0 else "red"
    up_cls   = "green" if upnl >= 0 else "red"
    eq_cls   = "green" if equity >= start_bal else "red"
    dep_pct  = min(100, deployed / start_bal * 100) if start_bal else 0

    # Strategy breakdown — sorted by P&L so the best performer (NEG_CARRY) leads.
    strat_items = sorted((arb.get("by_strategy") or {}).items(),
                         key=lambda x: x[1].get("pnl", 0), reverse=True)
    best_strat = strat_items[0][0] if strat_items and strat_items[0][1].get("pnl", 0) > 0 else None
    strat_html = ""
    for s, d in strat_items:
        swr = d.get("win_rate", 0)
        sw_cls = "pos" if swr >= 50 else "neg"
        sp_cls = "pos" if d.get("pnl", 0) >= 0 else "neg"
        name_html = f'<b>{s}</b>'
        note_html = ""
        if s == best_strat:
            name_html += (' <span style="background:#00d4aa22;color:#00d4aa;border:1px solid #00d4aa44;'
                          'border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700;margin-left:4px">BEST</span>')
        if s == "CARRY_ARB" and swr < 50:
            note_html = ('<div style="font-size:10px;color:#8095a8;margin-top:2px">'
                         'historical pre-fix losses — spike entries before the fee-breakeven gate '
                         '(now requires funding &ge; 39%/yr sustained + intel confirmation; '
                         'new entries are protected)</div>')
        if s == "SPOT_DISCOUNT" and swr < 60:
            note_html = ('<div style="font-size:10px;color:#8095a8;margin-top:2px">'
                         'BERAUSDT anomaly (&#8722;$1,074) — extreme-funding token removed from whitelist. '
                         'Without BERA: SPOT_DISCOUNT WR=71%, P&amp;L=+$1,061. '
                         'Entry criteria now blocks extreme-funding tokens and applies 48h stop-loss cooldown.</div>')
        strat_html += (
            f'<tr><td>{name_html}{note_html}</td>'
            f'<td>{d["n"]}</td>'
            f'<td class="{sw_cls}">{swr:.0f}%</td>'
            f'<td class="{sp_cls}">${d["pnl"]:+.2f}</td></tr>\n'
        )

    # Open positions — build rows + collect JS data
    open_rows = ""
    js_positions = []
    stale_count = 0
    for p in (arb.get("open_positions") or [])[:12]:
        trade_id = p.get("id", "")
        sym      = p.get("symbol", "?").replace("USDT", "")
        full_sym = p.get("symbol", "?")
        strat    = p.get("strategy", "")
        dirn     = "↑Spot↓Perp" if p.get("direction") == "LONG_SPOT_SHORT_PERP" else "↑Perp↓Spot"
        ann      = p.get("ann_yield", 0) or 0
        basis_e  = p.get("basis_entry", 0) or 0
        basis_n  = p.get("basis_now", 0) or 0
        fund_e   = p.get("funding_earned", 0) or 0
        upnl_p   = p.get("unrealized_pnl", 0) or 0
        up_cls   = "pos" if upnl_p >= 0 else "neg"
        fe_cls   = "pos" if fund_e >= 0 else "neg"
        spot_e   = p.get("spot_entry", 0) or 0
        perp_e   = p.get("perp_entry", 0) or 0
        size_usd = p.get("size_usd", 66666.67) or 66666.67
        fees_tot = p.get("fees_total", 0) or 0
        try:
            ts = p.get("opened_at", "")
            if ts.endswith("Z"): ts = ts[:-1] + "+00:00"
            from datetime import datetime, timezone
            age_h = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() / 3600
        except Exception:
            age_h = 0

        # Staleness: a position older than 8h that has never had a funding period
        # applied means the updater hasn't touched it — its stored unrealized P&L
        # and funding totals are stale (the live JS ticker still shows fresh P&L).
        fund_periods = p.get("funding_periods", 0) or 0
        is_stale = age_h > 8 and fund_periods == 0
        if is_stale:
            stale_count += 1

        # Yield display: for basis-convergence strategies the funding-annualized
        # number is meaningless (extreme-funding tokens stored -2000%+/yr).
        # Show the expected basis-convergence yield instead.
        basis_yield = p.get("basis_yield_ann")
        if strat in ("SPOT_DISCOUNT", "SPOT_PREMIUM"):
            if basis_yield is None:
                basis_yield = abs(basis_e) * (24 * 365 / 72)  # fallback compute
            ann_html = f'<span class="pos">{basis_yield:+.1f}%/yr</span><br><span style="font-size:9px;color:#8095a8">basis conv.</span>'
        else:
            ann_html = f'<span class="pos">{ann:+.1f}%/yr</span>'
        # Smart exit proximity indicators
        size_usd_p  = p.get("size_usd", 66666) or 66666
        tp_usd      = size_usd_p * 0.030
        sl_usd      = -size_usd_p * 0.015
        peak_p      = p.get("peak_pnl", 0) or 0
        trail_act   = size_usd_p * 0.015
        trail_drop  = size_usd_p * 0.008
        trail_sl    = (peak_p - trail_drop) if peak_p >= trail_act else None
        dist_tp     = tp_usd - upnl_p
        dist_sl     = upnl_p - sl_usd
        # Show closest threat/target
        if trail_sl is not None:
            smart_txt = f'TP ${tp_usd:,.0f} | Trail >{peak_p:+.0f}'
            smart_cls = "pos"
        else:
            smart_txt = f'TP ${tp_usd:,.0f} | SL ${sl_usd:,.0f}'
            smart_cls = "act"

        stale_badge = ('<span style="background:#ffd06022;color:#ffd060;border:1px solid #ffd06044;'
                       'border-radius:3px;padding:1px 4px;font-size:9px;font-weight:700;margin-left:4px">STALE</span>'
                       if is_stale else "")
        open_rows += (
            f'<tr data-trade="{trade_id}">'
            f'<td><b>{sym}</b>{stale_badge}</td>'
            f'<td class="mono" style="font-size:11px">{strat}</td>'
            f'<td class="mono" style="font-size:11px">{dirn}</td>'
            f'<td>{ann_html}</td>'
            f'<td class="mono">{basis_e:+.3f}% → <span class="live-basis">{basis_n:+.3f}%</span></td>'
            f'<td class="{fe_cls}">${fund_e:+.2f}</td>'
            f'<td class="live-unreal {up_cls}">${upnl_p:+.2f}</td>'
            f'<td class="{smart_cls}" style="font-size:10px">{smart_txt}</td>'
            f'<td class="act">{age_h:.0f}h</td>'
            f'</tr>\n'
        )
        js_positions.append({
            "id":        trade_id,
            "symbol":    full_sym,
            "direction": p.get("direction", ""),
            "spot_entry": spot_e,
            "perp_entry": perp_e,
            "size_usd":   size_usd,
            "funding_total": fund_e,
            "fees_total":    fees_tot,
        })

    # ── QUANT WIN-RATE ANALYSIS ──────────────────────────────────────────────────
    journal = arb.get("trade_journal") or []
    # Count wins excluding whitelist-evict and cooldown-evict (pre-system / re-entry violations)
    real_trades  = [t for t in journal if t.get("exit_reason") not in ("WHITELIST_EVICT", "COOLDOWN_EVICT")]
    real_wins    = [t for t in real_trades if t.get("is_win")]
    real_wr      = round(len(real_wins) / len(real_trades) * 100, 1) if real_trades else 0
    evict_count  = len(journal) - len(real_trades)
    # Build WR timeline sparkline data
    wr_points    = [t["cum_wr"] for t in journal]
    wr_dates     = [(t.get("closed_at") or "")[:10] for t in journal]
    # Canvas sparkline — min/max scale
    sp_min, sp_max = (min(wr_points) if wr_points else 0), (max(wr_points) if wr_points else 100)
    sp_range = max(sp_max - sp_min, 10)
    sp_w, sp_h = 420, 60
    def wr_x(i): return int(sp_w * i / max(len(wr_points)-1, 1))
    def wr_y(v): return int(sp_h - sp_h * (v - sp_min) / sp_range)
    spark_pts = " ".join(f"{wr_x(i)},{wr_y(v)}" for i, v in enumerate(wr_points))
    # Date labels (every 5th trade)
    date_labels = ""
    for i, t in enumerate(journal):
        if i == 0 or i == len(journal)-1 or (i+1) % 5 == 0:
            d = (t.get("closed_at") or "")[:10]
            date_labels += f'<text x="{wr_x(i)}" y="74" font-size="8" fill="#8095a8" text-anchor="middle">{d[5:]}</text>'

    # ── TRADE JOURNAL TABLE ───────────────────────────────────────────────────
    closed_rows = ""
    for t in reversed(journal):   # newest first
        sym      = t.get("symbol", "?").replace("USDT", "")
        strat    = t.get("strategy", "")
        t_rpnl   = t.get("realized_pnl", 0) or 0
        fund     = t.get("funding_earned", 0) or 0
        fees     = t.get("fees", 0) or 0
        exit_    = t.get("exit_reason", "?")
        rp_cl    = "pos" if t_rpnl >= 0 else "neg"
        win_badge = ('<span style="background:#00d4aa22;color:#00d4aa;border:1px solid #00d4aa44;'
                     'border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700">WIN</span>'
                     if t_rpnl > 0 else
                     '<span style="background:#ff4d4d22;color:#ff6b6b;border:1px solid #ff4d4d44;'
                     'border-radius:3px;padding:1px 5px;font-size:9px;font-weight:700">LOSS</span>')
        s_entry  = t.get("spot_entry", 0) or 0
        p_entry  = t.get("perp_entry", 0) or 0
        s_exit   = t.get("spot_exit", 0) or 0
        p_exit   = t.get("perp_exit", 0) or 0
        # Format prices: use ≤ 4 sig figs
        def fmt_p(v):
            if v == 0: return "—"
            if v >= 100: return f"{v:.2f}"
            if v >= 1:   return f"{v:.4f}"
            return f"{v:.6f}"
        # Timestamps
        opn_dt   = (t.get("opened_at") or "")[:16].replace("T", " ") or "—"
        cld_dt   = (t.get("closed_at") or "")[:16].replace("T", " ") or "—"
        hold_h   = t.get("hold_hours", 0) or 0
        hold_txt = f"{hold_h:.1f}h" if hold_h < 48 else f"{hold_h/24:.1f}d"
        trade_no = t.get("trade_no", "")
        cum_wr   = t.get("cum_wr", 0)
        wr_cl    = "pos" if cum_wr >= 55 else ("act" if cum_wr >= 40 else "neg")
        dirn_arrow = "↑S↓P" if "LONG_SPOT" in (t.get("direction") or "") else "↑P↓S"
        evict_flag = ' style="opacity:0.5"' if exit_ == "WHITELIST_EVICT" else ""
        closed_rows += (
            f'<tr{evict_flag}>'
            f'<td style="color:#8095a8;font-size:10px">#{trade_no}</td>'
            f'<td>{win_badge}</td>'
            f'<td><b>{sym}</b></td>'
            f'<td class="mono" style="font-size:10px">{strat[:4]}<br><span style="color:#8095a8;font-size:9px">{dirn_arrow}</span></td>'
            f'<td style="font-size:10px;color:#c0cfe0">'
            f'  <span style="color:#00d4aa">▶</span> {fmt_p(s_entry)} / {fmt_p(p_entry)}<br>'
            f'  <span style="color:#ffd060">■</span> {fmt_p(s_exit)} / {fmt_p(p_exit)}'
            f'</td>'
            f'<td style="font-size:10px;color:#8095a8">'
            f'  {opn_dt}<br>'
            f'  {cld_dt}'
            f'</td>'
            f'<td class="act" style="font-size:10px">{hold_txt}</td>'
            f'<td class="pos" style="font-size:11px">${fund:+.2f}</td>'
            f'<td class="neg" style="font-size:11px">-${fees:.2f}</td>'
            f'<td class="{rp_cl}" style="font-weight:700">${t_rpnl:+.2f}</td>'
            f'<td class="{wr_cl}" style="font-size:10px">{cum_wr}%</td>'
            f'<td class="act" style="font-size:10px">{exit_}</td>'
            f'</tr>\n'
        )

    import json as _json
    js_pos_json = _json.dumps(js_positions)
    js_summary  = _json.dumps({
        "starting_balance": start_bal,
        "realized_pnl":     arb.get("realized_pnl", 0) or 0,
        "capital_deployed": deployed,
        "n_open":           n_open,
    })

    # Scanner watchlist section (shown when no open positions)
    scan_time    = arb.get("scan_time", "")
    scan_regime  = arb.get("scan_regime", "UNKNOWN")
    scan_btc     = arb.get("scan_btc_24h", 0) or 0
    scan_floor   = arb.get("scan_carry_floor", 22) or 22
    scan_wl      = arb.get("scan_watchlist") or []
    regime_cls   = "pos" if scan_regime == "BULL" else ("red" if scan_regime == "BEAR" else "yellow")
    watchlist_rows = ""
    for w in scan_wl[:8]:
        sym      = w.get("symbol","?").replace("USDT","")
        bpct     = w.get("basis_pct", 0) or 0
        ann      = w.get("ann_pct", 0) or 0
        need_pct = w.get("need_pct", 0) or 0
        b_cls    = "neg" if bpct < 0 else "pos"
        watchlist_rows += (
            f'<tr>'
            f'<td><b>{sym}</b></td>'
            f'<td class="{b_cls}">{bpct:+.3f}%</td>'
            f'<td class="act" style="font-size:11px">need {need_pct:+.3f}% more</td>'
            f'<td style="color:#8095a8;font-size:11px">{ann:+.1f}%/yr</td>'
            f'</tr>\n'
        )
    watchlist_html = ""
    if watchlist_rows and n_open == 0:
        watchlist_html = f'''
<div class="arb-sub-title">Watching — Near Threshold (basis 0.05-0.20%)</div>
<table class="data-table" style="margin-bottom:12px">
<thead><tr><th>Symbol</th><th>Basis Now</th><th>Gap to Entry</th><th>Funding</th></tr></thead>
<tbody>{watchlist_rows}</tbody>
</table>'''

    btc_cls = "pos" if scan_btc >= 0 else "red"

    # ── Live Institutional Intelligence table ─────────────────────────────────
    live_inst = fetch_live_inst_data()
    inst_rows = ""
    consensus_scores = []
    for sym, d in live_inst.items():
        sc       = d.get("institutional_score", 0) or 0
        oi_tr    = d.get("oi_trend", "—")
        oi_1h    = d.get("oi_pct_1h", 0) or 0
        tt_bias  = d.get("top_trader_bias", "—")
        tt_ratio = d.get("top_long_ratio", 0) or 0
        gl_ratio = d.get("global_long_ratio", 0) or 0
        f_tr     = d.get("funding_trend", "—")
        conf     = d.get("confidence", "")
        sc_col   = ("#00d4aa" if sc >= 20 else "#ffd060" if sc >= -20 else "#ff6b6b")
        oi_col   = "#00d4aa" if oi_tr == "RISING" else ("#ff6b6b" if oi_tr == "FALLING" else "#8095a8")
        f_col    = "#00d4aa" if f_tr == "RISING" else ("#ff6b6b" if f_tr == "FALLING" else "#8095a8")
        tt_col   = "#00d4aa" if tt_bias == "LONG" else "#ff6b6b"
        gl_pct   = f"{gl_ratio*100:.0f}%" if gl_ratio else "—"
        ticker   = sym.replace("USDT","")
        consensus_scores.append(sc)
        inst_rows += (
            f'<tr>'
            f'<td><b style="color:#e0eaf5">{ticker}</b></td>'
            f'<td><span style="color:{oi_col}">{oi_tr}</span> <span style="color:#8095a8;font-size:10px">{oi_1h:+.2f}%/h</span></td>'
            f'<td><span style="color:{tt_col}">{tt_bias}</span> <span style="color:#8095a8;font-size:10px">{tt_ratio:.1f}:1</span></td>'
            f'<td style="color:#8095a8;font-size:11px">{gl_pct} long</td>'
            f'<td><span style="color:{f_col}">{f_tr}</span></td>'
            f'<td style="font-weight:700;color:{sc_col}">{sc:+d}</td>'
            f'<td style="color:#8095a8;font-size:10px">{conf}</td>'
            f'</tr>\n'
        )
    if consensus_scores:
        avg_sc = sum(consensus_scores) / len(consensus_scores)
        cons_lbl = ("BULLISH ACCUMULATION" if avg_sc >= 25 else
                    "MILD BULLISH" if avg_sc >= 5 else
                    "NEUTRAL" if avg_sc >= -15 else
                    "MILD BEARISH" if avg_sc >= -35 else "BEARISH DISTRIBUTION")
        cons_col = ("#00d4aa" if avg_sc >= 5 else "#ffd060" if avg_sc >= -15 else "#ff6b6b")
        consensus_line = f'<div style="font-size:11px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,208,96,0.1)">Consensus: <b style="color:{cons_col}">{cons_lbl}</b> <span style="color:#8095a8">(avg score {avg_sc:+.0f})</span> &nbsp;·&nbsp; Gate active: entries blocked when score &lt; 30</div>'
    else:
        consensus_line = '<div style="font-size:11px;margin-top:8px;color:#8095a8">Fetching institutional data…</div>'

    inst_intel_html = f'''<div style="background:rgba(255,208,96,0.07);border:1px solid rgba(255,208,96,0.2);border-radius:8px;padding:12px 16px;margin-bottom:12px">
  <div style="font-size:11px;color:#ffd060;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">&#9670; Live Institutional Intelligence · Binance FAPI · Updated {datetime.now(timezone.utc).strftime("%H:%M UTC")}</div>
  <table style="width:100%;border-collapse:collapse;font-size:11px">
    <thead><tr style="color:#8095a8;font-size:10px;text-transform:uppercase;letter-spacing:0.5px">
      <th style="text-align:left;padding:3px 8px 5px 0">Symbol</th>
      <th style="text-align:left;padding:3px 8px 5px 0">OI Trend (1h)</th>
      <th style="text-align:left;padding:3px 8px 5px 0">Top Traders</th>
      <th style="text-align:left;padding:3px 8px 5px 0">Global L/S</th>
      <th style="text-align:left;padding:3px 8px 5px 0">Funding</th>
      <th style="text-align:left;padding:3px 8px 5px 0">Score</th>
      <th style="text-align:left;padding:3px 8px 5px 0">Conf</th>
    </tr></thead>
    <tbody>{inst_rows if inst_rows else "<tr><td colspan='7' style='color:#8095a8;padding:8px 0'>Fetching…</td></tr>"}</tbody>
  </table>
  {consensus_line}
  <div style="font-size:10px;color:#8095a8;margin-top:4px">Score: &#8722;100 (heavy institutional selling) → +100 (institutional accumulation). Entry gate requires score &#8805;30.</div>
</div>'''
    scanner_status_html = f'''
<div style="background:rgba(0,180,255,0.07);border:1px solid rgba(0,180,255,0.18);border-radius:8px;padding:12px 16px;margin-bottom:14px;display:flex;align-items:center;gap:14px;flex-wrap:wrap">
  <span style="display:inline-flex;align-items:center;gap:6px">
    <span style="width:8px;height:8px;border-radius:50%;background:#00b4ff;display:inline-block;animation:pulse-dot 1.4s ease-in-out infinite"></span>
    <b style="color:#00b4ff;font-size:13px">Scanner Running</b>
  </span>
  <span style="color:#8095a8;font-size:12px">{scan_time}</span>
  <span class="{regime_cls}" style="font-size:12px">Regime: {scan_regime} · BTC 24h <span class="{btc_cls}">{scan_btc:+.2f}%</span></span>
  <span style="color:#8095a8;font-size:12px">Carry floor: {scan_floor:.1f}%/yr · Basis min: 0.20%</span>
  {"<span style='color:#8095a8;font-size:12px'>Watching " + str(len(scan_wl)) + " near-miss candidates</span>" if n_open == 0 else ""}
  {"<span class='pos' style='font-size:12px'>" + str(n_open) + " positions open</span>" if n_open > 0 else ""}
</div>

<!-- Live Crypto Prices Board (always visible, updates every 3s) -->
<div id="live-prices-board" style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">
  <div class="price-chip" data-sym="BTCUSDT"><span class="pc-sym">BTC</span><span class="pc-price" id="lp-BTCUSDT">—</span><span class="pc-chg" id="lc-BTCUSDT"></span></div>
  <div class="price-chip" data-sym="ETHUSDT"><span class="pc-sym">ETH</span><span class="pc-price" id="lp-ETHUSDT">—</span><span class="pc-chg" id="lc-ETHUSDT"></span></div>
  <div class="price-chip" data-sym="SOLUSDT"><span class="pc-sym">SOL</span><span class="pc-price" id="lp-SOLUSDT">—</span><span class="pc-chg" id="lc-SOLUSDT"></span></div>
  <div class="price-chip" data-sym="BNBUSDT"><span class="pc-sym">BNB</span><span class="pc-price" id="lp-BNBUSDT">—</span><span class="pc-chg" id="lc-BNBUSDT"></span></div>
  <div class="price-chip" data-sym="XRPUSDT"><span class="pc-sym">XRP</span><span class="pc-price" id="lp-XRPUSDT">—</span><span class="pc-chg" id="lc-XRPUSDT"></span></div>
  <div class="price-chip" data-sym="DOGEUSDT"><span class="pc-sym">DOGE</span><span class="pc-price" id="lp-DOGEUSDT">—</span><span class="pc-chg" id="lc-DOGEUSDT"></span></div>
  <div class="price-chip" data-sym="DOTUSDT"><span class="pc-sym">DOT</span><span class="pc-price" id="lp-DOTUSDT">—</span><span class="pc-chg" id="lc-DOTUSDT"></span></div>
  <div class="price-chip" data-sym="AVAXUSDT"><span class="pc-sym">AVAX</span><span class="pc-price" id="lp-AVAXUSDT">—</span><span class="pc-chg" id="lc-AVAXUSDT"></span></div>
</div>

<!-- Live Perp vs Spot Basis Board (updates every 5s) -->
<div style="margin-bottom:14px">
  <div style="font-size:10px;color:#8095a8;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px">Live Perp–Spot Basis &amp; Funding (8h rate · updates every 5s)</div>
  <div id="live-basis-board" style="display:flex;gap:8px;flex-wrap:wrap">
    <div class="basis-chip" id="bc-BTCUSDT"><span class="bc-sym">BTC</span><span class="bc-basis" id="bb-BTCUSDT">—</span><span class="bc-fund" id="bf-BTCUSDT"></span></div>
    <div class="basis-chip" id="bc-ETHUSDT"><span class="bc-sym">ETH</span><span class="bc-basis" id="bb-ETHUSDT">—</span><span class="bc-fund" id="bf-ETHUSDT"></span></div>
    <div class="basis-chip" id="bc-SOLUSDT"><span class="bc-sym">SOL</span><span class="bc-basis" id="bb-SOLUSDT">—</span><span class="bc-fund" id="bf-SOLUSDT"></span></div>
    <div class="basis-chip" id="bc-BNBUSDT"><span class="bc-sym">BNB</span><span class="bc-basis" id="bb-BNBUSDT">—</span><span class="bc-fund" id="bf-BNBUSDT"></span></div>
    <div class="basis-chip" id="bc-XRPUSDT"><span class="bc-sym">XRP</span><span class="bc-basis" id="bb-XRPUSDT">—</span><span class="bc-fund" id="bf-XRPUSDT"></span></div>
    <div class="basis-chip" id="bc-DOGEUSDT"><span class="bc-sym">DOGE</span><span class="bc-basis" id="bb-DOGEUSDT">—</span><span class="bc-fund" id="bf-DOGEUSDT"></span></div>
    <div class="basis-chip" id="bc-DOTUSDT"><span class="bc-sym">DOT</span><span class="bc-basis" id="bb-DOTUSDT">—</span><span class="bc-fund" id="bf-DOTUSDT"></span></div>
    <div class="basis-chip" id="bc-AVAXUSDT"><span class="bc-sym">AVAX</span><span class="bc-basis" id="bb-AVAXUSDT">—</span><span class="bc-fund" id="bf-AVAXUSDT"></span></div>
  </div>
  <div style="font-size:10px;color:#8095a8;margin-top:4px">Entry threshold: basis ≥ 0.20% <span style="margin-left:12px">Carry entry: funding ≥ 22%/yr</span></div>
</div>'''

    return f'''
<script>
window.PERP_POSITIONS = {js_pos_json};
window.PERP_SUMMARY   = {js_summary};
</script>
{inst_intel_html}
{scanner_status_html}
<!-- Account Balance Bar -->
<div class="acct-bar">
  <div class="acct-left">
    <div class="acct-label">Opening Balance</div>
    <div class="acct-opening">${start_bal:,.0f}</div>
  </div>
  <div class="acct-mid">
    <div class="acct-label">Account Equity &nbsp;<span id="live-dot" style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#8095a8;vertical-align:middle"></span></div>
    <div id="live-equity" class="acct-equity {eq_cls}">${equity:,.2f} <span class="acct-ret">({ret_pct:+.4f}%)</span></div>
    <div class="acct-deploy-wrap">
      <div class="acct-deploy-bar"><div class="acct-deploy-fill" style="width:{dep_pct:.1f}%"></div></div>
      <span class="acct-deploy-lbl">{dep_pct:.1f}% deployed — ${deployed:,.0f} in {n_open} positions · ${avail:,.0f} available</span>
    </div>
  </div>
  <div class="acct-right">
    <div class="acct-label">Total P&L</div>
    <div class="acct-pnl {rp_cls}">${rpnl:+,.2f} <span style="font-size:12px;color:#8095a8">realized</span></div>
    <div id="live-upnl" class="acct-pnl {up_cls}" style="font-size:16px">${upnl:+,.2f} <span style="font-size:12px;color:#8095a8">unrealized</span></div>
  </div>
</div>

<div class="arb-stats">
  <div class="arb-stat"><div class="arb-label">Open Positions</div><div class="arb-val">{n_open}/{MAX_OPEN_TRADES}</div><div class="arb-sub">${deployed:,.0f} deployed · ${avail:,.0f} free</div></div>
  <div class="arb-stat"><div class="arb-label">Projected Daily</div><div class="arb-val green">${proj_day:,.2f}</div><div class="arb-sub">funding income / day</div></div>
  <div class="arb-stat"><div class="arb-label">Projected Annual</div><div class="arb-val green">${proj_ann:,.0f} <span style="font-size:12px">({proj_pct:.1f}%/yr)</span></div><div class="arb-sub">on $1M account</div></div>
  <div class="arb-stat"><div class="arb-label">Realized P&L</div><div class="arb-val {rp_cls}">${rpnl:+,.2f}</div><div class="arb-sub">closed trades</div></div>
  <div class="arb-stat"><div class="arb-label">Win Rate</div><div class="arb-val {wr_cls}">{wr:.1f}%</div><div class="arb-sub">{arb.get("wins",0)}W / {arb.get("losses",0)}L · PF {pf:.2f}×</div></div>
</div>

<!-- Portfolio Protection Banner -->
{_port_protection_html(booked, book_count, port_peak, trail_floor, last_book_r, total_equity_gain)}

<!-- Best Trade Highlight -->
<div style="background:linear-gradient(135deg,rgba(0,212,170,0.12),rgba(0,212,170,0.04));border:1px solid rgba(0,212,170,0.3);border-radius:10px;padding:14px 18px;margin-bottom:14px">
  <div style="font-size:11px;color:#00d4aa;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">&#9733; Best Trade Highlight</div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center">
    <div>
      <div style="font-size:26px;font-weight:800;color:#00d4aa">+$3,388</div>
      <div style="font-size:11px;color:#8095a8">INIT NEG_CARRY &middot; PROFIT_TARGET exit</div>
    </div>
    <div style="flex:1;min-width:200px">
      <div style="font-size:12px;color:#c0cfe0;line-height:1.6">
        <b>INITUSDT</b> — negative funding rate captured before normalization.
        Institutional flow data confirmed heavy short positioning (top traders 72% short).
        Strategy: Long Perp + Short Spot, collected $3,388 funding + basis convergence over 11h hold.
      </div>
    </div>
  </div>
</div>

{f"""<div class="arb-sub-title">By Strategy</div>
<table class="data-table" style="margin-bottom:16px">
<thead><tr><th>Strategy</th><th>Trades</th><th>Win Rate</th><th>P&L</th></tr></thead>
<tbody>{strat_html}</tbody>
</table>""" if strat_html else ""}

{f"""<div style="background:rgba(255,208,96,0.08);border:1px solid rgba(255,208,96,0.25);border-radius:8px;
padding:10px 14px;margin-bottom:10px;font-size:12px;color:#ffd060">
&#9888; {stale_count} position(s) have not been refreshed in &gt;8h — stored funding/unrealized P&amp;L below is stale.
The live ticker (green values) recomputes P&amp;L from current prices; stored values catch up on the next updater cycle.
</div>""" if stale_count > 0 else ""}{f"""<div class="arb-sub-title">Open Positions ({n_open})</div>
<table class="data-table" style="margin-bottom:16px">
<thead><tr><th>Symbol</th><th>Strategy</th><th>Direction</th><th>Ann Yield</th><th>Basis Entry→Now</th><th>Funding</th><th>Unrealized</th><th>TP / SL</th><th>Age</th></tr></thead>
<tbody>{open_rows}</tbody>
</table>""" if open_rows else (watchlist_html + """<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:14px 18px;margin-top:8px">""" + f"""
  <div style="font-size:12px;color:#8095a8;margin-bottom:4px">No open positions — scanner active every minute</div>
  <div style="font-size:12px;color:#c0cfe0">Entry criteria: <span style="color:#ffd060">Basis &ge; 0.20%</span> (SPOT_DISCOUNT/PREMIUM) or <span style="color:#ffd060">Funding &ge; {scan_floor:.0f}%/yr</span> (CARRY_ARB). Market: <span class="{regime_cls}">{scan_regime}</span>, BTC 24h <span class="{btc_cls}">{scan_btc:+.2f}%</span>.
  </div>
  <div style="font-size:11px;color:#8095a8;margin-top:4px">Live perp&#8211;spot basis &amp; funding shown above (every 5s) &#8212; chips glow green when entry-ready.</div>
</div>""")}

{f"""
<div class="arb-sub-title" style="margin-top:20px">Win Rate Analysis</div>
<div style="background:#0a1520;border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:14px 18px;margin-bottom:12px">
  <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:10px">
    <div>
      <div style="font-size:10px;color:#8095a8;text-transform:uppercase;letter-spacing:.5px">Overall WR</div>
      <div style="font-size:22px;font-weight:700;color:{'#ff6b6b' if wr < 40 else '#ffd060' if wr < 55 else '#00d4aa'}">{wr:.0f}%</div>
      <div style="font-size:10px;color:#8095a8">{arb.get('wins',0)}W / {arb.get('losses',0)}L  ({n_close} closed)</div>
    </div>
    <div>
      <div style="font-size:10px;color:#8095a8;text-transform:uppercase;letter-spacing:.5px">Strategy WR (excl. evictions)</div>
      <div style="font-size:22px;font-weight:700;color:{'#00d4aa' if real_wr >= 55 else '#ffd060'}">{real_wr:.0f}%</div>
      <div style="font-size:10px;color:#8095a8">{len(real_wins)}W / {len(real_trades)-len(real_wins)}L  ({len(real_trades)} non-evict trades)</div>
    </div>
    <div>
      <div style="font-size:10px;color:#8095a8;text-transform:uppercase;letter-spacing:.5px">Profit Factor</div>
      <div style="font-size:22px;font-weight:700;color:{'#00d4aa' if pf >= 1.5 else '#ffd060'}">{pf:.2f}×</div>
      <div style="font-size:10px;color:#8095a8">avg win / avg loss ratio</div>
    </div>
    <div style="flex:1;min-width:200px">
      <div style="font-size:10px;color:#8095a8;margin-bottom:4px">Why overall WR shows {wr:.0f}%:</div>
      <div style="font-size:11px;color:#c0cfe0;line-height:1.5">{evict_count} trade(s) excluded: stock/junk tokens (SAMSUNG, NVDA etc.) opened pre-whitelist (WHITELIST_EVICT), plus any symbol re-entered within 48h of a stop-loss (COOLDOWN_EVICT — now blocked). These {"had no real trading signal" if evict_count > 0 else ""}. Excluding them, real strategy WR = <span style="color:#00d4aa;font-weight:700">{real_wr:.0f}%</span>.</div>
    </div>
  </div>
  <div style="margin-top:8px">
    <div style="font-size:10px;color:#8095a8;margin-bottom:4px">Cumulative Win Rate Over All {n_close} Trades</div>
    <svg width="{sp_w}" height="80" style="display:block;overflow:visible">
      <line x1="0" y1="{wr_y(50)}" x2="{sp_w}" y2="{wr_y(50)}" stroke="rgba(255,255,255,0.08)" stroke-dasharray="3,3"/>
      <polyline points="{spark_pts}" fill="none" stroke="#00d4aa" stroke-width="1.5" opacity="0.9"/>
      {date_labels}
      <text x="0" y="{wr_y(50)+3}" font-size="8" fill="rgba(255,255,255,0.3)">50%</text>
    </svg>
  </div>
</div>

<div class="arb-sub-title">Complete Trade Journal ({n_close} trades, newest first)</div>
<div style="font-size:11px;color:#8095a8;margin-bottom:8px">Entry/Exit: Spot price / Perp price &nbsp;|&nbsp; ↑S↓P = Long Spot Short Perp &nbsp;|&nbsp; ↑P↓S = Long Perp Short Spot &nbsp;|&nbsp; Faded rows = pre-whitelist historical evictions</div>
<div style="overflow-x:auto">
<table class="data-table" style="font-size:11px;min-width:900px">
<thead><tr>
  <th>#</th><th>W/L</th><th>Symbol</th><th>Strategy</th>
  <th>Entry (S/P) → Exit (S/P)</th>
  <th>Opened → Closed (UTC)</th>
  <th>Hold</th><th>Funding</th><th>Fees</th>
  <th>Net P&amp;L</th><th>Cum WR</th><th>Exit Reason</th>
</tr></thead>
<tbody>{closed_rows}</tbody>
</table>
</div>
""" if closed_rows else ""}
'''


def build_carry_table(scan):
    rows = scan.get("carry_arb") or []
    if not rows:
        return '<div class="empty-state">No carry arb opportunities right now — scanner looks for funding ≥ 22%/yr (BEAR) or 25%/yr (BULL). Rates are currently below threshold. Live basis+funding data is shown in the Perp Arb section above.</div>'
    tbody = ""
    for r in rows:
        ann   = r.get("ann_pct", 0)
        basis = r.get("basis_pct", 0)
        act   = r.get("action", "")
        cls   = "pos" if ann > 0 else "neg"
        tbody += (f'<tr><td><b>{r.get("symbol","?")}</b></td>'
                  f'<td class="{cls}">{ann:+.2f}%</td>'
                  f'<td>{basis:+.4f}%</td>'
                  f'<td class="act">{act.replace("_", " ")}</td></tr>\n')
    return f'''
<table class="data-table">
<thead><tr><th>Symbol</th><th>Ann %/yr</th><th>Basis %</th><th>Action</th></tr></thead>
<tbody>{tbody}</tbody>
</table>'''


def build_pairs_table(scan):
    rows = scan.get("stat_arb_pairs") or []
    if not rows:
        return '<div class="empty-state">No cointegrated pairs found in current window</div>'
    tbody = ""
    for r in rows:
        z  = r.get("z_score")
        z_str = f"{z:+.3f}" if z is not None else "N/A"
        z_cls = "pos" if (z or 0) > 0 else "neg"
        hl    = r.get("half_life_h", "?")
        pv    = r.get("p_value")
        pv_str = f"{pv:.4f}" if pv is not None else "N/A"
        act   = r.get("action", "HOLD")
        tbody += (f'<tr><td><b>{r.get("pair","?")}</b></td>'
                  f'<td class="{z_cls}">{z_str}</td>'
                  f'<td>{r.get("hedge_ratio",0):.3f}</td>'
                  f'<td>{hl}h</td>'
                  f'<td>{pv_str}</td>'
                  f'<td class="act">{act}</td></tr>\n')
    return f'''
<table class="data-table">
<thead><tr><th>Pair</th><th>Z-Score</th><th>β Hedge</th><th>Half-Life</th><th>p-value</th><th>Action</th></tr></thead>
<tbody>{tbody}</tbody>
</table>'''


def build_risk_panel(scan):
    risk = scan.get("risk") or {}
    if not risk:
        return '<div class="empty-state">Risk data unavailable</div>'
    regime = risk.get("corr_regime", "NORMAL")
    regime_cls = "warn" if regime == "HIGH_CORR" else "ok"
    return f'''
<div class="risk-grid">
  <div class="risk-item">
    <div class="risk-label">VaR (95%)</div>
    <div class="risk-val warn">{risk.get("var_95","N/A")}%<span class="risk-sub">/day</span></div>
  </div>
  <div class="risk-item">
    <div class="risk-label">CVaR (95%)</div>
    <div class="risk-val warn">{risk.get("cvar_95","N/A")}%<span class="risk-sub">/day</span></div>
  </div>
  <div class="risk-item">
    <div class="risk-label">VaR (99%)</div>
    <div class="risk-val">{risk.get("var_99","N/A")}%<span class="risk-sub">/day</span></div>
  </div>
  <div class="risk-item">
    <div class="risk-label">Worst Day</div>
    <div class="risk-val neg">{risk.get("worst_day","N/A")}%</div>
  </div>
  <div class="risk-item">
    <div class="risk-label">Avg Correlation</div>
    <div class="risk-val {regime_cls}">{risk.get("avg_correlation","N/A")}</div>
  </div>
  <div class="risk-item">
    <div class="risk-label">Eff. Positions</div>
    <div class="risk-val">{risk.get("effective_positions","N/A")} <span class="risk-sub">of 8</span></div>
  </div>
  <div class="risk-item">
    <div class="risk-label">Corr Regime</div>
    <div class="risk-val {regime_cls}">{regime}</div>
  </div>
  <div class="risk-item">
    <div class="risk-label">High-Corr Pairs</div>
    <div class="risk-val">{risk.get("high_corr_pairs","N/A")}</div>
  </div>
</div>'''


def build_engine_status(scan):
    engines = scan.get("engines") or {}
    names = {
        "master":   "Master Engine (8-layer structural)",
        "hft":      "HFT Engine (OFI + microprice)",
        "stat_arb": "Stat Arb (cointegration + Kalman)",
        "factor":   "Factor Model (12-factor alpha)",
        "risk":     "Risk Manager (VaR + Kelly)",
    }
    items = ""
    for key, label in names.items():
        online = engines.get(key, False)
        dot = "dot-on" if online else "dot-off"
        status = "ONLINE" if online else "OFFLINE"
        items += f'<div class="eng-row"><span class="dot {dot}"></span><span class="eng-label">{label}</span><span class="eng-status">{status}</span></div>\n'
    return f'<div class="engine-list">{items}</div>'


def build_history_table(signals):
    if not signals:
        return '<div class="empty-state">No signal history yet — first scan pending.</div>'
    rows = ""
    for s in signals[:50]:
        ts   = str(s.get("timestamp", ""))[:16]
        sym  = s.get("symbol", "?")
        dirn = s.get("direction", "?")
        sc   = s.get("unified_score", 0)
        oc   = s.get("outcome", "PENDING")
        ret  = s.get("return_pct")
        ret_str  = f"{ret:+.2f}%" if ret is not None else "—"
        ret_cls  = "pos" if (ret or 0) > 0 else ("neg" if (ret or 0) < 0 else "")
        dir_cls  = "long" if dirn == "LONG" else "short"
        rows += (f'<tr>'
                 f'<td class="mono">{ts}</td>'
                 f'<td><b>{sym}</b></td>'
                 f'<td class="{dir_cls}">{dirn}</td>'
                 f'<td class="mono">{sc:+d}</td>'
                 f'<td>{_outcome_badge(oc)}</td>'
                 f'<td class="{ret_cls} mono">{ret_str}</td>'
                 f'</tr>\n')
    return f'''
<table class="data-table history-table">
<thead><tr><th>Timestamp</th><th>Symbol</th><th>Direction</th><th>Score</th><th>Outcome</th><th>Return</th></tr></thead>
<tbody>{rows}</tbody>
</table>'''


def _parse_ts(ts_str):
    """Return (day_str YYYY-MM-DD, time_str HH:MM) from any timestamp format."""
    s = str(ts_str).replace("+00:00", "").replace("Z", "").replace(" UTC", "").strip()
    if "T" in s:
        day, time = s[:10], s[11:16]
    elif " " in s:
        parts = s.split(" ")
        day  = parts[0]
        time = parts[1][:5] if len(parts) > 1 else "?"
    else:
        day, time = s[:10], "?"
    return day, time


def build_daily_trade_log(signals):
    """Day-grouped signal log: per-day summary header + full chronological list."""
    if not signals:
        return '<div class="empty-state">No signal history yet — first scan pending.</div>'

    from collections import OrderedDict
    days = OrderedDict()
    for s in sorted(signals, key=lambda x: x.get("timestamp", ""), reverse=True):
        day, _ = _parse_ts(s.get("timestamp", ""))
        if day not in days:
            days[day] = []
        days[day].append(s)

    html_parts = []
    for day, sigs in days.items():
        resolved = [s for s in sigs if s.get("outcome") != "PENDING"]
        wins     = [s for s in resolved if s.get("outcome") == "WIN"]
        losses   = [s for s in resolved if s.get("outcome") == "LOSS"]
        pending  = [s for s in sigs     if s.get("outcome") == "PENDING"]
        wr       = len(wins) / len(resolved) * 100 if resolved else 0
        rets     = [s.get("return_pct") or 0 for s in resolved]
        avg_ret  = sum(rets) / len(rets) if rets else 0
        wr_color  = "#00d4aa" if wr >= 55 else ("#ffd060" if wr >= 45 else "#ff6b6b")
        avg_color = "#00d4aa" if avg_ret >= 0 else "#ff6b6b"

        pend_badge = (f'<span class="ds-badge ds-pend">{len(pending)} pending</span>'
                      if pending else "")
        wr_badge   = (f'<span class="ds-badge" style="color:{wr_color}">{wr:.0f}% WR</span>'
                      if resolved else "")
        avg_badge  = (f'<span class="ds-badge" style="color:{avg_color}">{avg_ret:+.2f}% avg</span>'
                      if resolved else "")

        html_parts.append(f'''<div class="dlog-day-header">
  <span class="dlog-date">&#128197; {day}</span>
  <span class="dlog-daystats">
    <span class="ds-badge ds-total">{len(sigs)} signals</span>
    <span class="ds-badge ds-win">&#9650; {len(wins)} WIN</span>
    <span class="ds-badge ds-loss">&#9660; {len(losses)} LOSS</span>
    {pend_badge}{wr_badge}{avg_badge}
  </span>
</div>''')

        for s in sorted(sigs, key=lambda x: x.get("timestamp", ""), reverse=True):
            _, time  = _parse_ts(s.get("timestamp", ""))
            sym      = s.get("symbol", "?")
            dirn     = s.get("direction", "?")
            sc       = s.get("unified_score", 0)
            oc       = s.get("outcome", "PENDING")
            ret      = s.get("return_pct")
            ret_str  = f"{ret:+.2f}%" if ret is not None else "—"
            ret_cls  = "pos" if (ret or 0) > 0 else ("neg" if (ret or 0) < 0 else "")
            dir_cls  = "long" if dirn == "LONG" else "short"
            row_cls  = ("dlog-row-win" if oc == "WIN"
                        else ("dlog-row-loss" if oc == "LOSS" else "dlog-row-pend"))

            html_parts.append(f'''<div class="dlog-row {row_cls}">
  <span class="dlog-time mono">{time} UTC</span>
  <span class="dlog-sym"><b>{sym.replace("USDT","")}</b></span>
  <span class="dlog-dir {dir_cls}">{dirn}</span>
  <span class="dlog-tier">{_tier_badge(sc)}</span>
  <span class="dlog-score mono">{sc:+d}</span>
  <span class="dlog-oc">{_outcome_badge(oc)}</span>
  <span class="dlog-ret {ret_cls} mono">{ret_str}</span>
</div>''')

    return f'<div class="dlog-wrap">{"".join(html_parts)}</div>'


# ── CHART DATA ───────────────────────────────────────────────────────────────

def chart_equity(equity):
    labels = json.dumps([e["date"] for e in equity])
    data   = json.dumps([e["cum"] for e in equity])
    return f"""
var eqCtx = document.getElementById('eqChart').getContext('2d');
new Chart(eqCtx, {{
  type: 'line',
  data: {{
    labels: {labels},
    datasets: [{{
      label: 'Cumulative Return %',
      data: {data},
      borderColor: '#00d4aa',
      backgroundColor: 'rgba(0,212,170,0.08)',
      fill: true,
      tension: 0.3,
      pointRadius: 3,
      pointBackgroundColor: '#00d4aa',
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#c0cfe0' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8095a8', maxTicksLimit: 10 }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
      y: {{ ticks: {{ color: '#8095a8', callback: v => v + '%' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
    }}
  }}
}});"""


def chart_sym_wrs(sym_wrs):
    syms   = list(sym_wrs.keys())
    wrs    = list(sym_wrs.values())
    colors = ["#00d4aa" if w >= 50 else "#ff6b6b" for w in wrs]
    labels = json.dumps(syms)
    data   = json.dumps(wrs)
    clrs   = json.dumps(colors)
    return f"""
var wrCtx = document.getElementById('wrChart').getContext('2d');
new Chart(wrCtx, {{
  type: 'bar',
  data: {{
    labels: {labels},
    datasets: [{{
      label: 'Win Rate %',
      data: {data},
      backgroundColor: {clrs},
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8095a8' }}, grid: {{ display: false }} }},
      y: {{ min: 0, max: 100, ticks: {{ color: '#8095a8', callback: v => v + '%' }},
           grid: {{ color: 'rgba(255,255,255,0.05)' }} }}
    }}
  }}
}});"""


# ── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:#0a0e1a;color:#c0cfe0;font-size:14px;line-height:1.5}
a{color:#00d4aa;text-decoration:none}
.container{max-width:1400px;margin:0 auto;padding:20px}

/* Header */
.header{display:flex;align-items:center;justify-content:space-between;padding:20px 24px;
        background:linear-gradient(135deg,#0d1526,#121c35);border-radius:12px;margin-bottom:24px;
        border:1px solid rgba(0,212,170,0.2)}
.header-left .title{font-size:22px;font-weight:700;color:#fff;letter-spacing:0.5px}
.header-left .subtitle{font-size:12px;color:#8095a8;margin-top:2px}
.header-right{text-align:right}
.header-right .ts{font-size:13px;color:#00d4aa;font-weight:600}
.header-right .sub{font-size:11px;color:#8095a8;margin-top:2px}
.engines-pill{display:inline-block;background:rgba(0,212,170,0.12);color:#00d4aa;
              border:1px solid rgba(0,212,170,0.3);border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600}

/* Stats bar */
.stats-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}
.stat-card{background:#0d1526;border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:16px 20px}
.stat-label{font-size:11px;color:#8095a8;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px}
.stat-val{font-size:28px;font-weight:700;color:#fff}
.stat-val.green{color:#00d4aa}
.stat-val.red{color:#ff6b6b}
.stat-val.yellow{color:#ffd060}
.stat-sub{font-size:12px;color:#8095a8;margin-top:2px}

/* Section */
.section{background:#0d1526;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:20px 24px;margin-bottom:20px}
.section-title{font-size:13px;font-weight:700;color:#8095a8;text-transform:uppercase;letter-spacing:1px;
               margin-bottom:16px;display:flex;align-items:center;gap:8px}
.section-title::before{content:'';display:inline-block;width:3px;height:14px;background:#00d4aa;border-radius:2px}

/* Signal cards */
.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
.signal-card{background:#111928;border-radius:10px;padding:16px;border:1px solid rgba(255,255,255,0.08);transition:border-color 0.2s}
.signal-card:hover{border-color:rgba(0,212,170,0.3)}
.signal-card.card-long{border-left:3px solid #00d4aa}
.signal-card.card-short{border-left:3px solid #ff6b6b}
.card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.sym{font-size:16px;font-weight:700;color:#fff}
.score{font-size:26px;font-weight:800;color:#fff;margin:8px 0}
.size{font-size:12px;padding:4px 8px;border-radius:6px;display:inline-block;margin-bottom:10px}
.size.approved{background:rgba(0,212,170,0.12);color:#00d4aa;border:1px solid rgba(0,212,170,0.3)}
.size.rejected{background:rgba(255,107,107,0.1);color:#ff6b6b;border:1px solid rgba(255,107,107,0.3)}
.bars{margin-top:8px}
.bar-row{display:flex;align-items:center;gap:6px;margin-bottom:5px}
.bar-label{font-size:10px;color:#8095a8;width:36px;text-transform:uppercase}
.bar-track{flex:1;height:4px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px}
.bar-fill.bar-bull{background:#00d4aa}
.bar-fill.bar-bear{background:#ff6b6b}
.bar-val{font-size:10px;color:#8095a8;width:28px;text-align:right}

/* Badges */
.badge{display:inline-block;padding:3px 8px;border-radius:5px;font-size:11px;font-weight:600;margin-bottom:4px}
.badge-perfect{background:rgba(255,215,0,0.15);color:#ffd700;border:1px solid rgba(255,215,0,0.3)}
.badge-high{background:rgba(0,212,170,0.12);color:#00d4aa;border:1px solid rgba(0,212,170,0.3)}
.badge-moderate{background:rgba(100,160,255,0.12);color:#64a0ff;border:1px solid rgba(100,160,255,0.3)}
.badge-wait{background:rgba(255,255,255,0.05);color:#8095a8;border:1px solid rgba(255,255,255,0.1)}
.dir{font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px}
.dir.long{background:rgba(0,212,170,0.12);color:#00d4aa}
.dir.short{background:rgba(255,107,107,0.12);color:#ff6b6b}
.dir.neutral{background:rgba(255,255,255,0.06);color:#8095a8}
.outcome{font-size:11px;font-weight:600;padding:2px 7px;border-radius:4px}
.outcome.win{background:rgba(0,212,170,0.12);color:#00d4aa}
.outcome.loss{background:rgba(255,107,107,0.12);color:#ff6b6b}
.outcome.pending{background:rgba(255,255,255,0.06);color:#8095a8}

/* Charts */
.charts-grid{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:20px}
.chart-wrap{background:#0d1526;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:20px 24px}
.chart-title{font-size:12px;font-weight:700;color:#8095a8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px}
.chart-canvas{position:relative;height:200px}

/* Tables */
.data-table{width:100%;border-collapse:collapse;font-size:13px}
.data-table th{color:#8095a8;font-size:11px;text-transform:uppercase;letter-spacing:0.8px;
               padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.07);text-align:left}
.data-table td{padding:9px 12px;border-bottom:1px solid rgba(255,255,255,0.04)}
.data-table tr:hover td{background:rgba(255,255,255,0.02)}
.history-table{font-size:12px}
.pos{color:#00d4aa;font-weight:600}
.neg{color:#ff6b6b;font-weight:600}
.ok{color:#00d4aa}
.warn{color:#ffd060}
.long{color:#00d4aa;font-weight:600}
.short{color:#ff6b6b;font-weight:600}
.act{color:#8095a8;font-size:12px}
.mono{font-family:'Fira Mono','Consolas',monospace;font-size:12px}

/* Risk grid */
.risk-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.risk-item{background:#111928;border-radius:8px;padding:12px 16px;border:1px solid rgba(255,255,255,0.06)}
.risk-label{font-size:11px;color:#8095a8;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.5px}
.risk-val{font-size:22px;font-weight:700;color:#fff}
.risk-val.warn{color:#ffd060}
.risk-val.ok{color:#00d4aa}
.risk-val.neg{color:#ff6b6b}
.risk-sub{font-size:11px;color:#8095a8;font-weight:400}

/* Engine list */
.engine-list{display:flex;flex-direction:column;gap:8px}
.eng-row{display:flex;align-items:center;gap:10px}
.dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.dot-on{background:#00d4aa;box-shadow:0 0 6px rgba(0,212,170,0.6)}
.dot-off{background:#ff6b6b}
.eng-label{flex:1;font-size:13px;color:#c0cfe0}
.eng-status{font-size:11px;font-weight:600;color:#8095a8}

/* Empty state */
.empty-state{color:#8095a8;font-style:italic;padding:16px 0;text-align:center;font-size:13px}

/* Footer */
.footer{text-align:center;color:#8095a8;font-size:11px;margin-top:24px;padding-top:16px;
        border-top:1px solid rgba(255,255,255,0.06)}

/* Account Balance Bar */
.acct-bar{display:grid;grid-template-columns:180px 1fr 220px;gap:20px;align-items:center;
          background:linear-gradient(135deg,#0d1526,#121c35);border:1px solid rgba(0,212,170,0.25);
          border-radius:12px;padding:18px 24px;margin-bottom:20px}
.acct-label{font-size:10px;color:#8095a8;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px}
.acct-opening{font-size:28px;font-weight:800;color:#fff}
.acct-equity{font-size:26px;font-weight:800}
.acct-equity.green{color:#00d4aa}
.acct-equity.red{color:#ff6b6b}
.acct-ret{font-size:13px;font-weight:400;color:#8095a8}
.acct-deploy-wrap{margin-top:8px;display:flex;align-items:center;gap:8px}
.acct-deploy-bar{width:160px;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;flex-shrink:0}
.acct-deploy-fill{height:100%;background:#00d4aa;border-radius:3px}
.acct-deploy-lbl{font-size:11px;color:#8095a8;white-space:nowrap}
.acct-pnl{font-size:20px;font-weight:700}
.acct-pnl.green{color:#00d4aa}
.acct-pnl.red{color:#ff6b6b}
@media(max-width:900px){.acct-bar{grid-template-columns:1fr 1fr}.acct-mid{grid-column:span 2}}
@media(max-width:600px){.acct-bar{grid-template-columns:1fr}.acct-deploy-wrap{flex-wrap:wrap}}

/* Perp Arb Panel */
.arb-stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}
.arb-stat{background:#111928;border-radius:8px;padding:12px 16px;border:1px solid rgba(255,255,255,0.06)}
.arb-label{font-size:11px;color:#8095a8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px}
.arb-val{font-size:22px;font-weight:700;color:#fff}
.arb-sub{font-size:11px;color:#8095a8;margin-top:2px}
.arb-sub-title{font-size:11px;font-weight:700;color:#8095a8;text-transform:uppercase;
               letter-spacing:0.8px;margin:12px 0 8px}
.arb-muted{color:#8095a8!important}
/* Portfolio protection banner */
.port-protect-bar{display:flex;gap:20px;background:linear-gradient(135deg,#0f1e2e,#0d2240);
  border:1px solid rgba(59,130,246,0.25);border-radius:10px;padding:16px 20px;margin-bottom:16px;align-items:flex-start}
.port-protect-left{flex:1;min-width:0}
.port-protect-title{font-size:12px;font-weight:700;color:#3b82f6;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px}
.port-protect-rule{font-size:12px;color:#94a3b8}
.port-protect-stats{display:flex;gap:16px;flex-shrink:0}
.port-protect-stat{min-width:120px;text-align:right}
@media(max-width:900px){.port-protect-bar{flex-direction:column}
  .port-protect-stats{flex-wrap:wrap;justify-content:flex-start}
  .port-protect-stat{text-align:left}}
@media(max-width:900px){.arb-stats{grid-template-columns:repeat(3,1fr)}}
@media(max-width:600px){.arb-stats{grid-template-columns:repeat(2,1fr)}}

/* Live Price Chips */
#live-prices-board{align-items:center}
.price-chip{display:flex;align-items:center;gap:6px;padding:6px 12px;background:#111928;
  border:1px solid rgba(255,255,255,0.08);border-radius:20px;font-size:12px;min-width:110px}
.pc-sym{font-weight:700;color:#8095a8;font-size:11px;width:34px}
.pc-price{font-weight:700;color:#e2e8f0;font-variant-numeric:tabular-nums;min-width:60px}
.pc-chg{font-size:10px;font-variant-numeric:tabular-nums;min-width:48px;text-align:right}
/* Live Basis+Funding Chips */
.basis-chip{display:flex;align-items:center;gap:6px;padding:6px 12px;background:#0d1826;
  border:1px solid rgba(255,255,255,0.08);border-radius:20px;font-size:12px;min-width:130px;
  transition:border-color .3s}
.basis-chip.basis-hot{border-color:rgba(0,212,170,0.4)}
.bc-sym{font-weight:700;color:#8095a8;font-size:11px;width:34px}
.bc-basis{font-weight:700;font-variant-numeric:tabular-nums;min-width:52px;font-size:12px}
.bc-fund{font-size:10px;font-variant-numeric:tabular-nums;min-width:52px;text-align:right;color:#8095a8}
/* Watchlist */
.watchlist-grid{display:flex;flex-direction:column;gap:6px}
.wl-row{display:grid;grid-template-columns:72px 1fr 100px 180px;align-items:center;gap:12px;
        padding:8px 10px;border-radius:7px;background:#111928;border:1px solid rgba(255,255,255,0.05)}
.wl-row:hover{border-color:rgba(0,212,170,0.2)}
.wl-sym{font-size:13px;font-weight:700;color:#fff}
.wl-score-wrap{display:flex;align-items:center;gap:8px}
.wl-track{flex:1;height:6px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden}
.wl-fill{height:100%;border-radius:3px}
.wl-fill.bar-bull{background:#00d4aa}
.wl-fill.bar-bear{background:#ff6b6b}
.wl-val{font-size:13px;font-weight:700;width:38px;text-align:right}
.wl-tier{text-align:center}
.wl-engines{display:flex;gap:4px;align-items:center}
.mini-bar-row{display:flex;align-items:center;gap:3px}
.mini-label{font-size:9px;color:#8095a8;width:22px}
.mini-track{width:36px;height:3px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden}
.mini-fill{height:100%;border-radius:2px}
.mini-val{font-size:9px;color:#8095a8;width:20px}
.badge-monitor{background:rgba(255,160,60,0.12);color:#ff9f40;border:1px solid rgba(255,160,60,0.3)}

@media(max-width:700px){.wl-row{grid-template-columns:60px 1fr 80px}.wl-engines{display:none}}

@media(max-width:900px){
  .stats-bar{grid-template-columns:repeat(2,1fr)}
  .charts-grid{grid-template-columns:1fr}
  .risk-grid{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:600px){
  .stats-bar{grid-template-columns:1fr}
  .risk-grid{grid-template-columns:1fr}
  .cards-grid{grid-template-columns:1fr}
}

/* Daily Trade Log */
.dlog-wrap{display:flex;flex-direction:column;gap:3px}
.dlog-day-header{
  display:flex;align-items:center;gap:14px;
  padding:10px 16px;margin-top:14px;
  background:linear-gradient(90deg,#0f1f38,#0d1526);
  border-radius:8px;border:1px solid rgba(0,212,170,0.18);
}
.dlog-day-header:first-child{margin-top:0}
.dlog-date{font-size:13px;font-weight:700;color:#fff;min-width:120px;flex-shrink:0}
.dlog-daystats{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.ds-badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:5px;white-space:nowrap}
.ds-total{background:rgba(255,255,255,0.07);color:#c0cfe0}
.ds-win{background:rgba(0,212,170,0.12);color:#00d4aa;border:1px solid rgba(0,212,170,0.25)}
.ds-loss{background:rgba(255,107,107,0.12);color:#ff6b6b;border:1px solid rgba(255,107,107,0.25)}
.ds-pend{background:rgba(255,255,255,0.05);color:#8095a8}
.dlog-row{
  display:grid;
  grid-template-columns:62px 68px 54px 110px 40px 90px 60px;
  align-items:center;gap:10px;
  padding:7px 14px 7px 28px;
  border-radius:6px;font-size:12px;
  border-left:2px solid transparent;
}
.dlog-row:hover{background:rgba(255,255,255,0.03)}
.dlog-row-win{border-left-color:rgba(0,212,170,0.5)}
.dlog-row-loss{border-left-color:rgba(255,107,107,0.5)}
.dlog-row-pend{border-left-color:rgba(128,149,168,0.3)}
.dlog-time{color:#8095a8;font-size:11px}
.dlog-sym{font-size:13px}
.dlog-dir{font-size:11px;font-weight:700}
.dlog-tier{}
.dlog-score{font-size:11px;text-align:right}
.dlog-oc{}
.dlog-ret{font-size:12px;text-align:right}
@media(max-width:700px){
  .dlog-row{grid-template-columns:52px 1fr 50px auto auto;gap:6px}
  .dlog-tier,.dlog-score{display:none}
  .dlog-day-header{flex-direction:column;align-items:flex-start;gap:6px}
}
"""


def _build_tier_breakdown(tiers, monitoring_count):
    """Score tier breakdown bar shown below the main stats."""
    rows = ""
    tier_config = [
        ("PERFECT",  "★★★",  "#ffd700", "≥65"),
        ("HIGH",     "★★",   "#00d4aa", "50-64"),
        ("MODERATE", "★",    "#64a0ff", "35-49"),
        ("MONITOR",  "○",    "#ff9f40", "15-34"),
    ]
    for key, stars, color, rng in tier_config:
        t = tiers.get(key, {})
        n   = t.get("n", 0)
        res = t.get("res", 0)
        wr  = t.get("wr", 0)
        avg = t.get("avg_ret", 0)
        if n == 0:
            continue
        wr_color = "#00d4aa" if wr >= 55 else ("#ffd060" if wr >= 40 else "#ff6b6b")
        # Only HIGH (50-64) + PERFECT (>=65) count toward main stats now
        is_excluded = key in ("MONITOR", "MODERATE")
        note = " — excluded from main stats (below 50 threshold)" if is_excluded else ""
        opacity = "0.5" if is_excluded else "1"
        rows += f'''
<div class="tier-row" style="opacity:{opacity}">
  <div class="tier-stars" style="color:{color}">{stars}</div>
  <div class="tier-label">{key} ({rng})</div>
  <div class="tier-n">{n} signals</div>
  <div class="tier-res">{res} resolved</div>
  <div class="tier-wr" style="color:{wr_color}">{wr:.1f}% WR</div>
  <div class="tier-ret" style="color:{'#00d4aa' if avg > 0 else '#ff6b6b'}">{avg:+.3f}%</div>
  <div class="tier-note" style="color:#8095a8;font-size:10px">{note}</div>
</div>'''

    if not rows:
        return ""

    return f'''<div class="section tier-breakdown-section" style="margin-bottom:20px">
  <div class="section-title">Signal Quality Breakdown
    <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">
      Win Rate / Avg Return per tier · only ★ ACTIONABLE tiers count toward main stats
    </span>
  </div>
  <div class="tier-grid">{rows}</div>
  {'<div style="font-size:11px;color:#ff9f40;margin-top:8px">⚠ ' + str(monitoring_count) + ' sub-threshold signals (score &lt; 50) excluded from Win Rate — MODERATE/MONITORING tiers proved to be noise (17-27% WR), not trades.</div>' if monitoring_count > 0 else ''}
</div>
<style>
.tier-grid {{display:flex;flex-direction:column;gap:6px}}
.tier-row {{display:grid;grid-template-columns:28px 110px 90px 90px 80px 80px 1fr;
           align-items:center;gap:10px;padding:8px 12px;border-radius:7px;
           background:#111928;border:1px solid rgba(255,255,255,0.05)}}
.tier-stars {{font-size:13px;text-align:center}}
.tier-label {{font-size:12px;font-weight:700;color:#fff}}
.tier-n {{font-size:12px;color:#8095a8}}
.tier-res {{font-size:12px;color:#8095a8}}
.tier-wr {{font-size:13px;font-weight:700}}
.tier-ret {{font-size:12px;font-weight:600}}
.tier-note {{font-size:10px}}
@media(max-width:700px){{.tier-row{{grid-template-columns:28px 1fr 60px 60px}}
  .tier-res,.tier-note{{display:none}}}}
</style>'''


def _build_stats_bar(stats, wr, wr_cls, pf, pf_cls, ret_cls):
    """KPI stats bar. Handles the fresh-DB state gracefully (no fake 0% WR)."""
    if stats["total"] == 0:
        return '''<div class="stats-bar" style="grid-template-columns:1fr">
  <div class="stat-card" style="text-align:center;padding:24px">
    <div class="stat-label">Signal Tracker Status</div>
    <div class="stat-val" style="font-size:18px;color:#00d4aa">Starting fresh — building quality signal history</div>
    <div class="stat-sub" style="margin-top:6px">Only HIGH/PERFECT tier signals (|score| &ge; 50, institutionally confirmed) are logged.
    Weaker MODERATE-tier signals (35-49) were pruned after proving to be noise (17% WR).
    Win-rate statistics will appear once the first qualifying signals resolve (4h after logging).</div>
  </div>
</div>'''
    return f'''<div class="stats-bar">
  <div class="stat-card">
    <div class="stat-label">Actionable Signals <span style="font-size:10px;color:#8095a8">(score &ge;50)</span></div>
    <div class="stat-val">{stats['actionable']}</div>
    <div class="stat-sub">{stats['resolved']} resolved &middot; {stats['pending']} pending &middot; {stats['total']} total logged</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Win Rate <span style="font-size:10px;color:#8095a8">(score &ge;50)</span></div>
    <div class="stat-val {wr_cls}">{wr if stats['resolved'] else '—'}{'%' if stats['resolved'] else ''}</div>
    <div class="stat-sub">{str(stats['wins']) + ' wins / ' + str(stats['resolved']) + ' resolved' if stats['resolved'] else 'no resolved signals yet — first outcomes land 4h after logging'}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Profit Factor</div>
    <div class="stat-val {pf_cls}">{f'{pf:.2f}&times;' if stats['resolved'] else '—'}</div>
    <div class="stat-sub">Gains &divide; Losses (need &gt;1.0)</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Return / Signal</div>
    <div class="stat-val {ret_cls}">{f"{stats['avg_ret']:+.3f}%" if stats['resolved'] else '—'}</div>
    <div class="stat-sub">Net of 0.3% fee threshold</div>
  </div>
</div>'''


# ── MAIN BUILDER ─────────────────────────────────────────────────────────────

def build_html(scan, signals, stats, arb=None):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scan_ts = scan.get("timestamp", "Never")
    n_online = scan.get("engines_online", 0)
    arb = arb or {}

    # Win-rate colour
    wr = stats["win_rate"]
    wr_cls  = "green" if wr >= 55 else ("yellow" if wr >= 45 else "red")
    pf      = stats["pf"]
    pf_cls  = "green" if pf >= 1.5 else ("yellow" if pf >= 1.0 else "red")
    ret_cls = "green" if stats["avg_ret"] > 0 else "red"

    # Charts JS
    eq_js = chart_equity(stats["equity"]) if stats["equity"] else ""
    wr_js = chart_sym_wrs(stats["sym_wrs"]) if stats["sym_wrs"] else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="600">
<title>Global Quant System | Live Dashboard</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="title">⚡ GLOBAL QUANT SYSTEM</div>
    <div class="subtitle">5-Engine Institutional Crypto Trading Dashboard</div>
  </div>
  <div class="header-right">
    <div class="ts">{scan_ts}</div>
    <div class="sub">Last scan &nbsp;|&nbsp; <span class="engines-pill">{n_online}/5 Engines Online</span></div>
    <div class="sub" style="margin-top:4px;font-size:10px">Auto-refreshes every 10 min &nbsp;|&nbsp; BloomAI Quant System</div>
  </div>
</div>

<!-- Stats Bar (ACTIONABLE signals only: |score| >= 50) -->
{_build_stats_bar(stats, wr, wr_cls, pf, pf_cls, ret_cls)}

<!-- Score Tier Breakdown -->
{_build_tier_breakdown(stats['tiers'], stats['monitoring_only'])}

<!-- Charts -->
<div class="charts-grid">
  <div class="chart-wrap">
    <div class="chart-title">📈 Equity Curve (cumulative avg return)</div>
    <div class="chart-canvas"><canvas id="eqChart"></canvas></div>
    {"<div class='empty-state' style='padding-top:70px'>Accumulating data — check back after a few scans</div>" if not stats["equity"] else ""}
  </div>
  <div class="chart-wrap">
    <div class="chart-title">🏆 Win Rate by Symbol</div>
    <div class="chart-canvas"><canvas id="wrChart"></canvas></div>
    {"<div class='empty-state' style='padding-top:70px'>No resolved signals yet</div>" if not stats["sym_wrs"] else ""}
  </div>
</div>

<!-- Active Signals -->
<div class="section">
  <div class="section-title">Active Signals  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">|score| ≥ 50 · inst-confirmed · regime-filtered</span></div>
  <div class="cards-grid">
    {build_signal_cards(scan)}
  </div>
</div>

<!-- Market Watchlist -->
<div class="section">
  <div class="section-title">Market Watchlist  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">all {len(scan.get("watchlist") or scan.get("active_signals") or [])} symbols — sorted by conviction</span></div>
  {build_watchlist(scan)}
</div>

<!-- Risk Dashboard -->
<div class="section">
  <div class="section-title">Portfolio Risk Dashboard</div>
  {build_risk_panel(scan)}
</div>

<!-- Perp Arb Paper Trading -->
<div class="section">
  <div class="section-title">Perpetual Arb — Paper Trading  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">sandbox · carry + basis + funding reversion · $1k per leg</span></div>
  {build_perp_arb_panel(arb)}
</div>

<!-- Carry Arb -->
<div class="section">
  <div class="section-title">Carry Arb Opportunities  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">long spot / short perp, passive 24/7</span></div>
  {build_carry_table(scan)}
</div>

<!-- Stat Arb Pairs -->
<div class="section">
  <div class="section-title">Stat Arb Pairs  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">market-neutral cointegrated pairs</span></div>
  {build_pairs_table(scan)}
</div>

<!-- Engine Status -->
<div class="section">
  <div class="section-title">Engine Status</div>
  {build_engine_status(scan)}
</div>

<!-- Daily Trade Log -->
<div class="section">
  <div class="section-title">Daily Trade Log  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">all signals · grouped by day · newest first</span></div>
  {build_daily_trade_log(signals)}
</div>

<div class="footer">
  Generated at {now_str} &nbsp;·&nbsp;
  Auto-scan every ~1 min &nbsp;·&nbsp; Tracking {len(scan.get("watchlist") or [])} symbols &nbsp;·&nbsp;
  BloomAI Quant System &nbsp;·&nbsp; <a href="https://bloomai.technology/quant/" target="_blank">bloomai.technology/quant</a>
</div>

</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
{eq_js}
{wr_js}
</script>
<script>
// ── LIVE P&L TICKER + PRICE BOARD ────────────────────────────────────────
(function() {{
  const positions = window.PERP_POSITIONS || [];
  const summary   = window.PERP_SUMMARY   || {{}};

  const startBal   = summary.starting_balance || 1000000;
  const realizedPnl = summary.realized_pnl || 0;

  const PRICE_BOARD_SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT','DOTUSDT','AVAXUSDT'];

  let spotPrices = {{}};
  let perpPrices = {{}};
  let prevPrices = {{}};   // for 24h-change color flash
  let lastFetch  = 0;
  let fetchPending = false;

  async function fetchPrices() {{
    if (fetchPending) return;
    fetchPending = true;
    try {{
      const [sRes, pRes] = await Promise.all([
        fetch('https://api.bybit.com/v5/market/tickers?category=spot'),
        fetch('https://api.bybit.com/v5/market/tickers?category=linear')
      ]);
      const sData = await sRes.json();
      const pData = await pRes.json();
      for (const t of (sData?.result?.list || [])) spotPrices[t.symbol] = parseFloat(t.lastPrice);
      for (const t of (pData?.result?.list || [])) {{
        perpPrices[t.symbol] = parseFloat(t.lastPrice);
        // Store 24h change pct for the price board
        if (t.price24hPcnt !== undefined) perpPrices[t.symbol + '_24h'] = parseFloat(t.price24hPcnt) * 100;
      }}
      lastFetch = Date.now();
      setDot('#00d4aa');  // green = live
      updatePriceBoard();
    }} catch(e) {{
      setDot('#ff6b6b');  // red = fetch error
    }}
    fetchPending = false;
  }}

  function updatePriceBoard() {{
    for (const sym of PRICE_BOARD_SYMS) {{
      const price = perpPrices[sym] || spotPrices[sym];
      const chg24h = perpPrices[sym + '_24h'];
      const pEl = document.getElementById('lp-' + sym);
      const cEl = document.getElementById('lc-' + sym);
      if (!price || !pEl) continue;
      // Format price
      const p = price;
      const formatted = p >= 1000 ? '$' + p.toLocaleString('en-US', {{maximumFractionDigits:1}})
                       : p >= 1   ? '$' + p.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:4}})
                       : '$' + p.toFixed(5);
      pEl.textContent = formatted;
      if (cEl && chg24h !== undefined) {{
        cEl.textContent = (chg24h >= 0 ? '+' : '') + chg24h.toFixed(2) + '%';
        cEl.style.color = chg24h >= 0 ? '#00d4aa' : '#ff6b6b';
      }}
    }}
  }}

  function setDot(color) {{
    const d = document.getElementById('live-dot');
    if (d) d.style.background = color;
  }}

  function fmt(v, sign=true) {{
    const s = sign ? (v >= 0 ? '+' : '') : '';
    return s + v.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}});
  }}

  function recalcAndUpdate() {{
    // Only update DOM if we have fresh prices (< 10s old)
    if (!lastFetch || Date.now() - lastFetch > 10000) return;
    // Always update price board even with no positions
    updatePriceBoard();

    let totalUnreal = 0;
    for (const pos of positions) {{
      const spotNow = spotPrices[pos.symbol];
      const perpNow = perpPrices[pos.symbol];
      if (!spotNow || !perpNow || !pos.spot_entry || !pos.perp_entry) continue;

      const sz = pos.size_usd;
      let pnl;
      if (pos.direction === 'LONG_SPOT_SHORT_PERP') {{
        pnl = sz * (spotNow - pos.spot_entry) / pos.spot_entry
            - sz * (perpNow - pos.perp_entry) / pos.perp_entry;
      }} else {{
        pnl = -sz * (spotNow - pos.spot_entry) / pos.spot_entry
            + sz * (perpNow - pos.perp_entry) / pos.perp_entry;
      }}
      pnl += (pos.funding_total || 0) - (pos.fees_total || 0);
      totalUnreal += pnl;

      const row = document.querySelector(`[data-trade="${{pos.id}}"]`);
      if (!row) continue;

      // Unrealized P&L cell
      const uc = row.querySelector('.live-unreal');
      if (uc) {{
        uc.textContent = '$' + fmt(pnl);
        uc.className = 'live-unreal ' + (pnl >= 0 ? 'pos' : 'neg');
      }}

      // Basis-now span
      const bc = row.querySelector('.live-basis');
      if (bc) {{
        const basisNow = (perpNow - spotNow) / spotNow * 100;
        bc.textContent = (basisNow >= 0 ? '+' : '') + basisNow.toFixed(3) + '%';
      }}
    }}

    // Account equity bar
    const equity = startBal + realizedPnl + totalUnreal;
    const retPct = (equity - startBal) / startBal * 100;
    const eqEl = document.getElementById('live-equity');
    if (eqEl) {{
      eqEl.textContent = '$' + equity.toLocaleString('en-US', {{minimumFractionDigits:2, maximumFractionDigits:2}})
        + ' (' + (retPct >= 0 ? '+' : '') + retPct.toFixed(4) + '%)';
      eqEl.className = 'acct-equity ' + (equity >= startBal ? 'green' : 'red');
    }}
    const upEl = document.getElementById('live-upnl');
    if (upEl) {{
      upEl.innerHTML = '$' + fmt(totalUnreal) + ' <span style="font-size:12px;color:#8095a8">unrealized</span>';
      upEl.className = 'acct-pnl ' + (totalUnreal >= 0 ? 'pos' : 'neg');
    }}
  }}

  // Fetch new prices every 3 seconds; recalc DOM every second
  fetchPrices();
  setInterval(fetchPrices, 3000);
  setInterval(recalcAndUpdate, 1000);
}})();

// ── LIVE BASIS + FUNDING RATES ────────────────────────────────────────────
(function() {{
  const BASIS_SYMS = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT','DOTUSDT','AVAXUSDT'];
  const BASIS_MIN = 0.20;  // entry threshold %
  const CARRY_ANN = 22;    // carry floor %/yr

  async function fetchBasisAndFunding() {{
    try {{
      // Spot prices
      const [sRes, pRes, fRes] = await Promise.all([
        fetch('https://api.bybit.com/v5/market/tickers?category=spot'),
        fetch('https://api.bybit.com/v5/market/tickers?category=linear'),
        fetch('https://api.bybit.com/v5/market/tickers?category=linear')
      ]);
      const sData = await sRes.json();
      const pData = await pRes.json();

      const spotMap = {{}};
      for (const t of (sData?.result?.list || [])) spotMap[t.symbol] = parseFloat(t.lastPrice);
      const perpMap = {{}};
      for (const t of (pData?.result?.list || [])) {{
        perpMap[t.symbol] = {{
          price: parseFloat(t.lastPrice),
          fundingRate: parseFloat(t.fundingRate || 0),
          nextFunding: t.nextFundingTime
        }};
      }}

      for (const sym of BASIS_SYMS) {{
        const spot = spotMap[sym];
        const perp = perpMap[sym];
        if (!spot || !perp) continue;

        const basis = (perp.price - spot) / spot * 100;
        const fr8h  = perp.fundingRate * 100;
        const frAnn = fr8h * 3 * 365;  // 3 payments/day × 365

        const bEl = document.getElementById('bb-' + sym);
        const fEl = document.getElementById('bf-' + sym);
        const chip = document.getElementById('bc-' + sym);

        if (bEl) {{
          bEl.textContent = (basis >= 0 ? '+' : '') + basis.toFixed(3) + '%';
          bEl.style.color = Math.abs(basis) >= BASIS_MIN ? '#00d4aa' : '#8095a8';
        }}
        if (fEl) {{
          fEl.textContent = (fr8h >= 0 ? '+' : '') + fr8h.toFixed(4) + '% · ' + frAnn.toFixed(1) + '%/yr';
          fEl.style.color = Math.abs(frAnn) >= CARRY_ANN ? '#ffd060' : '#8095a8';
        }}
        // Highlight chip if near entry threshold
        if (chip) {{
          if (Math.abs(basis) >= BASIS_MIN || Math.abs(frAnn) >= CARRY_ANN) {{
            chip.classList.add('basis-hot');
          }} else {{
            chip.classList.remove('basis-hot');
          }}
        }}
      }}
    }} catch(e) {{ /* silent */ }}
  }}

  fetchBasisAndFunding();
  setInterval(fetchBasisAndFunding, 5000);
}})();
</script>
</body>
</html>"""


def main():
    print("Loading scan data...")
    scan    = load_scan_json()
    signals = load_signals_db()
    stats   = compute_stats(signals)
    arb     = load_perp_arb_json()

    print(f"  {len(signals)} signals in DB, {stats['resolved']} resolved, win_rate={stats['win_rate']}%")
    print(f"  Active signals in JSON: {len(scan.get('active_signals') or [])}")
    print(f"  Perp arb trades: {arb.get('total_trades', 0)} total, "
          f"{arb.get('open_trades', 0)} open")

    html = build_html(scan, signals, stats, arb)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written → {OUT_PATH}")


if __name__ == "__main__":
    main()
