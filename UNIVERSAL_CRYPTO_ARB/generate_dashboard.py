"""
Dashboard Generator
Reads accuracy_db/signals.db + accuracy_db/latest_scan.json
Writes docs/index.html — served free via GitHub Pages
"""
import sqlite3, json, os, sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE, "accuracy_db", "signals.db")
JSON_PATH = os.path.join(BASE, "accuracy_db", "latest_scan.json")
OUT_DIR   = os.path.join(BASE, "docs")
OUT_PATH  = os.path.join(OUT_DIR, "index.html")

os.makedirs(OUT_DIR, exist_ok=True)


# ── DATA LOADING ─────────────────────────────────────────────────────────────

def load_scan_json():
    if not os.path.exists(JSON_PATH):
        return {}
    try:
        with open(JSON_PATH) as f:
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


def compute_stats(signals):
    resolved = [s for s in signals if s.get("outcome") != "PENDING"]
    pending  = [s for s in signals if s.get("outcome") == "PENDING"]
    wins     = [s for s in resolved if s.get("outcome") == "WIN"]

    win_rate = len(wins) / len(resolved) if resolved else 0.0
    returns  = [s.get("return_pct") or 0 for s in resolved]
    avg_ret  = sum(returns) / len(returns) if returns else 0.0

    gains  = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    pf     = gains / losses if losses > 0 else (99.0 if gains > 0 else 0.0)

    # Equity curve: cumulative avg return per day
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

    # Per-symbol win rates
    sym_stats = {}
    for s in resolved:
        sym = s.get("symbol", "?")
        if sym not in sym_stats:
            sym_stats[sym] = {"n": 0, "wins": 0}
        sym_stats[sym]["n"] += 1
        if s.get("outcome") == "WIN":
            sym_stats[sym]["wins"] += 1
    sym_wrs = {k: round(v["wins"] / v["n"] * 100, 1) for k, v in sym_stats.items() if v["n"] > 0}

    return {
        "total":     len(signals),
        "resolved":  len(resolved),
        "pending":   len(pending),
        "wins":      len(wins),
        "win_rate":  round(win_rate * 100, 1),
        "avg_ret":   round(avg_ret, 3),
        "pf":        round(pf, 2),
        "equity":    equity,
        "sym_wrs":   sym_wrs,
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
    if not active:
        return '<div class="empty-state">No active signals above threshold — market in distribution, wait for setup.</div>'

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


def build_carry_table(scan):
    rows = scan.get("carry_arb") or []
    if not rows:
        return '<div class="empty-state">No carry arb opportunities (funding rates neutral)</div>'
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
"""


# ── MAIN BUILDER ─────────────────────────────────────────────────────────────

def build_html(scan, signals, stats):
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scan_ts = scan.get("timestamp", "Never")
    n_online = scan.get("engines_online", 0)

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
    <div class="sub" style="margin-top:4px;font-size:10px">Auto-refreshes every 10 min &nbsp;|&nbsp; <a href="https://github.com/YNW1606/UNIVERSAL_CRYPTO_ARB" target="_blank">GitHub ↗</a></div>
  </div>
</div>

<!-- Stats Bar -->
<div class="stats-bar">
  <div class="stat-card">
    <div class="stat-label">Total Signals Logged</div>
    <div class="stat-val">{stats['total']}</div>
    <div class="stat-sub">{stats['resolved']} resolved · {stats['pending']} pending</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Win Rate</div>
    <div class="stat-val {wr_cls}">{wr}%</div>
    <div class="stat-sub">{stats['wins']} wins / {stats['resolved']} resolved</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Profit Factor</div>
    <div class="stat-val {pf_cls}">{pf:.2f}×</div>
    <div class="stat-sub">Gains ÷ Losses (need &gt;1.0)</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Avg Return / Signal</div>
    <div class="stat-val {ret_cls}">{stats['avg_ret']:+.3f}%</div>
    <div class="stat-sub">Net of 0.3% fee threshold</div>
  </div>
</div>

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
  <div class="section-title">Active Signals  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">|score| ≥ 35 (actionable)</span></div>
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

<!-- Signal History -->
<div class="section">
  <div class="section-title">Signal History  <span style="color:#8095a8;font-size:11px;text-transform:none;letter-spacing:0">last 50 signals</span></div>
  {build_history_table(signals)}
</div>

<div class="footer">
  Generated at {now_str} &nbsp;·&nbsp;
  GitHub Actions scan every ~1 min &nbsp;·&nbsp; Tracking {len(scan.get("watchlist") or [])} symbols &nbsp;·&nbsp;
  <a href="https://github.com/YNW1606/UNIVERSAL_CRYPTO_ARB" target="_blank">Source Code</a>
</div>

</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
{eq_js}
{wr_js}
</script>
</body>
</html>"""


def main():
    print("Loading scan data...")
    scan    = load_scan_json()
    signals = load_signals_db()
    stats   = compute_stats(signals)

    print(f"  {len(signals)} signals in DB, {stats['resolved']} resolved, win_rate={stats['win_rate']}%")
    print(f"  Active signals in JSON: {len(scan.get('active_signals') or [])}")

    html = build_html(scan, signals, stats)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written → {OUT_PATH}")
    print(f"GitHub Pages URL: https://YNW1606.github.io/UNIVERSAL_CRYPTO_ARB/")


if __name__ == "__main__":
    main()
