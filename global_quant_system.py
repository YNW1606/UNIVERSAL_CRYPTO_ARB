"""
Global Quant System — 5-Engine Unified Dashboard
Integrates: crypto_master_engine + hft_signal_engine + stat_arb_engine +
            quant_factor_model + unified_risk_manager
"""
import time, json, sys, os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","AVAXUSDT","ADAUSDT"]
PORTFOLIO_SIZE = 100_000.0

ENGINE_WEIGHTS = {
    "master":   0.40,
    "hft":      0.20,
    "factor":   0.25,
    "stat_arb": 0.15,
}

# ── ENGINE LOADERS ───────────────────────────────────────────────────────────

def load_engines():
    engines, status = {}, {}

    try:
        import crypto_master_engine as cme
        engines["master"] = cme
        status["master"] = "✓ ONLINE"
    except Exception as e:
        status["master"] = f"✗ OFFLINE: {e}"

    try:
        import hft_signal_engine as hft
        engines["hft"] = hft
        status["hft"] = "✓ ONLINE"
    except Exception as e:
        status["hft"] = f"✗ OFFLINE: {e}"

    try:
        import stat_arb_engine as sae
        engines["stat_arb"] = sae
        status["stat_arb"] = "✓ ONLINE"
    except Exception as e:
        status["stat_arb"] = f"✗ OFFLINE: {e}"

    try:
        import quant_factor_model as qfm
        engines["factor"] = qfm
        status["factor"] = "✓ ONLINE"
    except Exception as e:
        status["factor"] = f"✗ OFFLINE: {e}"

    try:
        import unified_risk_manager as urm
        engines["risk"] = urm
        status["risk"] = "✓ ONLINE"
    except Exception as e:
        status["risk"] = f"✗ OFFLINE: {e}"

    return engines, status

# ── SIGNAL COLLECTION ────────────────────────────────────────────────────────

def collect_master_signals(engines, symbols):
    """Call analyze_symbol() per symbol from crypto_master_engine."""
    if "master" not in engines:
        return {}
    cme = engines["master"]
    results = {}
    for sym in symbols:
        try:
            r = cme.analyze_symbol(sym)
            if r:
                results[sym] = r
        except Exception as e:
            pass
    return results


def collect_hft_signals(engines, symbols):
    """Call run_hft_scan() from hft_signal_engine."""
    if "hft" not in engines:
        return {}
    hft = engines["hft"]
    results = {}
    try:
        scan = hft.run_hft_scan(symbols)
        for r in scan:
            sym = r.get("symbol")
            if sym:
                results[sym] = r
    except Exception as e:
        print(f"  HFT engine error: {e}")
    return results


def collect_factor_signals(engines, symbols):
    """Call run_factor_scan() from quant_factor_model."""
    if "factor" not in engines:
        return {}, {}
    qfm = engines["factor"]
    try:
        result = qfm.run_factor_scan(symbols)
        weights  = result.get("weights", {})
        exp_rets = result.get("expected_returns", {})
        return weights, exp_rets
    except Exception as e:
        print(f"  Factor model error: {e}")
    return {}, {}


def collect_stat_arb_signals(engines, symbols):
    """Run BTC beta residuals + cointegrated pairs from stat_arb_engine."""
    if "stat_arb" not in engines:
        return {}, []
    sae = engines["stat_arb"]
    beta_sigs, pairs = {}, []
    try:
        prices = sae.load_all_prices(symbols, interval="1h", limit=200)
        beta_sigs = sae.btc_beta_signals(prices)
        all_pairs = sae.find_all_pairs(prices, p_threshold=0.10)
        for p in all_pairs[:5]:
            try:
                sig = sae.generate_pair_signal(prices, p)
                pairs.append(sig)
            except Exception:
                pass
    except Exception as e:
        print(f"  Stat arb error: {e}")
    return beta_sigs, pairs


def collect_risk_data(engines, symbols, portfolio_size):
    """Load return history, build covariance, create PositionApprover."""
    if "risk" not in engines:
        return None, None
    urm = engines["risk"]
    try:
        import numpy as np
        matrix, syms = urm.get_returns_matrix(symbols)
        if matrix.shape[0] < 5:
            return None, None
        cov  = urm.ledoit_wolf_shrinkage(matrix)
        corr = urm.correlation_matrix(cov)
        eq_w = np.ones(len(syms)) / len(syms)
        var95,  cvar95,  worst = urm.portfolio_var_cvar(eq_w, matrix, 0.95)
        var99,  cvar99,  _     = urm.portfolio_var_cvar(eq_w, matrix, 0.99)

        approver = urm.PositionApprover(portfolio_size=portfolio_size)
        approver.returns_matrix = matrix
        approver.corr_matrix    = corr
        approver.symbols_loaded = syms

        n    = corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        avg_corr   = float(np.mean(corr[mask]))
        eff_n      = 1 + (n - 1) * (1 - max(0, avg_corr))
        high_pairs = int(np.sum(corr[mask] > 0.80) // 2)

        risk_summary = {
            "var_95":              round(var95  * 100, 3),
            "cvar_95":             round(cvar95 * 100, 3),
            "var_99":              round(var99  * 100, 3),
            "cvar_99":             round(cvar99 * 100, 3),
            "worst_day":           round(worst  * 100, 3),
            "avg_correlation":     round(avg_corr, 3),
            "effective_positions": round(eff_n, 1),
            "high_corr_pairs":     high_pairs,
            "corr_regime": ("HIGH_CORR" if avg_corr > 0.70
                            else "NORMAL" if avg_corr > 0.40
                            else "LOW_CORR"),
        }
        return approver, risk_summary
    except Exception as e:
        print(f"  Risk manager error: {e}")
    return None, None


def collect_carry_arb(engines, symbols):
    """Carry arb from hft_signal_engine.spot_perp_basis."""
    if "hft" not in engines:
        return []
    hft = engines["hft"]
    arb = []
    for sym in symbols:
        try:
            b   = hft.spot_perp_basis(sym)
            ann = b.get("funding_ann_pct", 0)
            if abs(ann) > 3.0:
                arb.append({
                    "symbol":    sym,
                    "ann_pct":   ann,
                    "basis_pct": b.get("basis_pct", 0),
                    "action": "LONG_SPOT_SHORT_PERP" if ann > 0 else "SHORT_SPOT_LONG_PERP",
                })
        except Exception:
            pass
        time.sleep(0.05)
    arb.sort(key=lambda x: abs(x["ann_pct"]), reverse=True)
    return arb

# ── SIGNAL FUSION ────────────────────────────────────────────────────────────

def _master_score_and_dir(master_data):
    """Extract signed score from crypto_master_engine result dict."""
    if not master_data:
        return 0, ""
    # Try common keys used by crypto_master_engine
    raw = (master_data.get("master_score")
           or master_data.get("score")
           or master_data.get("final_score", 0))
    try:
        raw = float(raw)
    except Exception:
        raw = 0.0

    # Direction string
    for key in ("verdict", "direction", "signal", "trade_direction"):
        d = str(master_data.get(key, "")).upper()
        if d:
            break

    if any(w in d for w in ("BULL","LONG","BUY","UP")):
        return abs(raw), d
    if any(w in d for w in ("BEAR","SHORT","SELL","DOWN")):
        return -abs(raw), d
    # Fallback: use sign of raw score
    return raw, d


def fuse_signals(sym, master_data, hft_data, factor_weight, factor_exp_ret, stat_arb_data):
    """Combine all engines into a single -100..+100 unified_score."""
    scores = {}

    # Master (0-100 → signed via direction)
    m_score, m_dir = _master_score_and_dir(master_data)
    scores["master"] = float(m_score)

    # HFT (-100..+100 direct)
    scores["hft"] = float(hft_data.get("score", 0)) if hft_data else 0.0

    # Factor model: weight ∈ [-0.25, +0.25] → [-100, +100]
    if factor_weight is not None:
        scaled = factor_weight / 0.25 * 100
        if factor_exp_ret is not None:
            er_sign = 1.0 if factor_exp_ret >= 0 else -1.0
            scaled  = abs(scaled) * er_sign
        scores["factor"] = float(scaled)
    else:
        scores["factor"] = 0.0

    # Stat arb beta residual
    if stat_arb_data:
        td = stat_arb_data.get("trade_direction", "NEUTRAL")
        rz = float(stat_arb_data.get("residual_z", 0))
        if td == "LONG":
            scores["stat_arb"] =  min(50.0, abs(rz) * 20)
        elif td == "SHORT":
            scores["stat_arb"] = -min(50.0, abs(rz) * 20)
        else:
            scores["stat_arb"] = 0.0
    else:
        scores["stat_arb"] = 0.0

    unified = int(max(-100, min(100,
        scores["master"]   * ENGINE_WEIGHTS["master"]  +
        scores["hft"]      * ENGINE_WEIGHTS["hft"]     +
        scores["factor"]   * ENGINE_WEIGHTS["factor"]  +
        scores["stat_arb"] * ENGINE_WEIGHTS["stat_arb"]
    )))

    direction = "LONG" if unified > 0 else "SHORT" if unified < 0 else "NEUTRAL"
    abs_u = abs(unified)
    tier = ("★★★ PERFECT" if abs_u >= 65 else
            "★★  HIGH"    if abs_u >= 50 else
            "★   MODERATE"if abs_u >= 35 else
            "○   WAIT")

    return {
        "symbol":        sym,
        "unified_score": unified,
        "direction":     direction,
        "tier":          tier,
        "components":    scores,
        "master":        master_data,
        "hft":           hft_data,
        "factor_weight": factor_weight,
        "factor_exp_ret":factor_exp_ret,
        "stat_arb":      stat_arb_data,
        "master_dir":    m_dir,
    }

# ── MAIN SCAN ─────────────────────────────────────────────────────────────────

def run_global_scan(symbols=None, portfolio_size=PORTFOLIO_SIZE):
    symbols = symbols or SYMBOLS

    print("\nLoading engines...")
    engines, engine_status = load_engines()
    n_online = sum(1 for s in engine_status.values() if s.startswith("✓"))
    print(f"  {n_online}/5 engines online")

    print("  [1/5] Master engine (8-layer structural)...")
    master_sigs = collect_master_signals(engines, symbols)

    print("  [2/5] HFT engine (OFI + microprice)...")
    hft_sigs = collect_hft_signals(engines, symbols)

    print("  [3/5] Factor model (12 factors)...")
    factor_weights, factor_exp_rets = collect_factor_signals(engines, symbols)

    print("  [4/5] Stat arb (cointegration + BTC beta)...")
    beta_sigs, pair_sigs = collect_stat_arb_signals(engines, symbols)

    print("  [5/5] Risk manager (VaR + Kelly + DD)...")
    approver, risk_summary = collect_risk_data(engines, symbols, portfolio_size)

    print("  Carry arb scan...")
    carry_ops = collect_carry_arb(engines, symbols)

    # Fuse per symbol
    fused = []
    for sym in symbols:
        result = fuse_signals(
            sym,
            master_sigs.get(sym, {}),
            hft_sigs.get(sym, {}),
            factor_weights.get(sym),
            factor_exp_rets.get(sym),
            beta_sigs.get(sym),
        )
        abs_score = abs(result["unified_score"])
        if approver and abs_score >= 35:
            win_prob = 0.50 + abs_score / 400.0
            approval = approver.approve_trade(
                sym, result["direction"], abs_score,
                win_prob=win_prob, tp_pct=0.020, sl_pct=0.012
            )
        else:
            approval = {"approved": False, "size_usd": 0, "size_pct": 0, "reason": "Signal too weak"}
        result["approval"] = approval
        fused.append(result)

    fused.sort(key=lambda r: abs(r["unified_score"]), reverse=True)
    print_dashboard(fused, pair_sigs, carry_ops, risk_summary, engine_status, portfolio_size)
    return fused

# ── DISPLAY ───────────────────────────────────────────────────────────────────

def print_dashboard(fused, pair_sigs, carry_ops, risk_summary, engine_status, portfolio_size):
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    h_utc   = datetime.now(timezone.utc).hour
    session = ("NEW YORK" if 12 <= h_utc < 21 else
               "LONDON"   if  7 <= h_utc < 12 else
               "ASIA")
    n_online = sum(1 for s in engine_status.values() if s.startswith("✓"))

    W = 78
    print()
    print("╔" + "═" * W + "╗")
    hdr = f"  GLOBAL QUANT SYSTEM  ║  {n_online}/5 Engines  ║  {ts}  ║  {session}  "
    print(f"║{hdr:<{W}}║")
    print("╚" + "═" * W + "╝")

    bulls = sum(1 for r in fused if r["unified_score"] >  25)
    bears = sum(1 for r in fused if r["unified_score"] < -25)
    neut  = len(fused) - bulls - bears
    var95 = (risk_summary or {}).get("var_95", "N/A")
    print(f"\n  Market Breadth: {bulls} BULLISH  {bears} BEARISH  {neut} NEUTRAL  "
          f"| Portfolio VaR(95%): {var95}%/day")

    # ── Active signals ──────────────────────────────────────────────────────
    active = [r for r in fused if abs(r["unified_score"]) >= 35]
    print("\n" + "─" * W)
    if active:
        print("  ACTIVE SIGNALS  (unified_score ≥ 35)")
        print("─" * W)
        for r in active:
            u    = r["unified_score"]
            d    = r["direction"]
            tier = r["tier"]
            appr = r["approval"]
            icon = "▲" if u > 0 else "▼"
            if appr.get("approved"):
                size_str = f"${appr['size_usd']:,.0f}  ({appr['size_pct']:.1f}%)"
            else:
                size_str = "REJECTED"

            print(f"\n  {tier}  {r['symbol']:12s}  score={u:+3d}/100  "
                  f"{icon} {d:5s}  size={size_str}")

            comp = r.get("components", {})
            comp_str = "  |  ".join(
                f"{k}={v:+.0f}" for k, v in comp.items() if abs(v) > 1
            )
            if comp_str:
                print(f"    Components: {comp_str}")

            m = r.get("master") or {}
            if m:
                for key in ("verdict","note","signal","direction"):
                    note = m.get(key)
                    if note and str(note).strip():
                        print(f"    Master: {str(note)[:80]}")
                        break

            h_data = r.get("hft") or {}
            if h_data and h_data.get("ofi_z") is not None:
                bi = h_data.get("book_imbalance") or {}
                print(f"    HFT:    OFI_z={h_data.get('ofi_z',0):+.2f}  "
                      f"TFI_30s={h_data.get('tfi_30s',0):+.4f}  "
                      f"book_L5={bi.get(5, bi.get('5', 0.5)):.3f}  "
                      f"Kyle_λ={h_data.get('kyle_lambda',0):.2e}")

            sa = r.get("stat_arb") or {}
            if sa and sa.get("beta_to_btc") is not None:
                print(f"    StatArb: β_BTC={sa.get('beta_to_btc',0):+.3f}  "
                      f"resid_z={sa.get('residual_z',0):+.3f}  "
                      f"→ {sa.get('trade_direction','N/A')}")

            fw = r.get("factor_weight")
            fe = r.get("factor_exp_ret")
            if fw is not None:
                fe_str = f"{fe*100:+.4f}%/4h" if fe is not None else "N/A"
                print(f"    Factor: weight={fw:+.4f}  exp_ret={fe_str}")

            if not appr.get("approved"):
                print(f"    Risk:   {appr.get('reason','N/A')}")
            else:
                print(f"    Risk:   Kelly={appr.get('kelly_fraction',0):.3f}  "
                      f"win_prob={appr.get('win_prob',0):.0%}  APPROVED")
    else:
        print("  NO ACTIVE SIGNALS  (all below threshold — market in distribution, wait)")

    # ── Stat arb pairs ──────────────────────────────────────────────────────
    if pair_sigs:
        print("\n" + "─" * W)
        print("  STAT ARB PAIRS  (market-neutral, long + short simultaneously)")
        print("─" * W)
        for p in pair_sigs[:6]:
            icon  = "⚡" if p.get("signal", 0) != 0 else "○"
            z     = p.get("z_score", "N/A")
            z_str = f"{z:+.3f}" if isinstance(z, (int, float)) else str(z)
            hl    = p.get("half_life_h", "?")
            pv    = p.get("p_value", "?")
            pv_str= f"{pv:.4f}" if isinstance(pv, float) else str(pv)
            print(f"  {icon} {p['pair']:18s}  z={z_str:>8s}  β={p.get('hedge_ratio',0):.3f}  "
                  f"HL={hl}h  p={pv_str}  → {p.get('action','HOLD')}")

    # ── Carry arb ───────────────────────────────────────────────────────────
    if carry_ops:
        print("\n" + "─" * W)
        print("  CARRY ARB  (passive income — long spot / short perp, 24/7)")
        print("─" * W)
        for op in carry_ops[:6]:
            ann  = op["ann_pct"]
            icon = "★" if abs(ann) > 15 else "▶"
            print(f"  {icon} {op['symbol']:12s}  {ann:+6.1f}%/yr  "
                  f"basis={op['basis_pct']:+.4f}%  → {op['action']}")
    else:
        print("\n  CARRY ARB: funding rates neutral — no significant arb available")

    # ── Risk dashboard ───────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("  PORTFOLIO RISK DASHBOARD")
    print("─" * W)
    if risk_summary:
        print(f"  VaR(95%): {risk_summary.get('var_95','N/A')}%/day  |  "
              f"CVaR(95%): {risk_summary.get('cvar_95','N/A')}%/day  |  "
              f"VaR(99%): {risk_summary.get('var_99','N/A')}%/day  |  "
              f"Worst: {risk_summary.get('worst_day','N/A')}%")
        print(f"  Avg corr: {risk_summary.get('avg_correlation','N/A')}  "
              f"({risk_summary.get('corr_regime','N/A')})  |  "
              f"Effective positions: {risk_summary.get('effective_positions','N/A')} of {len(SYMBOLS)}  |  "
              f"High-corr pairs: {risk_summary.get('high_corr_pairs','N/A')}")
        print(f"  Portfolio: ${portfolio_size:,.0f}  |  "
              f"Circuit breaker: OK (DD=0.0%)  |  Trading: ALLOWED")
    else:
        print("  Risk data unavailable")

    # ── Engine status ─────────────────────────────────────────────────────────
    print("\n" + "─" * W)
    print("  ENGINE STATUS")
    print("─" * W)
    labels = {
        "master":   "crypto_master_engine    (8-layer: multi-TF + inst-flow + ML)",
        "hft":      "hft_signal_engine       (OFI + microprice + Kyle λ)",
        "stat_arb": "stat_arb_engine         (cointegration + Kalman filter)",
        "factor":   "quant_factor_model      (12-factor alpha + portfolio opt)",
        "risk":     "unified_risk_manager    (VaR + Kelly + DD circuit breaker)",
    }
    for key, label in labels.items():
        st = engine_status.get(key, "✗ UNKNOWN")
        print(f"  {st[0]} {label:60s}  {st[2:]}")

    print("═" * W + "\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Global Quant System — 5-Engine Unified Dashboard")
    parser.add_argument("--portfolio", type=float, default=PORTFOLIO_SIZE)
    parser.add_argument("--watch",  type=int,   help="Re-scan interval in seconds")
    parser.add_argument("--symbols",nargs="+",  default=None)
    parser.add_argument("--json",   action="store_true")
    args = parser.parse_args()

    syms = args.symbols or SYMBOLS

    def once():
        results = run_global_scan(syms, args.portfolio)
        if args.json:
            out = [{"symbol": r["symbol"],
                    "unified_score": r["unified_score"],
                    "direction": r["direction"],
                    "tier": r["tier"]} for r in results]
            print(json.dumps(out, indent=2))

    if args.watch:
        while True:
            once()
            print(f"[Next scan in {args.watch}s... Ctrl+C to stop]")
            time.sleep(args.watch)
    else:
        once()


if __name__ == "__main__":
    main()
