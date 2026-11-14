import numpy as np, requests, time, json, os
from datetime import datetime, timezone
from collections import deque

BINANCE_SPOT = "https://api.binance.com"
SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","AVAXUSDT","ADAUSDT"]

DD_WARNING_PCT   = 0.07
DD_LIMIT_PCT     = 0.10
DD_HARD_STOP_PCT = 0.15
MAX_CORR_EXPOSURE = 0.60
KELLY_SHRINKAGE  = 0.25
MAX_SINGLE_POS   = 0.15
RISK_FREE_RATE   = 0.045

# Correlation-adjusted sizing: above this pairwise corr with open positions,
# new position size is reduced proportionally (up to 50% at corr=1.0).
NEW_POS_CORR_LIMIT = 0.80

# Drawdown acceleration brake: this many consecutive negative PnL days
# cuts all new position sizes by LOSS_STREAK_SIZE_CUT until a green day.
LOSS_STREAK_DAYS     = 3
LOSS_STREAK_SIZE_CUT = 0.40

CACHE_DIR = "risk_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ── DATA FETCHING ─────────────────────────────────────────────────────────────

def fetch_klines(symbol, interval="1d", limit=90):
    from data_utils import fetch_klines as _fetch
    return _fetch(symbol, interval, limit, cache_dir=CACHE_DIR, cache_secs=3600)


def get_returns_matrix(symbols, interval="1d", limit=90):
    """Returns (returns_matrix [n_bars-1, n_assets], symbols_used)."""
    all_returns = {}
    for sym in symbols:
        try:
            klines = fetch_klines(sym, interval, limit)
            closes = np.array([float(k[4]) for k in klines])
            if len(closes) < 2:
                continue
            log_ret = np.diff(np.log(closes))
            all_returns[sym] = log_ret
        except Exception as e:
            print(f"  {sym}: {e}")

    if not all_returns:
        return np.zeros((0, 0)), []

    min_len = min(len(r) for r in all_returns.values())
    syms = list(all_returns.keys())
    matrix = np.column_stack([all_returns[s][-min_len:] for s in syms])
    return matrix, syms


# ── LEDOIT-WOLF SHRINKAGE COV (pure numpy) ────────────────────────────────────

def ledoit_wolf_shrinkage(X):
    """OAS-style shrinkage toward scaled identity. X: (n_samples, n_assets)."""
    n, p = X.shape
    if p == 1:
        return np.array([[float(np.var(X, ddof=1))]])

    sample_cov = np.cov(X, rowvar=False)

    trace_S = np.trace(sample_cov)
    mu_target = trace_S / p
    target = mu_target * np.eye(p)

    diff = sample_cov - target
    denom = np.sum(diff ** 2)

    if denom < 1e-20:
        return sample_cov

    beta_sq = 0.0
    for i in range(n):
        xi = X[i, :]
        outer = np.outer(xi, xi)
        dev = outer - sample_cov
        beta_sq += np.sum(dev ** 2)
    beta_sq /= (n ** 2)

    alpha = float(np.clip(beta_sq / denom, 0.0, 1.0))
    return (1.0 - alpha) * sample_cov + alpha * target


def correlation_matrix(cov):
    stds = np.sqrt(np.diag(cov))
    stds = np.where(stds < 1e-10, 1e-10, stds)
    corr = cov / np.outer(stds, stds)
    return np.clip(corr, -1.0, 1.0)


# ── VAR AND CVAR ──────────────────────────────────────────────────────────────

def portfolio_var_cvar(weights, returns_matrix, confidence=0.95):
    """Historical VaR and CVaR. Returns (var, cvar, worst_day) as fractions."""
    if returns_matrix.shape[0] < 10:
        return 0.05, 0.08, 0.10

    w = np.array(weights, dtype=float)
    port_rets = returns_matrix @ w
    sorted_rets = np.sort(port_rets)

    var_idx = int(np.floor((1.0 - confidence) * len(sorted_rets)))
    var_idx = max(0, min(var_idx, len(sorted_rets) - 1))
    var = -float(sorted_rets[var_idx])

    tail = sorted_rets[:var_idx + 1]
    cvar = -float(np.mean(tail)) if len(tail) > 0 else var
    worst_day = -float(sorted_rets[0])

    return max(0.0, var), max(0.0, cvar), max(0.0, worst_day)


def marginal_var(weights, returns_matrix, confidence=0.95):
    """Component VaR contributions for each asset."""
    cov = ledoit_wolf_shrinkage(returns_matrix)
    w = np.array(weights, dtype=float)
    port_std = float(np.sqrt(max(w @ cov @ w, 1e-20)))
    z = 1.645  # 95% one-tailed
    marginal = z * (cov @ w) / port_std
    return w * marginal


# ── KELLY SIZING ──────────────────────────────────────────────────────────────

def kelly_position_size(win_prob, win_return, loss_return,
                        shrinkage=KELLY_SHRINKAGE,
                        max_size=MAX_SINGLE_POS):
    """Fractional Kelly. loss_return must be negative (e.g. -0.01)."""
    if win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    if loss_return >= 0.0:
        return float(max_size)

    b = abs(win_return / loss_return)
    loss_prob = 1.0 - win_prob
    full_kelly = (win_prob * b - loss_prob) / b
    fractional = full_kelly * shrinkage
    return float(np.clip(fractional, 0.0, max_size))


def correlation_adjusted_size(base_size, corr_matrix, n_positions):
    """Scale down position size in a highly correlated portfolio."""
    if n_positions <= 1:
        return float(base_size)

    n = corr_matrix.shape[0]
    if n < 2:
        return float(base_size)

    mask = ~np.eye(n, dtype=bool)
    avg_corr = float(np.mean(corr_matrix[mask]))
    avg_corr = max(0.0, avg_corr)

    effective_n = 1.0 + (n_positions - 1) * (1.0 - avg_corr)
    effective_n = max(1.0, effective_n)
    scale = effective_n / n_positions

    return float(np.clip(base_size * scale, 0.0, MAX_SINGLE_POS))


# ── DRAWDOWN CIRCUIT BREAKER ──────────────────────────────────────────────────

class DrawdownController:
    def __init__(self, initial_equity=100_000.0,
                 warn_pct=DD_WARNING_PCT,
                 halt_pct=DD_LIMIT_PCT,
                 hard_stop_pct=DD_HARD_STOP_PCT):
        self.equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.warn_pct = warn_pct
        self.halt_pct = halt_pct
        self.hard_stop_pct = hard_stop_pct
        self.equity_history = deque([float(initial_equity)], maxlen=1000)
        self.halted = False
        self.halt_reason = ""
        self.halt_time = None
        self.cooldown_hours = 24

    def update(self, new_equity):
        """Returns (status, message, action)."""
        self.equity = float(new_equity)
        self.equity_history.append(self.equity)
        self.peak_equity = max(self.peak_equity, self.equity)

        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0

        if self.halted and self.halt_time:
            elapsed = (datetime.now(timezone.utc) - self.halt_time).total_seconds() / 3600
            if elapsed >= self.cooldown_hours:
                self.halted = False
                self.halt_reason = ""
                self.halt_time = None

        if dd >= self.hard_stop_pct:
            self.halted = True
            self.halt_reason = f"HARD STOP: DD={dd:.1%} >= {self.hard_stop_pct:.0%}"
            self.halt_time = datetime.now(timezone.utc)
            return "HARD_STOP", self.halt_reason, "CLOSE_ALL"

        if dd >= self.halt_pct:
            self.halted = True
            self.halt_reason = f"DD HALT: DD={dd:.1%} >= {self.halt_pct:.0%}"
            self.halt_time = datetime.now(timezone.utc)
            return "HALT", self.halt_reason, "NO_NEW_TRADES"

        if self.halted:
            return "HALTED", self.halt_reason, "NO_NEW_TRADES"

        if dd >= self.warn_pct:
            return "WARNING", f"DD={dd:.1%} approaching limit {self.halt_pct:.0%}", "REDUCE_SIZE"

        return "OK", f"DD={dd:.1%}", "NORMAL"

    def current_drawdown(self):
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    def max_drawdown_history(self):
        eq = list(self.equity_history)
        if len(eq) < 2:
            return 0.0
        peak = eq[0]
        max_dd = 0.0
        for e in eq:
            peak = max(peak, e)
            if peak > 0:
                dd = (peak - e) / peak
                max_dd = max(max_dd, dd)
        return max_dd

    def status_dict(self):
        dd = self.current_drawdown()
        max_dd = self.max_drawdown_history()
        return {
            "equity": round(self.equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "current_drawdown_pct": round(dd * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "trading_allowed": not self.halted,
            "warn_threshold_pct": round(self.warn_pct * 100, 1),
            "halt_threshold_pct": round(self.halt_pct * 100, 1),
        }


# ── POSITION APPROVER ─────────────────────────────────────────────────────────

class PositionApprover:
    """Single entry point for all position sizing decisions."""

    def __init__(self, portfolio_size=100_000.0):
        self.portfolio_size = float(portfolio_size)
        self.dd_controller = DrawdownController(self.portfolio_size)
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.returns_matrix = None
        self.corr_matrix = None
        self.symbols_loaded = []
        # Drawdown acceleration brake state
        self.daily_pnl_history = deque(maxlen=30)
        self.consecutive_loss_days = 0

    # ── Drawdown acceleration brake ──────────────────────────────────────────

    def record_daily_pnl(self, day_pnl):
        """
        Record one closed trading day's PnL. 3+ consecutive negative days
        activates the drawdown acceleration brake (-40% on all new sizes)
        until a positive day resets it.
        """
        try:
            day_pnl = float(day_pnl)
            self.daily_pnl_history.append(day_pnl)
            if day_pnl < 0:
                self.consecutive_loss_days += 1
            else:
                self.consecutive_loss_days = 0
        except Exception:
            pass

    def _loss_streak_brake_active(self):
        return self.consecutive_loss_days >= LOSS_STREAK_DAYS

    # ── Correlation-adjusted new-position sizing ─────────────────────────────

    def _max_corr_with_open_positions(self, symbol):
        """
        Max pairwise correlation between the candidate symbol and currently
        open positions, from the loaded correlation matrix.
        Returns 0.0 when unavailable (never raises).
        """
        try:
            if (self.corr_matrix is None or not self.open_positions
                    or not self.symbols_loaded):
                return 0.0
            if symbol not in self.symbols_loaded:
                return 0.0
            i = self.symbols_loaded.index(symbol)
            max_corr = 0.0
            for open_sym in self.open_positions:
                if open_sym == symbol or open_sym not in self.symbols_loaded:
                    continue
                j = self.symbols_loaded.index(open_sym)
                max_corr = max(max_corr, float(self.corr_matrix[i, j]))
            return max_corr
        except Exception:
            return 0.0

    def load_risk_data(self, symbols=None):
        syms = symbols or SYMBOLS
        try:
            self.returns_matrix, self.symbols_loaded = get_returns_matrix(syms)
            if self.returns_matrix.shape[0] > 5:
                cov = ledoit_wolf_shrinkage(self.returns_matrix)
                self.corr_matrix = correlation_matrix(cov)
        except Exception as e:
            print(f"  Risk data load error: {e}")

    def approve_trade(self, symbol, direction, signal_score,
                      win_prob=0.55, tp_pct=0.02, sl_pct=0.01):
        status, msg, action = self.dd_controller.update(self._current_equity())

        if action == "CLOSE_ALL":
            return {"approved": False, "reason": f"CIRCUIT BREAKER: {msg}", "size_usd": 0}
        if action == "NO_NEW_TRADES":
            return {"approved": False, "reason": f"HALTED: {msg}", "size_usd": 0}

        if signal_score < 45:
            return {"approved": False, "reason": f"Signal too weak: {signal_score}/100", "size_usd": 0}

        base_kelly = kelly_position_size(win_prob, tp_pct, -sl_pct)
        score_scale = float(np.clip(0.5 + (signal_score - 45) / 100.0, 0.5, 1.5))
        kelly_frac = min(base_kelly * score_scale, MAX_SINGLE_POS)

        n_open = len(self.open_positions)
        if self.corr_matrix is not None and n_open > 0:
            kelly_frac = correlation_adjusted_size(kelly_frac, self.corr_matrix, n_open + 1)

        # Correlation-adjusted new-position sizing: if this symbol is highly
        # correlated (> NEW_POS_CORR_LIMIT) with any existing position, cut the
        # new size by (corr - 0.80) / 0.20 x 50% (max 50% cut at corr = 1.0).
        corr_with_open = self._max_corr_with_open_positions(symbol)
        corr_reduction = 0.0
        if corr_with_open > NEW_POS_CORR_LIMIT:
            corr_reduction = min(
                (corr_with_open - NEW_POS_CORR_LIMIT) / (1.0 - NEW_POS_CORR_LIMIT) * 0.50,
                0.50)
            kelly_frac *= (1.0 - corr_reduction)

        # Drawdown acceleration brake: 3+ consecutive red days → -40% sizing.
        loss_brake = self._loss_streak_brake_active()
        if loss_brake:
            kelly_frac *= (1.0 - LOSS_STREAK_SIZE_CUT)

        if action == "REDUCE_SIZE":
            kelly_frac *= 0.5

        size_usd = min(kelly_frac * self.portfolio_size,
                       self.portfolio_size * MAX_SINGLE_POS)

        return {
            "approved": True,
            "symbol": symbol,
            "direction": direction,
            "size_usd": round(size_usd, 2),
            "size_pct": round(kelly_frac * 100, 2),
            "kelly_fraction": round(kelly_frac, 4),
            "win_prob": win_prob,
            "portfolio_size": self.portfolio_size,
            "signal_score": signal_score,
            "dd_status": status,
            "corr_with_open": round(corr_with_open, 3),
            "corr_size_cut_pct": round(corr_reduction * 100, 1),
            "loss_streak_brake": loss_brake,
            "consecutive_loss_days": self.consecutive_loss_days,
            "reason": "APPROVED" + (" [LOSS-STREAK BRAKE -40%]" if loss_brake else "")
                      + (f" [CORR CUT -{corr_reduction*100:.0f}%]" if corr_reduction > 0 else ""),
        }

    def _current_equity(self):
        return self.portfolio_size + self.daily_pnl

    def add_position(self, symbol, size_usd, direction, entry_price):
        self.open_positions[symbol] = {
            "size_usd": size_usd, "direction": direction,
            "entry": entry_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def remove_position(self, symbol, exit_price=None):
        if symbol in self.open_positions:
            pos = self.open_positions.pop(symbol)
            if exit_price and pos.get("entry") and pos["entry"] > 0:
                pnl_pct = (exit_price - pos["entry"]) / pos["entry"]
                if pos["direction"] == "SHORT":
                    pnl_pct = -pnl_pct
                self.daily_pnl += pnl_pct * pos["size_usd"]

    def risk_report(self):
        dd_status = self.dd_controller.status_dict()
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_size": self.portfolio_size,
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_pnl_pct": round(self.daily_pnl / self.portfolio_size * 100, 3),
            "open_positions": len(self.open_positions),
            "positions": self.open_positions,
            "drawdown": dd_status,
            "loss_streak": {
                "consecutive_loss_days": self.consecutive_loss_days,
                "brake_active": self._loss_streak_brake_active(),
                "size_cut_pct": LOSS_STREAK_SIZE_CUT * 100 if self._loss_streak_brake_active() else 0.0,
            },
        }

        if self.returns_matrix is not None and self.returns_matrix.shape[0] > 5:
            n = self.returns_matrix.shape[1]
            eq_w = np.ones(n) / n
            var95, cvar95, worst = portfolio_var_cvar(eq_w, self.returns_matrix, 0.95)
            var99, cvar99, _    = portfolio_var_cvar(eq_w, self.returns_matrix, 0.99)
            report["risk_metrics"] = {
                "var_95_pct":  round(var95  * 100, 3),
                "cvar_95_pct": round(cvar95 * 100, 3),
                "var_99_pct":  round(var99  * 100, 3),
                "cvar_99_pct": round(cvar99 * 100, 3),
                "worst_day_pct": round(worst * 100, 3),
            }

            if self.corr_matrix is not None:
                nc = self.corr_matrix.shape[0]
                mask = ~np.eye(nc, dtype=bool)
                avg_corr = float(np.mean(self.corr_matrix[mask]))
                high_corr_pairs = int(np.sum(self.corr_matrix[mask] > 0.80) // 2)
                report["correlation"] = {
                    "avg_pairwise": round(avg_corr, 3),
                    "high_corr_pairs": high_corr_pairs,
                    "effective_positions": round(1 + (nc - 1) * (1 - avg_corr), 2),
                    "regime": ("HIGH_CORR_MARKET" if avg_corr > 0.7
                               else "NORMAL_CORR" if avg_corr > 0.4
                               else "LOW_CORR_OPPORTUNITY"),
                }

        return report


# ── STANDALONE SCAN ───────────────────────────────────────────────────────────

def run_risk_scan(symbols=None, portfolio_size=100_000.0):
    syms = symbols or SYMBOLS

    print(f"Loading 90-day return history for {len(syms)} symbols...")
    returns_matrix, symbols_used = get_returns_matrix(syms, interval="1d", limit=90)

    if returns_matrix.shape[0] < 10:
        print("Insufficient data for risk analysis.")
        return

    n = len(symbols_used)
    print(f"  Loaded: {returns_matrix.shape[0]} days x {n} assets")

    cov  = ledoit_wolf_shrinkage(returns_matrix)
    corr = correlation_matrix(cov)
    eq_w = np.ones(n) / n

    var95, cvar95, worst = portfolio_var_cvar(eq_w, returns_matrix, 0.95)
    var99, cvar99, _     = portfolio_var_cvar(eq_w, returns_matrix, 0.99)

    print("\n" + "=" * 70)
    print(f"  UNIFIED RISK MANAGER  |  {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")
    print("=" * 70)

    print("\n-- PORTFOLIO VAR/CVAR (Equal-Weight Baseline) ----------------------")
    print(f"  Daily VaR  95%: {var95*100:6.3f}%  = ${var95*portfolio_size:>12,.0f}")
    print(f"  Daily CVaR 95%: {cvar95*100:6.3f}%  = ${cvar95*portfolio_size:>12,.0f}  <- expected loss worst 5% days")
    print(f"  Daily VaR  99%: {var99*100:6.3f}%  = ${var99*portfolio_size:>12,.0f}")
    print(f"  Daily CVaR 99%: {cvar99*100:6.3f}%  = ${cvar99*portfolio_size:>12,.0f}  <- expected loss worst 1% days")
    print(f"  Worst single day: {worst*100:6.3f}%  = ${worst*portfolio_size:>12,.0f}")

    print("\n-- CORRELATION MATRIX ----------------------------------------------")
    header = f"{'':8s}" + "".join(f"{s[:6]:>8s}" for s in symbols_used)
    print(header)
    for i, si in enumerate(symbols_used):
        row = f"{si[:8]:8s}"
        for j in range(n):
            row += f"{corr[i,j]:+8.3f}"
        print(row)

    mask = ~np.eye(n, dtype=bool)
    avg_corr     = float(np.mean(corr[mask]))
    high_corr    = int(np.sum(corr[mask] > 0.80) // 2)
    effective_pos = 1.0 + (n - 1) * (1.0 - avg_corr)

    print(f"\n  Avg pairwise correlation : {avg_corr:.3f}")
    print(f"  High-correlation pairs (>0.80): {high_corr}")
    print(f"  Effective independent positions: {effective_pos:.1f} of {n}")
    if avg_corr > 0.70:
        print("  WARNING: HIGH CORRELATION REGIME -- reduce sizes, broad selloff risk")

    print("\n-- KELLY POSITION SIZING GUIDE -------------------------------------")
    print(f"  {'Score':>6s}  {'WinProb':>8s}  {'TP':>5s}  {'SL':>5s}  "
          f"{'Full Kelly':>11s}  {'Frac Kelly':>11s}  {'USD($100k)':>12s}")
    test_cases = [
        (80, 0.60, 0.025, 0.012),
        (70, 0.57, 0.020, 0.012),
        (60, 0.54, 0.018, 0.012),
        (50, 0.52, 0.015, 0.010),
    ]
    for score, wp, tp, sl in test_cases:
        fk = kelly_position_size(wp, tp, -sl, shrinkage=1.0)
        qk = kelly_position_size(wp, tp, -sl, shrinkage=KELLY_SHRINKAGE)
        usd = qk * portfolio_size
        print(f"  {score:>6d}  {wp:>8.0%}  {tp:>5.1%}  {sl:>5.1%}  "
              f"{fk:>11.2%}  {qk:>11.2%}  ${usd:>11,.0f}")
    print(f"\n  Kelly shrinkage factor: {KELLY_SHRINKAGE:.0%}  (25% of full Kelly = ruin avoidance)")

    print("\n-- POSITION APPROVER - SAMPLE APPROVALS ----------------------------")
    approver = PositionApprover(portfolio_size=portfolio_size)
    approver.returns_matrix = returns_matrix
    approver.corr_matrix    = corr
    approver.symbols_loaded = symbols_used

    test_trades = [
        ("BTCUSDT", "LONG",  82),
        ("ETHUSDT", "LONG",  71),
        ("SOLUSDT", "SHORT", 55),
        ("XRPUSDT", "LONG",  35),
    ]
    for sym, direction, score in test_trades:
        result = approver.approve_trade(sym, direction, score)
        icon = "OK" if result["approved"] else "NO"
        if result["approved"]:
            print(f"  [{icon}] {sym:12s} {direction:5s} score={score:3d}  "
                  f"-> SIZE {result['size_pct']:5.1f}%  ${result['size_usd']:>10,.0f}  "
                  f"Kelly={result['kelly_fraction']:.4f}")
        else:
            print(f"  [{icon}] {sym:12s} {direction:5s} score={score:3d}  "
                  f"-> REJECTED: {result['reason']}")

    print("\n-- CIRCUIT BREAKER THRESHOLDS --------------------------------------")
    print(f"  Warning at:   {DD_WARNING_PCT:.0%} drawdown -> Reduce position size 50%")
    print(f"  Halt at:      {DD_LIMIT_PCT:.0%} drawdown -> No new trades allowed")
    print(f"  Hard stop at: {DD_HARD_STOP_PCT:.0%} drawdown -> Close all, 24h cooldown")
    print("=" * 70 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified Risk Manager")
    parser.add_argument("--portfolio", type=float, default=100_000.0,
                        help="Portfolio size in USD")
    parser.add_argument("--json", action="store_true", help="JSON risk metrics output")
    args = parser.parse_args()

    if args.json:
        matrix, syms = get_returns_matrix(SYMBOLS)
        if matrix.shape[0] < 5:
            print(json.dumps({"error": "insufficient data"}))
            return
        cov  = ledoit_wolf_shrinkage(matrix)
        corr = correlation_matrix(cov)
        n    = len(syms)
        eq_w = np.ones(n) / n
        var95, cvar95, worst = portfolio_var_cvar(eq_w, matrix, 0.95)
        var99, cvar99, _     = portfolio_var_cvar(eq_w, matrix, 0.99)
        mask = ~np.eye(n, dtype=bool)
        print(json.dumps({
            "var_95":  round(var95  * 100, 3),
            "cvar_95": round(cvar95 * 100, 3),
            "var_99":  round(var99  * 100, 3),
            "cvar_99": round(cvar99 * 100, 3),
            "worst_day": round(worst * 100, 3),
            "avg_pairwise_corr": round(float(np.mean(corr[mask])), 3),
        }, indent=2))
    else:
        run_risk_scan(portfolio_size=args.portfolio)


if __name__ == "__main__":
    main()
