import numpy as np, requests, time, json, os
from collections import deque
from datetime import datetime, timezone

BINANCE_SPOT = "https://api.binance.com"
BINANCE_FAPI = "https://fapi.binance.com"
DEFAULT_SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","AVAXUSDT","ADAUSDT"]


# ── DATA FETCHING ────────────────────────────────────────────────────────────

def fetch_orderbook(symbol, depth=20):
    """Returns bids=[[price,qty],...] asks=[[price,qty],...]"""
    r = requests.get(f"{BINANCE_SPOT}/api/v3/depth",
                     params={"symbol": symbol, "limit": depth}, timeout=5)
    d = r.json()
    bids = [[float(p), float(q)] for p, q in d.get("bids", [])]
    asks = [[float(p), float(q)] for p, q in d.get("asks", [])]
    return bids, asks


def fetch_agg_trades(symbol, limit=1000):
    """Returns list of dicts: {price, qty, is_buyer_maker, time_ms}"""
    r = requests.get(f"{BINANCE_SPOT}/api/v3/aggTrades",
                     params={"symbol": symbol, "limit": limit}, timeout=5)
    trades = []
    for t in r.json():
        trades.append({
            "price": float(t["p"]),
            "qty":   float(t["q"]),
            "is_buyer_maker": bool(t["m"]),
            "time_ms": int(t["T"])
        })
    return trades


def fetch_funding(symbol):
    """Returns {lastFundingRate, nextFundingTime, markPrice, indexPrice}"""
    r = requests.get(f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
                     params={"symbol": symbol}, timeout=5)
    return r.json()


def fetch_perp_price(symbol):
    r = requests.get(f"{BINANCE_FAPI}/fapi/v1/ticker/price",
                     params={"symbol": symbol}, timeout=5)
    return float(r.json().get("price", 0))


# ── OFI CALCULATOR ──────────────────────────────────────────────────────────

class OFICalculator:
    """Cont, Kukanov, Stoikov (2014) Order Flow Imbalance."""

    def __init__(self, maxlen=2000):
        self.history = deque(maxlen=maxlen)
        self._prev_bid = self._prev_bid_vol = None
        self._prev_ask = self._prev_ask_vol = None

    def update(self, bid, bid_vol, ask, ask_vol):
        if self._prev_bid is None:
            self._prev_bid, self._prev_bid_vol = bid, bid_vol
            self._prev_ask, self._prev_ask_vol = ask, ask_vol
            self.history.append(0.0)
            return 0.0

        bid_ofi = (bid_vol - self._prev_bid_vol) if bid >= self._prev_bid else -self._prev_bid_vol
        ask_ofi = (ask_vol - self._prev_ask_vol) if ask <= self._prev_ask else self._prev_ask_vol
        ofi = bid_ofi - ask_ofi

        self.history.append(float(ofi))
        self._prev_bid, self._prev_bid_vol = bid, bid_vol
        self._prev_ask, self._prev_ask_vol = ask, ask_vol
        return float(ofi)

    def cumulative(self, window=200):
        h = list(self.history)
        return float(np.sum(h[-window:]))

    def zscore(self, window=500):
        h = list(self.history)[-window:]
        if len(h) < 20:
            return 0.0
        arr = np.array(h)
        std = np.std(arr)
        return float((arr[-1] - np.mean(arr)) / std) if std > 1e-10 else 0.0

    def momentum(self, short=50, long_w=200):
        h = list(self.history)
        if len(h) < long_w:
            return 0.0
        return float(np.mean(h[-short:]) - np.mean(h[-long_w:-short]))


# ── MICROPRICE & BOOK IMBALANCE ──────────────────────────────────────────────

def compute_microprice(bids, asks):
    """Probability-weighted midpoint."""
    if not bids or not asks:
        return None
    bp, bq = bids[0][0], bids[0][1]
    ap, aq = asks[0][0], asks[0][1]
    total = bq + aq
    return float((bq * ap + aq * bp) / total) if total > 0 else float((bp + ap) / 2)


def compute_book_imbalance(bids, asks, levels=(1, 5, 10, 20)):
    """Returns {1: 0.62, 5: 0.58, 10: 0.55, 20: 0.52}"""
    result = {}
    for L in levels:
        bid_vol = sum(b[1] for b in bids[:L])
        ask_vol = sum(a[1] for a in asks[:L])
        total = bid_vol + ask_vol
        result[L] = float(bid_vol / total) if total > 0 else 0.5
    return result


def book_pressure(bids, asks, depth=10):
    """
    Total bid USD vs ask USD within depth levels.
    Returns (bid_usd, ask_usd, imbalance)
    """
    bid_usd = sum(p * q for p, q in bids[:depth])
    ask_usd = sum(p * q for p, q in asks[:depth])
    total = bid_usd + ask_usd
    imbalance = (bid_usd - ask_usd) / total if total > 0 else 0.0
    return bid_usd, ask_usd, imbalance


# ── TRADE FLOW IMBALANCE ─────────────────────────────────────────────────────

class TradeFlowTracker:
    WINDOWS_MS = [500, 2000, 10000, 30000, 60000]

    def __init__(self):
        self.trades = deque(maxlen=5000)   # (time_ms, qty, is_buy)

    def add(self, time_ms, qty, is_buyer_maker):
        is_buy = not is_buyer_maker   # aggressor is buyer when maker=False
        self.trades.append((time_ms, float(qty), is_buy))

    def imbalance(self, window_ms, now_ms=None):
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_ms
        buy = sell = 0.0
        for ts, qty, is_buy in self.trades:
            if ts >= cutoff:
                if is_buy:
                    buy += qty
                else:
                    sell += qty
        total = buy + sell
        return float((buy - sell) / total) if total > 0 else 0.0

    def all_imbalances(self, now_ms=None):
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        return {w: self.imbalance(w, now_ms) for w in self.WINDOWS_MS}

    def volume_acceleration(self, now_ms=None):
        """Current 60s volume vs prior 60s volume."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cur = sum(q for ts, q, _ in self.trades if ts >= now_ms - 60_000)
        prior = sum(q for ts, q, _ in self.trades if now_ms - 120_000 <= ts < now_ms - 60_000)
        return float(cur / prior) if prior > 0 else 1.0

    def large_trade_count(self, min_qty=10.0, now_ms=None):
        """Count trades above threshold in last 60s."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - 60_000
        return sum(1 for ts, qty, _ in self.trades if ts >= cutoff and qty >= min_qty)

    def buy_sell_ratio(self, window_ms=60000, now_ms=None):
        """buy_count / sell_count in window."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_ms
        buys = sum(1 for ts, _, is_buy in self.trades if ts >= cutoff and is_buy)
        sells = sum(1 for ts, _, is_buy in self.trades if ts >= cutoff and not is_buy)
        return buys / sells if sells > 0 else 1.0


# ── KYLE'S LAMBDA ────────────────────────────────────────────────────────────

def compute_kyle_lambda(trades, window=500):
    """
    OLS: Δprice = λ × net_signed_volume
    Returns lambda (price impact per unit volume).
    Small λ = liquid (easy to move), large λ = illiquid.
    """
    if len(trades) < 50:
        return 0.0

    recent = trades[-window:]
    prices = np.array([t["price"] for t in recent])
    signed_vol = np.array([t["qty"] if not t["is_buyer_maker"] else -t["qty"] for t in recent])

    dp = np.diff(prices)
    cum_vol = np.cumsum(signed_vol)[:-1]

    if len(dp) < 10 or np.std(cum_vol) < 1e-10:
        return 0.0

    X = cum_vol.reshape(-1, 1)
    XtX = float((X.T @ X).flat[0])
    if XtX < 1e-10:
        return 0.0
    lam = float((X.T @ dp).flat[0] / XtX)
    return lam


# ── ADVERSE SELECTION ────────────────────────────────────────────────────────

def compute_adverse_selection(trades):
    """
    Measure adverse selection from trade data.
    High adverse selection = informed flow dominating.
    Returns: float in [-1, 1]
    """
    if len(trades) < 100:
        return 0.0

    prices = [t["price"] for t in trades]
    sides  = [1 if not t["is_buyer_maker"] else -1 for t in trades]

    # For each trade, measure price impact 10 trades later
    impacts = []
    for i in range(len(trades) - 10):
        future_price = prices[i + 10]
        cur_price    = prices[i]
        d = sides[i]
        impact = d * (future_price - cur_price) / (cur_price + 1e-12)
        impacts.append(impact)

    if not impacts:
        return 0.0

    avg_impact = float(np.mean(impacts))
    # Positive = informed (trades predict future price), negative = uninformed
    return float(np.clip(avg_impact * 1000, -1, 1))


# ── SPOT-PERP BASIS ──────────────────────────────────────────────────────────

def spot_perp_basis(symbol):
    """Compare spot vs perpetual price. Returns basis signal dict."""
    try:
        spot_r = requests.get(f"{BINANCE_SPOT}/api/v3/ticker/price",
                              params={"symbol": symbol}, timeout=5)
        spot_price = float(spot_r.json()["price"])

        perp_price = fetch_perp_price(symbol)
        fund_data  = fetch_funding(symbol)
        funding_rate = float(fund_data.get("lastFundingRate", 0))

        basis_pct = (perp_price - spot_price) / spot_price * 100
        ann_funding = funding_rate * 3 * 365 * 100

        if basis_pct > 0.15:
            sig, strength = "PERP_PREMIUM_BEARISH", -1
        elif basis_pct < -0.15:
            sig, strength = "PERP_DISCOUNT_BULLISH", 1
        else:
            sig, strength = "NEUTRAL", 0

        return {
            "spot": spot_price,
            "perp": perp_price,
            "basis_pct": round(basis_pct, 4),
            "funding_rate_pct": round(funding_rate * 100, 5),
            "funding_ann_pct": round(ann_funding, 1),
            "signal": sig,
            "strength": strength,
        }
    except Exception as e:
        return {"error": str(e), "strength": 0}


# ── QUEUE DEPLETION ──────────────────────────────────────────────────────────

def detect_queue_depletion(bids_before, asks_before, bids_after, asks_after, threshold=0.65):
    """
    Detect when top-of-book is rapidly depleted (iceberg absorbing flow).
    Returns: ('BID_DEPLETION_BULLISH'|'ASK_DEPLETION_BEARISH'|None, pct)
    """
    bid_vol_before = sum(b[1] for b in bids_before[:3])
    bid_vol_after  = sum(b[1] for b in bids_after[:3])
    ask_vol_before = sum(a[1] for a in asks_before[:3])
    ask_vol_after  = sum(a[1] for a in asks_after[:3])

    if bid_vol_before > 0:
        bid_drop = (bid_vol_before - bid_vol_after) / bid_vol_before
        if bid_drop > threshold:
            return "BID_DEPLETION_BULLISH", round(bid_drop, 3)

    if ask_vol_before > 0:
        ask_drop = (ask_vol_before - ask_vol_after) / ask_vol_before
        if ask_drop > threshold:
            return "ASK_DEPLETION_BEARISH", round(ask_drop, 3)

    return None, 0.0


# ── SPOOF DETECTION (static) ─────────────────────────────────────────────────

def detect_spoof_static(bids, asks, spot, threshold_multiple=8.0, neighbour_multiple=4.0):
    """
    Static order-book spoof detection.
    Looks for isolated large orders far exceeding their neighbours.
    Returns list of detected spoofs.
    """
    spoofs = []

    def check_side(levels, side_label):
        if len(levels) < 5:
            return
        vols = [l[1] for l in levels]
        median_vol = float(np.median(vols))
        if median_vol < 1e-10:
            return
        for i, (price, vol) in enumerate(levels):
            if vol < threshold_multiple * median_vol:
                continue
            # Check neighbours
            neighbour_vols = []
            if i > 0:
                neighbour_vols.append(levels[i-1][1])
            if i < len(levels) - 1:
                neighbour_vols.append(levels[i+1][1])
            if not neighbour_vols:
                continue
            avg_neighbour = float(np.mean(neighbour_vols))
            if avg_neighbour < 1e-10:
                continue
            if vol >= neighbour_multiple * avg_neighbour:
                dist_pct = abs(price - spot) / spot * 100
                spoofs.append({
                    "side": side_label,
                    "price": round(price, 6),
                    "vol": round(vol, 4),
                    "multiple_of_median": round(vol / median_vol, 1),
                    "dist_from_spot_pct": round(dist_pct, 3),
                    "interpretation": "FAKE_ASK_BULLISH" if side_label == "ASK" else "FAKE_BID_BEARISH",
                })

    check_side(asks, "ASK")
    check_side(bids, "BID")
    return spoofs


# ── SIGNAL FUSION ────────────────────────────────────────────────────────────

def compute_hft_score(ofi_z, tfi_dict, vol_accel, book_imbal, basis_strength, adv_sel):
    """
    Combine all HFT signals into a -100 to +100 score.

    Weights (research-validated):
    - OFI:              40%  (most predictive 1-10s)
    - TFI(2s+30s):      30%
    - Book imbal L5:    10%
    - Adverse sel:      10%
    - Vol accel:         5%
    - Basis:             5%
    """
    ofi_sig = float(np.clip(ofi_z / 2.5, -1.0, 1.0))

    tfi_2s  = tfi_dict.get(2000,  0.0)
    tfi_30s = tfi_dict.get(30000, 0.0)
    tfi_sig = tfi_2s * 0.55 + tfi_30s * 0.45

    bi5     = book_imbal.get(5, 0.5)
    bi_sig  = (bi5 - 0.5) * 2.0

    vol_sig  = float(np.clip((vol_accel - 1.0) * 1.5, -1.0, 1.0))
    basis_sg = float(basis_strength) * 0.5
    adv_sg   = float(np.clip(adv_sel, -1.0, 1.0))

    raw = (ofi_sig  * 0.40
         + tfi_sig  * 0.30
         + bi_sig   * 0.10
         + adv_sg   * 0.10
         + vol_sig  * 0.05
         + basis_sg * 0.05)

    return int(np.clip(raw * 100, -100, 100))


# ── SNAPSHOT SIGNAL (single REST scan, no WebSocket) ─────────────────────────

def snapshot_signal(symbol):
    """
    Full HFT signal snapshot for one symbol using REST endpoints.
    Returns complete signal dict.
    """
    bids, asks = fetch_orderbook(symbol, depth=20)
    trades     = fetch_agg_trades(symbol, limit=1000)

    ofi_calc = OFICalculator()
    flow     = TradeFlowTracker()

    now_ms = int(time.time() * 1000)
    for t in trades:
        flow.add(t["time_ms"], t["qty"], t["is_buyer_maker"])

    # Simulate OFI from trade-level bid/ask approximation
    if len(trades) >= 20:
        for i in range(1, min(200, len(trades))):
            t = trades[i]
            if t["is_buyer_maker"]:
                bid_p, bid_v = t["price"], t["qty"]
                ask_p, ask_v = t["price"] * 1.0001, t["qty"] * 0.5
            else:
                ask_p, ask_v = t["price"], t["qty"]
                bid_p, bid_v = t["price"] * 0.9999, t["qty"] * 0.5
            ofi_calc.update(bid_p, bid_v, ask_p, ask_v)

    # Feed actual live book top-of-book
    if bids and asks:
        ofi_calc.update(bids[0][0], bids[0][1], asks[0][0], asks[0][1])

    ofi_z    = ofi_calc.zscore()
    ofi_cum  = ofi_calc.cumulative()
    tfi      = flow.all_imbalances(now_ms)
    vol_acc  = flow.volume_acceleration(now_ms)
    bi       = compute_book_imbalance(bids, asks)
    mp       = compute_microprice(bids, asks)
    bid_usd, ask_usd, pressure = book_pressure(bids, asks, depth=10)
    kyle_lam = compute_kyle_lambda(trades)
    adv_sel  = compute_adverse_selection(trades)
    spoofs   = detect_spoof_static(bids, asks, spot=bids[0][0] if bids else 1.0)

    try:
        basis         = spot_perp_basis(symbol)
        basis_strength = basis.get("strength", 0)
    except Exception:
        basis          = {}
        basis_strength = 0

    score = compute_hft_score(ofi_z, tfi, vol_acc, bi, basis_strength, adv_sel)

    if   score >=  60: verdict = "STRONG_BUY"
    elif score >=  30: verdict = "BUY"
    elif score <= -60: verdict = "STRONG_SELL"
    elif score <= -30: verdict = "SELL"
    else:              verdict = "NEUTRAL"

    spread     = asks[0][0] - bids[0][0] if bids and asks else 0.0
    spread_bps = spread / bids[0][0] * 10_000 if bids else 0.0

    return {
        "symbol":           symbol,
        "score":            score,
        "verdict":          verdict,
        "ofi_z":            round(ofi_z, 3),
        "ofi_cumulative":   round(ofi_cum, 2),
        "tfi_500ms":        round(tfi.get(500,   0), 4),
        "tfi_2s":           round(tfi.get(2000,  0), 4),
        "tfi_10s":          round(tfi.get(10000, 0), 4),
        "tfi_30s":          round(tfi.get(30000, 0), 4),
        "tfi_60s":          round(tfi.get(60000, 0), 4),
        "vol_acceleration": round(vol_acc, 3),
        "adverse_selection":round(adv_sel, 4),
        "microprice":       round(mp, 6) if mp else None,
        "spread_bps":       round(spread_bps, 3),
        "book_bid_usd":     round(bid_usd, 0),
        "book_ask_usd":     round(ask_usd, 0),
        "book_pressure":    round(pressure, 4),
        "book_imbalance":   {k: round(v, 4) for k, v in bi.items()},
        "kyle_lambda":      round(kyle_lam, 8),
        "spoofs":           spoofs,
        "basis":            basis,
    }


# ── FULL SCAN ────────────────────────────────────────────────────────────────

def run_hft_scan(symbols=None):
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    results = []
    for sym in symbols:
        try:
            sig = snapshot_signal(sym)
            results.append(sig)
        except Exception as e:
            results.append({"symbol": sym, "score": 0, "verdict": "ERROR", "error": str(e)})
        time.sleep(0.12)

    results.sort(key=lambda r: abs(r.get("score", 0)), reverse=True)
    return results


# ── CARRY ARB SCAN ───────────────────────────────────────────────────────────

def carry_arb_scan(symbols=None):
    """
    Identify best carry arbitrage opportunities.
    Long spot + Short perp → collect funding every 8h.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS

    opportunities = []
    for sym in symbols:
        try:
            b = spot_perp_basis(sym)
            if "error" in b:
                continue
            ann = b.get("funding_ann_pct", 0)
            if ann > 5.0:   # Only show meaningful carry
                opportunities.append({
                    "symbol": sym,
                    "funding_ann_pct": ann,
                    "funding_per_8h_pct": round(b.get("funding_rate_pct", 0), 5),
                    "basis_pct": b.get("basis_pct", 0),
                    "action": "LONG_SPOT_SHORT_PERP" if ann > 0 else "SHORT_SPOT_LONG_PERP",
                })
        except Exception:
            pass
        time.sleep(0.05)

    opportunities.sort(key=lambda x: abs(x["funding_ann_pct"]), reverse=True)
    return opportunities


# ── DISPLAY ──────────────────────────────────────────────────────────────────

def print_hft_report(results):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    print("\n" + "═" * 74)
    print(f"  HFT MICROSTRUCTURE SIGNALS  |  {ts}")
    print("═" * 74)

    for r in results:
        if r.get("verdict") == "ERROR":
            print(f"  {r['symbol']:12s}  ERROR: {str(r.get('error',''))[:55]}")
            continue

        score   = r.get("score", 0)
        verdict = r.get("verdict", "NEUTRAL")
        icon    = "⚡" if abs(score) >= 50 else "▶" if abs(score) >= 25 else "○"
        arrow   = "▲" if score > 0 else "▼" if score < 0 else "─"

        print(f"\n{icon} {r['symbol']:12s}  score={score:+4d}/100  {arrow} {verdict}")
        print(f"   OFI z={r.get('ofi_z',0):+.3f}  cum={r.get('ofi_cumulative',0):+.1f}  "
              f"| TFI 2s={r.get('tfi_2s',0):+.4f}  30s={r.get('tfi_30s',0):+.4f}  "
              f"60s={r.get('tfi_60s',0):+.4f}")

        bi = r.get("book_imbalance", {})
        print(f"   Book imbal: L1={bi.get(1,0.5):.3f}  L5={bi.get(5,0.5):.3f}  "
              f"L10={bi.get(10,0.5):.3f}  L20={bi.get(20,0.5):.3f}  "
              f"| pressure={r.get('book_pressure',0):+.4f}")

        print(f"   Vol accel={r.get('vol_acceleration',1):.2f}x  "
              f"adv_sel={r.get('adverse_selection',0):+.4f}  "
              f"spread={r.get('spread_bps',0):.2f}bps  "
              f"Kyle_λ={r.get('kyle_lambda',0):.2e}")

        basis = r.get("basis", {})
        if basis and "basis_pct" in basis:
            print(f"   Basis={basis['basis_pct']:+.4f}%  "
                  f"Funding={basis.get('funding_ann_pct',0):+.1f}%/yr  "
                  f"→ {basis.get('signal','N/A')}")

        spoofs = r.get("spoofs", [])
        if spoofs:
            for s in spoofs[:2]:
                print(f"   ⚠ SPOOF {s['side']} @{s['price']}  {s['multiple_of_median']}× median  "
                      f"→ {s['interpretation']}")

    print("\n" + "═" * 74)
    buys  = sum(1 for r in results if r.get("score", 0) >  25)
    sells = sum(1 for r in results if r.get("score", 0) < -25)
    neut  = len(results) - buys - sells
    print(f"  Breadth: {buys} BUY  |  {sells} SELL  |  {neut} NEUTRAL")
    print("═" * 74 + "\n")


def print_carry_report(opportunities):
    print("\n" + "─" * 60)
    print("  CARRY ARB OPPORTUNITIES (Long Spot + Short Perp)")
    print("─" * 60)
    if not opportunities:
        print("  No significant carry opportunities right now.")
    for o in opportunities:
        print(f"  {o['symbol']:12s}  {o['funding_ann_pct']:+6.1f}%/yr  "
              f"({o['funding_per_8h_pct']:+.5f}%/8h)  "
              f"basis={o['basis_pct']:+.4f}%  → {o['action']}")
    print("─" * 60 + "\n")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HFT Microstructure Signal Engine")
    parser.add_argument("--symbol",  help="Single symbol deep dive")
    parser.add_argument("--watch",   type=int, help="Rescan interval in seconds")
    parser.add_argument("--arb",     action="store_true", help="Show carry arb opportunities only")
    parser.add_argument("--json",    action="store_true", help="JSON output")
    args = parser.parse_args()

    syms = [args.symbol] if args.symbol else DEFAULT_SYMBOLS

    def run_once():
        if args.arb:
            opps = carry_arb_scan(syms)
            if args.json:
                print(json.dumps(opps, indent=2))
            else:
                print_carry_report(opps)
            return

        print(f"Scanning {len(syms)} symbol(s) for HFT signals...")
        results = run_hft_scan(syms)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print_hft_report(results)
            # Always show carry arb below the main report
            opps = carry_arb_scan(syms)
            print_carry_report(opps)

    if args.watch:
        while True:
            run_once()
            print(f"[Next scan in {args.watch}s... Ctrl+C to stop]")
            time.sleep(args.watch)
    else:
        run_once()


if __name__ == "__main__":
    main()
