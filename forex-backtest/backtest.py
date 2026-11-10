#!/usr/bin/env python3
"""Backtester for forex-market. Offline, standard library only. Never touches a broker.

    python backtest.py run --bars trader/bars/EURUSD_M5.json --equity 300 --risk 1
    python backtest.py run --bars FILE --rr 2 --flat-hour 16 --write

Honesty rules baked in:
  * Signals are read on a CLOSED bar and filled at the NEXT bar's open (no peeking).
  * Bars are BID prices. Buys fill at bid + spread; shorts close at bid + spread. Spread comes from
    each bar's own recorded spread, plus extra slippage on every stop and market fill.
  * If a bar touches both stop and target, the STOP is assumed to hit first.
  * A stop that gaps is filled at the open, not at the stop price.
  * Lots are sized from risk % and the stop distance, rounded DOWN to the broker's lot step. A trade that
    would need less than the broker's minimum lot is SKIPPED and counted, not quietly oversized.
  * The history is split in time: the first part is where you tune, the last part is out-of-sample.
"""

import argparse
import json
import math
import os
import sys
import time

DAY = 86400


class BacktestError(Exception):
    pass


# ------------------------------------------------------------------ costs and fills

def bar_spread_px(bar, point, fallback_points=0, floor_points=0):
    sp = bar.get("spread") or fallback_points or 0
    return max(sp, floor_points) * point


def simulate_trade(bars, j, side, sl_pips, tp_pips, spec, slip_pips, flat_ts=None, floor_points=0):
    """Enter at bar j's open. Returns dict with entry/exit prices, exit bar index, reason, pips (after costs)."""
    pip, point = spec["pip"], spec["point"]
    fb = spec.get("spread_points") or 0
    slip = slip_pips * pip
    b = bars[j]
    sp0 = bar_spread_px(b, point, fb, floor_points)
    long_ = side == "long"
    entry = b["o"] + sp0 + slip if long_ else b["o"] - slip   # long buys the ask; short sells the bid
    sl = entry - sl_pips * pip if long_ else entry + sl_pips * pip   # levels in BID terms
    tp = entry + tp_pips * pip if long_ else entry - tp_pips * pip
    for k in range(j, len(bars)):
        c = bars[k]
        sp = bar_spread_px(c, point, fb, floor_points)
        if flat_ts is not None and k > j and c["t"] >= flat_ts:
            return _done(side, entry, c["o"] if long_ else c["o"] + sp, k, "flat", pip, slip, long_)
        if long_:
            if k > j and c["o"] <= sl:
                return _done(side, entry, c["o"] - slip, k, "stop-gap", pip, 0, long_)
            if k > j and c["o"] >= tp:
                return _done(side, entry, c["o"], k, "target-gap", pip, 0, long_)
            if c["l"] <= sl:
                return _done(side, entry, sl - slip, k, "stop", pip, 0, long_)
            if c["h"] >= tp:
                return _done(side, entry, tp, k, "target", pip, 0, long_)
        else:
            if k > j and c["o"] + sp >= sl:
                return _done(side, entry, c["o"] + sp + slip, k, "stop-gap", pip, 0, long_)
            if k > j and c["o"] + sp <= tp:
                return _done(side, entry, c["o"] + sp, k, "target-gap", pip, 0, long_)
            if c["h"] + sp >= sl:
                return _done(side, entry, sl + slip, k, "stop", pip, 0, long_)
            if c["l"] + sp <= tp:
                return _done(side, entry, tp, k, "target", pip, 0, long_)
    last = bars[-1]
    sp = bar_spread_px(last, point, fb, floor_points)
    return _done(side, entry, last["c"] if long_ else last["c"] + sp, len(bars) - 1, "end-of-data", pip, 0, long_)


def _done(side, entry, exit_, k, reason, pip, _slip, long_):
    pips = (exit_ - entry) / pip if long_ else (entry - exit_) / pip
    return {"side": side, "entry": entry, "exit": exit_, "exit_index": k, "reason": reason, "pips": pips}


def size_lots(equity, risk_pct, stop_pips, spec):
    """Lots so that hitting the stop loses about risk_pct of equity. Rounded down. 0.0 means: too small to take."""
    pv = spec.get("pip_value_per_lot")
    if not pv or stop_pips <= 0:
        raise BacktestError("need pip_value_per_lot and a positive stop (re-pull bars so the spec is saved)")
    raw = equity * risk_pct / 100.0 / (stop_pips * pv)
    step = spec.get("volume_step") or 0.01
    lots = math.floor(raw / step + 1e-9) * step
    lots = min(lots, spec.get("volume_max") or lots)
    if lots + 1e-9 < (spec.get("volume_min") or step):
        return 0.0
    return round(lots, 8)


# ------------------------------------------------------------------ shared helpers

def daily_adr(bars, n=20, min_days=5):
    """day -> average daily range (price units) of the PREVIOUS n days. Causal: today's own range is not used."""
    hl = {}
    for b in bars:
        d = b["t"] // DAY
        r = hl.setdefault(d, [b["h"], b["l"]])
        r[0], r[1] = max(r[0], b["h"]), min(r[1], b["l"])
    out, prev = {}, []
    for d in sorted(hl):
        if len(prev) >= min_days:
            out[d] = sum(prev[-n:]) / len(prev[-n:])
        prev.append(hl[d][0] - hl[d][1])
    return out


def resample(bars, minutes):
    """Aggregate bars into bigger ones (e.g. M5 -> M15). Spread: the widest seen in the group."""
    step, out = minutes * 60, []
    for b in bars:
        t = b["t"] // step * step
        if out and out[-1]["t"] == t:
            o = out[-1]
            o["h"], o["l"], o["c"] = max(o["h"], b["h"]), min(o["l"], b["l"]), b["c"]
            o["v"] += b["v"]
            o["spread"] = max(o["spread"], b.get("spread") or 0)
        else:
            out.append({"t": t, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"], "v": b["v"],
                        "spread": b.get("spread") or 0})
    return out


def ema_series(values, n):
    k, out, e = 2.0 / (n + 1), [], None
    for v in values:
        e = v if e is None else e + k * (v - e)
        out.append(e)
    return out


def atr_series(bars, n=14):
    out, a, prev_c = [], None, None
    for b in bars:
        tr = b["h"] - b["l"] if prev_c is None else max(b["h"] - b["l"], abs(b["h"] - prev_c), abs(b["l"] - prev_c))
        a = tr if a is None else a + (tr - a) / n
        out.append(a)
        prev_c = b["c"]
    return out


# ------------------------------------------------------------------ strategy: session range breakout

class RangeBreakout:
    """Build a range over [range_start, range_end) UTC (default: the quiet Asian hours), then trade the first
    closed bar that settles beyond it, inside [range_end, trade_end). Stop at the range midpoint, target rr x
    the stop. One trade a day, flat by flat_hour."""

    def __init__(self, tf_min, spec, range_start=0, range_end=7, trade_end=12, flat_hour=16,
                 rr=1.5, buffer_pips=1.0, min_range_pips=15, max_range_pips=60, min_stop_pips=8,
                 adr_ref_pips=None):
        # adr_ref_pips: if set, the pip thresholds are written for a symbol whose average daily range is that
        # many pips (EURUSD ~70) and are scaled to each symbol's own range, so one setting fits gold and JPY too.
        self.adr_ref = adr_ref_pips
        self.adr = {}
        self.tf = tf_min * 60
        self.pip = spec["pip"]
        self.p = dict(range_start=range_start, range_end=range_end, trade_end=trade_end, flat_hour=flat_hour,
                      rr=rr, buffer_pips=buffer_pips, min_range_pips=min_range_pips,
                      max_range_pips=max_range_pips, min_stop_pips=min_stop_pips)
        self.ranges = {}
        self.traded = set()

    def prepare(self, bars):
        p = self.p
        if self.adr_ref:
            self.adr = daily_adr(bars)
        for b in bars:
            day, sec = divmod(b["t"], DAY)
            if p["range_start"] * 3600 <= sec < p["range_end"] * 3600:
                r = self.ranges.setdefault(day, [b["h"], b["l"], 0])
                r[0], r[1], r[2] = max(r[0], b["h"]), min(r[1], b["l"]), r[2] + 1

    def flat_ts(self, t):
        return (t // DAY) * DAY + self.p["flat_hour"] * 3600

    def on_bar(self, bars, i):
        p = self.p
        b = bars[i]
        day, sec = divmod(b["t"] + self.tf, DAY)      # decide at the bar's CLOSE time
        if not (p["range_end"] * 3600 <= sec < p["trade_end"] * 3600) or day in self.traded:
            return None
        r = self.ranges.get(day)
        if not r:
            return None
        hi, lo, n = r
        need = (p["range_end"] - p["range_start"]) * 3600 // self.tf
        if n < need * 0.8:                               # holes in the range: do not trade it
            return None
        k = 1.0
        if self.adr_ref:
            if day not in self.adr:
                return None
            k = self.adr[day] / (self.adr_ref * self.pip)
        width = (hi - lo) / self.pip
        if not (p["min_range_pips"] * k <= width <= p["max_range_pips"] * k):
            return None
        buf = p["buffer_pips"] * k * self.pip
        side = "long" if b["c"] > hi + buf else "short" if b["c"] < lo - buf else None
        if not side:
            return None
        mid = (hi + lo) / 2.0
        stop = abs(b["c"] - mid) / self.pip
        if stop < p["min_stop_pips"] * k:
            return None
        self.traded.add(day)
        return {"side": side, "sl_pips": stop, "tp_pips": stop * p["rr"]}


# ------------------------------------------------------------------ strategy: trend pullback

class TrendPullback:
    """Trade with the trend after a dip. Trend: EMA(fast) above EMA(slow) for longs (below for shorts).
    Trigger: the previous bar closed on the wrong side of EMA(pull) and this bar closes back on the trend side.
    Stop atr_mult x ATR(14), target rr x the stop. Only inside [start_hour, end_hour) UTC, flat by flat_hour,
    at most max_per_day trades a day."""

    def __init__(self, tf_min, spec, fast=50, slow=200, pull=20, atr_mult=1.5, rr=1.5,
                 start_hour=7, end_hour=17, flat_hour=20, max_per_day=2):
        self.tf = tf_min * 60
        self.pip = spec["pip"]
        self.p = dict(fast=fast, slow=slow, pull=pull, atr_mult=atr_mult, rr=rr, start_hour=start_hour,
                      end_hour=end_hour, flat_hour=flat_hour, max_per_day=max_per_day)
        self.count = {}

    def prepare(self, bars):
        closes = [b["c"] for b in bars]
        self.ef, self.es = ema_series(closes, self.p["fast"]), ema_series(closes, self.p["slow"])
        self.ep, self.atr = ema_series(closes, self.p["pull"]), atr_series(bars)

    def flat_ts(self, t):
        return (t // DAY) * DAY + self.p["flat_hour"] * 3600

    def on_bar(self, bars, i):
        p = self.p
        if i < p["slow"] * 3:                              # let the slow EMA settle
            return None
        day, sec = divmod(bars[i]["t"] + self.tf, DAY)
        if not (p["start_hour"] * 3600 <= sec < p["end_hour"] * 3600):
            return None
        if self.count.get(day, 0) >= p["max_per_day"]:
            return None
        c, c1 = bars[i]["c"], bars[i - 1]["c"]
        side = None
        if self.ef[i] > self.es[i] and c1 < self.ep[i - 1] and c > self.ep[i] and c > self.es[i]:
            side = "long"
        elif self.ef[i] < self.es[i] and c1 > self.ep[i - 1] and c < self.ep[i] and c < self.es[i]:
            side = "short"
        if not side:
            return None
        stop = p["atr_mult"] * self.atr[i] / self.pip
        if stop <= 0:
            return None
        self.count[day] = self.count.get(day, 0) + 1
        return {"side": side, "sl_pips": stop, "tp_pips": stop * p["rr"]}


# ------------------------------------------------------------------ engine

def run_backtest(bars, spec, strategy, equity=300.0, risk_pct=1.0, slip_pips=0.3, floor_spread_points=0,
                 commission_per_lot=0.0):
    if len(bars) < 100:
        raise BacktestError("too few bars (%d)" % len(bars))
    strategy.prepare(bars)
    pv = spec["pip_value_per_lot"]
    eq, peak, max_dd = equity, equity, 0.0
    trades, skipped = [], 0
    i = 0
    while i < len(bars) - 1:
        sig = strategy.on_bar(bars, i)
        if not sig:
            i += 1
            continue
        stop_cost = sig["sl_pips"] + slip_pips          # size for the stop plus slippage
        lots = size_lots(eq, risk_pct, stop_cost, spec)
        if lots <= 0:
            skipped += 1
            i += 1
            continue
        t = simulate_trade(bars, i + 1, sig["side"], sig["sl_pips"], sig["tp_pips"], spec, slip_pips,
                           strategy.flat_ts(bars[i + 1]["t"]), floor_spread_points)
        risk_money = lots * stop_cost * pv
        pnl = t["pips"] * pv * lots - commission_per_lot * lots
        eq += pnl
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 1.0)
        t.update({"t_entry": bars[i + 1]["t"], "t_exit": bars[t["exit_index"]]["t"], "lots": lots,
                  "pnl": pnl, "r": pnl / risk_money if risk_money else 0.0, "equity": eq})
        trades.append(t)
        if eq <= 0:
            break
        i = max(t["exit_index"], i + 1)
    return {"trades": trades, "skipped_too_small": skipped, "start_equity": equity, "end_equity": eq,
            "max_drawdown": max_dd, "bars": len(bars)}


def summarize(trades, start_equity, end_equity):
    n = len(trades)
    if not n:
        return {"trades": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] <= 0)
    streak = cur = 0
    for t in trades:
        cur = cur + 1 if t["pnl"] <= 0 else 0
        streak = max(streak, cur)
    return {"trades": n, "win_rate": len(wins) / n, "avg_r": sum(t["r"] for t in trades) / n,
            "profit_factor": (gp / gl) if gl else float("inf"), "net": sum(t["pnl"] for t in trades),
            "return_pct": (end_equity / start_equity - 1) * 100 if start_equity else 0.0,
            "worst_losing_streak": streak, "avg_pips": sum(t["pips"] for t in trades) / n}


def split_trades(trades, cut_ts):
    return [t for t in trades if t["t_entry"] < cut_ts], [t for t in trades if t["t_entry"] >= cut_ts]


def verdict(s_in, s_out):
    if s_out["trades"] < 30:
        return "NOT ENOUGH TRADES out-of-sample (%d) to conclude anything. Get more history before believing any number." % s_out["trades"]
    if s_out["avg_r"] <= 0:
        return "NO EDGE SHOWN: out-of-sample average R is %.2f. Do not trade this." % s_out["avg_r"]
    if s_in["trades"] >= 30 and s_in["avg_r"] <= 0:
        return "INCONSISTENT: lost in-sample (%.2f R) but won out-of-sample (%.2f R). That looks like luck or a regime change, not an edge." % (s_in["avg_r"], s_out["avg_r"])
    if s_in["trades"] and s_in["avg_r"] > 0 and s_out["avg_r"] < 0.5 * s_in["avg_r"]:
        return "FRAGILE: out-of-sample R (%.2f) is less than half of in-sample (%.2f), a sign of overfitting." % (s_out["avg_r"], s_in["avg_r"])
    return "Positive out-of-sample (%.2f R). Still not proof: paper trade on demo before any real money." % s_out["avg_r"]


# ------------------------------------------------------------------ CLI

def load_bars_file(path):
    with open(path) as f:
        doc = json.load(f)
    if not doc.get("spec"):
        raise BacktestError("%s has no symbol spec; re-pull it with forex-data/mt5_data.py bars --write" % path)
    return doc


def fmt(s):
    if not s["trades"]:
        return "  no trades"
    pf = "inf" if s["profit_factor"] == float("inf") else "%.2f" % s["profit_factor"]
    return ("  trades %d | win %.0f%% | avg %.2f R | profit factor %s | net %+.2f | avg %.1f pips | worst losing streak %d"
            % (s["trades"], s["win_rate"] * 100, s["avg_r"], pf, s["net"], s["avg_pips"], s["worst_losing_streak"]))


def cmd_run(a):
    doc = load_bars_file(a.bars)
    bars, spec = doc["bars"], doc["spec"]
    tf_min = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}[doc["tf"]]
    if a.resample:
        if a.resample <= tf_min or a.resample % tf_min:
            raise BacktestError("--resample must be a multiple of the file's timeframe (%d min)" % tf_min)
        bars, tf_min = resample(bars, a.resample), a.resample
    # scale pip-based settings to this symbol's own daily range (reference: a 70-pip-a-day pair)
    adr = sorted(daily_adr(bars).values())
    k = (adr[len(adr) // 2] / (a.adr_ref * spec["pip"])) if adr and a.adr_ref else 1.0
    slip = a.slip * k
    if a.strategy == "breakout":
        st = RangeBreakout(tf_min, spec, a.range_start, a.range_end, a.trade_end, a.flat_hour, a.rr,
                           a.buffer_pips, a.min_range, a.max_range, a.min_stop, a.adr_ref or None)
    else:
        st = TrendPullback(tf_min, spec, rr=a.rr, atr_mult=a.atr_mult)
    res = run_backtest(bars, spec, st, a.equity, a.risk, slip, int(a.min_spread_pips * spec["pip"] / spec["point"]),
                       a.commission)
    print("strategy: %s | slippage %.2f pips a fill" % (a.strategy, slip))
    trades = res["trades"]
    first, last = bars[0]["t"], bars[-1]["t"]
    cut = first + (last - first) * a.split
    ins, oos = split_trades(trades, cut)
    fm = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t))
    print("%s %s  %s -> %s  (%d bars, ~%d days)" % (doc["symbol"], doc["tf"], fm(first), fm(last), len(bars), (last - first) // DAY))
    print("equity %.2f -> %.2f | risk %.1f%% a trade | max drawdown %.1f%% | skipped (below min lot): %d"
          % (res["start_equity"], res["end_equity"], a.risk, res["max_drawdown"] * 100, res["skipped_too_small"]))
    print("ALL:\n" + fmt(summarize(trades, res["start_equity"], res["end_equity"])))
    s_in = summarize(ins, res["start_equity"], ins[-1]["equity"] if ins else res["start_equity"])
    s_out = summarize(oos, ins[-1]["equity"] if ins else res["start_equity"], res["end_equity"])
    print("IN-SAMPLE (to %s):\n%s" % (fm(cut), fmt(s_in)))
    print("OUT-OF-SAMPLE (after):\n" + fmt(s_out))
    reasons = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    print("exits: " + ", ".join("%s %d" % kv for kv in sorted(reasons.items())))
    print("VERDICT: " + verdict(s_in, s_out) if s_out["trades"] else "VERDICT: no out-of-sample trades.")
    if a.write:
        path = os.path.join("trader", "backtests", "%s_%s_%s.json" % (doc["symbol"], doc["tf"], a.strategy))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"result": res, "params": vars(a)}, f)
        print("wrote " + path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Forex backtester (session range breakout).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--bars", required=True)
    p.add_argument("--strategy", choices=["breakout", "pullback"], default="breakout")
    p.add_argument("--resample", type=int, default=0, help="aggregate bars to this many minutes first")
    p.add_argument("--adr-ref", type=float, default=70.0,
                   help="pip settings assume this average daily range and are scaled per symbol; 0 = off")
    p.add_argument("--atr-mult", type=float, default=1.5, help="pullback: stop = this x ATR(14)")
    p.add_argument("--equity", type=float, default=300.0)
    p.add_argument("--risk", type=float, default=1.0, help="percent of equity risked per trade")
    p.add_argument("--rr", type=float, default=1.5)
    p.add_argument("--range-start", type=int, default=0)
    p.add_argument("--range-end", type=int, default=7)
    p.add_argument("--trade-end", type=int, default=12)
    p.add_argument("--flat-hour", type=int, default=16)
    p.add_argument("--buffer-pips", type=float, default=1.0)
    p.add_argument("--min-range", type=float, default=15)
    p.add_argument("--max-range", type=float, default=60)
    p.add_argument("--min-stop", type=float, default=8)
    p.add_argument("--slip", type=float, default=0.3, help="extra pips lost on each market fill / stop")
    p.add_argument("--min-spread-pips", type=float, default=0.0)
    p.add_argument("--commission", type=float, default=0.0, help="per lot, account currency")
    p.add_argument("--split", type=float, default=0.6, help="fraction of history that is in-sample")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_run)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except (BacktestError, OSError, KeyError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
