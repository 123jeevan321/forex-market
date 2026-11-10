#!/usr/bin/env python3
"""Order-flow strategies on a tick-rule PROXY. Offline.

    python ../forex-data/mt5_data.py bars EURUSD --tf M5 --count 99999 --write
    python ../forex-data/mt5_data.py flow EURUSD --days 365
    python orderflow.py trader/bars/EURUSD_M5.json trader/flow/EURUSD_M5.json

IMPORTANT: retail forex has no order book and no trade tape. "Delta" here is (ticks where the mid price
moved up) minus (ticks where it moved down) inside each bar. It is quote pressure, not real buying and
selling volume. Real order flow / footprint needs exchange data (currency futures).

Three scalps, all in the London / New York opening hours, out after 45 minutes:
  divergence  price makes a new high but cumulative delta is lower than at the previous high -> short
  absorption  heavy activity and strongly positive delta, yet the bar fails to close up     -> short
  momentum    strongly positive delta AND the bar closes at its high through recent highs    -> long
(each mirrored for the other side). Plus a direct check: does a bar's delta predict the next bars at all?
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

_spec = importlib.util.spec_from_file_location("bt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

DAY = 86400


def merge_flow(bars, flow_rows):
    """Attach delta / tick count to each bar, keeping only the stretch that the flow data covers."""
    by_t = {r["t"]: r for r in flow_rows}
    if not by_t:
        return []
    lo, hi = min(by_t), max(by_t)
    out = []
    for b in bars:
        if not (lo <= b["t"] <= hi):
            continue
        r = by_t.get(b["t"])
        out.append(dict(b, delta=(r["up"] - r["dn"]) if r else 0, ticks=r["n"] if r else 0))
    return out


def rolling_z(values, n=20):
    """z-score of each value against the n values BEFORE it (causal). None until there is history."""
    out, s, s2 = [], 0.0, 0.0
    for i, v in enumerate(values):
        if i >= n:
            m = s / n
            var = max(s2 / n - m * m, 0.0)
            out.append((v - m) / math.sqrt(var) if var > 0 else 0.0)
        else:
            out.append(None)
        s += v
        s2 += v * v
        if i >= n:
            old = values[i - n]
            s -= old
            s2 -= old * old
    return out


def rolling_mean_prev(values, n=20):
    out, s = [], 0.0
    for i, v in enumerate(values):
        out.append(s / n if i >= n else None)
        s += v
        if i >= n:
            s -= values[i - n]
    return out


def correlation(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx, syy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


def predictive_power(bars, horizons=(1, 3, 6)):
    """Correlation between a bar's delta z-score and (a) its own move, (b) the move over the NEXT k bars.
    (a) is high by construction. (b) is the question: near zero means delta tells you nothing about what
    comes next."""
    z = rolling_z([b["delta"] for b in bars])
    idx = [i for i in range(len(bars)) if z[i] is not None]
    res = {"same_bar": correlation([z[i] for i in idx], [bars[i]["c"] - bars[i]["o"] for i in idx])}
    for k in horizons:
        ok = [i for i in idx if i + k < len(bars) and bars[i + k]["t"] - bars[i]["t"] <= (k + 2) * (bars[1]["t"] - bars[0]["t"])]
        res["next_%d" % k] = correlation([z[i] for i in ok], [bars[i + k]["c"] - bars[i]["c"] for i in ok])
        res["n"] = len(ok)
    return res


class _FlowBase:
    def __init__(self, tf_min, spec, rr=1.5, max_hold_min=45, buffer_pips=0.5, min_stop_pips=3, max_stop_pips=15,
                 adr_ref_pips=70, max_per_day=3):
        self.tf = tf_min * 60
        self.pip = spec["pip"]
        self.p = dict(rr=rr, max_hold=max_hold_min * 60, buffer_pips=buffer_pips, min_stop_pips=min_stop_pips,
                      max_stop_pips=max_stop_pips, adr_ref=adr_ref_pips, max_per_day=max_per_day)
        self.count = {}

    def prepare(self, bars):
        deltas = [b["delta"] for b in bars]
        self.adr = bt.daily_adr(bars)
        self.z = rolling_z(deltas)
        self.avg_ticks = rolling_mean_prev([b["ticks"] for b in bars])
        self.cum, c = [], 0
        for d in deltas:
            c += d
            self.cum.append(c)

    def flat_ts(self, t):
        return t + self.p["max_hold"]

    def _ready(self, bars, i):
        """(day, scale) if a trade may be considered on this bar, else None."""
        day, sec = divmod(bars[i]["t"] + self.tf, DAY)
        if not bt.in_kill_zone(sec) or day not in self.adr or self.z[i] is None:
            return None
        if self.count.get(day, 0) >= self.p["max_per_day"]:
            return None
        return day, self.adr[day] / (self.p["adr_ref"] * self.pip)

    def _order(self, day, k, side, stop_price_dist):
        stop = stop_price_dist / self.pip
        if not (self.p["min_stop_pips"] * k <= stop <= self.p["max_stop_pips"] * k):
            return None
        self.count[day] = self.count.get(day, 0) + 1
        return {"side": side, "sl_pips": stop, "tp_pips": stop * self.p["rr"]}


class DeltaDivergence(_FlowBase):
    """New price extreme without a new cumulative-delta extreme, confirmed by a bar closing against the move."""
    lookback = 12

    def on_bar(self, bars, i):
        n = self.lookback
        r = self._ready(bars, i) if i > n else None
        if not r:
            return None
        day, k = r
        buf = self.p["buffer_pips"] * k * self.pip
        b, prior = bars[i], range(i - n, i)
        jh = max(prior, key=lambda j: bars[j]["h"])
        jl = min(prior, key=lambda j: bars[j]["l"])
        if b["h"] > bars[jh]["h"] and b["c"] < b["o"] and self.cum[i] < self.cum[jh]:
            return self._order(day, k, "short", b["h"] + buf - b["c"])
        if b["l"] < bars[jl]["l"] and b["c"] > b["o"] and self.cum[i] > self.cum[jl]:
            return self._order(day, k, "long", b["c"] - (b["l"] - buf))
        return None


class Absorption(_FlowBase):
    """A busy bar with strongly one-sided delta that fails to close that way: the pressure was absorbed."""
    z_min, busy = 1.5, 1.5

    def on_bar(self, bars, i):
        r = self._ready(bars, i)
        if not r or not self.avg_ticks[i]:
            return None
        day, k = r
        b = bars[i]
        if b["ticks"] < self.busy * self.avg_ticks[i]:
            return None
        buf = self.p["buffer_pips"] * k * self.pip
        if self.z[i] >= self.z_min and b["c"] <= b["o"]:
            return self._order(day, k, "short", b["h"] + buf - b["c"])
        if self.z[i] <= -self.z_min and b["c"] >= b["o"]:
            return self._order(day, k, "long", b["c"] - (b["l"] - buf))
        return None


class ImbalanceMomentum(_FlowBase):
    """Strongly one-sided delta and the bar closes at its extreme through the recent range: go with it."""
    z_min, lookback = 2.0, 6

    def on_bar(self, bars, i):
        n = self.lookback
        r = self._ready(bars, i) if i > n else None
        if not r:
            return None
        day, k = r
        b = bars[i]
        rng = b["h"] - b["l"]
        if rng <= 0:
            return None
        buf = self.p["buffer_pips"] * k * self.pip
        prior = bars[i - n:i]
        if self.z[i] >= self.z_min and b["c"] > max(x["h"] for x in prior) and (b["h"] - b["c"]) <= 0.25 * rng:
            return self._order(day, k, "long", b["c"] - (b["l"] - buf))
        if self.z[i] <= -self.z_min and b["c"] < min(x["l"] for x in prior) and (b["c"] - b["l"]) <= 0.25 * rng:
            return self._order(day, k, "short", b["h"] + buf - b["c"])
        return None


STRATEGIES = (("divergence", DeltaDivergence), ("absorption", Absorption), ("momentum", ImbalanceMomentum))


def evaluate(bars, spec, cls, equity=5000.0, risk=1.0, slip=0.3, commission=0.0, split=0.6, **kw):
    st = cls(5, spec, **kw)
    res = bt.run_backtest(bars, spec, st, equity, risk, slip, 0, commission)
    cut = bars[0]["t"] + (bars[-1]["t"] - bars[0]["t"]) * split
    ins, oos = bt.split_trades(res["trades"], cut)
    s_all = bt.summarize(res["trades"], equity, res["end_equity"])
    s_in = bt.summarize(ins, equity, ins[-1]["equity"] if ins else equity)
    s_out = bt.summarize(oos, ins[-1]["equity"] if ins else equity, res["end_equity"])
    return res, s_all, s_in, s_out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Order-flow proxy strategies.")
    ap.add_argument("bars")
    ap.add_argument("flow")
    ap.add_argument("--equity", type=float, default=5000.0)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--spread-pips", type=float, default=None, help="fixed spread instead of the recorded one")
    ap.add_argument("--commission", type=float, default=0.0, help="per lot, round trip")
    ap.add_argument("--slip", type=float, default=0.3)
    ap.add_argument("--stop-min", type=float, default=3)
    ap.add_argument("--stop-max", type=float, default=15)
    ap.add_argument("--max-hold", type=int, default=45)
    ap.add_argument("--rr", type=float, default=1.5)
    ap.add_argument("--zero-cost", action="store_true", help="no spread, slippage or commission (diagnostic only)")
    a = ap.parse_args(argv)
    doc = json.load(open(a.bars))
    if doc["tf"] != "M5":
        print("error: this tool expects M5 bars", file=sys.stderr)
        return 1
    spec = doc["spec"]
    bars = merge_flow(doc["bars"], json.load(open(a.flow))["flow"])
    if len(bars) < 2000:
        print("error: only %d bars overlap the flow data" % len(bars), file=sys.stderr)
        return 1
    slip, com = a.slip, a.commission
    if a.zero_cost:
        a.spread_pips, slip, com = 0.0, 0.0, 0.0
    if a.spread_pips is not None:
        pts = a.spread_pips * spec["pip"] / spec["point"]
        bars = [dict(b, spread=pts) for b in bars]
        spec = dict(spec, spread_points=pts)
    adr = sorted(bt.daily_adr(bars).values())
    k = adr[len(adr) // 2] / (70 * spec["pip"]) if adr else 1.0
    fm = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t))
    print("%s M5 with order-flow proxy: %s -> %s (%d bars)" % (doc["symbol"], fm(bars[0]["t"]), fm(bars[-1]["t"]), len(bars)))
    pp = predictive_power(bars)
    print("does delta predict price?  same bar %+.2f | next bar %+.3f | next 3 bars %+.3f | next 6 bars %+.3f   (n=%d)"
          % (pp["same_bar"], pp["next_1"], pp["next_3"], pp["next_6"], pp["n"]))
    for name, cls in STRATEGIES:
        res, s_all, s_in, s_out = evaluate(bars, spec, cls, a.equity, a.risk, slip * k, com, rr=a.rr,
                                           max_hold_min=a.max_hold, min_stop_pips=a.stop_min, max_stop_pips=a.stop_max)
        if not s_all["trades"]:
            print("%-11s no trades" % name)
            continue
        g = lambda s: ("%+.2f R" % s["avg_r"]) if s["trades"] else "n/a"
        print("%-11s trades %4d | win %3.0f%% | avg %+.2f R (t %+.1f) | in-sample %s | out-of-sample %s | $%.0f -> $%.0f"
              % (name, s_all["trades"], s_all["win_rate"] * 100, s_all["avg_r"], s_all["t_stat"], g(s_in), g(s_out),
                 res["start_equity"], res["end_equity"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
