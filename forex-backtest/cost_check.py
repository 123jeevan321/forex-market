#!/usr/bin/env python3
"""Diagnostic: how much of a scalping strategy's result is the strategy, and how much is trading cost?

    python cost_check.py trader/bars/EURUSD_M5.json [more files...]

Runs the liquidity-sweep scalp three ways: with the broker's real spread, with zero costs (not achievable,
shown only to separate signal from cost), and at several reward:risk settings to show win rate vs payoff.
"""

import importlib.util
import json
import os
import sys

_spec = importlib.util.spec_from_file_location("bt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def run(bars, spec, rr, costs=True, equity=1e6):
    if not costs:
        bars = [dict(b, spread=0) for b in bars]
        spec = dict(spec, spread_points=0)
    st = bt.LiquiditySweep(5, spec, rr=rr)
    res = bt.run_backtest(bars, spec, st, equity=equity, risk_pct=0.1, slip_pips=0.3 if costs else 0.0)
    return bt.summarize(res["trades"], equity, res["end_equity"]), res["trades"]


def main(paths):
    for path in paths:
        doc = json.load(open(path))
        bars, spec = doc["bars"], doc["spec"]
        pip, point = spec["pip"], spec["point"]
        st = bt.LiquiditySweep(5, spec)
        st.prepare(bars)
        stops = [s["sl_pips"] for s in (st.on_bar(bars, i) for i in range(len(bars))) if s]
        spreads = sorted(b["spread"] * point / pip for b in bars if 7 * 3600 <= b["t"] % 86400 < 15 * 3600)
        med_stop = sorted(stops)[len(stops) // 2]
        med_spread = spreads[len(spreads) // 2]
        print("\n%s  median stop %.1f pips | median spread %.1f pips | cost of one round trip = %.0f%% of the risk"
              % (doc["symbol"], med_stop, med_spread, 100 * (med_spread + 0.6) / med_stop))
        print("  %-22s %7s %7s %9s %7s" % ("setting", "trades", "win %", "avg R", "t"))
        for label, rr, costs in (("target 0.5R, real cost", 0.5, True), ("target 1R,   real cost", 1.0, True),
                                 ("target 1.5R, real cost", 1.5, True), ("target 3R,   real cost", 3.0, True),
                                 ("target 0.5R, ZERO cost", 0.5, False), ("target 1.5R, ZERO cost", 1.5, False)):
            s, _ = run(bars, spec, rr, costs)
            print("  %-22s %7d %6.0f%% %+9.2f %7.1f" % (label, s["trades"], s["win_rate"] * 100, s["avg_r"], s["t_stat"]))


if __name__ == "__main__":
    main(sys.argv[1:])
