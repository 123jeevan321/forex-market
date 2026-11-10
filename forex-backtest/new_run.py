#!/usr/bin/env python3
"""Runner and reports for strategies D-H (and the old A-C for the unified comparison). Run from the repo root:

    python forex-backtest/new_run.py                  # everything: baselines, 3 cost scenarios, sensitivity, walk-forward
    python forex-backtest/new_run.py --kinds D        # one strategy
    python forex-backtest/new_run.py --kinds D --symbols EURUSD --no-tune     # quick check

Config: 1h regime/trend, 15m setup, 5m entry (G: 4h trend). Same account rules as the A/B/C tests:
$10,000, 0.5% risk, max 2 open, 1% daily loss stop, entries 06:00-20:00 UTC, flat 20:00 UTC.
Old reports (trader/reports) are never touched. New ones go to reports/ in the repo.
"""

import argparse
import csv
import json
import os
import sys
import time

import spec_run as R
import new_strategies as ns

se = R.se
bt = R.bt
DAY = 86400
CFG = R.CONFIGS[2]
SYMBOLS, FX = R.SYMBOLS, R.FX
FOLDERS = {"D": "strategy_D_vwap", "E": "strategy_E_volatility", "F": "strategy_F_failed_breakout",
           "G": "strategy_G_pullback", "H": "strategy_H_asian_range"}
NAMES = {"A": "A sweep+MSS+FVG", "B": "B trend continuation", "C": "C session breakout+retest",
         "D": "D VWAP mean reversion", "E": "E volatility squeeze", "F": "F failed breakout",
         "G": "G HTF pullback", "H": "H Asian range reversion"}
SCEN = {"xm": dict(), "funding_assumed": dict(fp=True), "zero_cost": dict(zero=True)}
GRIDS = {
    "D": [("sd", [1.5, 2.0, 2.5]), ("adx_max", [18.0, 20.0, 25.0]), ("tp_mode", ["vwap", "rr1.5", "rr2", "partial"]),
          ("slope_filter", [False, True]), ("atr_expansion", [False, True])],
    "E": [("squeeze_n", [5, 10, 20]), ("entry_mode", ["immediate", "retest"]), ("filt", ["none", "ema50", "adx", "vwap"]),
          ("rr", [1.5, 2.0, 2.5])],
    "F": [("target", ["mid", "opp", "vwap"]), ("levels", [("pd",), ("asian",), ("london",), ("pd", "asian", "london")])],
    "G": [("momentum", ["none", "rsi", "stoch", "macd", "ema"]), ("rr", [1.5, 2.0, 3.0]), ("depth_min", [0.3, 0.5, 1.0])],
    "H": [("session", ["london", "ny"]), ("target", ["mid", "opp"])],
}


def build(kind, pip, **kw):
    if kind in "ABC":
        return R.make(kind, CFG, pip, **kw)
    cls = {"D": ns.StrategyD, "E": ns.StrategyE, "F": ns.StrategyF, "G": ns.StrategyG, "H": ns.StrategyH}[kind]
    return cls(CFG["tf_e"], CFG["setup"], 240 if kind == "G" else CFG["trend"], pip=pip, **kw)


class Lab:
    def __init__(self, symbols):
        self.data = {s: R.load(s, CFG) for s in symbols}
        self.symbols = symbols
        self.fx = [s for s in symbols if s in FX]
        self.t0 = max(d[0][0]["t"] for d in self.data.values())
        self.t1 = min(d[0][-1]["t"] for d in self.data.values())
        self.cache = {}

    def strat(self, kind, sym, kw):
        key = (kind, sym, tuple(sorted((k, repr(v)) for k, v in kw.items() if k != "rr")))
        if key not in self.cache:
            bars, spec = self.data[sym]
            st = build(kind, spec["pip"], **{k: v for k, v in kw.items() if k != "rr"})
            st.prepare(bars)
            st._rr0 = st.p["rr"]
            self.cache[key] = st
        st = self.cache[key]
        st.p["rr"] = kw.get("rr", st._rr0)
        return st

    def trades(self, kind, sym, kw=None, scen="xm"):
        kw = kw or {}
        bars, spec = self.data[sym]
        tr, skipped, _ = R.run(sym, kind, CFG, bars, spec, strat=self.strat(kind, sym, kw), **SCEN[scen])
        return tr

    def group(self, kind, syms, kw=None, scen="xm"):
        return [t for s in syms for t in self.trades(kind, s, kw, scen)]

    def split_cut(self):
        return self.t0 + (self.t1 - self.t0) * 0.8


def est_costs(lab, trades, scen):
    tot = 0.0
    for t in trades:
        pv = lab.data[t["symbol"]][1]["pip_value_per_lot"]
        tot += t["cost_pips_est"] * t["lots"] * pv + (5.0 * t["lots"] if scen == "funding_assumed" else 0.0)
    return tot


def wf_windows(lab, trades, days=60):
    out, t = [], lab.t0
    while t + days * DAY <= lab.t1 + DAY:
        w = R.pooled([x for x in trades if t <= x["t_entry"] < t + days * DAY])
        out.append((time.strftime("%b%y", time.gmtime(t)), w["n"], w["exp"]))
        t += days * DAY
    return out


def row_for(lab, kind, group, trs, scen):
    taken, _, m = R.acct(trs, lab.t0, lab.t1)
    oos = R.acct(R.periods(trs, lab.t0, lab.t1)["test"], lab.t0, lab.t1)[2]
    wf = [w for w in wf_windows(lab, trs) if w[1] >= 5]
    r = {"strategy": NAMES[kind], "group": group, "scenario": scen, "trades": m.get("trades", 0)}
    if m.get("trades"):
        sr = m["session_r"]
        r.update({"win_rate": round(m["win_rate"], 4), "win_ci_lo": round(m["ci_lo"], 4), "win_ci_hi": round(m["ci_hi"], 4),
                  "avg_win_r": round(m["avg_win_r"] or 0, 3), "avg_loss_r": round(m["avg_loss_r"] or 0, 3),
                  "profit_factor": round(min(m["profit_factor"], 99), 3), "expectancy_r": round(m["expectancy_r"], 4),
                  "net_return_pct": round(m["return_pct"], 2), "max_drawdown_pct": round(m["max_dd_pct"], 2),
                  "sharpe": round(m["sharpe"], 2), "recovery_factor": round(min(m["recovery"], 99), 2),
                  "max_consec_losses": m["max_losses"], "avg_hold_min": round(m["avg_hold_min"], 1),
                  "est_costs_usd": round(est_costs(lab, trs, scen), 2),
                  "asian_r": round(sr["asian"][0], 3) if sr["asian"][0] is not None else "", "asian_n": sr["asian"][1],
                  "london_r": round(sr["london"][0], 3) if sr["london"][0] is not None else "", "london_n": sr["london"][1],
                  "ny_r": round(sr["ny"][0], 3) if sr["ny"][0] is not None else "", "ny_n": sr["ny"][1],
                  "long_r": round(m["long_r"], 3) if m["long_r"] is not None else "",
                  "short_r": round(m["short_r"], 3) if m["short_r"] is not None else "",
                  "wf_windows": len(wf), "wf_positive": sum(1 for w in wf if w[2] > 0)})
    if oos.get("trades"):
        r.update({"oos_trades": oos["trades"], "oos_win_rate": round(oos["win_rate"], 4),
                  "oos_expectancy_r": round(oos["expectancy_r"], 4), "oos_return_pct": round(oos["return_pct"], 2),
                  "oos_profit_factor": round(min(oos["profit_factor"], 99), 3)})
    return r, taken, m, oos


COLS = ["strategy", "group", "scenario", "trades", "win_rate", "win_ci_lo", "win_ci_hi", "avg_win_r", "avg_loss_r",
        "profit_factor", "expectancy_r", "net_return_pct", "max_drawdown_pct", "sharpe", "recovery_factor",
        "max_consec_losses", "avg_hold_min", "est_costs_usd", "asian_r", "asian_n", "london_r", "london_n", "ny_r",
        "ny_n", "long_r", "short_r", "wf_windows", "wf_positive", "oos_trades", "oos_win_rate", "oos_expectancy_r",
        "oos_return_pct", "oos_profit_factor"]


def cell(m):
    return "-" if not m.get("trades") else "%3d %3.0f%% %+5.2fR %+6.1f%%" % (m["trades"], m["win_rate"] * 100, m["expectancy_r"], m["return_pct"])


def fmt(m):
    if not m.get("trades"):
        return "no trades"
    pf = "inf" if m["profit_factor"] == float("inf") else "%.2f" % m["profit_factor"]
    return "n %4d | win %3.0f%% [%2.0f-%2.0f] | exp %+.2f R | PF %s | ret %+6.1f%% | maxDD %5.1f%% | sharpe %+.2f" % (
        m["trades"], m["win_rate"] * 100, m["ci_lo"] * 100, m["ci_hi"] * 100, m["expectancy_r"], pf, m["return_pct"],
        m["max_dd_pct"], m["sharpe"])


def report_kind(lab, kind, rows, tune, out_dir):
    L = []
    P = L.append
    P("%s | data %s -> %s | $10,000, 0.5%% risk, max 2 open, 1%% daily loss stop | split 60/20/20 chronological"
      % (NAMES[kind], time.strftime("%Y-%m-%d", time.gmtime(lab.t0)), time.strftime("%Y-%m-%d", time.gmtime(lab.t1))))
    P("BASELINE (rules fixed in advance, not tuned). cells: trades, win%, expectancy, return%")
    P("%-16s %-26s %-26s %-26s %-26s" % ("group", "ALL", "TRAIN", "VALIDATION", "OUT-OF-SAMPLE"))
    groups = {s: lab.trades(kind, s) for s in lab.symbols}
    groups["FX5 portfolio"] = [t for s in lab.fx for t in groups[s]]
    if len(lab.symbols) == len(SYMBOLS):
        groups["ALL6 portfolio"] = [t for s in lab.symbols for t in groups[s]]
    fx_taken = fx_m = None
    for g, trs in groups.items():
        taken, _, m = R.acct(trs, lab.t0, lab.t1)
        pr = {k: R.acct(v, lab.t0, lab.t1)[2] for k, v in R.periods(trs, lab.t0, lab.t1).items()}
        P("%-16s %-26s %-26s %-26s %-26s" % (g, cell(m), cell(pr["train"]), cell(pr["val"]), cell(pr["test"])))
        if g == "FX5 portfolio":
            fx_taken, fx_m = taken, m
    if fx_m and fx_m.get("trades"):
        m = fx_m
        P("FX5 detail: " + fmt(m))
        P("  long %s R (%d) | short %s R (%d) | sessions (R, n) %s | avg win %+.2f R | avg loss %+.2f R | max consecutive losses %d | avg hold %.0f min | %.1f trades/week"
          % ("%+.2f" % m["long_r"] if m["long_r"] is not None else "-", m["long_n"],
             "%+.2f" % m["short_r"] if m["short_r"] is not None else "-", m["short_n"],
             {k: (round(v[0], 2) if v[0] is not None else None, v[1]) for k, v in m["session_r"].items()},
             m["avg_win_r"] or 0, m["avg_loss_r"] or 0, m["max_losses"], m["avg_hold_min"], m["trades_per_week"]))
        R.charts(NAMES[kind] + " (FX5 portfolio, XM costs)", fx_taken, os.path.join(out_dir, "equity_drawdown_monthly_FX5.png"))
    # trade log (XM scenario, every instrument)
    R.write_csv(os.path.join(out_dir, "trades_xm_costs.csv"), [t for trs in (groups[s] for s in lab.symbols) for t in trs])
    with open(os.path.join(out_dir, "equity_curve_FX5.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exit_time_utc", "equity_usd"])
        for t in fx_taken or []:
            w.writerow([time.strftime("%Y-%m-%d %H:%M", time.gmtime(t["t_exit"])), "%.2f" % t["equity_after"]])
    # cost scenarios
    P("\nCOST SCENARIOS (same baseline signals). funding_assumed = 0.2-0.3 pip spread + $5/lot; the $5 commission is from the "
      "FundingPips help-centre text as reported by search, the spreads are NOT official.")
    for scen in SCEN:
        for g, syms in (("FX5 portfolio", lab.fx), ("GOLD.i# (separate)", ["GOLD.i#"])):
            syms = [s for s in syms if s in lab.symbols]
            if not syms:
                continue
            trs = lab.group(kind, syms, None, scen)
            r, _, m, oos = row_for(lab, kind, g.split(" (")[0], trs, scen)
            rows.append(r)
            P("  %-16s %-18s ALL %s | OOS %s" % (scen, g, fmt(m), cell(oos)))
    for s in lab.symbols:                                    # per-symbol rows for the CSV (xm only)
        r, *_ = row_for(lab, kind, s, groups[s], "xm")
        rows.append(r)
    for scen in ("funding_assumed", "zero_cost"):
        for s in lab.symbols:
            r, *_ = row_for(lab, kind, s, lab.trades(kind, s, None, scen), scen)
            rows.append(r)
    wf = wf_windows(lab, groups["FX5 portfolio"])
    P("\nWALK-FORWARD (rules never refitted, FX5 pooled, 2-month windows, XM costs): " +
      " | ".join("%s n%d %+.2fR" % w for w in wf))
    pos = [w for w in wf if w[1] >= 5 and w[2] > 0]
    P("  windows with 5+ trades and positive expectancy: %d of %d" % (len(pos), len([w for w in wf if w[1] >= 5])))
    if tune:
        tune_kind(lab, kind, P)
    P("\nLIMITS: tick-volume VWAP is broker-specific; candle data hides intrabar order (stop assumed first when both are touched); "
      "swap not modelled (all trades flat by 20:00 UTC); no news filter; 16 months of 5-minute data only (not 2 years).")
    return L, groups


def tune_kind(lab, kind, P):
    cut = lab.split_cut()
    P("\nPARAMETER SENSITIVITY (one factor at a time, FX5 pooled; choice uses train+validation ONLY, out-of-sample shown for honesty)")
    P("%-26s %-34s %-30s" % ("setting", "train+val (n, win, exp R, PF)", "out-of-sample (n, win, exp R)"))
    best = {}
    for param, vals in GRIDS[kind]:
        sc = []
        for v in vals:
            trs = lab.group(kind, lab.fx, {param: v})
            dev = R.pooled([t for t in trs if t["t_entry"] < cut])
            oos = R.pooled([t for t in trs if t["t_entry"] >= cut])
            P("%-26s n %4d win %3.0f%% %+.2f R PF %-5.2f   n %4d win %3.0f%% %+.2f R" % (
                "%s=%s" % (param, v), dev["n"], dev["win"] * 100, dev["exp"], min(dev["pf"], 99), oos["n"], oos["win"] * 100, oos["exp"]))
            sc.append((dev["exp"] if dev["n"] >= 30 else -9, v))
        best[param] = max(sc, key=lambda x: x[0])[1]
    trs = lab.group(kind, lab.fx, best)
    dev = R.pooled([t for t in trs if t["t_entry"] < cut])
    oos_trs = [t for t in trs if t["t_entry"] >= cut]
    oos = R.pooled(oos_trs)
    mo = R.acct(oos_trs, lab.t0, lab.t1)[2]
    P(">> TUNED on train+validation: %s\n   train+val n %d exp %+.2f R | OUT-OF-SAMPLE n %d win %.0f%% exp %+.2f R PF %.2f return %+.1f%%"
      % (best, dev["n"], dev["exp"], oos["n"], oos["win"] * 100, oos["exp"], min(oos["pf"], 99), mo.get("return_pct", 0.0)))


def coverage(symbols):
    L = ["DATA COVERAGE (exact, from the files used)", "%-10s %-4s %9s %-17s %-17s %6s" % ("symbol", "tf", "bars", "first (UTC)", "last (UTC)", "days")]
    for s in symbols:
        for tf in ("M1", "M5", "H1"):
            p = os.path.join("trader", "bars", "%s_%s.json" % (s, tf))
            if not os.path.exists(p):
                continue
            b = json.load(open(p))["bars"]
            L.append("%-10s %-4s %9d %-17s %-17s %6d" % (s, tf, len(b), time.strftime("%Y-%m-%d %H:%M", time.gmtime(b[0]["t"])),
                                                       time.strftime("%Y-%m-%d %H:%M", time.gmtime(b[-1]["t"])), (b[-1]["t"] - b[0]["t"]) // DAY))
    L.append("These strategies use 1h/4h regime, 15m setup and 5m entry, so they run on the M5 files (about 16 months). "
             "The M1 files (about 97 days) are NOT used and are not a two-year test.")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="DEFGH")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--old", action="store_true", help="also recompute A/B/C on the same engine for the unified table")
    a = ap.parse_args(argv)
    lab = Lab(a.symbols)
    rows, summary = [], {}
    cov = coverage(a.symbols)
    os.makedirs("reports", exist_ok=True)
    open(os.path.join("reports", "data_coverage.txt"), "w").write("\n".join(cov))
    print("\n".join(cov), flush=True)
    for kind in a.kinds:
        out = os.path.join("reports", FOLDERS[kind])
        os.makedirs(out, exist_ok=True)
        L, groups = report_kind(lab, kind, rows, not a.no_tune, out)
        open(os.path.join(out, "summary.txt"), "w", encoding="utf-8").write("\n".join(L))
        summary[kind] = groups
        print("\n".join(L), flush=True)
    if a.old or a.kinds == "DEFGH":
        for kind in "ABC":
            for s in lab.symbols:
                lab.trades(kind, s)
    # unified comparison, FX5 portfolio, XM costs, rules fixed in advance
    U = ["UNIFIED COMPARISON: FX5 portfolio (EURUSD GBPUSD USDJPY AUDUSD USDCAD), XM costs, fixed baseline rules",
         "%-26s %7s %7s %6s %8s %8s %9s %9s %8s %9s" % ("strategy", "trades", "win %", "PF", "exp R", "maxDD %", "return %", "OOS ret %", "OOS n", "OOS exp R")]
    ranking = []
    for kind in "ABCDEFGH":
        if kind not in "ABC" and kind not in a.kinds:
            continue
        trs = lab.group(kind, lab.fx)
        r, _, m, oos = row_for(lab, kind, "FX5 portfolio", trs, "xm")
        if kind in "ABC":
            rows.append(r)
        if not m.get("trades"):
            continue
        U.append("%-26s %7d %6.0f%% %6.2f %+8.2f %8.1f %+9.1f %+9.1f %8d %+9.2f" % (
            NAMES[kind], m["trades"], m["win_rate"] * 100, min(m["profit_factor"], 99), m["expectancy_r"], m["max_dd_pct"],
            m["return_pct"], oos.get("return_pct", 0.0), oos.get("trades", 0), oos.get("expectancy_r", 0.0)))
        ranking.append((oos.get("expectancy_r", -9) if oos.get("trades", 0) >= 20 else -9, NAMES[kind], m, oos))
    U.append("\nRANKING by out-of-sample expectancy (needs 20+ OOS trades), then drawdown. Not ranked by win rate.")
    for i, (e, nm, m, oos) in enumerate(sorted(ranking, key=lambda x: (-x[0], x[2]["max_dd_pct"])), 1):
        U.append("%d. %-26s OOS exp %+.2f R (n %d) | full-period exp %+.2f R | win %.0f%% | PF %.2f | maxDD %.1f%%"
                 % (i, nm, oos.get("expectancy_r", 0), oos.get("trades", 0), m["expectancy_r"], m["win_rate"] * 100,
                    min(m["profit_factor"], 99), m["max_dd_pct"]))
    open(os.path.join("reports", "unified_comparison.txt"), "w", encoding="utf-8").write("\n".join(U))
    print("\n".join(U))
    path = os.path.join("reports", "new_strategies_comparison.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path, len(rows), "rows")


if __name__ == "__main__":
    sys.exit(main())
