#!/usr/bin/env python3
"""Runner for the three spec strategies (A sweep+MSS+FVG, B trend continuation, C session breakout+retest).

    python forex-backtest/spec_run.py --config 2            # 1h trend / 15m setup / 5m entry  (16 months of data)
    python forex-backtest/spec_run.py --config 1            # 15m trend / 5m setup / 1m entry  (about 14 weeks)
    python forex-backtest/spec_run.py --config 2 --tune     # also: one-factor sensitivity + tuned set on train+validation

Needs trader/bars/<SYMBOL>_M5.json (config 2) or _M1.json (config 1) from forex-data/mt5_data.py.
Writes trader/reports/: summary, per-trade CSV, and charts. Account rules: $10,000, 0.5% risk, max 2 open,
1% daily loss stop, entries 06:00-20:00 UTC, flat 20:00 UTC. Costs: XM's recorded spread + slippage.
"""

import argparse
import csv
import json
import os
import sys
import time

import spec_engine as se

bt = se.bt
DAY = 86400
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "GOLD.i#"]
FX = SYMBOLS[:5]
FP_SPREAD_PIPS = {"EURUSD": 0.2, "GBPUSD": 0.3, "USDJPY": 0.3, "AUDUSD": 0.3, "USDCAD": 0.3, "GOLD.i#": 15.0}
CONFIGS = {
    1: dict(file="M1", tf_e=1, setup=5, trend=15, label="15m trend / 5m setup / 1m entry"),
    2: dict(file="M5", tf_e=5, setup=15, trend=60, label="1h trend / 15m setup / 5m entry"),
}
START, RISK = 10000.0, 0.5


class Frozen:
    """Lets the engine reuse a strategy that was already prepared (so reward:risk can be varied cheaply)."""

    def __init__(self, s):
        self.s = s

    def prepare(self, bars):
        pass

    def on_bar(self, bars, i):
        return self.s.on_bar(bars, i)

    def flat_ts(self, t):
        return self.s.flat_ts(t)


def make(kind, cfg, pip, **kw):
    cls = {"A": se.StrategyA, "B": se.StrategyB, "C": se.StrategyC}[kind]
    return cls(cfg["tf_e"], cfg["setup"], cfg["trend"], pip=pip, **kw)


def load(sym, cfg):
    doc = json.load(open(os.path.join("trader", "bars", "%s_%s.json" % (sym, cfg["file"]))))
    if not doc.get("spec"):
        raise SystemExit("%s has no spec; re-pull it" % sym)
    return doc["bars"], doc["spec"]


def slip_scale(bars, spec):
    adr = sorted(bt.daily_adr(bars).values())
    return adr[len(adr) // 2] / (70 * spec["pip"]) if adr else 1.0


def run(sym, kind, cfg, bars, spec, strat=None, rr=None, fp=False, zero=False, **kw):
    """One backtest. Returns trades tagged with everything the CSV needs."""
    pip = spec["pip"]
    if strat is None:
        strat = make(kind, cfg, pip, **kw)
        strat.prepare(bars)
    if rr is not None:
        strat.p["rr"] = rr
    b2, sp2, com = bars, spec, 0.0
    if fp:                                                  # FundingPips-style costs: tight spread + $5 a lot
        pts = FP_SPREAD_PIPS[sym] * spec["pip"] / spec["point"]
        b2 = [dict(b, spread=pts) for b in bars]
        sp2, com = dict(spec, spread_points=pts), 5.0
    slip = 0.3 * slip_scale(bars, spec)
    if zero:                                                # diagnostic only: no spread, slippage or commission
        b2, sp2, com, slip = [dict(b, spread=0) for b in bars], dict(spec, spread_points=0), 0.0, 0.0
    res = bt.run_backtest(b2, sp2, Frozen(strat), START, RISK, slip, 0, com)
    idx = {b["t"]: i for i, b in enumerate(bars)}
    out = []
    for t in res["trades"]:
        sg = strat.sig.get(t["i_signal"], {})
        lg = t["side"] == "long"
        t = dict(t, symbol=sym, strategy=kind, setup=sg.get("setup", ""),
                 sl_price=t["entry"] - t["sl_pips"] * pip if lg else t["entry"] + t["sl_pips"] * pip,
                 tp_price=t["entry"] + t["tp_pips"] * pip if lg else t["entry"] - t["tp_pips"] * pip)
        out.append(t)
    return out, res["skipped_too_small"], strat


def pooled(trades):
    """Plain R statistics, used for the sensitivity tables."""
    n = len(trades)
    if not n:
        return {"n": 0, "win": 0.0, "exp": 0.0, "pf": 0.0}
    w = sum(t["r"] for t in trades if t["r"] > 0)
    l = -sum(t["r"] for t in trades if t["r"] <= 0)
    return {"n": n, "win": sum(1 for t in trades if t["r"] > 0) / n, "exp": sum(t["r"] for t in trades) / n,
            "pf": (w / l) if l else float("inf")}


def periods(trades, t0, t1):
    tr, va, te = se.split_by_time(trades, t0, t1)
    return {"train": tr, "val": va, "test": te}


def acct(trades, t0, t1):
    taken, skipped = se.account_sim(trades, START, RISK, 1.0, 2)
    return taken, skipped, se.metrics(taken, START, (t1 - t0) / DAY)


def fmt_m(m):
    if not m.get("trades"):
        return "no trades"
    pf = "inf" if m["profit_factor"] == float("inf") else "%.2f" % m["profit_factor"]
    return "n %4d | win %3.0f%% [%2.0f-%2.0f] | exp %+.2f R | PF %s | ret %+7.1f%% | maxDD %5.1f%%" % (
        m["trades"], m["win_rate"] * 100, m["ci_lo"] * 100, m["ci_hi"] * 100, m["expectancy_r"], pf, m["return_pct"], m["max_dd_pct"])


def short(m):
    if not m.get("trades"):
        return "   -    "
    return "%3d %3.0f%% %+5.2fR %+6.1f%%" % (m["trades"], m["win_rate"] * 100, m["expectancy_r"], m["return_pct"])


def write_csv(path, trades):
    cols = ["instrument", "entry_time_utc", "direction", "entry_price", "stop_loss", "take_profit", "lots",
            "risk_usd", "gross_pnl_usd", "est_costs_usd", "net_pnl_usd", "r_multiple", "strategy", "setup_type",
            "session", "exit_reason", "exit_time_utc"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in sorted(trades, key=lambda x: x["t_entry"]):
            pv = 10.0 if t["symbol"] != "GOLD.i#" else 1.0
            cost = t["cost_pips_est"] * t["lots"] * (t["pnl"] / (t["pips"] * t["lots"]) if t["pips"] and t["lots"] else pv)
            h = (t["t_entry"] % DAY) // 3600
            w.writerow([t["symbol"], time.strftime("%Y-%m-%d %H:%M", time.gmtime(t["t_entry"])), t["side"],
                        "%.5f" % t["entry"], "%.5f" % t["sl_price"], "%.5f" % t["tp_price"], t["lots"],
                        "%.2f" % t["risk_money"], "%.2f" % (t["pnl"] + abs(cost)), "%.2f" % abs(cost), "%.2f" % t["pnl"],
                        "%.3f" % t["r"], t["strategy"], t["setup"],
                        "london" if 7 <= h < 12 else "new_york" if 12 <= h < 21 else "asian", t["reason"],
                        time.strftime("%Y-%m-%d %H:%M", time.gmtime(t["t_exit"]))])


def charts(name, taken, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not taken:
        return
    taken = sorted(taken, key=lambda t: t["t_exit"])
    xs = [t["t_exit"] / DAY for t in taken]
    eq = [START] + [t["equity_after"] for t in taken]
    peak, dd = START, []
    for e in eq[1:]:
        peak = max(peak, e)
        dd.append((e - peak) / peak * 100)
    months, mwin = {}, {}
    for t in taken:
        k = time.strftime("%Y-%m", time.gmtime(t["t_exit"]))
        months[k] = months.get(k, 0.0) + t["pnl_acct"] / t["equity_at_entry"] * 100
        a = mwin.setdefault(k, [0, 0])
        a[0] += t["pnl_acct"] > 0
        a[1] += 1
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    ax[0][0].plot(xs, eq[1:])
    ax[0][0].set_title("%s equity ($10,000 start)" % name)
    ax[0][1].fill_between(xs, dd, 0, color="tab:red", alpha=0.5)
    ax[0][1].set_title("drawdown %")
    ks = sorted(months)
    ax[1][0].bar(ks, [months[k] for k in ks], color=["tab:green" if months[k] > 0 else "tab:red" for k in ks])
    ax[1][0].set_title("monthly return % (sum of per-trade %)")
    ax[1][0].tick_params(axis="x", rotation=60, labelsize=7)
    ax[1][1].hist([a[0] / a[1] * 100 for a in mwin.values() if a[1] >= 3], bins=8, color="tab:blue")
    ax[1][1].axvline(60, color="k", ls="--")
    ax[1][1].set_title("monthly win-rate distribution (months with 3+ trades); dashed = 60% target")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=int, choices=[1, 2], default=2)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    a = ap.parse_args(argv)
    cfg = CONFIGS[a.config]
    data = {s: load(s, cfg) for s in a.symbols}
    fx = [s for s in a.symbols if s in FX]
    t0 = max(d[0][0]["t"] for d in data.values())
    t1 = min(d[0][-1]["t"] for d in data.values())
    fm = lambda t: time.strftime("%Y-%m-%d", time.gmtime(t))
    out = []
    P = out.append
    P("CONFIG %d: %s | data %s -> %s (%d days) | $%.0f, %.1f%% risk, max 2 open, 1%% daily loss stop"
      % (a.config, cfg["label"], fm(t0), fm(t1), (t1 - t0) // DAY, START, RISK))
    P("split 60/20/20: train to %s | validation to %s | out-of-sample after" % (
        fm(t0 + (t1 - t0) * 0.6), fm(t0 + (t1 - t0) * 0.8)))
    P("Costs: XM recorded spread + 0.3 pip slippage (scaled by volatility). Swap not modelled (all trades flat by 20:00 UTC). "
      "No news filter (no historical news data).\n")
    base = {}                                              # (kind, sym) -> (trades, strat)
    for kind in "ABC":
        for s in a.symbols:
            bars, spec = data[s]
            tr, skipped, st = run(s, kind, cfg, bars, spec)
            base[(kind, s)] = (tr, st, skipped)
            print("done baseline", kind, s, len(tr), flush=True)
    names = {"A": "A sweep+MSS+FVG", "B": "B trend continuation", "C": "C session breakout+retest"}
    results = []                                           # for the ranking
    P("=== BASELINE (fixed rules, no tuning). cells: trades, win%, expectancy, return%  ===")
    P("%-28s %-26s %-26s %-26s %-26s" % ("strategy / symbol", "ALL", "TRAIN", "VALIDATION", "OUT-OF-SAMPLE"))
    all_trades = []
    for kind in "ABC":
        groups = {s: base[(kind, s)][0] for s in a.symbols}
        groups["FX5 portfolio"] = [t for s in fx for t in groups[s]]
        groups["ALL6 portfolio"] = [t for s in a.symbols for t in groups[s]]
        for g, trs in groups.items():
            taken, _, m = acct(trs, t0, t1)
            pr = {k: acct(v, t0, t1)[2] for k, v in periods(trs, t0, t1).items()}
            P("%-28s %-26s %-26s %-26s %-26s" % ((kind + " " + g)[:28], short(m), short(pr["train"]), short(pr["val"]), short(pr["test"])))
            results.append((kind + " " + g, m, pr["test"]))
            if g == "FX5 portfolio":
                charts(names[kind] + " FX5", taken, os.path.join("trader", "reports", "config%d_%s_FX5.png" % (a.config, kind)))
                P("   %s FX5 detail: %s" % (kind, fmt_m(m)))
                P("   long %s R (%d) | short %s R (%d) | sessions(R,n) %s | sharpe %.2f | recovery %.2f | avg hold %.0f min | %.1f trades/wk"
                  % (("%+.2f" % m["long_r"]) if m["long_r"] is not None else "-", m["long_n"],
                     ("%+.2f" % m["short_r"]) if m["short_r"] is not None else "-", m["short_n"],
                     {k: (round(v[0], 2) if v[0] is not None else None, v[1]) for k, v in m["session_r"].items()},
                     m["sharpe"], m["recovery"], m["avg_hold_min"], m["trades_per_week"]))
        all_trades += groups["ALL6 portfolio"]
    write_csv(os.path.join("trader", "reports", "trades_config%d.csv" % a.config), all_trades)
    # FundingPips-style costs on the FX5 portfolio
    P("\n=== SAME RULES WITH FUNDINGPIPS-STYLE COSTS (0.2-0.3 pip spread + $5/lot; assumed, not verified) ===")
    for kind in "ABC":
        trs = []
        for s in fx:
            bars, spec = data[s]
            tr, _, _ = run(s, kind, cfg, bars, spec, strat=base[(kind, s)][1], fp=True)
            trs += tr
        _, _, m = acct(trs, t0, t1)
        _, _, mt = acct(periods(trs, t0, t1)["test"], t0, t1)
        P("%-24s ALL %s" % (names[kind], fmt_m(m)))
    # rr variants (cheap: signals are reused)
    P("\n=== TAKE-PROFIT VARIANTS, FX5 portfolio: fixed reward:risk (all periods | out-of-sample) ===")
    for kind in "ABC":
        row = []
        for rr in (1.5, 2.0, 2.5, 3.0):
            trs = []
            for s in fx:
                bars, spec = data[s]
                tr, _, _ = run(s, kind, cfg, bars, spec, strat=base[(kind, s)][1], rr=rr)
                trs += tr
            m = acct(trs, t0, t1)[2]
            mo = acct(periods(trs, t0, t1)["test"], t0, t1)[2]
            row.append("1:%.1f  win %2.0f%% exp %+.2f R ret %+6.1f%% | OOS n %d exp %+.2f R" % (
                rr, m.get("win_rate", 0) * 100, m.get("expectancy_r", 0), m.get("return_pct", 0), mo.get("trades", 0), mo.get("expectancy_r", 0)))
        P(names[kind])
        for r in row:
            P("   " + r)
        for s in fx:
            base[(kind, s)][1].p["rr"] = 2.0
    # ranking
    P("\n=== RANKING by full-period return % (best -> worst). OOS = the last 20%% of the data ===")
    rk = sorted(results, key=lambda r: r[1].get("return_pct", -999), reverse=True)
    P("%-4s %-30s %-10s %-9s %-9s %-10s %-9s %-9s" % ("#", "strategy / symbol", "return %", "win %", "exp R", "maxDD %", "OOS ret%", "OOS n"))
    for i, (nm, m, mt) in enumerate(rk, 1):
        if not m.get("trades"):
            continue
        P("%-4d %-30s %+9.1f %8.0f%% %+8.2f %9.1f %+9.1f %9d" % (i, nm, m["return_pct"], m["win_rate"] * 100,
                                                              m["expectancy_r"], m["max_dd_pct"], mt.get("return_pct", 0.0), mt.get("trades", 0)))
    if a.tune:
        tune(a, cfg, data, fx, t0, t1, P, names)
    text = "\n".join(out)
    os.makedirs(os.path.join("trader", "reports"), exist_ok=True)
    open(os.path.join("trader", "reports", "summary_config%d.txt" % a.config), "w", encoding="utf-8").write(text)
    print(text)


def tune(a, cfg, data, fx, t0, t1, P, names):
    """One-factor-at-a-time sensitivity, pooled over the five FX pairs, judged on train+validation only."""
    cut = t0 + (t1 - t0) * 0.8

    def pooled_run(kind, **kw):
        trs = []
        for s in fx:
            bars, spec = data[s]
            tr, _, _ = run(s, kind, cfg, bars, spec, **kw)
            trs += tr
        dev = [t for t in trs if t["t_entry"] < cut]
        oos = [t for t in trs if t["t_entry"] >= cut]
        return pooled(dev), pooled(oos), trs

    grids = {
        "A": [("fvg_atr", [0.1, 0.15, 0.2]), ("buf_atr", [0.05, 0.10, 0.15]), ("rr", [1.5, 2.0, 2.5, 3.0]),
              ("trend_mode", ["full", "struct", "none"])],
        "B": [("rr", [1.5, 2.0, 2.5]), ("rsi_filter", [False, True]), ("adx_min", [0.0, 20.0, 25.0]), ("atr_filter", [False, True])],
        "C": [("session", ["london", "overlap", "ny"]), ("range_min", [15, 30, 60]), ("rr", [1.5, 2.0, 2.5]),
              ("trend_mode", ["none", "struct", "full"])],
    }
    P("\n=== PARAMETER SENSITIVITY (one factor at a time; FX5 pooled; selection uses train+validation ONLY) ===")
    P("%-24s %-22s %-34s %-30s" % ("strategy", "setting", "train+validation (n, win, exp R, PF)", "out-of-sample (n, win, exp R)"))
    for kind in "ABC":
        best = {}
        for param, vals in grids[kind]:
            sc = []
            for v in vals:
                dev, oos, _ = pooled_run(kind, **{param: v})
                P("%-24s %-22s n %4d win %3.0f%% %+.2f R PF %-5.2f   n %4d win %3.0f%% %+.2f R" % (
                    names[kind][:24], "%s=%s" % (param, v), dev["n"], dev["win"] * 100, dev["exp"], min(dev["pf"], 99),
                    oos["n"], oos["win"] * 100, oos["exp"]))
                sc.append((dev["exp"] if dev["n"] >= 30 else -9, v))
            best[param] = max(sc, key=lambda x: x[0])[1]
        dev, oos, trs = pooled_run(kind, **best)
        taken, _, m = acct(trs, t0, t1)
        mo = acct([t for t in trs if t["t_entry"] >= cut], t0, t1)[2]
        P(">> %s TUNED on train+val: %s\n   train+val exp %+.2f R (n %d) | OUT-OF-SAMPLE n %d win %.0f%% exp %+.2f R PF %.2f ret %+.1f%%"
          % (names[kind], best, dev["exp"], dev["n"], oos["n"], oos["win"] * 100, oos["exp"], min(oos["pf"], 99), mo.get("return_pct", 0.0)))
    P("\n=== WALK-FORWARD (baseline rules, FX5 pooled; rules were never refitted, so every window is out-of-sample) ===")
    win = 60 * DAY if a.config == 2 else 21 * DAY
    for kind in "ABC":
        trs = []
        for s in fx:
            bars, spec = data[s]
            tr, _, _ = run(s, kind, cfg, bars, spec)
            trs += tr
        row, t = [], t0
        while t + win <= t1 + DAY:
            w = pooled([x for x in trs if t <= x["t_entry"] < t + win])
            row.append("%s n%d %+.2fR" % (time.strftime("%b%y", time.gmtime(t)), w["n"], w["exp"]))
            t += win
        P("%-24s %s" % (names[kind], " | ".join(row)))


if __name__ == "__main__":
    sys.exit(main())
