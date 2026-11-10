#!/usr/bin/env python3
"""Prop-firm challenge simulator. Offline. Answers: with THIS strategy's real backtested trades, how often
would a two-phase funded-account challenge be passed, and what is it worth after the fee?

    python backtest.py run --bars trader/bars/EURUSD_M5.json --equity 5000 --write
    python challenge.py trader/backtests/EURUSD_M5_breakout.json --risk 0.5 1 2 --fee 36

Method: the backtest's trades are grouped by day, and whole days are drawn at random (with replacement) to
build thousands of possible futures. Each trade risks a fixed percent of the STARTING balance.

The rules are parameters because firms differ and change them. CHECK THEM against the firm's current page:
  phase 1 target 8%, phase 2 target 5%, max total loss 10% of the starting balance (static),
  max daily loss 5% of the starting balance, 80% profit split, payout every 10 trading days.
The daily-loss check here uses closed trades only; real firms also count open (floating) losses, which is
stricter. Consistency rules, news rules and minimum-day rules are not modelled.
"""

import argparse
import json
import random
import sys

DAY = 86400


def load_days(path):
    """Backtest result file -> list of days, each a list of R multiples (results in units of risk)."""
    with open(path) as f:
        doc = json.load(f)
    days = {}
    for t in doc["result"]["trades"]:
        days.setdefault(t["t_entry"] // DAY, []).append(t["r"])
    return [days[d] for d in sorted(days)]


def calendar_stats(path, inactivity_days=30):
    """How the strategy's trades are spread over real time: calendar days per trading day, the longest wait
    between closed trades, and how many waits were longer than the firm's inactivity limit."""
    with open(path) as f:
        trades = json.load(f)["result"]["trades"]
    exits = sorted(t["t_exit"] for t in trades)
    span = max(exits[-1] - exits[0], DAY)
    gaps = [(b - a) / float(DAY) for a, b in zip(exits, exits[1:])]
    n_days = len(set(t["t_entry"] // DAY for t in trades))
    return {"cal_per_trading_day": span / float(DAY) / n_days, "trades_per_year": len(trades) * 365.0 * DAY / span,
            "longest_gap": max(gaps) if gaps else 0.0, "gaps_over_limit": sum(1 for g in gaps if g > inactivity_days),
            "years": span / (365.0 * DAY)}


def run_phase(days, rng, balance, risk_pct, target_pct, max_loss_pct, daily_loss_pct, max_days=3000):
    """Returns (passed, trading days used)."""
    eq, risk = balance, balance * risk_pct / 100.0
    floor, goal, day_cap = balance * (1 - max_loss_pct / 100.0), balance * (1 + target_pct / 100.0), balance * daily_loss_pct / 100.0
    for n in range(1, max_days + 1):
        day_pnl = 0.0
        for r in rng.choice(days):
            pnl = r * risk
            eq += pnl
            day_pnl += pnl
            if eq <= floor or day_pnl <= -day_cap:
                return False, n
            if eq >= goal:
                return True, n
    return False, max_days


def run_funded(days, rng, balance, risk_pct, max_loss_pct, daily_loss_pct, split_pct, payout_every, funded_days):
    """Returns (total paid out to the trader, survived to the end)."""
    eq, risk, paid = balance, balance * risk_pct / 100.0, 0.0
    floor, day_cap = balance * (1 - max_loss_pct / 100.0), balance * daily_loss_pct / 100.0
    for n in range(1, funded_days + 1):
        day_pnl = 0.0
        for r in rng.choice(days):
            pnl = r * risk
            eq += pnl
            day_pnl += pnl
            if eq <= floor or day_pnl <= -day_cap:
                return paid, False
        if n % payout_every == 0 and eq > balance:
            paid += (eq - balance) * split_pct / 100.0
            eq = balance                                   # profit withdrawn, account back to its start
    return paid, True


def simulate(days, risk_pct, a, seed=1):
    rng = random.Random(seed)
    p1 = p2 = got_paid = 0
    paid_total, days_to_fund = 0.0, []
    for _ in range(a.sims):
        ok, n1 = run_phase(days, rng, a.balance, risk_pct, a.target1, a.max_loss, a.daily_loss)
        if not ok:
            continue
        p1 += 1
        ok, n2 = run_phase(days, rng, a.balance, risk_pct, a.target2, a.max_loss, a.daily_loss)
        if not ok:
            continue
        p2 += 1
        days_to_fund.append(n1 + n2)
        paid, _ = run_funded(days, rng, a.balance, risk_pct, a.max_loss, a.daily_loss, a.split, a.payout_every,
                             a.funded_days)
        paid_total += paid
        got_paid += paid > 0
    n = float(a.sims)
    days_to_fund.sort()
    return {"risk": risk_pct, "pass1": p1 / n, "pass_both": p2 / n, "any_payout": got_paid / n,
            "avg_payout_per_attempt": paid_total / n, "ev_per_attempt": paid_total / n - a.fee,
            "median_days_to_fund": days_to_fund[len(days_to_fund) // 2] if days_to_fund else None}


def coin_flip_pass(target_pct, max_loss_pct):
    """A trader with NO edge and no costs reaches +target before -max_loss with this probability."""
    return max_loss_pct / float(max_loss_pct + target_pct)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prop-firm challenge simulator.")
    ap.add_argument("files", nargs="+", help="backtest result files (backtest.py run --write)")
    ap.add_argument("--risk", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="percent of balance per trade")
    ap.add_argument("--balance", type=float, default=5000.0)
    ap.add_argument("--target1", type=float, default=8.0)
    ap.add_argument("--target2", type=float, default=5.0)
    ap.add_argument("--max-loss", type=float, default=10.0)
    ap.add_argument("--daily-loss", type=float, default=5.0)
    ap.add_argument("--split", type=float, default=80.0)
    ap.add_argument("--payout-every", type=int, default=10, help="trading days between payouts")
    ap.add_argument("--funded-days", type=int, default=120, help="trading days simulated on the funded account")
    ap.add_argument("--fee", type=float, default=36.0, help="challenge fee, account currency")
    ap.add_argument("--inactivity", type=int, default=30, help="firm breaches the account after this many days without a closed trade")
    ap.add_argument("--sims", type=int, default=5000)
    a = ap.parse_args(argv)
    cf = coin_flip_pass(a.target1, a.max_loss) * coin_flip_pass(a.target2, a.max_loss)
    print("$%.0f account | phase 1 +%.0f%% | phase 2 +%.0f%% | max loss %.0f%% | daily loss %.0f%% | split %.0f%% | fee $%.0f"
          % (a.balance, a.target1, a.target2, a.max_loss, a.daily_loss, a.split, a.fee))
    print("Reference: a coin-flip trader with zero costs passes both phases %.0f%% of the time.\n" % (cf * 100))
    for path in a.files:
        days = load_days(path)
        rs = [r for d in days for r in d]
        if len(rs) < 30:
            print("%s: only %d trades, skipped" % (path, len(rs)))
            continue
        print("%s  (%d trades over %d trading days, average %+.2f R)" % (path, len(rs), len(days), sum(rs) / len(rs)))
        c = calendar_stats(path, a.inactivity)
        print("  %.0f trades a year | longest wait between trades %.0f days | waits over %d days: %d in %.1f years%s"
              % (c["trades_per_year"], c["longest_gap"], a.inactivity, c["gaps_over_limit"], c["years"],
                 "  <-- would breach the inactivity rule" if c["gaps_over_limit"] else ""))
        print("  %6s %10s %10s %12s %14s %14s %16s" % ("risk", "pass ph.1", "pass both", "any payout", "avg payout", "net of fee", "months to fund"))
        for risk in a.risk:
            s = simulate(days, risk, a)
            months = "%.0f" % (s["median_days_to_fund"] * c["cal_per_trading_day"] / 30.4) if s["median_days_to_fund"] else "-"
            print("  %5.1f%% %9.0f%% %9.0f%% %11.0f%% %14.2f %+14.2f %16s"
                  % (risk, s["pass1"] * 100, s["pass_both"] * 100, s["any_payout"] * 100,
                     s["avg_payout_per_attempt"], s["ev_per_attempt"], months))
        print("  (funded stage simulated: about %.1f years of this strategy's trading)"
              % (a.funded_days * c["cal_per_trading_day"] / 365.0))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
