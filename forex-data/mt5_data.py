#!/usr/bin/env python3
"""MT5 data layer for forex-market. READ-ONLY: it never places or changes an order.

Needs the MetaTrader 5 terminal (Windows) open and logged in, and `pip install MetaTrader5`.

    python mt5_data.py status                           # terminal + account (demo/real) + server
    python mt5_data.py symbols --match USD              # symbols your broker offers
    python mt5_data.py spec EURUSD                      # pip size, lot rules, spread, tick value
    python mt5_data.py bars EURUSD --tf M5 --count 500  # recent candles
    python mt5_data.py bars EURUSD --tf M5 --count 5000 --write   # -> trader/bars/EURUSD_M5.json
    python mt5_data.py bars --csv file.csv --tf M5      # offline: MT5 "Export Bars" CSV

Times are UTC (MT5 server time is converted using --server-offset hours, default read
from the newest tick). Bars are dicts: t (UTC unix), o, h, l, c, v, spread (points).
"""

import argparse
import csv
import json
import os
import sys
import time

TIMEFRAMES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


class DataError(Exception):
    pass


def load_mt5():
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise DataError("MetaTrader5 package missing: pip install MetaTrader5")
    return mt5


def connect(mt5=None):
    """Attach to the already-running terminal. Returns the module."""
    mt5 = mt5 or load_mt5()
    if not mt5.initialize():
        raise DataError("cannot reach MT5 terminal (open it and log in): %s" % (mt5.last_error(),))
    return mt5


def pip_size(digits, point):
    """A pip is 10 points on 3/5-digit symbols, else 1 point. (Gold, 2 digits: 1 point = 0.01.)"""
    return point * 10 if digits in (3, 5) else point


def symbol_spec(info, tick=None):
    """MT5 symbol_info (object or dict) -> plain dict the sizing code can use."""
    g = (lambda k, d=None: info.get(k, d)) if isinstance(info, dict) else (lambda k, d=None: getattr(info, k, d))
    digits, point = g("digits"), g("point")
    if digits is None or not point:
        raise DataError("symbol info is missing digits/point")
    pip = pip_size(digits, point)
    spec = {
        "symbol": g("name"), "digits": digits, "point": point, "pip": pip,
        "contract_size": g("trade_contract_size"),
        "tick_value": g("trade_tick_value"), "tick_size": g("trade_tick_size"),
        "volume_min": g("volume_min"), "volume_step": g("volume_step"), "volume_max": g("volume_max"),
        "spread_points": g("spread"),
        "stops_level_points": g("trade_stops_level"),
        "currency_profit": g("currency_profit"),
    }
    spec["spread_pips"] = round(spec["spread_points"] * point / pip, 2) if spec["spread_points"] is not None else None
    if tick is not None:
        tg = (lambda k: tick.get(k)) if isinstance(tick, dict) else (lambda k: getattr(tick, k, None))
        spec["bid"], spec["ask"] = tg("bid"), tg("ask")
    # value of one pip for a 1.00 lot, in the account/profit currency
    if spec["tick_value"] and spec["tick_size"]:
        spec["pip_value_per_lot"] = round(spec["tick_value"] * pip / spec["tick_size"], 4)
    return spec


def rates_to_bars(rates, server_offset_s=0):
    """MT5 rates (numpy structured array, or list of dicts) -> list of bar dicts, oldest first, UTC."""
    bars = []
    for r in rates:
        get = (lambda k: r[k]) if isinstance(r, dict) or hasattr(r, "dtype") else (lambda k: getattr(r, k))
        bars.append({
            "t": int(get("time")) - int(server_offset_s),
            "o": float(get("open")), "h": float(get("high")), "l": float(get("low")), "c": float(get("close")),
            "v": int(get("tick_volume")), "spread": int(get("spread")),
        })
    bars.sort(key=lambda b: b["t"])
    return bars


def check_bars(bars, tf_minutes):
    """Sanity-check candles: ordering, OHLC consistency, gaps. Returns a list of problems (empty = clean)."""
    problems = []
    step = tf_minutes * 60
    for i, b in enumerate(bars):
        if not (b["l"] <= min(b["o"], b["c"]) and b["h"] >= max(b["o"], b["c"])):
            problems.append("bar %d: OHLC inconsistent" % i)
        if i:
            d = b["t"] - bars[i - 1]["t"]
            if d <= 0:
                problems.append("bar %d: not strictly increasing" % i)
            elif d > step * 3 and d < 2 * 86400 - step:  # weekends are expected; flag other holes
                problems.append("bar %d: gap of %d min" % (i, d // 60))
    return problems


def read_csv_bars(path, delimiter=None):
    """MT5 'Export Bars' CSV: Date,Time,Open,High,Low,Close,TickVol,Vol,Spread (tab or comma, local server time)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        dl = delimiter or ("\t" if "\t" in sample.splitlines()[0] else ",")
        rows = list(csv.reader(f, delimiter=dl))
    out = []
    for row in rows:
        if not row or not row[0][:1].isdigit():
            continue  # header
        try:
            ts = time.mktime(time.strptime(row[0] + " " + row[1], "%Y.%m.%d %H:%M"))
            ts = int(ts) - time.timezone  # treat as UTC wall clock, not machine local
            out.append({"t": ts, "o": float(row[2]), "h": float(row[3]), "l": float(row[4]),
                        "c": float(row[5]), "v": int(float(row[6])),
                        "spread": int(float(row[8])) if len(row) > 8 and row[8] else 0})
        except (ValueError, IndexError):
            raise DataError("bad CSV row: %r" % (row,))
    if not out:
        raise DataError("no bars in %s" % path)
    return sorted(out, key=lambda b: b["t"])


def server_offset(mt5, symbol):
    """Seconds the broker's clock is ahead of UTC, from the newest tick, rounded to 30 min."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or not tick.time:
        return 0
    raw = tick.time - time.time()
    return int(round(raw / 1800.0) * 1800)


def fetch_bars(mt5, symbol, tf, count, offset=None):
    if tf not in TIMEFRAMES:
        raise DataError("timeframe must be one of %s" % ", ".join(TIMEFRAMES))
    if not mt5.symbol_select(symbol, True):
        raise DataError("symbol %s not available at this broker (try `symbols --match`)" % symbol)
    off = server_offset(mt5, symbol) if offset is None else int(offset * 3600)
    rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, "TIMEFRAME_" + tf), 0, count)
    if rates is None or len(rates) == 0:
        raise DataError("no bars for %s %s: %s" % (symbol, tf, mt5.last_error()))
    bars = rates_to_bars(rates, off)
    # the last bar is still forming; callers decide entries on closed bars only
    return bars, off


def write_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f)


def cmd_status(a):
    mt5 = connect()
    try:
        ti, ai = mt5.terminal_info(), mt5.account_info()
        mode = {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(getattr(ai, "trade_mode", None), "unknown")
        print("terminal : %s (connected=%s, algo trading allowed=%s)" % (ti.name, ti.connected, ti.trade_allowed))
        print("account  : %s on %s  [%s]" % (ai.login, ai.server, mode))
        print("balance  : %.2f %s  leverage 1:%s" % (ai.balance, ai.currency, ai.leverage))
        if mode == "REAL":
            print("NOTE: this is a REAL account. This tool is read-only, but use a DEMO account while developing.")
    finally:
        mt5.shutdown()


def cmd_symbols(a):
    mt5 = connect()
    try:
        names = sorted(s.name for s in (mt5.symbols_get() or []) if a.match.upper() in s.name.upper())
        print("\n".join(names) if names else "no symbols match")
    finally:
        mt5.shutdown()


def cmd_spec(a):
    mt5 = connect()
    try:
        mt5.symbol_select(a.symbol, True)
        info = mt5.symbol_info(a.symbol)
        if info is None:
            raise DataError("unknown symbol %s" % a.symbol)
        print(json.dumps(symbol_spec(info, mt5.symbol_info_tick(a.symbol)), indent=2))
    finally:
        mt5.shutdown()


def cmd_bars(a):
    if a.csv:
        bars, off = read_csv_bars(a.csv), 0
        symbol = a.symbol or os.path.splitext(os.path.basename(a.csv))[0]
    else:
        if not a.symbol:
            raise DataError("give a symbol, or --csv FILE")
        mt5 = connect()
        try:
            bars, off = fetch_bars(mt5, a.symbol, a.tf, a.count, a.server_offset)
        finally:
            mt5.shutdown()
        symbol = a.symbol
    problems = check_bars(bars, TIMEFRAMES[a.tf])
    print("%s %s: %d bars, %s -> %s UTC (server offset %+.1fh)" % (
        symbol, a.tf, len(bars), time.strftime("%Y-%m-%d %H:%M", time.gmtime(bars[0]["t"])),
        time.strftime("%Y-%m-%d %H:%M", time.gmtime(bars[-1]["t"])), off / 3600.0))
    print("data check: %s" % ("clean" if not problems else "%d problem(s), first: %s" % (len(problems), problems[0])))
    if a.write:
        path = os.path.join("trader", "bars", "%s_%s.json" % (symbol, a.tf))
        write_json(path, {"symbol": symbol, "tf": a.tf, "server_offset_s": off, "bars": bars})
        print("wrote %s" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only MT5 data layer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("symbols"); p.add_argument("--match", default=""); p.set_defaults(fn=cmd_symbols)
    p = sub.add_parser("spec"); p.add_argument("symbol"); p.set_defaults(fn=cmd_spec)
    p = sub.add_parser("bars")
    p.add_argument("symbol", nargs="?")
    p.add_argument("--tf", default="M5", choices=list(TIMEFRAMES))
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--csv")
    p.add_argument("--server-offset", type=float, default=None, help="broker clock minus UTC, hours")
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=cmd_bars)
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except DataError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
