import datetime
import random
import unittest

from _load import load

se = load("forex-backtest", "spec_engine")

DAY = 86400
PIP = 0.0001
SPEC = {"pip": PIP, "point": 0.00001, "spread_points": 0, "pip_value_per_lot": 10.0,
        "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0}


def mk(t, o, h, l, c):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": 10, "spread": 0}


def walk(days, seed=7, step=300, vol=0.00035):
    """A seeded random walk on M5, Monday..Friday only, with a 24h session."""
    rng = random.Random(seed)
    out, p = [], 1.1000
    t0 = 4 * DAY                                           # 1970-01-05 was a Monday
    d = 0
    while len({b["t"] // DAY for b in out}) < days:
        day = t0 + d * DAY
        d += 1
        if datetime.datetime.fromtimestamp(day, datetime.timezone.utc).weekday() >= 5:
            continue
        for i in range(288):
            o = p
            c = o + rng.gauss(0, vol) + (p - 1.1) * -0.002
            h, l = max(o, c) + abs(rng.gauss(0, vol / 2)), min(o, c) - abs(rng.gauss(0, vol / 2))
            out.append(mk(day + i * step, o, h, l, c))
            p = c
    return out


class SwingTests(unittest.TestCase):
    def test_swing_known_only_after_n_bars(self):
        hs = [1, 2, 3, 9, 3, 2, 1, 2, 3, 4]
        bars = [mk(i * 300, h - 0.5, h, h - 1, h - 0.5) for i, h in enumerate(hs)]
        sw = se.Swings(bars, 2)
        self.assertEqual(sw.known("hi", 4, 5), [])          # bar 3 is a peak but needs bars 4 and 5 to confirm
        self.assertEqual([x[1] for x in sw.known("hi", 5, 5)], [3])
        self.assertEqual(sw.known("hi", 5, 5)[0][0], 5)

    def test_future_bars_do_not_change_known_swings(self):
        a = walk(6)
        b = [dict(x) for x in a]
        for x in b[1000:]:
            x["h"] += 0.5
            x["c"] += 0.5
        sa, sb = se.Swings(a, 2), se.Swings(b, 2)
        self.assertEqual(sa.known("hi", 900, 10), sb.known("hi", 900, 10))
        self.assertEqual(sa.known("lo", 900, 10), sb.known("lo", 900, 10))


class StatsTests(unittest.TestCase):
    def test_wilson_interval(self):
        lo, hi = se.wilson(60, 100)
        self.assertAlmostEqual(lo, 0.502, places=2)
        self.assertAlmostEqual(hi, 0.691, places=2)
        self.assertEqual(se.wilson(0, 0), (0.0, 0.0))
        lo, hi = se.wilson(6, 10)
        self.assertLess(lo, 0.35)                           # 6 of 10 says almost nothing

    def t(self, entry, r, exit_=None):
        return {"t_entry": entry, "t_exit": exit_ or entry + 3600, "r": r, "side": "long"}

    def test_daily_loss_limit_stops_the_day(self):
        base = 10 * DAY
        trades = [self.t(base + 3600 * k, -1.0) for k in range(1, 6)]
        taken, skipped = se.account_sim(trades, 10000, 0.6, 1.0, 2)
        self.assertEqual(len(taken), 2)                     # two 0.6% losses pass the 1% daily cap
        self.assertEqual(skipped, 3)

    def test_next_day_resets(self):
        trades = [self.t(10 * DAY + 3600, -1), self.t(10 * DAY + 7200, -1), self.t(10 * DAY + 10800, -1),
                  self.t(11 * DAY + 3600, 1)]
        taken, _ = se.account_sim(trades, 10000, 0.6, 1.0, 2)
        self.assertEqual(len(taken), 3)

    def test_max_two_open(self):
        trades = [self.t(10 * DAY + 3600 * k, 1.0, 10 * DAY + 80000) for k in range(1, 5)]
        taken, skipped = se.account_sim(trades, 10000, 0.5, 1.0, 2)
        self.assertEqual((len(taken), skipped), (2, 2))

    def test_compounding_and_metrics(self):
        trades = [self.t(10 * DAY + k * 7200, 2.0 if k % 2 else -1.0) for k in range(1, 9)]
        taken, _ = se.account_sim(trades, 10000, 1.0, 100.0, 5)
        m = se.metrics(taken, 10000)
        self.assertEqual((m["trades"], m["wins"], m["losses"]), (8, 4, 4))
        self.assertAlmostEqual(m["win_rate"], 0.5)
        self.assertAlmostEqual(m["expectancy_r"], 0.5)
        self.assertGreater(m["return_pct"], 0)
        self.assertGreater(m["profit_factor"], 1.0)
        self.assertEqual(m["max_losses"], 1)

    def test_split_is_chronological(self):
        trades = [{"t_entry": i} for i in range(100)]
        a, b, c = se.split_by_time(trades, 0, 100)
        self.assertEqual((len(a), len(b), len(c)), (60, 20, 20))
        self.assertLess(max(x["t_entry"] for x in a), min(x["t_entry"] for x in b))
        self.assertLess(max(x["t_entry"] for x in b), min(x["t_entry"] for x in c))


class SessionTests(unittest.TestCase):
    def utc(self, day, name):
        return datetime.datetime.fromtimestamp(se.session_open_utc(day // DAY, name), datetime.timezone.utc)

    def test_dst_is_handled(self):
        jan = datetime.datetime(2026, 1, 14, tzinfo=datetime.timezone.utc).timestamp()
        jul = datetime.datetime(2026, 7, 14, tzinfo=datetime.timezone.utc).timestamp()
        mar = datetime.datetime(2026, 3, 20, tzinfo=datetime.timezone.utc).timestamp()   # US on summer time, UK not yet
        self.assertEqual(self.utc(jan, "london").hour, 8)
        self.assertEqual(self.utc(jul, "london").hour, 7)
        self.assertEqual(self.utc(jan, "ny").strftime("%H:%M"), "14:30")
        self.assertEqual(self.utc(jul, "ny").strftime("%H:%M"), "13:30")
        self.assertEqual(self.utc(mar, "london").hour, 8)
        self.assertEqual(self.utc(mar, "overlap").hour, 12)


class StrategyAScenario(unittest.TestCase):
    def build(self):
        """Three quiet days, then on day 4 a stop run under yesterday's low at 08:00 UTC, an up-move that breaks the
        last lower high and leaves a gap, a dip back into the gap and a bullish candle."""
        bars = []
        for d in range(3):
            for i in range(288):
                base = 1.1010 + 0.0030 * ((i // 36) % 2)
                bars.append(mk((10 + d) * DAY + i * 300, base, base + 0.0004, base - 0.0004, base))
        day = 13 * DAY
        for i in range(96):                                         # 00:00-08:00 quiet, 1.1005 area
            bars.append(mk(day + i * 300, 1.1005, 1.1008, 1.1003, 1.1005))
        # yesterday's low is 1.1006; build highs: peak 1.1030 with two lower bars each side, then decline
        path = [(1.1012, 1.1018, 1.1010, 1.1015), (1.1015, 1.1024, 1.1013, 1.1022), (1.1022, 1.1030, 1.1020, 1.1026),
                (1.1026, 1.1027, 1.1018, 1.1019), (1.1019, 1.1021, 1.1012, 1.1014), (1.1014, 1.1015, 1.1006, 1.1008),
                (1.1008, 1.1009, 1.1001, 1.1003)]
        for k, (o, h, l, c) in enumerate(path):
            bars.append(mk(day + (96 + k) * 300, o, h, l, c))
        n = 96 + len(path)
        after = [(1.1003, 1.1004, 1.0985, 1.1010),                   # sweep: wick under 1.1006 and 1.0985, closes back above
                 (1.1010, 1.1022, 1.1009, 1.1020),                   # displacement
                 (1.1020, 1.1045, 1.1019, 1.1042),                   # closes above the 1.1030 lower high: MSS
                 (1.1042, 1.1052, 1.1038, 1.1050),                   # candle 3: gap between candle 1 high and its low
                 (1.1050, 1.1051, 1.1016, 1.1020),                   # dips back into the gap
                 (1.1020, 1.1034, 1.1012, 1.1030),                   # bullish confirmation inside the gap
                 (1.1038, 1.1046, 1.1036, 1.1044)]
        for k, (o, h, l, c) in enumerate(after):
            bars.append(mk(day + (n + k) * 300, o, h, l, c))
        for i in range(n + len(after), 288):
            bars.append(mk(day + i * 300, 1.1044, 1.1046, 1.1042, 1.1044))
        return bars, day + n * 300

    def test_long_setup_found(self):
        bars, t_sweep = self.build()
        s = se.StrategyA(5, 5, 15, fvg_atr=0.05, trend_mode="none", pip=PIP)
        s.prepare(bars)
        self.assertTrue(s.setups, "no setup found")
        st = s.setups[0]
        self.assertEqual(st["side"], "long")
        self.assertLess(st["ext"], 1.0990)
        idx = {b["t"]: i for i, b in enumerate(bars)}
        self.assertTrue(s.sig, "no entry signal")
        i = min(s.sig)
        self.assertGreater(bars[i]["t"], t_sweep + 3 * 300)        # only after the sweep, MSS and FVG exist
        sig = s.on_bar(bars, i)
        self.assertEqual(sig["side"], "long")
        self.assertLess(sig["sl_price"], st["ext"])                 # stop under the sweep wick, with a buffer

    def test_no_setup_without_a_sweep(self):
        bars, _ = self.build()
        for b in bars[13 * 288 + 0:]:
            pass
        flat = [mk(b["t"], 1.1020, 1.1022, 1.1018, 1.1020) for b in bars]
        s = se.StrategyA(5, 5, 15, trend_mode="none", pip=PIP)
        s.prepare(flat)
        self.assertEqual(s.setups, [])


class NoLookAheadTests(unittest.TestCase):
    CUT = 9000

    def sigs(self, make, bars):
        s = make()
        s.prepare(bars)
        return s.sig

    def check(self, make, label):
        bars = walk(45)
        n = self.CUT
        full = self.sigs(make, bars)
        part = self.sigs(make, bars[:n])
        f = {i: v for i, v in full.items() if i < n - 400}
        q = {i: v for i, v in part.items() if i < n - 400}
        self.assertEqual(f, q, label + ": signals changed when the future was removed")
        return len(f)

    def test_strategy_a(self):
        k = self.check(lambda: se.StrategyA(5, 15, 60, fvg_atr=0.0, trend_mode="none", pip=PIP), "A")
        self.assertGreater(k, 0)

    def test_strategy_b(self):
        self.check(lambda: se.StrategyB(5, 15, 60, disp_atr=1.0, pip=PIP), "B")

    def test_strategy_c(self):
        k = self.check(lambda: se.StrategyC(5, 15, 60, session="london", range_min=30, pip=PIP), "C")
        self.assertGreater(k, 0)


class EngineIntegration(unittest.TestCase):
    def test_oversized_stop_is_rejected(self):
        bars, _ = StrategyAScenario().build()
        s = se.StrategyA(5, 5, 15, fvg_atr=0.05, trend_mode="none", max_stop_atr=2, pip=PIP)
        res = se.bt.run_backtest(bars, SPEC, s, equity=10000, risk_pct=0.5, slip_pips=0.0)
        self.assertEqual(len(res["trades"]), 0)

    def test_structural_stop_end_to_end(self):
        bars, _ = StrategyAScenario().build()
        s = se.StrategyA(5, 5, 15, fvg_atr=0.05, trend_mode="none", max_stop_atr=10, pip=PIP)
        res = se.bt.run_backtest(bars, SPEC, s, equity=10000, risk_pct=0.5, slip_pips=0.0)
        self.assertEqual(len(res["trades"]), 1)
        t = res["trades"][0]
        self.assertEqual(t["side"], "long")
        self.assertIn(t["reason"], ("target", "stop", "flat", "end-of-data"))
        planned = (t["entry"] - s.setups[0]["ext"] + 0.1 * s.setups[0]["atr"]) / PIP    # pips from fill to the structural stop
        self.assertAlmostEqual(t["pips"] / planned, t["r"], delta=0.02)


if __name__ == "__main__":
    unittest.main()
