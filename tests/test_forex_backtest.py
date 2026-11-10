import unittest

from _load import load

bt = load("forex-backtest", "backtest")

SPEC = {"pip": 0.0001, "point": 0.00001, "spread_points": 0, "pip_value_per_lot": 10.0,
        "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0}


def bar(t, o, h, l, c, spread=0):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": 1, "spread": spread}


def flat_bars(n, price=1.1000, t0=0, step=300, spread=0):
    return [bar(t0 + i * step, price, price + 0.00005, price - 0.00005, price, spread) for i in range(n)]


class FillTests(unittest.TestCase):
    def test_long_target(self):
        b = flat_bars(5)
        b[2] = bar(600, 1.1, 1.1020, 1.0999, 1.1015)  # +20 pips high
        t = bt.simulate_trade(b, 1, "long", 10, 15, SPEC, 0)
        self.assertEqual(t["reason"], "target")
        self.assertAlmostEqual(t["pips"], 15, places=4)

    def test_long_stop(self):
        b = flat_bars(5)
        b[2] = bar(600, 1.1, 1.1001, 1.0980, 1.0985)
        t = bt.simulate_trade(b, 1, "long", 10, 15, SPEC, 0)
        self.assertEqual(t["reason"], "stop")
        self.assertAlmostEqual(t["pips"], -10, places=4)

    def test_both_touched_stop_wins(self):
        b = flat_bars(5)
        b[2] = bar(600, 1.1, 1.1030, 1.0970, 1.1)
        t = bt.simulate_trade(b, 1, "long", 10, 15, SPEC, 0)
        self.assertEqual(t["reason"], "stop")

    def test_spread_costs_long_entry(self):
        # spread is paid through the fill: a long buys at the ask, so the bid must travel 2 pips further
        hit = bt.simulate_trade(self._spiked(0), 1, "long", 10, 15, SPEC, 0)
        miss = bt.simulate_trade(self._spiked(20), 1, "long", 10, 15, SPEC, 0)
        self.assertEqual(hit["reason"], "target")
        self.assertNotEqual(miss["reason"], "target")

    def _spiked(self, spread):
        b = flat_bars(5, spread=spread)
        b[2] = bar(600, 1.1, 1.1016, 1.0999, 1.1, spread)    # high is 16 pips up
        return b

    def test_slippage_hurts_stop(self):
        b = flat_bars(5)
        b[2] = bar(600, 1.1, 1.1001, 1.0980, 1.0985)
        t = bt.simulate_trade(b, 1, "long", 10, 15, SPEC, 0.5)
        self.assertAlmostEqual(t["pips"], -10.5, places=4)  # 0.5 pip worse fill on the stop

    def test_stop_gap_fills_at_open(self):
        b = flat_bars(5)
        b[2] = bar(600, 1.0950, 1.0955, 1.0940, 1.0945)  # gaps 50 pips under a 10 pip stop
        t = bt.simulate_trade(b, 1, "long", 10, 15, SPEC, 0)
        self.assertEqual(t["reason"], "stop-gap")
        self.assertAlmostEqual(t["pips"], -50, places=4)

    def test_short_target_and_spread(self):
        b = flat_bars(5, spread=20)
        b[2] = bar(600, 1.1, 1.1001, 1.0980, 1.0985, 20)   # low 20 pips down; buy back at ask (+2)
        t = bt.simulate_trade(b, 1, "short", 10, 15, SPEC, 0)
        self.assertEqual(t["reason"], "target")
        self.assertAlmostEqual(t["pips"], 15, places=4)

    def test_flat_time(self):
        b = flat_bars(6)
        t = bt.simulate_trade(b, 1, "long", 50, 50, SPEC, 0, flat_ts=b[3]["t"])
        self.assertEqual(t["reason"], "flat")
        self.assertEqual(t["exit_index"], 3)


class SizingTests(unittest.TestCase):
    def test_rounds_down(self):
        # $300 * 1% = $3 ; stop 20 pips * $10 = $200 per lot -> 0.015 -> 0.01
        self.assertAlmostEqual(bt.size_lots(300, 1, 20, SPEC), 0.01)

    def test_too_small_is_skipped(self):
        self.assertEqual(bt.size_lots(300, 1, 50, SPEC), 0.0)   # 0.006 lots < 0.01 minimum

    def test_big_account(self):
        self.assertAlmostEqual(bt.size_lots(10000, 1, 20, SPEC), 0.5)


class StrategyTests(unittest.TestCase):
    def day_bars(self, breakout_close):
        """One day of M5 bars: quiet 00-07 range of ~30 pips, then a bar closing at breakout_close at 07:05."""
        out = []
        for i in range(0, 12 * 7):                       # 00:00..07:00 exclusive
            hi, lo = (1.1015, 1.0985)
            out.append(bar(i * 300, 1.1, hi if i % 2 else 1.1005, lo if i % 3 == 0 else 1.0995, 1.1))
        out.append(bar(7 * 3600, 1.1, breakout_close + 0.0002, 1.1, breakout_close))   # closes 07:05
        out += [bar(7 * 3600 + 300 * (k + 1), breakout_close, breakout_close + 0.0001, breakout_close - 0.0001,
                    breakout_close) for k in range(200)]
        return out

    def test_signal_long_on_close_above_range(self):
        bars = self.day_bars(1.1040)
        s = bt.RangeBreakout(5, SPEC)
        s.prepare(bars)
        sig = s.on_bar(bars, 12 * 7)
        self.assertEqual(sig["side"], "long")
        self.assertGreater(sig["sl_pips"], 8)
        self.assertAlmostEqual(sig["tp_pips"], sig["sl_pips"] * 1.5)

    def test_no_signal_inside_range(self):
        bars = self.day_bars(1.1000)
        s = bt.RangeBreakout(5, SPEC)
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, 12 * 7))

    def test_one_trade_per_day(self):
        bars = self.day_bars(1.1040)
        s = bt.RangeBreakout(5, SPEC)
        s.prepare(bars)
        self.assertIsNotNone(s.on_bar(bars, 12 * 7))
        self.assertIsNone(s.on_bar(bars, 12 * 7 + 1))

    def test_no_future_peeking(self):
        # the signal for bar i must not change if later bars change
        a = self.day_bars(1.1040)
        b = [dict(x) for x in a]
        for x in b[12 * 7 + 5:]:
            x["c"] = x["h"] = x["o"] = x["l"] = 1.2
        sa, sb = bt.RangeBreakout(5, SPEC), bt.RangeBreakout(5, SPEC)
        sa.prepare(a); sb.prepare(b)
        self.assertEqual(sa.on_bar(a, 12 * 7), sb.on_bar(b, 12 * 7))


class EngineTests(unittest.TestCase):
    def test_runs_and_accounts(self):
        bars = StrategyTests().day_bars(1.1040)
        s = bt.RangeBreakout(5, SPEC, min_stop_pips=5)
        r = bt.run_backtest(bars, SPEC, s, equity=10000, risk_pct=1, slip_pips=0)
        self.assertEqual(len(r["trades"]), 1)
        t = r["trades"][0]
        self.assertAlmostEqual(r["end_equity"], 10000 + t["pnl"])

    def test_verdicts(self):
        few = {"trades": 5, "avg_r": 1.0}
        self.assertIn("NOT ENOUGH", bt.verdict(few, few))
        neg = {"trades": 50, "avg_r": -0.1}
        self.assertIn("NO EDGE", bt.verdict(neg, neg))
        self.assertIn("INCONSISTENT", bt.verdict({"trades": 50, "avg_r": -0.1}, {"trades": 50, "avg_r": 0.4}))
        self.assertIn("FRAGILE", bt.verdict({"trades": 50, "avg_r": 0.4}, {"trades": 50, "avg_r": 0.1}))


if __name__ == "__main__":
    unittest.main()
