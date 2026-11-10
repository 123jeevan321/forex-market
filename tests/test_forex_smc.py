import unittest

from _load import load

bt = load("forex-backtest", "backtest")

SPEC = {"pip": 0.0001, "point": 0.00001, "spread_points": 0, "pip_value_per_lot": 10.0,
        "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0}
DAY = 86400


def bar(t, o, h, l, c, v=10):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "spread": 0}


def quiet_day(d, lo=1.0980, hi=1.1050, n=288):
    """A full day of M5 bars that oscillates between lo and hi (a 70 pip day)."""
    out = []
    for i in range(n):
        up = (i // 24) % 2 == 0
        a, z = (lo, hi) if up else (hi, lo)
        f0, f1 = (i % 24) / 24.0, (i % 24 + 1) / 24.0
        o, c = a + (z - a) * f0, a + (z - a) * f1
        out.append(bar(d * DAY + i * 300, o, max(o, c), min(o, c), c))
    return out


def history(days=8):
    out = []
    for d in range(days):
        out += quiet_day(d)
    return out


class ProfileTests(unittest.TestCase):
    def test_poc_sits_where_the_volume_is(self):
        bars = [bar(i * 300, 1.10, 1.1002, 1.0998, 1.10, 100) for i in range(50)]      # heavy at 1.1000
        bars += [bar((50 + i) * 300, 1.11, 1.1102, 1.1098, 1.11, 5) for i in range(10)]  # thin at 1.1100
        poc, vah, val = bt.volume_profile(bars)
        self.assertAlmostEqual(poc, 1.10, places=3)
        self.assertLess(vah, 1.105)                      # the value area does not reach the thin prices
        self.assertLessEqual(val, poc)
        self.assertGreaterEqual(vah, poc)

    def test_flat_market_has_no_profile(self):
        self.assertIsNone(bt.volume_profile([bar(0, 1.1, 1.1, 1.1, 1.1)]))


class ContextTests(unittest.TestCase):
    def test_previous_day_only(self):
        bars = history(8)
        ctx = bt.DayContext(bars, 300)
        self.assertNotIn(0, ctx.prev)
        self.assertAlmostEqual(ctx.prev[7]["high"], 1.1050)
        changed = [dict(b) for b in bars]
        for b in changed[7 * 288:]:
            b["h"] += 1
        self.assertEqual(bt.DayContext(changed, 300).prev[7], ctx.prev[7])   # today cannot change "yesterday"


class SweepTests(unittest.TestCase):
    def setup(self, **kw):
        bars = history(8)
        t = 8 * DAY
        day = [bar(t + i * 300, 1.1020, 1.1022, 1.1018, 1.1020) for i in range(288)]
        bars += day
        self.i = 8 * 288 + 7 * 12 + 2                      # the bar closing at 07:15 (London kill zone)
        s = bt.LiquiditySweep(5, SPEC, **kw)
        return bars, s

    def test_sweep_of_yesterdays_high_goes_short(self):
        bars, s = self.setup()
        bars[self.i] = bar(bars[self.i]["t"], 1.1048, 1.1056, 1.1044, 1.1045)   # spikes over 1.1050, closes back
        s.prepare(bars)
        sig = s.on_bar(bars, self.i)
        self.assertEqual(sig["side"], "short")
        self.assertAlmostEqual(sig["tp_pips"], sig["sl_pips"] * 1.5)
        self.assertIsNone(s.on_bar(bars, self.i))            # the level is spent

    def test_sweep_of_yesterdays_low_goes_long(self):
        bars, s = self.setup()
        bars[self.i] = bar(bars[self.i]["t"], 1.0982, 1.0986, 1.0974, 1.0985)
        s.prepare(bars)
        self.assertEqual(s.on_bar(bars, self.i)["side"], "long")

    def test_breakout_that_holds_is_not_a_sweep(self):
        bars, s = self.setup()
        bars[self.i] = bar(bars[self.i]["t"], 1.1048, 1.1060, 1.1047, 1.1058)   # closes ABOVE the level
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, self.i))

    def test_outside_kill_zone_is_ignored(self):
        bars, s = self.setup()
        j = 8 * 288 + 3 * 12                                 # 03:00 UTC
        bars[j] = bar(bars[j]["t"], 1.1048, 1.1056, 1.1044, 1.1045)
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, j))

    def test_huge_spike_is_too_wide_for_a_scalp(self):
        bars, s = self.setup()
        bars[self.i] = bar(bars[self.i]["t"], 1.1048, 1.1100, 1.1044, 1.1045)   # 55 pip stop
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, self.i))

    def test_time_stop(self):
        bars, s = self.setup()
        self.assertEqual(s.flat_ts(1000), 1000 + 45 * 60)


class ValueReturnTests(unittest.TestCase):
    def test_close_back_inside_value_targets_the_poc(self):
        bars = history(8)
        t = 8 * DAY
        ctx = bt.DayContext(bars, 300)
        vah = bt.volume_profile(quiet_day(7))[1]
        up = vah + 0.0004                                    # sits just above yesterday's value area
        bars += [bar(t + i * 300, up, up + 0.0001, up - 0.0001, up) for i in range(288)]
        prev = bt.DayContext(bars, 300).prev[8]
        self.assertAlmostEqual(prev["vah"], vah)
        i = 8 * 288 + 7 * 12 + 2
        inside = prev["vah"] - 0.0003
        bars[i] = bar(bars[i]["t"], up, up + 0.0001, inside - 0.0001, inside)
        s = bt.ValueReturn(5, SPEC, min_rr=0.1)
        s.prepare(bars)
        sig = s.on_bar(bars, i)
        self.assertEqual(sig["side"], "short")
        self.assertAlmostEqual(sig["tp_pips"], (inside - prev["poc"]) / 0.0001, places=4)
        self.assertTrue(ctx.prev)


class FvgTests(unittest.TestCase):
    def build(self, retest_close):
        """H1 bars: 8 quiet days, then at 05:00-07:00 a strong up candle leaves a gap 1.1010-1.1040,
        and the 08:00 bar dips into it and closes at retest_close."""
        bars = []
        for d in range(8):
            bars += [bar(d * DAY + h * 3600, 1.1000, 1.1010, 1.0990, 1.1000) for h in range(24)]
        t = 8 * DAY
        bars += [bar(t + h * 3600, 1.1000, 1.1010, 1.0990, 1.1000) for h in range(5)]
        bars.append(bar(t + 5 * 3600, 1.1000, 1.1010, 1.0995, 1.1008))     # candle before: high 1.1010
        bars.append(bar(t + 6 * 3600, 1.1008, 1.1065, 1.1006, 1.1060))     # displacement
        bars.append(bar(t + 7 * 3600, 1.1060, 1.1070, 1.1040, 1.1065))     # candle after: low 1.1040
        bars.append(bar(t + 8 * 3600, 1.1065, 1.1068, 1.1030, retest_close))  # dips into the gap
        bars += [bar(t + h * 3600, 1.1060, 1.1062, 1.1058, 1.1060) for h in range(9, 24)]
        return bars, len(bars) - 16

    def test_first_retest_goes_long(self):
        bars, i = self.build(1.1066)
        s = bt.FvgRetrace(60, SPEC, adr_ref_pips=20)        # these toy days have a 20 pip range
        s.prepare(bars)
        self.assertEqual(s.gaps[i - 1][0], 1)
        sig = s.on_bar(bars, i)
        self.assertEqual(sig["side"], "long")
        self.assertAlmostEqual(sig["tp_pips"], sig["sl_pips"] * 2.0)
        self.assertIsNone(s.on_bar(bars, i))                 # a gap is traded once

    def test_no_entry_if_the_bar_closes_inside_the_gap(self):
        bars, i = self.build(1.1035)
        s = bt.FvgRetrace(60, SPEC, adr_ref_pips=20)
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, i))


if __name__ == "__main__":
    unittest.main()
