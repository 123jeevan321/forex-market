import math
import unittest

from _load import load

bt = load("forex-backtest", "backtest")

SPEC = {"pip": 0.0001, "point": 0.00001, "spread_points": 0, "pip_value_per_lot": 10.0,
        "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0}


def wave_bars(n, step=900, amp=0.0030, drift=0.000004):
    """A drifting sine wave: gives trends and pullbacks."""
    out, prev = [], 1.1
    for i in range(n):
        c = 1.1 + drift * i + amp * math.sin(i / 40.0) + 0.0004 * math.sin(i / 3.0)
        out.append({"t": i * step, "o": prev, "h": max(prev, c) + 0.0001, "l": min(prev, c) - 0.0001,
                    "c": c, "v": 1, "spread": 10})
        prev = c
    return out


class HelperTests(unittest.TestCase):
    def test_adr_is_causal(self):
        bars = wave_bars(96 * 12)
        a = bt.daily_adr(bars)
        changed = [dict(b) for b in bars]
        for b in changed[96 * 8:]:                  # blow up day 8 onwards
            b["h"] += 1.0
        b2 = bt.daily_adr(changed)
        self.assertEqual(a[8], b2[8])               # day 8's ADR only uses days before it
        self.assertNotIn(3, a)                      # not enough history yet

    def test_resample(self):
        bars = [{"t": i * 300, "o": 1 + i, "h": 2 + i, "l": i, "c": 1.5 + i, "v": 1, "spread": i} for i in range(6)]
        r = bt.resample(bars, 15)
        self.assertEqual(len(r), 2)
        self.assertEqual((r[0]["o"], r[0]["h"], r[0]["l"], r[0]["c"], r[0]["v"], r[0]["spread"]), (1, 4, 0, 3.5, 3, 2))

    def test_ema_and_atr_lengths(self):
        bars = wave_bars(50)
        self.assertEqual(len(bt.ema_series([b["c"] for b in bars], 10)), 50)
        self.assertTrue(all(x > 0 for x in bt.atr_series(bars)))


class PullbackTests(unittest.TestCase):
    def test_trades_both_ways_and_no_peeking(self):
        bars = wave_bars(4000)
        s = bt.TrendPullback(15, SPEC)
        s.prepare(bars)
        sigs = {i: s.on_bar(bars, i) for i in range(len(bars))}
        hits = {i: v for i, v in sigs.items() if v}
        self.assertTrue(any(v["side"] == "long" for v in hits.values()))
        first = min(hits)
        # same signal when everything after that bar is different
        cut = [dict(b) for b in bars[:first + 1]] + [dict(b, c=9.9, h=9.9, o=9.9, l=9.9) for b in bars[first + 1:]]
        s2 = bt.TrendPullback(15, SPEC)
        s2.prepare(cut)
        self.assertEqual(s2.on_bar(cut, first), hits[first])

    def test_respects_hours_and_daily_cap(self):
        bars = wave_bars(4000)
        s = bt.TrendPullback(15, SPEC, max_per_day=1)
        s.prepare(bars)
        days = {}
        for i in range(len(bars)):
            if s.on_bar(bars, i):
                d, sec = divmod(bars[i]["t"] + 900, 86400)
                self.assertTrue(7 * 3600 <= sec < 17 * 3600)
                days[d] = days.get(d, 0) + 1
        self.assertTrue(days and max(days.values()) == 1)

    def test_engine_runs_pullback(self):
        bars = wave_bars(4000)
        r = bt.run_backtest(bars, SPEC, bt.TrendPullback(15, SPEC), equity=10000)
        self.assertGreater(len(r["trades"]), 0)


class ScaledBreakoutTests(unittest.TestCase):
    def test_needs_adr_history(self):
        bars = wave_bars(96 * 3)
        s = bt.RangeBreakout(15, SPEC, adr_ref_pips=70)
        s.prepare(bars)
        self.assertTrue(all(s.on_bar(bars, i) is None for i in range(len(bars))))


if __name__ == "__main__":
    unittest.main()
