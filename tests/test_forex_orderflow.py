import unittest

from _load import load

of = load("forex-backtest", "orderflow")
md = load("forex-data", "mt5_data")

SPEC = {"pip": 0.0001, "point": 0.00001, "spread_points": 0, "pip_value_per_lot": 10.0,
        "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0}
DAY = 86400


def bar(t, o, h, l, c, delta=0, ticks=100):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": ticks, "spread": 0, "delta": delta, "ticks": ticks}


def history(days=8):
    """Quiet M5 days with a 70 pip daily range and small alternating delta."""
    out = []
    for d in range(days):
        for i in range(288):
            lo, hi = 1.0980, 1.1050
            up = (i // 24) % 2 == 0
            a, z = (lo, hi) if up else (hi, lo)
            o, c = a + (z - a) * (i % 24) / 24.0, a + (z - a) * (i % 24 + 1) / 24.0
            out.append(bar(d * DAY + i * 300, o, max(o, c), min(o, c), c, delta=3 if i % 2 else -3))
    return out


def today(price=1.1020):
    t = 8 * DAY
    return [bar(t + i * 300, price, price + 0.0002, price - 0.0002, price, delta=2 if i % 2 else -2) for i in range(288)]


I = 8 * 288 + 7 * 12 + 2          # the bar closing 07:15 UTC, inside the London window


class TickRuleTests(unittest.TestCase):
    def test_up_down_counts_and_spread(self):
        times = [0, 10, 20, 290, 300, 310]
        bids = [1.0, 1.1, 1.1, 1.0, 1.2, 1.1]
        asks = [b + 0.2 for b in bids]
        flow, last = md.ticks_to_flow(times, bids, asks, None, 300, point=0.1)
        self.assertEqual(flow[0][:3], [1, 1, 4])        # first tick has no direction; then up, flat, down
        self.assertEqual(flow[300][:3], [1, 1, 2])      # up from the previous bucket's last tick, then down
        self.assertAlmostEqual(flow[0][3], 4 * 2.0)     # spread 0.2 = 2 points, summed over 4 ticks
        self.assertAlmostEqual(last, 1.2)

    def test_chunks_chain(self):
        _, last = md.ticks_to_flow([0], [1.0], [1.0], None, 300)
        flow, _ = md.ticks_to_flow([300], [1.5], [1.5], last, 300)
        self.assertEqual(flow[300][:2], [1, 0])         # up versus the previous chunk's last price


class HelperTests(unittest.TestCase):
    def test_rolling_z_is_causal(self):
        vals = [1, -1] * 20 + [50]
        z = of.rolling_z(vals, 20)
        self.assertIsNone(z[19])
        self.assertGreater(z[40], 10)
        self.assertEqual(z[:40], of.rolling_z(vals[:40] + [-999], 20)[:40])

    def test_merge_keeps_only_covered_bars(self):
        bars = [bar(i * 300, 1, 1, 1, 1) for i in range(10)]
        rows = [{"t": 600, "up": 7, "dn": 2, "n": 9}, {"t": 1500, "up": 1, "dn": 4, "n": 6}]
        m = of.merge_flow(bars, rows)
        self.assertEqual([b["t"] for b in m], [600, 900, 1200, 1500])
        self.assertEqual((m[0]["delta"], m[0]["ticks"]), (5, 9))
        self.assertEqual(m[1]["delta"], 0)              # no ticks recorded for that bar

    def test_predictive_power_detects_a_real_signal(self):
        bars, p = [], 1.0
        for i in range(600):
            d = 10 if (i // 3) % 2 == 0 else -10       # delta leads: next bar moves the way this delta points
            prev_d = 10 if ((i - 1) // 3) % 2 == 0 else -10
            c = p + 0.0001 * (1 if prev_d > 0 else -1)
            bars.append(bar(i * 300, p, max(p, c), min(p, c), c, delta=d + (i % 5)))
            p = c
        self.assertGreater(of.predictive_power(bars)["next_1"], 0.3)


class StrategyTests(unittest.TestCase):
    def test_divergence_short(self):
        bars = history() + today()
        for j in range(I - 12, I):                      # heavy selling pressure while price holds up
            bars[j]["delta"] = -40
        bars[I - 6] = bar(bars[I - 6]["t"], 1.1020, 1.1030, 1.1018, 1.1028, delta=-40)   # earlier high
        bars[I] = bar(bars[I]["t"], 1.1030, 1.1036, 1.1024, 1.1026, delta=-40)           # higher high, closes down
        s = of.DeltaDivergence(5, SPEC)
        s.prepare(bars)
        sig = s.on_bar(bars, I)
        self.assertEqual(sig["side"], "short")
        self.assertAlmostEqual(sig["tp_pips"], sig["sl_pips"] * 1.5)

    def test_no_divergence_when_delta_confirms(self):
        bars = history() + today()
        bars[I - 6] = bar(bars[I - 6]["t"], 1.1020, 1.1030, 1.1018, 1.1028, delta=5)
        for j in range(I - 5, I + 1):
            bars[j]["delta"] = 40
        bars[I] = bar(bars[I]["t"], 1.1030, 1.1036, 1.1024, 1.1026, delta=40)
        s = of.DeltaDivergence(5, SPEC)
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, I))

    def test_absorption_short(self):
        bars = history() + today()
        bars[I] = bar(bars[I]["t"], 1.1024, 1.1030, 1.1018, 1.1021, delta=60, ticks=400)  # buying, closes down
        s = of.Absorption(5, SPEC)
        s.prepare(bars)
        self.assertEqual(s.on_bar(bars, I)["side"], "short")

    def test_absorption_needs_a_busy_bar(self):
        bars = history() + today()
        bars[I] = bar(bars[I]["t"], 1.1024, 1.1030, 1.1018, 1.1021, delta=60, ticks=100)
        s = of.Absorption(5, SPEC)
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, I))

    def test_momentum_long(self):
        bars = history() + today()
        bars[I] = bar(bars[I]["t"], 1.1020, 1.1030, 1.1019, 1.1029, delta=80)             # closes near its high
        s = of.ImbalanceMomentum(5, SPEC)
        s.prepare(bars)
        self.assertEqual(s.on_bar(bars, I)["side"], "long")

    def test_outside_trading_hours(self):
        bars = history() + today()
        j = 8 * 288 + 3 * 12
        bars[j] = bar(bars[j]["t"], 1.1020, 1.1030, 1.1019, 1.1029, delta=80)
        s = of.ImbalanceMomentum(5, SPEC)
        s.prepare(bars)
        self.assertIsNone(s.on_bar(bars, j))

    def test_evaluate_runs(self):
        bars = history(20)
        res, s_all, _, _ = of.evaluate(bars, SPEC, of.ImbalanceMomentum)
        self.assertIn("trades", s_all)


if __name__ == "__main__":
    unittest.main()
