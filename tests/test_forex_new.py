import datetime
import math
import unittest

from _load import load
from test_forex_spec import walk

ns = load("forex-backtest", "new_strategies")
bt = ns.bt

DAY = 86400
PIP = 0.0001
SPEC = {"pip": PIP, "point": 0.00001, "spread_points": 0, "pip_value_per_lot": 10.0,
        "volume_min": 0.01, "volume_step": 0.01, "volume_max": 50.0}


def mk(t, o, h, l, c, v=10, spread=0):
    return {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "spread": spread}


class IndicatorTests(unittest.TestCase):
    def test_vwap_and_bands(self):
        bars = [mk(0, 1, 1, 1, 1, v=1), mk(300, 3, 3, 3, 3, v=3), mk(DAY, 5, 5, 5, 5, v=2)]
        vw, sd, cnt = ns.vwap_bands(bars)
        self.assertAlmostEqual(vw[0], 1.0)
        self.assertAlmostEqual(vw[1], (1 * 1 + 3 * 3) / 4.0)            # volume weighted: 2.5
        var = (1 * 1 + 3 * 9) / 4.0 - 2.5 ** 2                         # E[x^2] - mean^2 = 0.75
        self.assertAlmostEqual(sd[1], math.sqrt(var))
        self.assertEqual(cnt, [1, 2, 1])
        self.assertAlmostEqual(vw[2], 5.0)                              # a new day resets the session

    def test_vwap_is_causal(self):
        bars = [mk(i * 300, 1 + i * 0.001, 1.002 + i * 0.001, 0.998 + i * 0.001, 1 + i * 0.001, v=5 + i % 7) for i in range(50)]
        a = ns.vwap_bands(bars)
        b = ns.vwap_bands(bars[:30])
        self.assertEqual(a[0][:30], b[0])
        self.assertEqual(a[1][:30], b[1])

    def test_rolling_mean_std(self):
        m, s = ns.roll_mean_std([1, 2, 3, 4, 5, 6], 3)
        self.assertEqual(m[:2], [None, None])
        self.assertAlmostEqual(m[2], 2.0)
        self.assertAlmostEqual(m[5], 5.0)
        self.assertAlmostEqual(s[2], math.sqrt(2 / 3.0))

    def test_stochastic_and_macd(self):
        up = [mk(i * 300, 1 + i * 0.01, 1.01 + i * 0.01, 0.99 + i * 0.01, 1.005 + i * 0.01) for i in range(60)]
        k, d = ns.stochastic(up)
        self.assertIsNone(k[10])
        self.assertGreater(k[-1], 70)                                    # a steady rise sits near the top of its range
        closes = [b["c"] for b in up]
        self.assertGreater(ns.macd_hist(closes)[-1] + 1, 0.9)            # finite and sensible
        down = [mk(i * 300, 2 - i * 0.01, 2.01 - i * 0.01, 1.99 - i * 0.01, 2 - i * 0.01) for i in range(60)]
        self.assertLess(ns.stochastic(down)[0][-1], 30)

    def test_window_utc_handles_dst(self):
        jan = datetime.datetime(2026, 1, 14, tzinfo=datetime.timezone.utc).timestamp() // DAY
        jul = datetime.datetime(2026, 7, 14, tzinfo=datetime.timezone.utc).timestamp() // DAY
        f = lambda day, spec: datetime.datetime.fromtimestamp(ns.window_utc(int(day), spec)[0], datetime.timezone.utc).strftime("%H:%M")
        self.assertEqual(f(jan, ns.ASIAN), "00:00")
        self.assertEqual(f(jul, ns.ASIAN), "00:00")                       # Tokyo has no daylight saving
        self.assertEqual(f(jan, ns.ENTRY_SESSIONS["london"]), "08:00")
        self.assertEqual(f(jul, ns.ENTRY_SESSIONS["london"]), "07:00")
        self.assertEqual(f(jan, ns.ENTRY_SESSIONS["ny"]), "13:00")
        self.assertEqual(f(jul, ns.ENTRY_SESSIONS["ny"]), "12:00")


class _Fake:
    """A strategy that emits one prepared signal, to test the engine's new price targets and partial exits."""

    def __init__(self, sig):
        self.sig = {5: sig}
        self.p = {"rr": 2.0}

    def prepare(self, bars):
        pass

    def on_bar(self, bars, i):
        return self.sig.get(i)

    def flat_ts(self, t):
        return None


def flat(n=140, price=1.1000):
    return [mk(i * 300, price, price + 0.00002, price - 0.00002, price) for i in range(n)]


class EngineTargetTests(unittest.TestCase):
    def run_sig(self, sig, bars):
        return bt.run_backtest(bars, SPEC, _Fake(sig), equity=10000, risk_pct=1.0, slip_pips=0.0)["trades"]

    def test_price_target_hit(self):
        b = flat()
        b[8] = mk(8 * 300, 1.1, 1.1021, 1.0999, 1.1015)
        t = self.run_sig({"side": "long", "sl_price": 1.0990, "tp_price": 1.1020, "rr": 2.0}, b)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["reason"], "target")
        self.assertAlmostEqual(t[0]["pips"], 20.0, places=3)
        self.assertAlmostEqual(t[0]["r"], 2.0, places=2)

    def test_too_little_reward_is_rejected(self):
        b = flat()
        t = self.run_sig({"side": "long", "sl_price": 1.0990, "tp_price": 1.1020, "min_rr": 3.0, "rr": 2.0}, b)
        self.assertEqual(t, [])                                          # 20 pips of reward for 10 of risk is only 2:1

    def test_target_on_the_wrong_side_is_rejected(self):
        b = flat()
        t = self.run_sig({"side": "long", "sl_price": 1.0990, "tp_price": 1.0995, "rr": 2.0}, b)
        self.assertEqual(t, [])

    def test_partial_exit_averages_the_legs(self):
        b = flat()
        b[8] = mk(8 * 300, 1.1, 1.1012, 1.0999, 1.1010)                  # reaches 1R (10 pips) but not the 20 pip target
        b[10] = mk(10 * 300, 1.1010, 1.1021, 1.1009, 1.1020)             # reaches the target
        t = self.run_sig({"side": "long", "sl_price": 1.0990, "tp_price": 1.1020, "partial_rr": 1.0, "rr": 2.0}, b)
        self.assertEqual(len(t), 1)
        self.assertAlmostEqual(t[0]["pips"], 15.0, places=3)             # half at +10, half at +20
        self.assertEqual(t[0]["exit_index"], 10)

    def test_partial_when_stopped_loses_full_risk(self):
        b = flat()
        b[8] = mk(8 * 300, 1.1, 1.1001, 1.0985, 1.0988)
        t = self.run_sig({"side": "long", "sl_price": 1.0990, "tp_price": 1.1020, "partial_rr": 1.0, "rr": 2.0}, b)
        self.assertAlmostEqual(t[0]["pips"], -10.0, places=3)


def quiet_day(d, lo=1.1000, hi=1.1040, n=288):
    out = []
    for i in range(n):
        up = (i // 24) % 2 == 0
        a, z = (lo, hi) if up else (hi, lo)
        o, c = a + (z - a) * (i % 24) / 24.0, a + (z - a) * (i % 24 + 1) / 24.0
        out.append(mk(d * DAY + i * 300, o, max(o, c), min(o, c), c))
    return out


class FailedBreakoutScenario(unittest.TestCase):
    def test_close_outside_then_back_inside_goes_long(self):
        bars = quiet_day(10) + quiet_day(11)
        t = 12 * DAY
        day = [mk(t + i * 300, 1.1020, 1.1022, 1.1018, 1.1020) for i in range(288)]
        j = 8 * 12                                                         # 08:00 UTC
        for k in range(3):                                                 # a full 15m block that closes below 1.1000
            day[j + k] = mk(t + (j + k) * 300, 1.0996, 1.0997, 1.0988, 1.0992)
        for k in range(3, 6):                                              # next block closes back inside
            day[j + k] = mk(t + (j + k) * 300, 1.0996, 1.1008, 1.0995, 1.1005)
        day[j + 6] = mk(t + (j + 6) * 300, 1.1005, 1.1012, 1.1003, 1.1010)  # bullish confirmation
        bars += day
        s = ns.StrategyF(5, 15, 60, target="mid", levels=("pd",), min_rr=0.3, pip=PIP)
        s.prepare(bars)
        self.assertTrue(s.sig, "no signal")
        i = min(s.sig)
        sig = s.on_bar(bars, i)
        self.assertEqual(sig["side"], "long")
        self.assertAlmostEqual(sig["tp_price"], 1.1020, places=4)         # the middle of yesterday's 1.1000-1.1040 range
        self.assertLess(sig["sl_price"], 1.0988)                           # beyond the failed-breakout extreme
        self.assertGreater(bars[i]["t"], t + 8 * 3600 + 20 * 60)           # only after the return inside

    def test_accepted_breakout_is_not_faded(self):
        bars = quiet_day(10) + quiet_day(11)
        t = 12 * DAY
        day = [mk(t + i * 300, 1.1020, 1.1022, 1.1018, 1.1020) for i in range(288)]
        for i in range(8 * 12, 8 * 12 + 40):                               # stays below the level for hours: accepted
            day[i] = mk(t + i * 300, 1.0990, 1.0992, 1.0986, 1.0990)
        s = ns.StrategyF(5, 15, 60, levels=("pd",), min_rr=0.3, pip=PIP)
        s.prepare(bars + day)
        self.assertEqual(s.sig, {})


class NoLookAheadTests(unittest.TestCase):
    CUT = 20000
    bars = None

    @classmethod
    def setUpClass(cls):
        cls.bars = walk(90, seed=11)

    def check(self, make, label):
        a = make()
        a.prepare(self.bars)
        b = make()
        b.prepare(self.bars[:self.CUT])
        f = {i: v for i, v in a.sig.items() if i < self.CUT - 500}
        q = {i: v for i, v in b.sig.items() if i < self.CUT - 500}
        self.assertEqual(f, q, label + ": signals changed when the future was removed")
        return len(f)

    def test_d(self):
        k = self.check(lambda: ns.StrategyD(5, 15, 60, adx_max=100, pip=PIP), "D")
        self.assertGreater(k, 0)

    def test_d_partial(self):
        self.check(lambda: ns.StrategyD(5, 15, 60, adx_max=100, tp_mode="partial", pip=PIP), "D partial")

    def test_e_immediate(self):
        self.check(lambda: ns.StrategyE(5, 15, 60, squeeze_n=5, pip=PIP), "E")

    def test_e_retest_with_filters(self):
        for f in ("ema50", "vwap", "adx"):
            self.check(lambda f=f: ns.StrategyE(5, 15, 60, squeeze_n=5, entry_mode="retest", filt=f, pip=PIP), "E " + f)

    def test_f(self):
        k = self.check(lambda: ns.StrategyF(5, 15, 60, min_rr=0.0, pip=PIP), "F")
        self.assertGreater(k, 0)

    def test_g_all_momentum_filters(self):
        for m in ("none", "rsi", "stoch", "macd", "ema"):
            self.check(lambda m=m: ns.StrategyG(5, 15, 240, momentum=m, pip=PIP), "G " + m)

    def test_h(self):
        for sess in ("london", "ny"):
            self.check(lambda sess=sess: ns.StrategyH(5, 15, 60, session=sess, min_rr=0.0, min_range_adr=0.0, max_range_adr=9, pip=PIP), "H " + sess)


class SignalSanity(unittest.TestCase):
    def test_signals_have_valid_stops_and_directions(self):
        bars = walk(90, seed=12)
        for s in (ns.StrategyD(5, 15, 60, adx_max=100, pip=PIP), ns.StrategyF(5, 15, 60, min_rr=0.0, pip=PIP),
                  ns.StrategyE(5, 15, 60, squeeze_n=5, pip=PIP)):
            s.prepare(bars)
            for i, g in s.sig.items():
                self.assertIn(g["side"], ("long", "short"))
                px = bars[i]["c"]
                self.assertTrue(g["sl_price"] < px if g["side"] == "long" else g["sl_price"] > px,
                                "stop on the wrong side of the signal bar's close")
                h = ((bars[i]["t"] + 300) % DAY) // 3600
                self.assertTrue(6 <= h < 20)                               # inside the entry-hours rule

    def test_engine_runs_each_strategy(self):
        bars = walk(90, seed=13)
        for s in (ns.StrategyD(5, 15, 60, adx_max=100, pip=PIP), ns.StrategyH(5, 15, 60, min_rr=0.0, min_range_adr=0, max_range_adr=9, pip=PIP),
                  ns.StrategyG(5, 15, 240, momentum="none", pip=PIP)):
            res = bt.run_backtest(bars, SPEC, s, equity=10000, risk_pct=0.5, slip_pips=0.0)
            self.assertIn("trades", res)


if __name__ == "__main__":
    unittest.main()
