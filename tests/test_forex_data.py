import os
import tempfile
import unittest

from _load import load

m = load("forex-data", "mt5_data")


class Info:
    name = "EURUSD"; digits = 5; point = 0.00001; trade_contract_size = 100000
    trade_tick_value = 1.0; trade_tick_size = 0.00001
    volume_min = 0.01; volume_step = 0.01; volume_max = 100.0
    spread = 12; trade_stops_level = 0; currency_profit = "USD"


class Gold(Info):
    name = "GOLD"; digits = 2; point = 0.01; trade_contract_size = 100
    trade_tick_value = 1.0; trade_tick_size = 0.01; spread = 25


class SpecTests(unittest.TestCase):
    def test_pip_size(self):
        self.assertAlmostEqual(m.pip_size(5, 0.00001), 0.0001)
        self.assertAlmostEqual(m.pip_size(3, 0.001), 0.01)   # USDJPY
        self.assertAlmostEqual(m.pip_size(2, 0.01), 0.01)    # gold

    def test_eurusd_spec(self):
        s = m.symbol_spec(Info())
        self.assertEqual(s["spread_pips"], 1.2)
        self.assertAlmostEqual(s["pip_value_per_lot"], 10.0)  # $10 a pip per lot

    def test_gold_spec(self):
        s = m.symbol_spec(Gold())
        self.assertAlmostEqual(s["pip_value_per_lot"], 1.0)

    def test_missing_info_raises(self):
        with self.assertRaises(m.DataError):
            m.symbol_spec({"name": "X"})


class BarTests(unittest.TestCase):
    def rates(self):
        return [{"time": 1000 + 300 * i + 7200, "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15,
                 "tick_volume": 10, "spread": 12} for i in range(4)][::-1]

    def test_offset_and_sort(self):
        bars = m.rates_to_bars(self.rates(), 7200)
        self.assertEqual([b["t"] for b in bars], [1000, 1300, 1600, 1900])

    def test_check_clean(self):
        self.assertEqual(m.check_bars(m.rates_to_bars(self.rates(), 7200), 5), [])

    def test_check_bad_ohlc_and_gap(self):
        bars = m.rates_to_bars(self.rates(), 7200)
        bars[1]["h"] = 1.0
        bars[3]["t"] += 5 * 3600   # holes over 2h that are not a weekend are flagged
        p = m.check_bars(bars, 5)
        self.assertTrue(any("OHLC" in x for x in p))
        self.assertTrue(any("gap" in x for x in p))

    def test_csv(self):
        text = "Date\tTime\tOpen\tHigh\tLow\tClose\tTickVol\tVol\tSpread\n" \
               "2026.11.02\t10:00\t1.1\t1.2\t1.0\t1.15\t50\t0\t12\n" \
               "2026.11.02\t10:05\t1.15\t1.2\t1.1\t1.18\t40\t0\t11\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(text)
        try:
            bars = m.read_csv_bars(f.name)
        finally:
            os.unlink(f.name)
        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[1]["t"] - bars[0]["t"], 300)
        self.assertEqual(bars[0]["spread"], 12)


class ReadOnlyTest(unittest.TestCase):
    def test_no_order_calls(self):
        src = open(m.__file__).read()
        self.assertNotIn("order_send", src)
        self.assertNotIn("order_check", src)


if __name__ == "__main__":
    unittest.main()
