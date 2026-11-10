import random
import unittest
from types import SimpleNamespace

from _load import load

ch = load("forex-backtest", "challenge")

ARGS = dict(balance=5000.0, target1=8.0, target2=5.0, max_loss=10.0, daily_loss=5.0, split=80.0,
            payout_every=10, funded_days=120, fee=36.0, sims=2000)


class ChallengeTests(unittest.TestCase):
    def test_coin_flip_reference(self):
        self.assertAlmostEqual(ch.coin_flip_pass(8, 10), 10 / 18.0)
        self.assertAlmostEqual(ch.coin_flip_pass(5, 10), 10 / 15.0)

    def test_always_winning_passes(self):
        ok, n = ch.run_phase([[1.0]], random.Random(1), 5000, 1.0, 8.0, 10.0, 5.0)
        self.assertTrue(ok)
        self.assertEqual(n, 8)                      # +1% a day -> +8% on day 8

    def test_always_losing_fails_on_total_loss(self):
        ok, n = ch.run_phase([[-1.0]], random.Random(1), 5000, 1.0, 8.0, 10.0, 5.0)
        self.assertFalse(ok)
        self.assertEqual(n, 10)

    def test_daily_loss_limit(self):
        ok, n = ch.run_phase([[-1.0] * 6], random.Random(1), 5000, 1.0, 8.0, 10.0, 5.0)
        self.assertFalse(ok)
        self.assertEqual(n, 1)                      # five 1% losses in one day breach the 5% daily limit

    def test_funded_payout_uses_the_split(self):
        paid, alive = ch.run_funded([[1.0]], random.Random(1), 5000, 1.0, 10.0, 5.0, 80.0, 10, 20)
        self.assertTrue(alive)
        self.assertAlmostEqual(paid, 2 * 0.8 * 500)  # two payouts of 80% of $500

    def test_zero_edge_matches_theory(self):
        s = ch.simulate([[1.0], [-1.0]], 1.0, SimpleNamespace(**ARGS))
        self.assertAlmostEqual(s["pass1"], 10 / 18.0, delta=0.04)
        self.assertAlmostEqual(s["pass_both"], (10 / 18.0) * (10 / 15.0), delta=0.04)

    def test_losing_strategy_rarely_passes(self):
        s = ch.simulate([[-1.0], [-1.0], [1.0]], 1.0, SimpleNamespace(**ARGS))
        self.assertLess(s["pass_both"], 0.02)
        self.assertLess(s["ev_per_attempt"], 0)


if __name__ == "__main__":
    unittest.main()
