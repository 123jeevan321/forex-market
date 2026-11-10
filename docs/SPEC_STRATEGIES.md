# Spec strategies A / B / C: exact rules, ambiguities, data

Code: `forex-backtest/spec_engine.py` (rules), `forex-backtest/spec_run.py` (runner, reports),
tests in `tests/test_forex_spec.py`. Run: `python forex-backtest/spec_run.py --config 2 --tune`.

## Configurations
| config | trend frame | setup frame | entry frame | history available |
| --- | --- | --- | --- | --- |
| 1 | 15m | 5m | 1m | about 14 weeks (MT5 keeps only 100,000 bars per timeframe) |
| 2 | 1h | 15m | 5m | about 16 months |

Neither reaches the 2-year minimum. To get more M1/M5 history, raise MT5 "Max bars in chart" (Tools, Options,
Charts) to Unlimited, restart the terminal and re-pull.

## No look-ahead
* A swing needs 2 bars on each side; it is known only 2 bars after it forms (`Swings.known`).
* Higher-frame values are read from the last COMPLETED bar at the moment of the decision.
* A setup is usable from the close of the bar that completes it. Entry = confirmation candle closes, fill at the next open.
* `tests/test_forex_spec.py::NoLookAheadTests` removes the future from the data and checks earlier signals are identical.

## Strategy A, plain language
1. Trend (trend frame): last two confirmed swing highs and lows both rising (bull) or falling (bear), ADX >= 15
   (else "sideways", no trade), and in the `full` mode price on the right side of EMA 50.
2. Liquidity levels known before the setup bar: previous day high/low, Asian high/low (after 07:00 UTC),
   London 07-13 high/low (after 13:00 UTC), the last 3 confirmed swing highs/lows, equal highs/lows (two swings within 0.15 ATR).
3. Sweep: a setup-frame bar wicks beyond a level and closes back on the other side. The wick extreme is recorded.
4. MSS: within 10 bars, a close beyond the latest confirmed swing on the far side of the sweep bar's close.
   Dead if the sweep extreme is broken again first.
5. FVG: three candles with a gap of at least `fvg_atr` x ATR, formed from the sweep bar onward. The setup exists from the close of the bar that completes MSS + FVG.
6. Entry: within 16 setup bars, price trades into the gap and an entry-frame candle closes in the trade direction
   (and above the gap's far edge). Enter at the next open. Dead if the stop level trades first.
7. Stop: sweep extreme -/+ `buf_atr` x ATR. Rejected if wider than 4 ATR or tighter than 0.3 ATR (cannot survive the spread).
8. Target: fixed reward:risk. Entries only 06:00-20:00 UTC; everything is flat at 20:00 UTC (so no swap).

## Ambiguities in the written rules and how they were resolved
* "Approaches a level": taken as the wick reaching through it.
* "Latest confirmed lower high": the newest confirmed swing high above the sweep bar's close.
* Which FVG: the newest one formed since the sweep.
* "1-minute confirmation candle": a candle closing in the trade direction after price has touched the gap.
* "Execute at the next available price": next bar open plus spread and slippage.
* Previous session: the Asian range (for London) and the London range (for New York).
* "Excessively sideways": ADX below 15 on the trend frame.
* Partial exit at 1:1 and "next opposing liquidity" targets are NOT implemented in this pass; only fixed 1:1.5 / 2 / 2.5 / 3.
* Not implemented: order blocks, premium/discount, VWAP, Volume Profile and tick-volume filters as extra filters on A/B/C
  (tick-volume order-flow proxies were tested separately, see `orderflow.py`), news filter (no historical news data).

## Data limits
* Tick volume is how often the quote changed, not traded volume. Spot FX has no central volume.
* Spreads are XM's recorded per-bar spread; slippage 0.3 pip (scaled to each symbol's volatility). Not verified against fills.
* Swap is not modelled because all trades close by 20:00 UTC.
* Candle data hides the order of events inside a bar: if one candle touches both stop and target, the stop is assumed first.

## Data availability for the CME order-flow brief (second brief)
| item | available here? |
| --- | --- |
| CME 6E/6B/6J/6A/6C executed trades, bid/ask classified | NO |
| CME footprint (volume at price), real delta, real CVD | NO |
| CME DOM / depth history | NO |
| Stacked diagonal imbalances on real prints | NO |
| Real session volume profile (POC, VAH, VAL) | NO |
| XM quote ticks (bid/ask changes) | YES, 1-2 years |
| Tick-rule delta proxy from XM quotes | YES (tested: no predictive power) |
| Tick volume, VWAP on tick volume | YES (not real volume) |

Those results cannot be produced without a purchased futures data feed (for example CME MDP 3.0 via a data vendor).
Nothing in this repo claims to have measured real order flow.
