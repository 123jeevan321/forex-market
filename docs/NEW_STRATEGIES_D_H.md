# Strategies D-H: rules, limits, results location

Code: `forex-backtest/new_strategies.py`, runner `forex-backtest/new_run.py`, tests `tests/test_forex_new.py`.
Run from the repo root: `python forex-backtest/new_run.py` (all) or `--kinds D --symbols EURUSD --no-tune`.
Outputs (old results in `trader/reports` are never touched): `reports/new_strategies_comparison.csv`,
`reports/unified_comparison.txt`, `reports/data_coverage.txt`, and one folder per strategy with `summary.txt`,
`trades_xm_costs.csv`, `equity_curve_FX5.csv` and an equity/drawdown/monthly chart.

Shared: frames 1h regime or trend (G: 4h), 15m setup, 5m entry; enter at the next open after a confirmation candle
closes in the trade direction; entries 06:00-20:00 UTC; flat 20:00 UTC; stops 0.3-4 ATR(15m); $10,000, 0.5% risk,
max 2 open, 1% daily loss stop; chronological 60/20/20 split; 16 months of M5 data. Stop assumed first if one candle
touches both stop and target.

| | rule |
| --- | --- |
| D | Session VWAP (resets 00:00 UTC) from broker TICK volume, volume-weighted SD bands (1.5/2/2.5). Range regime: 1h ADX below 18/20/25 (+ optional EMA50 slope, ATR expansion). A 15m bar wicks outside a band and closes back inside, then a confirming 5m candle. Stop beyond the rejection extreme + 0.1 ATR. Target: VWAP (rejected if reward < 0.8 x risk), 1:1.5, 1:2 or half at 1:1 + half at VWAP (stop not moved). |
| E | Bollinger(20,2) inside Keltner(20, 1.5 ATR) for 5/10/20 bars. A 15m close beyond the squeeze range with rising ATR and a displacement candle (range >= 1 ATR, body >= 50%). Immediate or retest entry; optional EMA50, 1h ADX, or VWAP direction filter. Stop 1.5 ATR. One entry per squeeze. |
| F | Ranges: previous day, Asian, London. A 15m bar CLOSES outside; within 3 bars one closes back inside (failed acceptance; a longer stay = accepted, no trade); confirmation candle. Stop beyond the excursion extreme. Target: range midpoint, opposite boundary or VWAP. No MSS/FVG. |
| G | 4h EMA50 > EMA200 with HH/HL swings; impulse >= 2 ATR on 15m; pullback 0.5 ATR to 80% of the impulse that keeps the 4h swing intact; momentum recovery by ONE of RSI / stochastic / MACD / EMA alignment (compared separately); stop beyond the pullback extreme. |
| H | Asian range from the Tokyo 09:00-16:00 session (time-zone aware), skipped if narrower than 0.25 or wider than 0.8 of the average daily range. A 15m bar moves outside and closes back inside within 3 bars, unless 1h ADX >= 25 with EMA50 sloping against; London or New York entries; target midpoint or opposite boundary. |

## Limits and failure conditions
* Tick-volume VWAP is broker-specific, not centralized FX volume.
* Only 16 months of 5-minute data (MT5 keeps 100,000 bars per timeframe). The 1-minute files (97 days) are not used.
* Costs: XM recorded spreads (actual), a FundingPips-style scenario (commission $5/lot from the help-centre text as surfaced by search;
  the 0.2-0.3 pip spreads are assumptions, NOT official), and a zero-cost diagnostic that cannot be achieved live.
* Swap not modelled (flat by 20:00 UTC). No news filter. Partial exits do not move the stop to breakeven.
* Sensitivity is one factor at a time; the "tuned" set is chosen on train+validation only and judged on the last 20%.
* A strategy with fewer than about 30 trades (G baseline, any single-symbol cell) says nothing statistically.
