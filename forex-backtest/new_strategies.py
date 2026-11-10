#!/usr/bin/env python3
"""Strategies D-H. Reuses spec_engine (frames, causal swings, session statistics, account simulator) and the
existing engine in backtest.py. Signals are computed in prepare() as {entry-bar index: signal}; every rule reads
only bars that are complete at the decision time (tests prove it by deleting the future).

  D  VWAP mean reversion + volatility regime       (1h regime / 15m setup / 5m entry)
  E  Volatility compression -> expansion           (Bollinger inside Keltner, then a breakout)
  F  Failed breakout reversal                      (failed ACCEPTANCE outside a range; no MSS/FVG)
  G  Higher-timeframe pullback + momentum recovery (4h trend / 15m pullback / 5m entry)
  H  Asian range mean reversion                    (Tokyo-session range; London or New York entries)

VWAP here is built from BROKER TICK VOLUME (how often the quote changed). It is not centralized FX volume.
"""

import bisect
import importlib.util
import math
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def _sib(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


se = _sib("spec_engine")
bt = se.bt
DAY = 86400
Frame = se.Frame


# ------------------------------------------------------------------ indicators

def vwap_bands(bars):
    """Session VWAP (resets at 00:00 UTC) weighted by tick volume, and the volume-weighted standard deviation.
    Returns (vwap, sd, bars_since_reset) per bar, using the bar itself and everything before it in the session."""
    vw, sd, cnt = [], [], []
    day, sv, spv, spv2, n = None, 0.0, 0.0, 0.0, 0
    for b in bars:
        d = b["t"] // DAY
        if d != day:
            day, sv, spv, spv2, n = d, 0.0, 0.0, 0.0, 0
        tp = (b["h"] + b["l"] + b["c"]) / 3.0
        v = b["v"] or 1
        sv, spv, spv2, n = sv + v, spv + tp * v, spv2 + tp * tp * v, n + 1
        m = spv / sv
        vw.append(m)
        sd.append(math.sqrt(max(spv2 / sv - m * m, 0.0)))
        cnt.append(n)
    return vw, sd, cnt


def roll_mean_std(vals, n):
    """Rolling mean and population standard deviation over the last n values (None until n values exist)."""
    mean, std, s, s2 = [], [], 0.0, 0.0
    for i, v in enumerate(vals):
        s, s2 = s + v, s2 + v * v
        if i >= n:
            s, s2 = s - vals[i - n], s2 - vals[i - n] ** 2
        if i >= n - 1:
            m = s / n
            mean.append(m)
            std.append(math.sqrt(max(s2 / n - m * m, 0.0)))
        else:
            mean.append(None)
            std.append(None)
    return mean, std


def macd_hist(closes, fast=12, slow=26, sig=9):
    m = [a - b for a, b in zip(bt.ema_series(closes, fast), bt.ema_series(closes, slow))]
    return [a - b for a, b in zip(m, bt.ema_series(m, sig))]


def stochastic(bars, n=14, ks=3, ds=3):
    """Slow stochastic: %K (smoothed over ks) and %D (smoothed over ds). None until enough history."""
    raw = []
    for i, b in enumerate(bars):
        if i < n - 1:
            raw.append(None)
            continue
        w = bars[i - n + 1:i + 1]
        hh, ll = max(x["h"] for x in w), min(x["l"] for x in w)
        raw.append(50.0 if hh == ll else (b["c"] - ll) / (hh - ll) * 100.0)

    def sma(vals, k):
        out = []
        for i in range(len(vals)):
            w = vals[i - k + 1:i + 1] if i >= k - 1 else []
            out.append(sum(w) / k if len(w) == k and None not in w else None)
        return out
    kk = sma(raw, ks)
    return kk, sma(kk, ds)


class _NS(se._Base):
    """Shared helpers for D-H."""

    def _lim(self, atr):
        return {"max_sl_pips": self.p["max_stop_atr"] * atr / self.pip, "min_sl_pips": self.p["min_stop_atr"] * atr / self.pip}

    def _confirm(self, bars, times, avail, valid_s, side, stop, atr, setup, **extra):
        """First entry-frame candle that closes in the trade direction after `avail`; dead if the stop level trades first."""
        i0 = bisect.bisect_left(times, avail)
        i1 = bisect.bisect_left(times, avail + valid_s)
        long_ = side == "long"
        for i in range(i0, min(i1, len(bars) - 1)):
            e = bars[i]
            if (e["l"] <= stop) if long_ else (e["h"] >= stop):
                return None
            if not self._hours_ok(e):
                continue
            if (long_ and e["c"] > e["o"]) or ((not long_) and e["c"] < e["o"]):
                self._emit(i, dict({"side": side, "sl_price": stop, "setup": setup}, **self._lim(atr), **extra))
                return i
        return None


# ------------------------------------------------------------------ D: VWAP mean reversion

class StrategyD(_NS):
    def __init__(self, tf_e, setup_min, trend_min, sd=2.0, adx_max=20.0, tp_mode="vwap", buf_atr=0.10,
                 slope_filter=False, atr_expansion=False, min_rr=0.8, valid_bars=4, min_stop_atr=0.3,
                 max_stop_atr=4.0, pip=0.0001):
        self.tf_e, self.pip = tf_e * 60, pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, sd=sd, adx_max=adx_max, tp_mode=tp_mode, buf_atr=buf_atr,
                      slope_filter=slope_filter, atr_expansion=atr_expansion, min_rr=min_rr, valid_bars=valid_bars,
                      min_stop_atr=min_stop_atr, max_stop_atr=max_stop_atr,
                      rr={"rr1.5": 1.5, "rr2": 2.0}.get(tp_mode, 2.0))
        self.sig = {}

    def prepare(self, bars):
        p = self.p
        self.sig = {}
        S = Frame(bars, p["setup_min"], 2)
        T = Frame(bars, p["trend_min"], 2, extras=True)
        vw, sd, cnt = vwap_bands(S.b)
        times = [x["t"] for x in bars]
        for j in range(40, len(S.b)):
            if cnt[j] < 8 or sd[j] <= 0:
                continue
            b, avail = S.b[j], S.close_t[j]
            k = T.done(avail)
            if k < 60 or T.adx[k] >= p["adx_max"]:                    # not range-bound: stay out
                continue
            if p["slope_filter"] and abs(T.ema50[k] - T.ema50[k - 5]) > 0.5 * T.atr[k]:
                continue
            if p["atr_expansion"] and S.atr[j] > 1.25 * S.atr[j - 12]:
                continue
            lo_b, up_b = vw[j] - p["sd"] * sd[j], vw[j] + p["sd"] * sd[j]
            if b["l"] < lo_b and b["c"] > lo_b:
                side, stop = "long", b["l"] - p["buf_atr"] * S.atr[j]
            elif b["h"] > up_b and b["c"] < up_b:
                side, stop = "short", b["h"] + p["buf_atr"] * S.atr[j]
            else:
                continue
            extra = {}
            if p["tp_mode"] in ("vwap", "partial"):
                extra = {"tp_price": vw[j], "min_rr": p["min_rr"]}
            if p["tp_mode"] == "partial":
                extra["partial_rr"] = 1.0
            self._confirm(bars, times, avail, p["valid_bars"] * S.step, side, stop, S.atr[j], "vwap-" + side, **extra)
        return self


# ------------------------------------------------------------------ E: volatility compression -> expansion

class StrategyE(_NS):
    def __init__(self, tf_e, setup_min, trend_min, squeeze_n=10, entry_mode="immediate", filt="none", atr_mult=1.5,
                 rr=2.0, disp_min=1.0, kc_mult=1.5, valid_bars=8, min_stop_atr=0.3, max_stop_atr=4.0, pip=0.0001):
        self.tf_e, self.pip = tf_e * 60, pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, squeeze_n=squeeze_n, entry_mode=entry_mode, filt=filt,
                      atr_mult=atr_mult, rr=rr, disp_min=disp_min, kc_mult=kc_mult, valid_bars=valid_bars,
                      min_stop_atr=min_stop_atr, max_stop_atr=max_stop_atr)
        self.sig = {}

    def prepare(self, bars):
        p = self.p
        self.sig = {}
        S = Frame(bars, p["setup_min"], 2, extras=True)
        T = Frame(bars, p["trend_min"], 2, extras=True)
        vw, _, _ = vwap_bands(S.b)
        closes = [x["c"] for x in S.b]
        mean, sdv = roll_mean_std(closes, 20)
        ema20 = bt.ema_series(closes, 20)
        times = [x["t"] for x in bars]
        run, armed, cool = 0, None, False                      # armed: range of the compression, as of the previous bar
        for j in range(60, len(S.b)):
            b = S.b[j]
            if armed is not None:
                broke = self._breakout(S, T, vw, j, armed, p)
                if broke:
                    self._enter(bars, times, S, j, broke, armed)
                    armed, cool = None, True                          # one entry per squeeze: wait for it to end
                elif armed["off"] is not None and j - armed["off"] > 12:
                    armed = None
            sq = (mean[j] is not None and mean[j] + 2 * sdv[j] < ema20[j] + p["kc_mult"] * S.atr[j]
                  and mean[j] - 2 * sdv[j] > ema20[j] - p["kc_mult"] * S.atr[j])
            run = run + 1 if sq else 0
            if not sq:
                cool = False
            if sq and run >= p["squeeze_n"] and not cool:
                w = S.b[j - run + 1:j + 1]
                armed = {"hi": max(x["h"] for x in w), "lo": min(x["l"] for x in w), "off": None}
            elif armed is not None and not sq and armed["off"] is None:
                armed["off"] = j
        return self

    def _breakout(self, S, T, vw, j, armed, p):
        b = S.b[j]
        rng = b["h"] - b["l"]
        if rng <= 0 or S.atr[j] <= S.atr[j - 1] or rng < p["disp_min"] * S.atr[j - 1] or abs(b["c"] - b["o"]) < 0.5 * rng:
            return None
        side = "long" if b["c"] > armed["hi"] else "short" if b["c"] < armed["lo"] else None
        if side is None:
            return None
        lg = side == "long"
        f = p["filt"]
        if f == "ema50" and not ((b["c"] > S.ema50[j] and S.ema50[j] > S.ema50[j - 3]) if lg else (b["c"] < S.ema50[j] and S.ema50[j] < S.ema50[j - 3])):
            return None
        if f == "vwap" and not (b["c"] > vw[j] if lg else b["c"] < vw[j]):
            return None
        if f == "adx":
            k = T.done(S.close_t[j])
            if k < 60 or T.adx[k] < 20:
                return None
        return side

    def _enter(self, bars, times, S, j, side, armed):
        p = self.p
        avail, atr = S.close_t[j], S.atr[j]
        lg = side == "long"
        if p["entry_mode"] == "immediate":
            i = bisect.bisect_left(times, avail) - 1
            if i < 0 or times[i] + self.tf_e != avail or not self._hours_ok(bars[i]):
                return
            ref = bars[i]["c"]
            stop = ref - p["atr_mult"] * atr if lg else ref + p["atr_mult"] * atr
            self._emit(i, dict({"side": side, "sl_price": stop, "setup": "squeeze-break"}, **self._lim(atr)))
            return
        edge = armed["hi"] if lg else armed["lo"]
        i0 = bisect.bisect_left(times, avail)
        i1 = bisect.bisect_left(times, avail + p["valid_bars"] * S.step)
        for i in range(i0, min(i1, len(bars) - 1)):
            e = bars[i]
            if (e["c"] < edge) if lg else (e["c"] > edge):                 # fell back through the range: failed
                return
            if not self._hours_ok(e):
                continue
            if ((e["l"] <= edge and e["c"] > edge and e["c"] > e["o"]) if lg else (e["h"] >= edge and e["c"] < edge and e["c"] < e["o"])):
                stop = e["c"] - p["atr_mult"] * atr if lg else e["c"] + p["atr_mult"] * atr
                self._emit(i, dict({"side": side, "sl_price": stop, "setup": "squeeze-retest"}, **self._lim(atr)))
                return


# ------------------------------------------------------------------ F: failed breakout (failed acceptance)

class StrategyF(_NS):
    def __init__(self, tf_e, setup_min, trend_min, target="mid", levels=("pd", "asian", "london"), accept_bars=3,
                 buf_atr=0.10, min_rr=0.8, valid_bars=4, rr=2.0, min_stop_atr=0.3, max_stop_atr=4.0, pip=0.0001):
        self.tf_e, self.pip = tf_e * 60, pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, target=target, levels=tuple(levels),
                      accept_bars=accept_bars, buf_atr=buf_atr, min_rr=min_rr, valid_bars=valid_bars, rr=rr,
                      min_stop_atr=min_stop_atr, max_stop_atr=max_stop_atr)
        self.sig = {}

    def prepare(self, bars):
        p = self.p
        self.sig = {}
        S = Frame(bars, p["setup_min"], 2)
        stats = se.DayStats(S.b, S.step)
        vw, _, cnt = vwap_bands(S.b)
        times = [x["t"] for x in bars]
        by_day = {}
        for j, b in enumerate(S.b):
            by_day.setdefault(b["t"] // DAY, []).append(j)
        for d, idxs in sorted(by_day.items()):
            ranges = []
            pv = stats.prev_day(d)
            if pv:
                ranges.append(("pd", pv[1], pv[0], 0))
            if d in stats.asian:
                ranges.append(("asian", stats.asian[d][1], stats.asian[d][0], 7 * 3600))
            if d in stats.london:
                ranges.append(("london", stats.london[d][1], stats.london[d][0], 13 * 3600))
            for name, lo, hi, start in ranges:
                if name in p["levels"] and hi > lo:
                    self._scan(bars, times, S, vw, cnt, idxs, name, lo, hi, start)
        return self

    def _scan(self, bars, times, S, vw, cnt, idxs, name, lo, hi, start):
        p = self.p
        st = {"long": None, "short": None}                     # [first bar outside, extreme] while outside
        done = set()
        for j in idxs:
            b = S.b[j]
            if b["t"] % DAY < start or j < 30:
                continue
            for side in ("long", "short"):
                if side in done:
                    continue
                lg = side == "long"
                out = (b["c"] < lo) if lg else (b["c"] > hi)
                inside = (b["c"] >= lo) if lg else (b["c"] <= hi)
                s = st[side]
                if s is None:
                    if out:
                        st[side] = [j, b["l"] if lg else b["h"]]
                    continue
                s[1] = min(s[1], b["l"]) if lg else max(s[1], b["h"])
                if inside:                                     # closed back inside: the breakout was not accepted
                    done.add(side)
                    ext = s[1]
                    atr = S.atr[j]
                    stop = ext - p["buf_atr"] * atr if lg else ext + p["buf_atr"] * atr
                    tgt = {"mid": (lo + hi) / 2.0, "opp": hi if lg else lo, "vwap": vw[j] if cnt[j] >= 8 else None}[p["target"]]
                    if tgt is None:
                        continue
                    self._confirm(bars, times, S.close_t[j], p["valid_bars"] * S.step, side, stop, atr, "failed-" + name,
                                  tp_price=tgt, min_rr=p["min_rr"])
                elif j - s[0] >= p["accept_bars"]:             # stayed outside too long: it was accepted
                    done.add(side)


# ------------------------------------------------------------------ G: higher-timeframe pullback

class StrategyG(_NS):
    def __init__(self, tf_e, setup_min, trend_min=240, momentum="rsi", rr=2.0, buf_atr=0.10, impulse_atr=2.0,
                 depth_min=0.5, depth_max_frac=0.8, window=32, valid_bars=3, min_stop_atr=0.3, max_stop_atr=4.0,
                 pip=0.0001):
        self.tf_e, self.pip = tf_e * 60, pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, momentum=momentum, rr=rr, buf_atr=buf_atr,
                      impulse_atr=impulse_atr, depth_min=depth_min, depth_max_frac=depth_max_frac, window=window,
                      valid_bars=valid_bars, min_stop_atr=min_stop_atr, max_stop_atr=max_stop_atr)
        self.sig = {}

    def _recovering(self, mode, side, j, S, rsi, kk, dd, mh, ema20):
        lg = side == "long"
        if mode == "none":
            return True
        if mode == "rsi":
            return (any(rsi[m] < 40 for m in range(j - 3, j)) and rsi[j] > 40) if lg else (any(rsi[m] > 60 for m in range(j - 3, j)) and rsi[j] < 60)
        if mode == "stoch":
            if None in (kk[j], dd[j], kk[j - 1], dd[j - 1]):
                return False
            if lg:
                return kk[j] > dd[j] and kk[j - 1] <= dd[j - 1] and min(x for x in kk[j - 3:j + 1] if x is not None) < 30
            return kk[j] < dd[j] and kk[j - 1] >= dd[j - 1] and max(x for x in kk[j - 3:j + 1] if x is not None) > 70
        if mode == "macd":
            return (mh[j] > mh[j - 1] and mh[j - 1] <= mh[j - 2] and mh[j - 1] < 0) if lg else (mh[j] < mh[j - 1] and mh[j - 1] >= mh[j - 2] and mh[j - 1] > 0)
        if mode == "ema":
            return (S.b[j]["c"] > ema20[j] and ema20[j] > S.ema50[j]) if lg else (S.b[j]["c"] < ema20[j] and ema20[j] < S.ema50[j])
        raise ValueError(mode)

    def prepare(self, bars):
        p = self.p
        self.sig = {}
        S = Frame(bars, p["setup_min"], 2, extras=True)
        T = Frame(bars, p["trend_min"], 2, extras=True)
        closes = [x["c"] for x in S.b]
        kk, dd = stochastic(S.b)
        mh, ema20 = macd_hist(closes), bt.ema_series(closes, 20)
        times = [x["t"] for x in bars]
        seen, W = set(), p["window"]
        for j in range(60, len(S.b)):
            avail = S.close_t[j]
            k = T.done(avail)
            if k < 210:
                continue
            hs, ls = T.sw.known("hi", k, 2), T.sw.known("lo", k, 2)
            if len(hs) < 2 or len(ls) < 2:
                continue
            up = T.ema50[k] > T.ema200[k] and hs[1][2] > hs[0][2] and ls[1][2] > ls[0][2]
            dn = T.ema50[k] < T.ema200[k] and hs[1][2] < hs[0][2] and ls[1][2] < ls[0][2]
            if not (up or dn):
                continue
            side = "long" if up else "short"
            lg = up
            win = S.b[j - W:j + 1]
            if lg:
                ih = j - W + max(range(len(win)), key=lambda m: win[m]["h"])
            else:
                ih = j - W + min(range(len(win)), key=lambda m: win[m]["l"])
            if j - ih < 2 or ih < 20 or (side, ih) in seen:
                continue
            base = S.b[ih - 16:ih + 1]
            ext_imp = S.b[ih]["h"] if lg else S.b[ih]["l"]
            start = min(x["l"] for x in base) if lg else max(x["h"] for x in base)
            imp = abs(ext_imp - start)
            if imp < p["impulse_atr"] * S.atr[ih]:
                continue
            leg = S.b[ih:j + 1]
            pull = min(x["l"] for x in leg) if lg else max(x["h"] for x in leg)
            depth = abs(ext_imp - pull)
            if depth < p["depth_min"] * S.atr[j] or depth > p["depth_max_frac"] * imp:
                continue
            if (lg and pull <= ls[-1][2]) or ((not lg) and pull >= hs[-1][2]):        # higher-timeframe structure broken
                continue
            if not self._recovering(p["momentum"], side, j, S, S.rsi, kk, dd, mh, ema20):
                continue
            seen.add((side, ih))
            atr = S.atr[j]
            stop = pull - p["buf_atr"] * atr if lg else pull + p["buf_atr"] * atr
            self._confirm(bars, times, avail, p["valid_bars"] * S.step, side, stop, atr, "pullback-" + p["momentum"])
        return self


# ------------------------------------------------------------------ H: Asian range mean reversion

ASIAN = ("Asia/Tokyo", 9, 0, 7.0)                                # 09:00-16:00 Tokyo (00:00-07:00 UTC; Tokyo has no DST)
ENTRY_SESSIONS = {"london": ("Europe/London", 8, 0, 4.0), "ny": ("America/New_York", 8, 0, 4.0)}


def window_utc(day, spec):
    zone, h, m, hours = spec
    d = datetime.fromtimestamp(day * DAY, timezone.utc)
    t0 = datetime(d.year, d.month, d.day, h, m, tzinfo=ZoneInfo(zone)).timestamp()
    return t0, t0 + hours * 3600


class StrategyH(_NS):
    def __init__(self, tf_e, setup_min, trend_min, session="london", target="mid", buf_atr=0.10, min_range_adr=0.25,
                 max_range_adr=0.80, adx_strong=25.0, accept_bars=3, min_rr=0.8, valid_bars=4, rr=2.0,
                 min_stop_atr=0.3, max_stop_atr=4.0, pip=0.0001):
        self.tf_e, self.pip = tf_e * 60, pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, session=session, target=target, buf_atr=buf_atr,
                      min_range_adr=min_range_adr, max_range_adr=max_range_adr, adx_strong=adx_strong,
                      accept_bars=accept_bars, min_rr=min_rr, valid_bars=valid_bars, rr=rr,
                      min_stop_atr=min_stop_atr, max_stop_atr=max_stop_atr)
        self.sig = {}

    def prepare(self, bars):
        p = self.p
        self.sig = {}
        S = Frame(bars, p["setup_min"], 2)
        T = Frame(bars, p["trend_min"], 2, extras=True)
        adr = bt.daily_adr(S.b)
        times = [x["t"] for x in bars]
        stimes = [x["t"] for x in S.b]
        need = 0.8 * ASIAN[3] * 3600 / S.step
        for day in range(stimes[0] // DAY, stimes[-1] // DAY + 1):
            if datetime.fromtimestamp(day * DAY, timezone.utc).weekday() >= 5 or day not in adr:
                continue
            a0, a1 = window_utc(day, ASIAN)
            ia, ib = bisect.bisect_left(stimes, a0), bisect.bisect_left(stimes, a1)
            if ib - ia < need:
                continue
            w = S.b[ia:ib]
            hi, lo = max(x["h"] for x in w), min(x["l"] for x in w)
            if not (p["min_range_adr"] * adr[day] <= hi - lo <= p["max_range_adr"] * adr[day]):   # abnormal width
                continue
            e0, e1 = window_utc(day, ENTRY_SESSIONS[p["session"]])
            self._day(bars, times, S, T, lo, hi, bisect.bisect_left(stimes, e0), bisect.bisect_left(stimes, e1))
        return self

    def _day(self, bars, times, S, T, lo, hi, j0, j1):
        p = self.p
        st, done = {"long": None, "short": None}, set()
        for j in range(max(j0, 30), min(j1, len(S.b))):
            b = S.b[j]
            for side in ("long", "short"):
                if side in done:
                    continue
                lg = side == "long"
                if st[side] is None and ((b["l"] < lo) if lg else (b["h"] > hi)):
                    st[side] = [j, b["l"] if lg else b["h"]]
                elif st[side] is not None:
                    s = st[side]
                    s[1] = min(s[1], b["l"]) if lg else max(s[1], b["h"])
                if st[side] is None:
                    continue
                s = st[side]
                inside = (lo <= b["c"] <= hi)
                if inside:
                    done.add(side)
                    if j - s[0] >= p["accept_bars"]:
                        continue
                    k = T.done(S.close_t[j])
                    if k < 60:
                        continue
                    slope = T.ema50[k] - T.ema50[k - 3]
                    if T.adx[k] >= p["adx_strong"] and ((slope < 0) if lg else (slope > 0)):   # strong trend against us
                        continue
                    atr = S.atr[j]
                    stop = s[1] - p["buf_atr"] * atr if lg else s[1] + p["buf_atr"] * atr
                    tgt = (lo + hi) / 2.0 if p["target"] == "mid" else (hi if lg else lo)
                    self._confirm(bars, times, S.close_t[j], p["valid_bars"] * S.step, side, stop, atr,
                                  "asian-" + p["session"], tp_price=tgt, min_rr=p["min_rr"])
                elif j - s[0] >= p["accept_bars"]:
                    done.add(side)
