#!/usr/bin/env python3
"""Spec strategies A / B / C, causal structure tools and the account simulator. Offline, standard library.

  A  Liquidity sweep -> market structure shift (MSS) -> fair value gap (FVG) retest
  B  Trend continuation: displacement candle, pullback into its FVG / origin, confirmation
  C  Session breakout + retest (time-zone aware: London and New York local clock, DST handled)

Look-ahead rules used everywhere
  * A swing point needs `n` bars on each side, so it is only KNOWN `n` bars after it forms.
  * Higher-timeframe values (EMA, ADX, swings, trend) are read from the last COMPLETED higher-timeframe bar.
  * A setup is usable only from the close of the bar that completes it; entries are filled at the NEXT open.
  * Every signal is computed in prepare() as a map {entry-bar index: signal}; a test proves that cutting off
    the future leaves earlier signals unchanged.
"""

import bisect
import importlib.util
import math
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_spec = importlib.util.spec_from_file_location("bt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

DAY = 86400
ENTRY_FROM_H, ENTRY_TO_H = 6, 20         # no entries in the thin / wide-spread hours around rollover
FLAT_H = 20                              # everything closed by 20:00 UTC: no overnight swap to model


# ------------------------------------------------------------------ indicators

def rsi_series(closes, n=14):
    out, ag, al, prev = [], None, None, None
    for c in closes:
        if prev is None:
            out.append(50.0)
        else:
            ch = c - prev
            g, l = max(ch, 0.0), max(-ch, 0.0)
            ag = g if ag is None else ag + (g - ag) / n
            al = l if al is None else al + (l - al) / n
            out.append(100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al))
        prev = c
    return out


def adx_series(bars, n=14):
    out, pdm, mdm, tr_s, adx, prev = [], 0.0, 0.0, 0.0, None, None
    for i, b in enumerate(bars):
        if prev is None:
            out.append(0.0)
            prev = b
            continue
        up, dn = b["h"] - prev["h"], prev["l"] - b["l"]
        pd = up if up > dn and up > 0 else 0.0
        md = dn if dn > up and dn > 0 else 0.0
        tr = max(b["h"] - b["l"], abs(b["h"] - prev["c"]), abs(b["l"] - prev["c"]))
        pdm, mdm, tr_s = pdm - pdm / n + pd, mdm - mdm / n + md, tr_s - tr_s / n + tr
        if tr_s > 0:
            pdi, mdi = 100 * pdm / tr_s, 100 * mdm / tr_s
            dx = 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
        else:
            dx = 0.0
        adx = dx if adx is None else adx + (dx - adx) / n
        out.append(adx)
        prev = b
    return out


# ------------------------------------------------------------------ confirmed swings

class Swings:
    """Fractal swings: a high that is above `n` bars on each side. Confirmed only at index i + n."""

    def __init__(self, bars, n):
        self.n = n
        self.hi, self.lo = [], []                  # (confirm_idx, idx, price), ascending
        for i in range(n, len(bars) - n):
            h, l = bars[i]["h"], bars[i]["l"]
            if all(h > bars[i - k]["h"] for k in range(1, n + 1)) and all(h >= bars[i + k]["h"] for k in range(1, n + 1)):
                self.hi.append((i + n, i, h))
            if all(l < bars[i - k]["l"] for k in range(1, n + 1)) and all(l <= bars[i + k]["l"] for k in range(1, n + 1)):
                self.lo.append((i + n, i, l))
        self._hc = [x[0] for x in self.hi]
        self._lc = [x[0] for x in self.lo]

    def known(self, kind, j, count):
        """The last `count` swings of kind 'hi'/'lo' that are CONFIRMED by the close of bar j (newest last)."""
        arr, conf = (self.hi, self._hc) if kind == "hi" else (self.lo, self._lc)
        k = bisect.bisect_right(conf, j)
        return arr[max(0, k - count):k]


class Frame:
    """One timeframe built from the entry bars, with the series strategies need."""

    def __init__(self, entry_bars, minutes, swing_n=2, extras=False):
        self.step = minutes * 60
        self.b = bt.resample(entry_bars, minutes) if minutes else entry_bars
        self.close_t = [x["t"] + self.step for x in self.b]
        self.atr = bt.atr_series(self.b)
        self.sw = Swings(self.b, swing_n)
        if extras:
            closes = [x["c"] for x in self.b]
            self.ema50, self.ema200 = bt.ema_series(closes, 50), bt.ema_series(closes, 200)
            self.rsi, self.adx = rsi_series(closes), adx_series(self.b)

    def done(self, t):
        """Index of the last bar that is COMPLETE at time t (-1 if none)."""
        return bisect.bisect_right(self.close_t, t) - 1


def trend_dir(tf, k, mode, adx_min=15.0):
    """+1 / -1 / 0 from the last completed bar k. mode: 'none' (always 0 = no filter), 'struct', 'full'."""
    if mode == "none" or k < 220:
        return 0 if mode == "none" else None
    hs, ls = tf.sw.known("hi", k, 2), tf.sw.known("lo", k, 2)
    if len(hs) < 2 or len(ls) < 2 or tf.adx[k] < adx_min:
        return None                                 # no structure yet, or excessively sideways
    d = 1 if (hs[1][2] > hs[0][2] and ls[1][2] > ls[0][2]) else -1 if (hs[1][2] < hs[0][2] and ls[1][2] < ls[0][2]) else None
    if d is None or mode == "struct":
        return d
    c = tf.b[k]["c"]
    return d if (d > 0 and c > tf.ema50[k]) or (d < 0 and c < tf.ema50[k]) else None


def side_ok(direction, side):
    return direction == 0 or (direction > 0 and side == "long") or (direction < 0 and side == "short")


class _Base:
    flat_h = FLAT_H

    def flat_ts(self, t):
        return bt.next_hour_ts(t, self.flat_h)

    def on_bar(self, bars, i):
        s = self.sig.get(i)
        return dict(s, rr=self.p["rr"]) if s else None

    def _hours_ok(self, e):
        h = ((e["t"] + self.tf_e) % DAY) // 3600
        return ENTRY_FROM_H <= h < ENTRY_TO_H

    def _emit(self, i, sig):
        if i not in self.sig:
            self.sig[i] = sig


# ------------------------------------------------------------------ Strategy A

class StrategyA(_Base):
    """Liquidity sweep + MSS + FVG. See the module docstring and the README section for the exact rules."""

    def __init__(self, tf_e, setup_min, trend_min, fvg_atr=0.15, buf_atr=0.10, rr=2.0, trend_mode="full",
                 max_stop_atr=4.0, mss_window=10, valid_bars=16, swing_n=2, min_stop_atr=0.3, pip=0.0001):
        self.tf_e = tf_e * 60
        self.pip = pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, fvg_atr=fvg_atr, buf_atr=buf_atr, rr=rr,
                      trend_mode=trend_mode, max_stop_atr=max_stop_atr, mss_window=mss_window,
                      valid_bars=valid_bars, swing_n=swing_n, min_stop_atr=min_stop_atr)
        self.sig, self.setups = {}, []

    # -- liquidity levels known BEFORE setup bar j opens
    def _levels(self, S, j, day_stats):
        b = S.b[j]
        d, sec = divmod(b["t"], DAY)
        lows, highs = [], []
        prev = day_stats.prev_day(d)
        if prev:
            highs.append(("pdh", prev[0])); lows.append(("pdl", prev[1]))
        a = day_stats.asian.get(d)
        if a and sec >= 7 * 3600:
            highs.append(("ash", a[0])); lows.append(("asl", a[1]))
        l = day_stats.london.get(d)
        if l and sec >= 13 * 3600:
            highs.append(("lsh", l[0])); lows.append(("lsl", l[1]))
        sh, sl = S.sw.known("hi", j - 1, 4), S.sw.known("lo", j - 1, 4)
        tol = 0.15 * S.atr[j - 1]
        for x in sh[-3:]:
            highs.append(("swing", x[2]))
        for x in sl[-3:]:
            lows.append(("swing", x[2]))
        for arr, out in ((sh, highs), (sl, lows)):
            for a1 in range(len(arr)):
                for a2 in range(a1 + 1, len(arr)):
                    if abs(arr[a1][2] - arr[a2][2]) <= tol:
                        out.append(("equal", max(arr[a1][2], arr[a2][2]) if out is highs else min(arr[a1][2], arr[a2][2])))
        return highs, lows

    def prepare(self, bars):
        p = self.p
        self.sig, self.setups = {}, []
        S = Frame(bars, p["setup_min"], p["swing_n"])
        T = Frame(bars, p["trend_min"], 2, extras=True)
        stats = DayStats(S.b, S.step)
        times = [x["t"] for x in bars]
        seen = set()
        for j in range(30, len(S.b)):
            b = S.b[j]
            highs, lows = self._levels(S, j, stats)
            for side in ("long", "short"):
                lv = None
                if side == "long":
                    cands = [(n, L) for n, L in lows if b["l"] < L and b["c"] > L]
                    lv = max(cands, key=lambda x: x[1]) if cands else None       # the level swept deepest-from-above
                else:
                    cands = [(n, L) for n, L in highs if b["h"] > L and b["c"] < L]
                    lv = min(cands, key=lambda x: x[1]) if cands else None
                if not lv:
                    continue
                self._follow(S, T, stats, j, side, lv, seen)
        self.setups.sort(key=lambda s: s["avail"])
        for s in self.setups:
            self._scan_entries(bars, times, s)
        return self

    def _follow(self, S, T, stats, j, side, lv, seen):
        """After a sweep at bar j, wait (causally) for the MSS and an FVG; emit the setup the moment both exist."""
        p, long_ = self.p, side == "long"
        b = S.b[j]
        ext = b["l"] if long_ else b["h"]
        pool = S.sw.known("hi" if long_ else "lo", j, 6)
        mss = None
        for x in reversed(pool):                                   # latest confirmed swing on the far side
            if (x[2] > b["c"]) if long_ else (x[2] < b["c"]):
                mss = x[2]
                break
        if mss is None:
            return
        mss_at, fvgs = None, []
        for m in range(j + 1, min(j + p["mss_window"] + 2, len(S.b))):
            c = S.b[m]
            if (c["l"] < ext) if long_ else (c["h"] > ext):        # sweep extreme broken again: setup dead
                return
            if mss_at is None and ((c["c"] > mss) if long_ else (c["c"] < mss)):
                mss_at = m
            if m >= j + 2:
                c1, c3 = S.b[m - 2], c
                if long_ and c1["h"] < c3["l"] and c3["l"] - c1["h"] >= p["fvg_atr"] * S.atr[m - 1]:
                    fvgs.append((c1["h"], c3["l"]))
                if (not long_) and c1["l"] > c3["h"] and c1["l"] - c3["h"] >= p["fvg_atr"] * S.atr[m - 1]:
                    fvgs.append((c3["h"], c1["l"]))
            if mss_at is not None and fvgs:
                bot, top = fvgs[-1]
                avail = S.close_t[m]
                key = (side, round(bot, 6), round(top, 6))
                if key in seen:
                    return
                seen.add(key)
                k = T.done(avail)
                if k >= 0:
                    d = trend_dir(T, k, p["trend_mode"])
                    if d is not None and side_ok(d, side):
                        self.setups.append({"side": side, "avail": avail, "expire": avail + p["valid_bars"] * S.step,
                                            "bot": bot, "top": top, "ext": ext, "atr": S.atr[m], "level": lv[0]})
                return

    def _scan_entries(self, bars, times, s):
        p = self.p
        i0 = bisect.bisect_left(times, s["avail"])
        i1 = bisect.bisect_left(times, s["expire"])
        long_ = s["side"] == "long"
        buf = p["buf_atr"] * s["atr"]
        stop = s["ext"] - buf if long_ else s["ext"] + buf
        touched = False
        for i in range(i0, min(i1, len(bars) - 1)):
            e = bars[i]
            if (e["l"] <= stop) if long_ else (e["h"] >= stop):
                return
            touched = touched or ((e["l"] <= s["top"]) if long_ else (e["h"] >= s["bot"]))
            if not touched or not self._hours_ok(e):
                continue
            if long_ and e["c"] > e["o"] and e["c"] > s["bot"]:
                self._emit(i, {"side": "long", "sl_price": stop, "max_sl_pips": p["max_stop_atr"] * s["atr"] / self.pip, "min_sl_pips": p["min_stop_atr"] * s["atr"] / self.pip,
                               "setup": "sweep-" + s["level"]})
                return
            if (not long_) and e["c"] < e["o"] and e["c"] < s["top"]:
                self._emit(i, {"side": "short", "sl_price": stop, "max_sl_pips": p["max_stop_atr"] * s["atr"] / self.pip, "min_sl_pips": p["min_stop_atr"] * s["atr"] / self.pip,
                               "setup": "sweep-" + s["level"]})
                return


# ------------------------------------------------------------------ Strategy B

class StrategyB(_Base):
    """Trend continuation: EMA50/200 + HH/HL on the trend frame, a displacement candle on the setup frame,
    pullback into its FVG (or its origin), bullish/bearish confirmation, stop beyond the pullback swing."""

    def __init__(self, tf_e, setup_min, trend_min, disp_atr=1.5, buf_atr=0.10, rr=2.0, rsi_filter=False,
                 adx_min=0.0, atr_filter=False, valid_bars=16, max_stop_atr=4.0, min_stop_atr=0.3, pip=0.0001):
        self.tf_e = tf_e * 60
        self.pip = pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, disp_atr=disp_atr, buf_atr=buf_atr, rr=rr,
                      rsi_filter=rsi_filter, adx_min=adx_min, atr_filter=atr_filter, valid_bars=valid_bars,
                      max_stop_atr=max_stop_atr, min_stop_atr=min_stop_atr)
        self.sig = {}

    def prepare(self, bars):
        p = self.p
        self.sig = {}
        S = Frame(bars, p["setup_min"], 2)
        T = Frame(bars, p["trend_min"], 2, extras=True)
        times = [x["t"] for x in bars]
        atr_med = None
        for d in range(60, len(S.b) - 2):
            b = S.b[d]
            rng = b["h"] - b["l"]
            if rng <= 0 or rng < p["disp_atr"] * S.atr[d - 1] or abs(b["c"] - b["o"]) < 0.6 * rng:
                continue
            side = "long" if b["c"] > b["o"] else "short"
            long_ = side == "long"
            nxt = S.b[d + 1]
            prev = S.b[d - 1]
            if long_ and prev["h"] < nxt["l"]:
                bot, top, avail, kind = prev["h"], nxt["l"], S.close_t[d + 1], "fvg"
            elif (not long_) and prev["l"] > nxt["h"]:
                bot, top, avail, kind = nxt["h"], prev["l"], S.close_t[d + 1], "fvg"
            else:
                bot, top = (b["l"], b["o"]) if long_ else (b["o"], b["h"])         # origin of the displacement
                avail, kind = S.close_t[d], "origin"
            k = T.done(avail)
            if k < 220 or not (T.ema200[k] and T.ema50[k]):
                continue
            hs, ls = T.sw.known("hi", k, 2), T.sw.known("lo", k, 2)
            if len(hs) < 2 or len(ls) < 2:
                continue
            up = T.ema50[k] > T.ema200[k] and hs[1][2] > hs[0][2] and ls[1][2] > ls[0][2]
            dn = T.ema50[k] < T.ema200[k] and hs[1][2] < hs[0][2] and ls[1][2] < ls[0][2]
            if (long_ and not up) or ((not long_) and not dn):
                continue
            if p["rsi_filter"] and ((long_ and T.rsi[k] <= 50) or ((not long_) and T.rsi[k] >= 50)):
                continue
            if T.adx[k] < p["adx_min"]:
                continue
            if p["atr_filter"]:
                w = sorted(S.atr[max(0, d - 100):d])
                if S.atr[d] < w[len(w) // 2]:
                    continue
            self._entries(bars, times, S, d, side, bot, top, avail, kind)
        return self

    def _entries(self, bars, times, S, d, side, bot, top, avail, kind):
        p = self.p
        i0 = bisect.bisect_left(times, avail)
        i1 = bisect.bisect_left(times, avail + p["valid_bars"] * S.step)
        long_ = side == "long"
        atr = S.atr[d]
        extreme = None                                           # pullback swing: lowest low (long) since the setup
        touched = False
        for i in range(i0, min(i1, len(bars) - 1)):
            e = bars[i]
            extreme = (e["l"] if extreme is None else min(extreme, e["l"])) if long_ else (e["h"] if extreme is None else max(extreme, e["h"]))
            if (e["c"] < bot) if long_ else (e["c"] > top):      # closed through the whole zone: not a pullback any more
                return
            touched = touched or ((e["l"] <= top) if long_ else (e["h"] >= bot))
            if not touched or not self._hours_ok(e):
                continue
            if (long_ and e["c"] > e["o"]) or ((not long_) and e["c"] < e["o"]):
                stop = extreme - p["buf_atr"] * atr if long_ else extreme + p["buf_atr"] * atr
                self._emit(i, {"side": side, "sl_price": stop, "max_sl_pips": p["max_stop_atr"] * atr / self.pip, "min_sl_pips": p["min_stop_atr"] * atr / self.pip,
                               "setup": "trend-" + kind})
                return


# ------------------------------------------------------------------ Strategy C

SESSIONS = {                              # name: (time zone, local open hour, minute, window hours)
    "london": ("Europe/London", 8, 0, 4.0),
    "overlap": ("America/New_York", 8, 0, 3.5),
    "ny": ("America/New_York", 9, 30, 4.0),
}


def session_open_utc(day, name):
    """UTC timestamp of the session's local opening time on UTC date `day` (days since 1970). DST handled."""
    zone, h, m, _ = SESSIONS[name]
    d = datetime.fromtimestamp(day * DAY, timezone.utc)
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ZoneInfo(zone)).timestamp()


class StrategyC(_Base):
    """Session range breakout, then the retest: a candle closes beyond the opening range, price comes back to
    the broken edge, the edge holds, a confirmation candle closes, enter on the next open."""

    def __init__(self, tf_e, setup_min, trend_min, session="london", range_min=30, buf_atr=0.10, rr=2.0,
                 trend_mode="none", min_break_atr=0.0, max_break_atr=2.0, max_stop_atr=4.0, min_stop_atr=0.3, pip=0.0001):
        self.tf_e = tf_e * 60
        self.pip = pip
        self.p = dict(setup_min=setup_min, trend_min=trend_min, session=session, range_min=range_min,
                      buf_atr=buf_atr, rr=rr, trend_mode=trend_mode, min_break_atr=min_break_atr,
                      max_break_atr=max_break_atr, max_stop_atr=max_stop_atr, min_stop_atr=min_stop_atr)
        self.sig = {}

    def prepare(self, bars):
        p = self.p
        self.sig = {}
        S = Frame(bars, p["setup_min"], 2)
        T = Frame(bars, p["trend_min"], 2, extras=True)
        times = [x["t"] for x in bars]
        need = 0.8 * p["range_min"] * 60 / self.tf_e
        for day in range(times[0] // DAY, times[-1] // DAY + 1):
            if datetime.fromtimestamp(day * DAY, timezone.utc).weekday() >= 5:
                continue
            o = session_open_utc(day, p["session"])
            r_end = o + p["range_min"] * 60
            w_end = o + SESSIONS[p["session"]][3] * 3600
            a, b_, c_ = bisect.bisect_left(times, o), bisect.bisect_left(times, r_end), bisect.bisect_left(times, w_end)
            if b_ - a < need or b_ >= len(bars):
                continue
            hi, lo = max(x["h"] for x in bars[a:b_]), min(x["l"] for x in bars[a:b_])
            self._day(bars, S, T, hi, lo, b_, min(c_, len(bars) - 1))
        return self

    def _day(self, bars, S, T, hi, lo, i0, i1):
        p = self.p
        state, brk = None, None                                # state: 'long' / 'short' once a breakout closed
        for i in range(i0, i1):
            e = bars[i]
            tclose = e["t"] + self.tf_e
            sk = S.done(tclose)
            if sk < 30:
                continue
            atr = S.atr[sk]
            if state is None:
                if e["c"] > hi + p["min_break_atr"] * atr and e["c"] - hi <= p["max_break_atr"] * atr:
                    state, brk = "long", i
                elif e["c"] < lo - p["min_break_atr"] * atr and lo - e["c"] <= p["max_break_atr"] * atr:
                    state, brk = "short", i
                elif e["c"] > hi + p["max_break_atr"] * atr or e["c"] < lo - p["max_break_atr"] * atr:
                    return                                     # over-extended breakout: nothing to retest
                continue
            long_ = state == "long"
            edge = hi if long_ else lo
            if (e["c"] < edge) if long_ else (e["c"] > edge):  # fell back into the range: breakout failed
                return
            if i == brk or not self._hours_ok(e):
                continue
            hit = (e["l"] <= edge) if long_ else (e["h"] >= edge)
            if hit and ((long_ and e["c"] > e["o"]) or ((not long_) and e["c"] < e["o"])):
                k = T.done(tclose)
                d = trend_dir(T, k, p["trend_mode"]) if k >= 0 else None
                if d is None or not side_ok(d, state):
                    return
                ext = min(x["l"] for x in bars[brk:i + 1]) if long_ else max(x["h"] for x in bars[brk:i + 1])
                stop = ext - p["buf_atr"] * atr if long_ else ext + p["buf_atr"] * atr
                self._emit(i, {"side": state, "sl_price": stop, "max_sl_pips": p["max_stop_atr"] * atr / self.pip, "min_sl_pips": p["min_stop_atr"] * atr / self.pip,
                               "setup": "break-retest-" + p["session"]})
                return


# ------------------------------------------------------------------ day statistics for the liquidity levels

class DayStats:
    def __init__(self, sbars, step):
        self.hl, self.asian, self.london = {}, {}, {}
        need_a, need_l = 0.8 * 7 * 3600 / step, 0.8 * 6 * 3600 / step
        for b in sbars:
            d, sec = divmod(b["t"], DAY)
            r = self.hl.setdefault(d, [b["h"], b["l"]])
            r[0], r[1] = max(r[0], b["h"]), min(r[1], b["l"])
            if sec < 7 * 3600:
                a = self.asian.setdefault(d, [b["h"], b["l"], 0])
                a[0], a[1], a[2] = max(a[0], b["h"]), min(a[1], b["l"]), a[2] + 1
            elif 7 * 3600 <= sec < 13 * 3600:
                a = self.london.setdefault(d, [b["h"], b["l"], 0])
                a[0], a[1], a[2] = max(a[0], b["h"]), min(a[1], b["l"]), a[2] + 1
        self.asian = {d: v for d, v in self.asian.items() if v[2] >= need_a}
        self.london = {d: v for d, v in self.london.items() if v[2] >= need_l}
        self.days = sorted(self.hl)

    def prev_day(self, d):
        k = bisect.bisect_left(self.days, d)
        return tuple(self.hl[self.days[k - 1]]) if k > 0 else None


# ------------------------------------------------------------------ account simulator and statistics

def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / float(n)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def account_sim(trades, start=10000.0, risk_pct=0.5, daily_loss_pct=1.0, max_open=2):
    """Apply the account rules to trades given in R multiples. Fractional sizing (ideal lots).
    Rules: risk_pct of current equity per trade; no new trade once the day's closed loss reaches daily_loss_pct
    of the day's starting equity; at most max_open trades open at once. Returns (accepted trades, skipped)."""
    tr = sorted(trades, key=lambda t: t["t_entry"])
    eq, taken, open_, skipped = start, [], [], 0
    closed_pnl, day_start = {}, {}
    pending = []                                           # accepted, not yet closed: (t_exit, pnl)
    for t in tr:
        for x in sorted([x for x in pending if x[0] <= t["t_entry"]]):
            eq += x[1]
            dd = x[0] // DAY
            closed_pnl[dd] = closed_pnl.get(dd, 0.0) + x[1]
        pending = [x for x in pending if x[0] > t["t_entry"]]
        d = t["t_entry"] // DAY
        day_start.setdefault(d, eq)
        if len(pending) >= max_open or closed_pnl.get(d, 0.0) <= -daily_loss_pct / 100.0 * day_start[d]:
            skipped += 1
            continue
        pnl = t["r"] * risk_pct / 100.0 * eq
        pending.append((t["t_exit"], pnl))
        taken.append(dict(t, pnl_acct=pnl, equity_at_entry=eq))
    for x in sorted(pending):
        eq += x[1]
    # equity curve by exit time
    taken.sort(key=lambda t: t["t_exit"])
    e = start
    for t in taken:
        e += t["pnl_acct"]
        t["equity_after"] = e
    return taken, skipped


def metrics(taken, start=10000.0, days_span=None):
    n = len(taken)
    if n == 0:
        return {"trades": 0}
    wins = [t for t in taken if t["pnl_acct"] > 0]
    losses = [t for t in taken if t["pnl_acct"] <= 0]
    gp, gl = sum(t["pnl_acct"] for t in wins), -sum(t["pnl_acct"] for t in losses)
    net = sum(t["pnl_acct"] for t in taken)
    peak, dd_max, dd_abs, e = start, 0.0, 0.0, start
    for t in taken:
        e = t["equity_after"]
        peak = max(peak, e)
        dd_max = max(dd_max, (peak - e) / peak)
        dd_abs = max(dd_abs, peak - e)
    cw = cl = bw = bl = 0
    for t in taken:
        if t["pnl_acct"] > 0:
            cw, cl = cw + 1, 0
        else:
            cl, cw = cl + 1, 0
        bw, bl = max(bw, cw), max(bl, cl)
    lo, hi = wilson(len(wins), n)
    span = days_span or max((taken[-1]["t_exit"] - taken[0]["t_entry"]) / DAY, 1.0)
    daily = {}
    for t in taken:
        daily[t["t_exit"] // DAY] = daily.get(t["t_exit"] // DAY, 0.0) + t["pnl_acct"] / t["equity_at_entry"]
    nd = max(int(span * 5 / 7), len(daily), 2)
    vals = list(daily.values()) + [0.0] * (nd - len(daily))
    mu = sum(vals) / nd
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (nd - 1)) if nd > 1 else 0.0
    sessions = {"asian": [], "london": [], "ny": []}
    for t in taken:
        h = (t["t_entry"] % DAY) // 3600
        sessions["london" if 7 <= h < 12 else "ny" if 12 <= h < 21 else "asian"].append(t["r"])
    side = lambda s: [t["r"] for t in taken if t["side"] == s]
    avg = lambda v: sum(v) / len(v) if v else None
    return {"trades": n, "wins": len(wins), "losses": len(losses), "win_rate": len(wins) / n, "ci_lo": lo, "ci_hi": hi,
            "avg_win_r": avg([t["r"] for t in wins]), "avg_loss_r": avg([t["r"] for t in losses]),
            "avg_win_usd": gp / len(wins) if wins else 0.0, "avg_loss_usd": -gl / len(losses) if losses else 0.0,
            "net": net, "return_pct": net / start * 100, "profit_factor": gp / gl if gl else float("inf"),
            "max_dd_pct": dd_max * 100, "max_dd_usd": dd_abs, "max_wins": bw, "max_losses": bl,
            "expectancy_r": sum(t["r"] for t in taken) / n, "expectancy_usd": net / n,
            "avg_hold_min": sum(t["t_exit"] - t["t_entry"] for t in taken) / n / 60.0,
            "trades_per_day": n / (span * 5 / 7), "trades_per_week": n / (span / 7.0),
            "sharpe": mu / sd * math.sqrt(252) if sd else 0.0,
            "recovery": net / dd_abs if dd_abs else float("inf"),
            "long_r": avg(side("long")), "short_r": avg(side("short")),
            "long_n": len(side("long")), "short_n": len(side("short")),
            "session_r": {k: (avg(v), len(v)) for k, v in sessions.items()}}


def split_by_time(trades, t0, t1, fractions=(0.6, 0.2, 0.2)):
    """Chronological train / validation / out-of-sample split by entry time. Never shuffled."""
    c1 = t0 + (t1 - t0) * fractions[0]
    c2 = t0 + (t1 - t0) * (fractions[0] + fractions[1])
    return ([t for t in trades if t["t_entry"] < c1], [t for t in trades if c1 <= t["t_entry"] < c2],
            [t for t in trades if t["t_entry"] >= c2])
