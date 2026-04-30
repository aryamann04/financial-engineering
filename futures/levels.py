from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from config.sessions import SESSIONS, SessionWindow, filter_session_bars, ET
from futures.atr import compute_atr, compute_vwap, get_trend_alignment, resample_to_15m
from futures.data import FuturesData


@dataclass
class SessionLevels:
    name: str
    high: float | None
    low: float | None
    bars: int

    def mid(self) -> float | None:
        if self.high is not None and self.low is not None:
            return (self.high + self.low) / 2.0
        return None

    def range(self) -> float | None:
        if self.high is not None and self.low is not None:
            return self.high - self.low
        return None


@dataclass
class FuturesLevels:
    symbol: str
    display_name: str
    current_price: float | None
    prev_day_high: float | None
    prev_day_low: float | None
    prev_day_close: float | None
    today_high: float | None
    today_low: float | None
    asia: SessionLevels
    london: SessionLevels
    new_york: SessionLevels
    ny_open_range: SessionLevels
    vwap: float | None
    atr_5m: float | None
    atr_15m: float | None
    trend_5m: str
    trend_15m: str
    trend_align: str
    context_lines: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _session_hl(df5: pd.DataFrame, session_key: str, ref_date: date) -> SessionLevels:
    sess = SESSIONS[session_key]
    bars = filter_session_bars(df5, sess, ref_date)
    if bars.empty:
        return SessionLevels(sess.name, None, None, 0)
    return SessionLevels(
        sess.name,
        float(bars["High"].max()),
        float(bars["Low"].min()),
        len(bars),
    )


def _build_context(
    spot: float | None,
    levels: "FuturesLevels",
) -> list[str]:
    lines: list[str] = []
    atr = levels.atr_5m
    atr15 = levels.atr_15m

    # Trend alignment
    trend_labels = {"bullish": "Bullish", "bearish": "Bearish", "mixed": "Mixed", "neutral": "Neutral"}
    lines.append(
        f"5m/15m trend: {trend_labels.get(levels.trend_5m, 'N/A')} / "
        f"{trend_labels.get(levels.trend_15m, 'N/A')} — "
        f"alignment: {trend_labels.get(levels.trend_align, 'N/A')}"
    )

    if spot is None:
        return lines

    # Proximity to key levels (within 0.5x 5m ATR)
    named = {
        "Prev Day High": levels.prev_day_high,
        "Prev Day Low":  levels.prev_day_low,
        "Asia High":     levels.asia.high,
        "Asia Low":      levels.asia.low,
        "London High":   levels.london.high,
        "London Low":    levels.london.low,
        "VWAP":          levels.vwap,
        "NY Open Range High": levels.ny_open_range.high,
        "NY Open Range Low":  levels.ny_open_range.low,
    }
    if atr and atr > 0:
        threshold = 0.5 * atr
        for name, level in named.items():
            if level is None:
                continue
            diff = abs(spot - level)
            if diff <= threshold:
                side = "above" if spot >= level else "below"
                lines.append(
                    f"Price is {side} {name} by {diff:.5g} ({diff / atr:.2f}x 5m ATR) — near key level"
                )

    # Extension from breakout levels
    if atr15 and atr15 > 0:
        breakouts = {
            "London High": levels.london.high,
            "London Low":  levels.london.low,
            "Asia High":   levels.asia.high,
            "Asia Low":    levels.asia.low,
            "Prev Day High": levels.prev_day_high,
            "Prev Day Low":  levels.prev_day_low,
        }
        for name, lvl in breakouts.items():
            if lvl is None:
                continue
            ext = (spot - lvl) / atr15
            if abs(ext) > 1.5:
                side = "above" if ext > 0 else "below"
                msg = (
                    "extended — prefer pullback or fade setup"
                    if abs(ext) > 2.0
                    else "watch for pullback or retest"
                )
                lines.append(
                    f"Price is {abs(ext):.1f}x 15m ATR {side} {name} — {msg}"
                )
                break

    # VWAP relationship
    if levels.vwap:
        if spot > levels.vwap:
            lines.append(f"Price is above VWAP ({levels.vwap:.5g}) — short-term bullish bias")
        else:
            lines.append(f"Price is below VWAP ({levels.vwap:.5g}) — short-term bearish bias")

    # Suggested mode
    if levels.trend_align == "bullish":
        lines.append("Suggested mode: Continuation longs / wait for pullback to support")
    elif levels.trend_align == "bearish":
        lines.append("Suggested mode: Continuation shorts / wait for pullback to resistance")
    elif levels.trend_align == "mixed":
        lines.append("Suggested mode: Caution — trends not aligned; fade extremes only or wait")
    else:
        lines.append("Suggested mode: Neutral — insufficient data for directional bias")

    return lines


def compute_levels(data: FuturesData, open_range_minutes: int = 30) -> FuturesLevels:
    errors = list(data.errors)

    # --- Previous / today from daily bars ---
    prev_high = prev_low = prev_close = None
    today_high = today_low = None

    if not data.daily_df.empty:
        daily = data.daily_df
        if len(daily) >= 2:
            pb = daily.iloc[-2]
            prev_high  = float(pb["High"])
            prev_low   = float(pb["Low"])
            prev_close = float(pb["Close"])
        tb = daily.iloc[-1]
        today_high = float(tb["High"])
        today_low  = float(tb["Low"])

    # --- Intraday session levels ---
    asia   = SessionLevels("Asia", None, None, 0)
    london = SessionLevels("London", None, None, 0)
    ny     = SessionLevels("New York", None, None, 0)
    ny_or  = SessionLevels("NY Open Range", None, None, 0)
    vwap = atr_5m = atr_15m = None
    trend_5m = trend_15m = trend_align = "neutral"
    ref_date: date | None = None

    df5 = data.intraday_5m

    if not df5.empty:
        # Ensure ET index for all session math
        idx = df5.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx_et = idx.tz_convert(ET)
        df5_et = df5.copy()
        df5_et.index = idx_et
        ref_date = idx_et[-1].date()

        asia   = _session_hl(df5_et, "asia",     ref_date)
        london = _session_hl(df5_et, "london",   ref_date)
        ny     = _session_hl(df5_et, "new_york", ref_date)

        # Custom NY open-range window
        ny_sess = SESSIONS["new_york"]
        or_end_total_min = ny_sess.start_hour * 60 + ny_sess.start_minute + open_range_minutes
        or_end_h, or_end_m = divmod(or_end_total_min, 60)
        or_window = SessionWindow(
            "NY Open Range",
            ny_sess.start_hour, ny_sess.start_minute,
            or_end_h, or_end_m,
        )
        or_bars = filter_session_bars(df5_et, or_window, ref_date)
        if not or_bars.empty:
            ny_or = SessionLevels(
                "NY Open Range",
                float(or_bars["High"].max()),
                float(or_bars["Low"].min()),
                len(or_bars),
            )

        # VWAP from CME session start (prev calendar day 18:00 ET)
        yesterday = ref_date - timedelta(days=1)
        dates_et = pd.Series(idx_et.date, index=df5_et.index)
        cme_mask = (
            ((dates_et == yesterday) & (idx_et.hour >= 18)) |
            (dates_et == ref_date)
        )
        cme_bars = df5_et[cme_mask.values]
        vwap = compute_vwap(cme_bars)

        # ATR
        atr_5m  = compute_atr(df5_et, period=14)
        df15    = resample_to_15m(df5_et)
        atr_15m = compute_atr(df15, period=14)

        trend_5m, trend_15m, trend_align = get_trend_alignment(df5_et, df15)
    else:
        errors.append("No intraday data — session levels and ATR unavailable.")

    levels = FuturesLevels(
        symbol=data.yf_symbol,
        display_name=data.display_name,
        current_price=data.spot,
        prev_day_high=prev_high,
        prev_day_low=prev_low,
        prev_day_close=prev_close,
        today_high=today_high,
        today_low=today_low,
        asia=asia,
        london=london,
        new_york=ny,
        ny_open_range=ny_or,
        vwap=vwap,
        atr_5m=atr_5m,
        atr_15m=atr_15m,
        trend_5m=trend_5m,
        trend_15m=trend_15m,
        trend_align=trend_align,
        errors=errors,
    )
    levels.context_lines = _build_context(data.spot, levels)
    return levels


def format_levels_display(lvl: FuturesLevels) -> str:
    """Render a levels snapshot as a clean terminal string."""
    NA = "  N/A"

    def fmt(val: float | None, decimals: int = 4) -> str:
        if val is None:
            return NA
        return f"{val:>{12}.{decimals}f}"

    def fmt_atr_dist(price: float | None, atr: float | None, mult: float, direction: int) -> str:
        if price is None or atr is None:
            return NA
        return f"{price + direction * mult * atr:>{12}.4f}  ({mult}x ATR)"

    lines = []
    sep = "─" * 60

    lines.append(sep)
    lines.append(f"  {lvl.display_name}  ({lvl.symbol})")
    lines.append(sep)

    price_str = f"{lvl.current_price:.5g}" if lvl.current_price is not None else "N/A"
    lines.append(f"  Current Price   : {price_str}")
    atr5_str  = f"{lvl.atr_5m:.5g}"  if lvl.atr_5m  is not None else "N/A"
    atr15_str = f"{lvl.atr_15m:.5g}" if lvl.atr_15m is not None else "N/A"
    lines.append(f"  5m  ATR         : {atr5_str}")
    lines.append(f"  15m ATR         : {atr15_str}")
    lines.append("")

    lines.append("  KEY LEVELS")
    lines.append("  " + "─" * 42)
    rows = [
        ("Previous Day High",   lvl.prev_day_high),
        ("Previous Day Low",    lvl.prev_day_low),
        ("Previous Day Close",  lvl.prev_day_close),
        ("Today High",          lvl.today_high),
        ("Today Low",           lvl.today_low),
        ("Asia High",           lvl.asia.high),
        ("Asia Low",            lvl.asia.low),
        ("London High",         lvl.london.high),
        ("London Low",          lvl.london.low),
        ("NY Open Range High",  lvl.ny_open_range.high),
        ("NY Open Range Low",   lvl.ny_open_range.low),
        ("NY Session High",     lvl.new_york.high),
        ("NY Session Low",      lvl.new_york.low),
        ("VWAP",                lvl.vwap),
    ]
    for name, val in rows:
        val_str = f"{val:.5g}" if val is not None else "N/A"
        lines.append(f"  {name:<26}: {val_str}")

    lines.append("")

    if lvl.current_price is not None and (lvl.atr_5m or lvl.atr_15m):
        ref_atr = lvl.atr_15m or lvl.atr_5m
        atr_label = "15m ATR" if lvl.atr_15m else "5m ATR"
        price = lvl.current_price
        lines.append(f"  ATR TARGETS FROM CURRENT PRICE  (using {atr_label} = {ref_atr:.5g})")
        lines.append("  " + "─" * 42)
        for label, mult, sign in [
            ("Long  1x", 1, +1), ("Long  2x", 2, +1), ("Long  3x", 3, +1),
            ("Short 1x", 1, -1), ("Short 2x", 2, -1), ("Short 3x", 3, -1),
        ]:
            tgt = price + sign * mult * ref_atr
            lines.append(f"  {label} ATR target  : {tgt:.5g}")
        lines.append("")

    lines.append("  CONTEXT")
    lines.append("  " + "─" * 42)
    for cl in lvl.context_lines:
        lines.append(f"  - {cl}")

    if lvl.errors:
        lines.append("")
        lines.append("  WARNINGS")
        lines.append("  " + "─" * 42)
        for e in lvl.errors:
            lines.append(f"  ! {e}")

    lines.append(sep)
    return "\n".join(lines)
