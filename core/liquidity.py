from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from config.sessions import ET, SESSIONS, filter_session_bars
from core.structure import SwingPoint, detect_swings


@dataclass(frozen=True)
class KeyLevel:
    name: str
    price: float
    kind: str
    timeframe: str
    source: str


@dataclass(frozen=True)
class LiquidityEvent:
    event_type: str
    direction: str
    level_name: str
    level_price: float
    price: float
    timestamp: object
    description: str
    source: str


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out = df.copy()
    out.index = idx.tz_convert(ET)
    return out


def _equal_levels(swings: list[SwingPoint], tolerance: float, kind: str) -> list[KeyLevel]:
    points = [s for s in swings if s.kind == kind]
    levels: list[KeyLevel] = []
    for left, right in zip(points, points[1:]):
        if abs(left.price - right.price) <= tolerance:
            levels.append(
                KeyLevel(
                    name=f"Equal {'Highs' if kind == 'high' else 'Lows'}",
                    price=(left.price + right.price) / 2.0,
                    kind="equal_highs" if kind == "high" else "equal_lows",
                    timeframe="intraday",
                    source="swing comparison",
                )
            )
    return levels[-2:]


def detect_key_levels(daily_df: pd.DataFrame, intraday_df: pd.DataFrame, atr: float | None = None) -> list[KeyLevel]:
    levels: list[KeyLevel] = []
    tol = (atr or 0.0) * 0.15
    df = _normalize_index(intraday_df)

    if daily_df is not None and len(daily_df) >= 2:
        prev = daily_df.iloc[-2]
        levels.extend(
            [
                KeyLevel("Previous Day High", float(prev["High"]), "previous_day_high", "1d", "daily bars"),
                KeyLevel("Previous Day Low", float(prev["Low"]), "previous_day_low", "1d", "daily bars"),
            ]
        )

    if not df.empty:
        ref_date = df.index[-1].date()
        asia = filter_session_bars(df, SESSIONS["asia"], ref_date)
        new_york = filter_session_bars(df, SESSIONS["new_york"], ref_date)
        overnight = df[(df.index >= pd.Timestamp(ref_date, tz=ET) - timedelta(hours=6)) & (df.index < pd.Timestamp(ref_date, tz=ET) + timedelta(hours=9, minutes=30))]

        session_map = [
            ("Current Session High", new_york["High"].max() if not new_york.empty else None, "current_session_high", "session"),
            ("Current Session Low", new_york["Low"].min() if not new_york.empty else None, "current_session_low", "session"),
            ("Overnight High", overnight["High"].max() if not overnight.empty else None, "overnight_high", "session"),
            ("Overnight Low", overnight["Low"].min() if not overnight.empty else None, "overnight_low", "session"),
            ("Asia High", asia["High"].max() if not asia.empty else None, "asia_high", "session"),
            ("Asia Low", asia["Low"].min() if not asia.empty else None, "asia_low", "session"),
        ]
        for name, price, kind, source in session_map:
            if price is not None and pd.notna(price):
                levels.append(KeyLevel(name, float(price), kind, "intraday", source))

        swings = detect_swings(df.tail(120), lookback=2)
        for swing in swings[-8:]:
            levels.append(
                KeyLevel(
                    name=f"Swing {swing.kind.title()}",
                    price=float(swing.price),
                    kind=f"swing_{swing.kind}",
                    timeframe="intraday",
                    source="swing structure",
                )
            )
        if tol > 0:
            levels.extend(_equal_levels(swings[-10:], tol, "high"))
            levels.extend(_equal_levels(swings[-10:], tol, "low"))

    unique: dict[tuple[str, int], KeyLevel] = {}
    for level in levels:
        key = (level.name, round(level.price, 4))
        unique[key] = level
    return list(unique.values())


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    key_levels: list[KeyLevel],
    atr: float | None = None,
    lookback_bars: int = 40,
) -> list[LiquidityEvent]:
    if df is None or df.empty or not key_levels:
        return []

    window = _normalize_index(df).tail(lookback_bars)
    tol = (atr or 0.0) * 0.1
    events: list[LiquidityEvent] = []

    for ts, row in window.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        open_ = float(row["Open"])
        candle_range = max(high - low, 1e-9)
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        wick_heavy_up = upper_wick / candle_range >= 0.45
        wick_heavy_down = lower_wick / candle_range >= 0.45

        for level in key_levels:
            price = level.price
            if high > price + tol and close < price and wick_heavy_up:
                events.append(
                    LiquidityEvent(
                        "sweep",
                        "bearish",
                        level.name,
                        price,
                        close,
                        ts,
                        f"Took {level.name} then rejected back below it with an upper wick.",
                        f"based on current bars vs {level.source}",
                    )
                )
            elif low < price - tol and close > price and wick_heavy_down:
                events.append(
                    LiquidityEvent(
                        "sweep",
                        "bullish",
                        level.name,
                        price,
                        close,
                        ts,
                        f"Took {level.name} then reclaimed back above it with a lower wick.",
                        f"based on current bars vs {level.source}",
                    )
                )

    events.sort(key=lambda event: event.timestamp)
    return events[-8:]
