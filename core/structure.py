from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    timestamp: object
    price: float
    kind: str


@dataclass(frozen=True)
class StructureEvent:
    event: str
    direction: str
    level: float
    timestamp: object
    detail: str


@dataclass
class StructureState:
    regime: str
    bias: str
    swings: list[SwingPoint] = field(default_factory=list)
    events: list[StructureEvent] = field(default_factory=list)
    summary: str = ""


def detect_swings(df: pd.DataFrame, lookback: int = 2) -> list[SwingPoint]:
    if df is None or len(df) < lookback * 2 + 1:
        return []

    highs = df["High"].reset_index(drop=True)
    lows = df["Low"].reset_index(drop=True)
    swings: list[SwingPoint] = []

    for idx in range(lookback, len(df) - lookback):
        high = float(highs.iloc[idx])
        low = float(lows.iloc[idx])
        left_highs = highs.iloc[idx - lookback:idx]
        right_highs = highs.iloc[idx + 1:idx + lookback + 1]
        left_lows = lows.iloc[idx - lookback:idx]
        right_lows = lows.iloc[idx + 1:idx + lookback + 1]

        if high >= float(left_highs.max()) and high >= float(right_highs.max()):
            swings.append(SwingPoint(df.index[idx], high, "high"))
        if low <= float(left_lows.min()) and low <= float(right_lows.min()):
            swings.append(SwingPoint(df.index[idx], low, "low"))

    swings.sort(key=lambda swing: swing.timestamp)
    return swings


def detect_structure(df: pd.DataFrame, lookback: int = 2) -> StructureState:
    swings = detect_swings(df, lookback=lookback)
    if len(swings) < 4 or df is None or df.empty:
        return StructureState("range", "neutral", swings=swings, summary="Insufficient swing history.")

    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    events: list[StructureEvent] = []
    last_close = float(df["Close"].iloc[-1])

    if len(highs) >= 2:
        prev_high, last_high = highs[-2], highs[-1]
        if last_close > prev_high.price:
            events.append(StructureEvent("BOS", "bullish", prev_high.price, df.index[-1], "Price closed above the prior swing high."))
    if len(lows) >= 2:
        prev_low, last_low = lows[-2], lows[-1]
        if last_close < prev_low.price:
            events.append(StructureEvent("BOS", "bearish", prev_low.price, df.index[-1], "Price closed below the prior swing low."))

    higher_highs = len(highs) >= 2 and highs[-1].price > highs[-2].price
    higher_lows = len(lows) >= 2 and lows[-1].price > lows[-2].price
    lower_highs = len(highs) >= 2 and highs[-1].price < highs[-2].price
    lower_lows = len(lows) >= 2 and lows[-1].price < lows[-2].price

    if higher_highs and higher_lows:
        regime = "trend continuation"
        bias = "bullish"
    elif lower_highs and lower_lows:
        regime = "trend continuation"
        bias = "bearish"
    else:
        range_high = max(s.price for s in swings[-6:] if s.kind == "high") if any(s.kind == "high" for s in swings[-6:]) else None
        range_low = min(s.price for s in swings[-6:] if s.kind == "low") if any(s.kind == "low" for s in swings[-6:]) else None
        if range_high is not None and range_low is not None:
            if last_close > range_high:
                regime = "breakout"
                bias = "bullish"
            elif last_close < range_low:
                regime = "breakout"
                bias = "bearish"
            else:
                regime = "range"
                bias = "neutral"
        else:
            regime = "range"
            bias = "neutral"

    if events:
        last_event = events[-1]
        if bias != "neutral" and last_event.direction != bias:
            events.append(StructureEvent("CHoCH", last_event.direction, last_event.level, last_event.timestamp, "Break conflicted with the prevailing swing sequence."))
            regime = "failed breakout" if regime == "breakout" else "sweep/reversal attempt"

    summary = f"{regime.title()} with {bias} lean." if bias != "neutral" else f"{regime.title()}."
    return StructureState(regime=regime, bias=bias, swings=swings, events=events[-4:], summary=summary)
