from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures.atr import compute_atr, compute_ema, compute_vwap


@dataclass(frozen=True)
class TrendSnapshot:
    trend: str
    ema_fast: float | None
    ema_slow: float | None
    close: float | None


def trend_snapshot(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> TrendSnapshot:
    if df is None or len(df) < slow:
        return TrendSnapshot("neutral", None, None, None)
    close = df["Close"]
    ema_fast = float(compute_ema(close, fast).iloc[-1])
    ema_slow = float(compute_ema(close, slow).iloc[-1])
    last_close = float(close.iloc[-1])
    if last_close > ema_fast > ema_slow:
        trend = "bullish"
    elif last_close < ema_fast < ema_slow:
        trend = "bearish"
    else:
        trend = "neutral"
    return TrendSnapshot(trend, ema_fast, ema_slow, last_close)


def rolling_range(df: pd.DataFrame, lookback: int = 20) -> float | None:
    if df is None or len(df) < lookback:
        return None
    window = df.tail(lookback)
    return float(window["High"].max() - window["Low"].min())


def session_range(df: pd.DataFrame) -> tuple[float | None, float | None]:
    if df is None or df.empty:
        return None, None
    return float(df["High"].max()), float(df["Low"].min())


def atr_value(df: pd.DataFrame, period: int = 14) -> float | None:
    return compute_atr(df, period=period)


def vwap_value(df: pd.DataFrame) -> float | None:
    return compute_vwap(df)
