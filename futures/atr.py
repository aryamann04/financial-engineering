from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """
    Wilder's ATR using exponential smoothing (alpha = 1/period).
    Requires High, Low, Close columns. Returns None if insufficient bars.
    """
    if df is None or len(df) < period:
        return None
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev  = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period).mean().iloc[-1]
    return float(atr) if np.isfinite(atr) else None


def compute_vwap(df: pd.DataFrame) -> float | None:
    """
    Volume-weighted average price over all bars in df.
    Requires High, Low, Close, Volume columns.
    """
    if df is None or df.empty:
        return None
    try:
        typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = df["Volume"].clip(lower=0)
        total_vol = vol.sum()
        if total_vol == 0:
            return float(typical.iloc[-1])
        return float((typical * vol).sum() / total_vol)
    except Exception:
        return None


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def get_trend_alignment(
    df5: pd.DataFrame | None,
    df15: pd.DataFrame | None,
) -> tuple[str, str, str]:
    """
    Classify trend on 5m and 15m timeframes using EMA 9/21 and price.
    Returns (trend_5m, trend_15m, alignment).
    Each trend is 'bullish' | 'bearish' | 'neutral'.
    Alignment is 'bullish' | 'bearish' | 'mixed' | 'neutral'.
    """

    def _classify(df: pd.DataFrame | None) -> str:
        if df is None or len(df) < 21:
            return "neutral"
        close = df["Close"]
        ema9  = compute_ema(close, 9).iloc[-1]
        ema21 = compute_ema(close, 21).iloc[-1]
        price = close.iloc[-1]
        if price > ema9 > ema21:
            return "bullish"
        if price < ema9 < ema21:
            return "bearish"
        return "neutral"

    t5  = _classify(df5)
    t15 = _classify(df15)

    if t5 == t15 and t5 != "neutral":
        align = t5
    elif t5 == "neutral" and t15 == "neutral":
        align = "neutral"
    else:
        align = "mixed"

    return t5, t15, align


def resample_to_15m(df5: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-minute OHLCV bars to 15-minute bars."""
    if df5 is None or df5.empty:
        return pd.DataFrame()
    return (
        df5.resample("15min")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(subset=["Open"])
    )
