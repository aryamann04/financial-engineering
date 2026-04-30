from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return (
        df.resample(rule)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Open", "High", "Low", "Close"])
    )


def recent_window(df: pd.DataFrame, bars: int | None = None) -> pd.DataFrame:
    if df is None or df.empty or not bars:
        return df.copy() if hasattr(df, "copy") else df
    return df.tail(int(bars)).copy()
