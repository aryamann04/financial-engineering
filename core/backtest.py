from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.indicators import atr_value, trend_snapshot


@dataclass(frozen=True)
class BacktestSummary:
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_r: float
    source: str


def run_bias_backtest(df: pd.DataFrame, timeframe: str = "5m", hold_bars: int = 12) -> BacktestSummary | None:
    if df is None or len(df) < 80:
        return None

    atr = atr_value(df)
    if atr is None:
        return None

    trades = wins = losses = 0
    r_values: list[float] = []

    for idx in range(30, len(df) - hold_bars):
        window = df.iloc[: idx + 1]
        trend = trend_snapshot(window)
        if trend.trend == "neutral":
            continue
        entry = float(df["Close"].iloc[idx])
        stop = entry - atr if trend.trend == "bullish" else entry + atr
        target = entry + (2 * atr) if trend.trend == "bullish" else entry - (2 * atr)
        future = df.iloc[idx + 1 : idx + hold_bars + 1]
        trades += 1

        hit_target = False
        hit_stop = False
        for _, row in future.iterrows():
            if trend.trend == "bullish":
                hit_target = hit_target or float(row["High"]) >= target
                hit_stop = hit_stop or float(row["Low"]) <= stop
            else:
                hit_target = hit_target or float(row["Low"]) <= target
                hit_stop = hit_stop or float(row["High"]) >= stop
            if hit_target or hit_stop:
                break

        if hit_target and not hit_stop:
            wins += 1
            r_values.append(2.0)
        else:
            losses += 1
            r_values.append(-1.0)

    if not trades:
        return None

    return BacktestSummary(
        trades=trades,
        wins=wins,
        losses=losses,
        win_rate=(wins / trades) * 100.0,
        avg_r=sum(r_values) / len(r_values),
        source=f"based on heuristic {timeframe} bias follow-through test",
    )
