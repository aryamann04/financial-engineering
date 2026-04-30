from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import pandas as pd


@dataclass
class FairValueGap:
    """
    A Fair Value Gap (imbalance) identified from three consecutive candles.

    Bullish FVG: candle[i].high < candle[i+2].low  → gap = (c1.high, c3.low)
    Bearish FVG: candle[i].low  > candle[i+2].high → gap = (c3.high, c1.low)

    The gap represents an area where price moved so quickly that only one side
    traded — it may act as a magnet or support/resistance on retests.
    """
    timeframe: str
    direction: str       # 'bullish' | 'bearish'
    lower: float
    upper: float
    midpoint: float
    start_time: object
    end_time: object
    formed_at: object    # pd.Timestamp of the middle candle
    size: float          # upper - lower in price points
    atr_mult: float | None = None  # size / ATR at detection time
    is_filled: bool = False
    fill_pct: float = 0.0
    is_near: bool = False          # within 1x ATR of midpoint
    dist_from_price: float = 0.0
    age_bars: int = 0              # bars since formation


def _update_status(fvg: FairValueGap, current_price: float, atr: float | None) -> None:
    fvg.dist_from_price = abs(current_price - fvg.midpoint)

    if fvg.direction == "bullish":
        if current_price <= fvg.lower:
            fvg.is_filled = True
            fvg.fill_pct  = 100.0
        elif current_price < fvg.upper:
            fvg.fill_pct = (fvg.upper - current_price) / max(fvg.size, 1e-12) * 100
        else:
            fvg.fill_pct = 0.0
    else:  # bearish
        if current_price >= fvg.upper:
            fvg.is_filled = True
            fvg.fill_pct  = 100.0
        elif current_price > fvg.lower:
            fvg.fill_pct = (current_price - fvg.lower) / max(fvg.size, 1e-12) * 100
        else:
            fvg.fill_pct = 0.0

    if atr and atr > 0:
        fvg.is_near = fvg.dist_from_price <= atr
    else:
        fvg.is_near = fvg.dist_from_price <= fvg.size * 3


def detect_fvgs(
    df: pd.DataFrame,
    timeframe: str,
    current_price: float,
    atr: float | None = None,
    min_size_points: float = 0.0,
    min_size_atr: float = 0.12,
    min_gap_to_avg_range: float = 0.20,
    min_displacement_fraction: float = 0.20,
    max_fvgs: int = 12,
) -> list[FairValueGap]:
    """
    Scan df for unfilled Fair Value Gaps and return them sorted by proximity
    to current_price. Filled FVGs are excluded.
    """
    if df is None or len(df) < 3:
        return []

    fvgs: list[FairValueGap] = []
    n = len(df)

    for i in range(n - 2):
        c1 = df.iloc[i]
        c2 = df.iloc[i + 1]
        c3 = df.iloc[i + 2]

        # Bullish FVG
        gap_size = float(c3["Low"]) - float(c1["High"])
        if gap_size > min_size_points and _is_valid_bullish_fvg(
            c1, c2, c3, gap_size, atr, min_size_points, min_size_atr,
            min_gap_to_avg_range, min_displacement_fraction,
        ):
            lower = float(c1["High"])
            upper = float(c3["Low"])
            fvg = FairValueGap(
                timeframe=timeframe,
                direction="bullish",
                lower=lower, upper=upper,
                midpoint=(lower + upper) / 2.0,
                start_time=df.index[i],
                end_time=df.index[i + 2],
                formed_at=df.index[i + 1],
                size=upper - lower,
                atr_mult=(upper - lower) / atr if atr and atr > 0 else None,
                age_bars=n - i - 2,
            )
            _update_status(fvg, current_price, atr)
            if not fvg.is_filled:
                fvgs.append(fvg)

        # Bearish FVG
        gap_size = float(c1["Low"]) - float(c3["High"])
        if gap_size > min_size_points and _is_valid_bearish_fvg(
            c1, c2, c3, gap_size, atr, min_size_points, min_size_atr,
            min_gap_to_avg_range, min_displacement_fraction,
        ):
            lower = float(c3["High"])
            upper = float(c1["Low"])
            fvg = FairValueGap(
                timeframe=timeframe,
                direction="bearish",
                lower=lower, upper=upper,
                midpoint=(lower + upper) / 2.0,
                start_time=df.index[i],
                end_time=df.index[i + 2],
                formed_at=df.index[i + 1],
                size=upper - lower,
                atr_mult=(upper - lower) / atr if atr and atr > 0 else None,
                age_bars=n - i - 2,
            )
            _update_status(fvg, current_price, atr)
            if not fvg.is_filled:
                fvgs.append(fvg)

    # Sort: unfilled, nearest first; prefer more recent FVGs when tied
    fvgs.sort(key=lambda f: (f.dist_from_price, f.age_bars))
    return fvgs[:max_fvgs]


def _passes_common_filters(
    c1: pd.Series,
    c2: pd.Series,
    c3: pd.Series,
    gap_size: float,
    atr: float | None,
    min_size_points: float,
    min_size_atr: float,
    min_gap_to_avg_range: float,
) -> bool:
    if gap_size <= max(min_size_points, 0.0):
        return False

    if atr and atr > 0 and gap_size < atr * min_size_atr:
        return False

    ranges = [
        float(c1["High"]) - float(c1["Low"]),
        float(c2["High"]) - float(c2["Low"]),
        float(c3["High"]) - float(c3["Low"]),
    ]
    avg_range = mean(max(r, 1e-12) for r in ranges)
    if gap_size < avg_range * min_gap_to_avg_range:
        return False

    return True


def _is_valid_bullish_fvg(
    c1: pd.Series,
    c2: pd.Series,
    c3: pd.Series,
    gap_size: float,
    atr: float | None,
    min_size_points: float,
    min_size_atr: float,
    min_gap_to_avg_range: float,
    min_displacement_fraction: float,
) -> bool:
    if not _passes_common_filters(
        c1, c2, c3, gap_size, atr, min_size_points, min_size_atr, min_gap_to_avg_range
    ):
        return False

    c1_high = float(c1["High"])
    c2_high = float(c2["High"])
    c2_close = float(c2["Close"])
    c3_close = float(c3["Close"])
    threshold = c1_high + gap_size * min_displacement_fraction

    return (
        c2_high >= threshold and
        c2_close > c1_high and
        c3_close >= threshold
    )


def _is_valid_bearish_fvg(
    c1: pd.Series,
    c2: pd.Series,
    c3: pd.Series,
    gap_size: float,
    atr: float | None,
    min_size_points: float,
    min_size_atr: float,
    min_gap_to_avg_range: float,
    min_displacement_fraction: float,
) -> bool:
    if not _passes_common_filters(
        c1, c2, c3, gap_size, atr, min_size_points, min_size_atr, min_gap_to_avg_range
    ):
        return False

    c1_low = float(c1["Low"])
    c2_low = float(c2["Low"])
    c2_close = float(c2["Close"])
    c3_close = float(c3["Close"])
    threshold = c1_low - gap_size * min_displacement_fraction

    return (
        c2_low <= threshold and
        c2_close < c1_low and
        c3_close <= threshold
    )


def fvgs_above(fvgs: list[FairValueGap], price: float) -> list[FairValueGap]:
    return sorted(
        [f for f in fvgs if f.lower > price and not f.is_filled],
        key=lambda f: f.lower,
    )


def fvgs_below(fvgs: list[FairValueGap], price: float) -> list[FairValueGap]:
    return sorted(
        [f for f in fvgs if f.upper < price and not f.is_filled],
        key=lambda f: f.upper, reverse=True,
    )


def fvg_alignment_across_timeframes(
    fvgs_5m: list[FairValueGap],
    fvgs_15m: list[FairValueGap],
    fvgs_1h: list[FairValueGap],
    price: float,
    atr: float | None,
) -> list[str]:
    """
    Return context lines describing FVG alignment across timeframes.
    Cross-timeframe FVG clusters are higher-priority zones.
    """
    tol = atr * 0.5 if atr else 999
    lines: list[str] = []

    def _cluster(direction: str) -> None:
        all_fvgs = {
            "5m": [f for f in fvgs_5m  if f.direction == direction and not f.is_filled],
            "15m":[f for f in fvgs_15m if f.direction == direction and not f.is_filled],
            "1h": [f for f in fvgs_1h  if f.direction == direction and not f.is_filled],
        }
        for tf_a, list_a in all_fvgs.items():
            for tf_b, list_b in all_fvgs.items():
                if tf_b <= tf_a:
                    continue
                for fa in list_a:
                    for fb in list_b:
                        if abs(fa.midpoint - fb.midpoint) <= tol:
                            side = "above" if fa.midpoint > price else "below"
                            lines.append(
                                f"{direction.capitalize()} FVG cluster {side} price: "
                                f"{tf_a} + {tf_b} both at ~{fa.midpoint:.5g}"
                            )

    _cluster("bullish")
    _cluster("bearish")
    return lines
