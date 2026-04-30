from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from futures.fvg import detect_fvgs


@dataclass(frozen=True)
class FVGZone:
    timeframe: str
    direction: str
    lower: float
    upper: float
    midpoint: float
    status: str
    fill_pct: float
    distance: float
    formed_at: object
    source: str


def detect_fvg_zones(
    df: pd.DataFrame,
    timeframe: str,
    current_price: float,
    atr: float | None,
    min_size_atr: float,
) -> list[FVGZone]:
    if df is None or df.empty:
        return []
    raw = detect_fvgs(
        df,
        timeframe=timeframe,
        current_price=current_price,
        atr=atr,
        min_size_atr=min_size_atr,
        max_fvgs=16,
    )
    zones: list[FVGZone] = []
    for fvg in raw:
        if fvg.is_filled:
            status = "fully mitigated"
        elif fvg.fill_pct > 0:
            status = "partially filled"
        else:
            status = "unfilled"
        zones.append(
            FVGZone(
                timeframe=fvg.timeframe,
                direction=fvg.direction,
                lower=fvg.lower,
                upper=fvg.upper,
                midpoint=fvg.midpoint,
                status=status,
                fill_pct=fvg.fill_pct,
                distance=fvg.dist_from_price,
                formed_at=fvg.formed_at,
                source=f"based on current {timeframe} bars",
            )
        )
    zones.sort(key=lambda zone: (zone.distance, zone.timeframe))
    return zones
