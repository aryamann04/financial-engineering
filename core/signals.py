from __future__ import annotations

from dataclasses import dataclass

from core.fvg import FVGZone
from core.indicators import TrendSnapshot
from core.structure import StructureState


@dataclass(frozen=True)
class TimeframeSnapshot:
    timeframe: str
    trend: str
    bias: str
    price_position: str
    atr: float | None
    recent_range: float | None
    session_high: float | None
    session_low: float | None
    vwap: float | None
    key_levels: list[str]
    structure: str
    source: str


def summarize_timeframe(
    timeframe: str,
    trend: TrendSnapshot,
    structure: StructureState,
    current_price: float | None,
    atr: float | None,
    recent_range: float | None,
    session_high: float | None,
    session_low: float | None,
    vwap: float | None,
    nearby_fvgs: list[FVGZone],
) -> TimeframeSnapshot:
    if current_price is None:
        price_position = "No live price"
    elif vwap is None:
        price_position = "VWAP unavailable"
    elif current_price > vwap:
        price_position = "Above VWAP"
    elif current_price < vwap:
        price_position = "Below VWAP"
    else:
        price_position = "At VWAP"

    key_levels = [
        f"{zone.direction} FVG {zone.lower:.5g}-{zone.upper:.5g}"
        for zone in nearby_fvgs[:2]
    ]

    bias = trend.trend if trend.trend != "neutral" else structure.bias
    return TimeframeSnapshot(
        timeframe=timeframe,
        trend=trend.trend,
        bias=bias,
        price_position=price_position,
        atr=atr,
        recent_range=recent_range,
        session_high=session_high,
        session_low=session_low,
        vwap=vwap,
        key_levels=key_levels,
        structure=structure.regime,
        source=f"based on current {timeframe} bars",
    )


def alignment_status(snapshots: dict[str, TimeframeSnapshot]) -> str:
    directions = [snap.bias for snap in snapshots.values() if snap.bias in {"bullish", "bearish"}]
    if not directions:
        return "neutral"
    if all(direction == directions[0] for direction in directions):
        return directions[0]
    return "conflicted"
