from __future__ import annotations

from dataclasses import dataclass

from core.fvg import FVGZone
from core.liquidity import KeyLevel


@dataclass(frozen=True)
class RiskPlan:
    entry_zone: str
    invalidation: str
    targets: list[str]
    r_multiple: float | None
    method: str
    source: str


def build_risk_plan(
    current_price: float | None,
    direction: str,
    atr: float | None,
    key_levels: list[KeyLevel],
    nearby_fvgs: list[FVGZone],
    entry_low: float | None = None,
    entry_high: float | None = None,
) -> RiskPlan | None:
    if current_price is None:
        return None

    entry_low = current_price if entry_low is None else entry_low
    entry_high = current_price if entry_high is None else entry_high
    entry_mid = (entry_low + entry_high) / 2.0
    sign = 1 if direction == "long" else -1

    reference_level = None
    if direction == "long":
        below_levels = [level for level in key_levels if level.price < entry_low]
        reference_level = max(below_levels, key=lambda level: level.price) if below_levels else None
        if nearby_fvgs:
            bullish = [zone for zone in nearby_fvgs if zone.direction == "bullish" and zone.lower <= entry_low]
            if bullish:
                reference_level = reference_level or KeyLevel("Bullish FVG boundary", bullish[0].lower, "fvg", bullish[0].timeframe, bullish[0].source)
    else:
        above_levels = [level for level in key_levels if level.price > entry_high]
        reference_level = min(above_levels, key=lambda level: level.price) if above_levels else None
        if nearby_fvgs:
            bearish = [zone for zone in nearby_fvgs if zone.direction == "bearish" and zone.upper >= entry_high]
            if bearish:
                reference_level = reference_level or KeyLevel("Bearish FVG boundary", bearish[0].upper, "fvg", bearish[0].timeframe, bearish[0].source)

    padding = (atr or 0.0) * 0.5
    if reference_level is not None:
        invalidation_price = reference_level.price - padding if direction == "long" else reference_level.price + padding
        method = f"{reference_level.name} with ATR padding"
    else:
        invalidation_price = entry_mid - sign * (atr or entry_mid * 0.002)
        method = "ATR fallback"

    targets: list[float] = []
    aligned_levels = sorted(
        [level.price for level in key_levels if (level.price > entry_mid if direction == "long" else level.price < entry_mid)],
        reverse=direction == "short",
    )
    for price in aligned_levels[:3]:
        targets.append(price)
    while len(targets) < 3 and atr:
        targets.append(entry_mid + sign * atr * (len(targets) + 1))

    risk = abs(entry_mid - invalidation_price)
    reward = abs(targets[0] - entry_mid) if targets else None
    r_multiple = (reward / risk) if reward is not None and risk > 0 else None

    return RiskPlan(
        entry_zone=f"{entry_low:.5g} - {entry_high:.5g}",
        invalidation=f"{invalidation_price:.5g}",
        targets=[f"{target:.5g}" for target in targets[:3]],
        r_multiple=r_multiple,
        method=method,
        source="based on ATR, nearby structure, and liquidity levels",
    )
