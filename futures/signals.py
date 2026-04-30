from __future__ import annotations

from dataclasses import dataclass, field

from futures.bias import BiasResult, ConfluenceZone
from futures.fvg import FairValueGap
from futures.levels import FuturesLevels
from futures.volume import VolumeProfile


@dataclass
class AlertMessage:
    level: str          # info | caution | bullish | bearish
    message: str


@dataclass
class ExitGuidance:
    direction: str
    current_r: float | None
    distance_to_stop: float
    distance_to_target: float | None
    distance_to_1x: float | None
    distance_to_2x: float | None
    distance_to_3x: float | None
    nearest_barrier: str
    management_style: str
    warnings: list[str] = field(default_factory=list)


def build_alerts(
    spot: float | None,
    levels: FuturesLevels | None,
    bias: BiasResult | None,
    prior_bias: str | None,
    confluence: list[ConfluenceZone],
    fvgs: list[FairValueGap],
    volume_profile: VolumeProfile | None,
    atr: float | None,
    volume_spike: bool,
) -> list[AlertMessage]:
    if spot is None:
        return [AlertMessage("caution", "No current price available for alerts.")]

    alerts: list[AlertMessage] = []
    tol = (atr or 0) * 0.25 if atr else 0.0

    if bias and prior_bias and prior_bias != bias.bias and bias.bias in ("bullish", "bearish"):
        alerts.append(AlertMessage(bias.bias, f"Bias changed from {prior_bias} to {bias.bias}."))

    if levels:
        for name, price in [
            ("session high", levels.new_york.high),
            ("session low", levels.new_york.low),
            ("VWAP", levels.vwap),
        ]:
            if price is not None and abs(spot - price) <= max(tol, abs(price) * 0.0003):
                alerts.append(AlertMessage("info", f"Price is near {name} ({price:.5g})."))

    for zone in confluence[:3]:
        if zone.lower <= spot <= zone.upper or min(abs(spot - zone.lower), abs(spot - zone.upper)) <= tol:
            alerts.append(AlertMessage("info", f"Price is interacting with confluence zone {zone.lower:.5g}-{zone.upper:.5g}."))
            break

    for fvg in fvgs[:6]:
        if fvg.lower <= spot <= fvg.upper:
            direction = "bullish" if fvg.direction == "bullish" else "bearish"
            alerts.append(AlertMessage(direction, f"Price entered {fvg.timeframe} {fvg.direction} FVG {fvg.lower:.5g}-{fvg.upper:.5g}."))
            break

    if volume_profile and volume_profile.is_inside_value(spot):
        alerts.append(AlertMessage("caution", "Price is inside value area; expect more chop unless value breaks."))

    if volume_spike:
        alerts.append(AlertMessage("info", "Relative volume spike detected on the latest bars."))

    return alerts[:6]


def build_exit_guidance(
    direction: str,
    entry: float,
    stop: float,
    current_price: float,
    target: float | None,
    atr: float | None,
    levels: FuturesLevels | None,
    volume_profile: VolumeProfile | None,
    fvgs: list[FairValueGap],
) -> ExitGuidance:
    sign = 1 if direction == "long" else -1
    risk = abs(entry - stop) or 1e-9
    current_r = sign * (current_price - entry) / risk

    barriers: list[tuple[str, float]] = []
    if levels:
        for name, price in [
            ("prev day high", levels.prev_day_high),
            ("prev day low", levels.prev_day_low),
            ("London high", levels.london.high),
            ("London low", levels.london.low),
            ("VWAP", levels.vwap),
        ]:
            if price is not None:
                barriers.append((name, price))
    if volume_profile:
        barriers.extend([("POC", volume_profile.poc), ("VAH", volume_profile.vah), ("VAL", volume_profile.val)])
    for fvg in fvgs[:6]:
        barriers.append((f"{fvg.timeframe} {fvg.direction} FVG", fvg.midpoint))

    if direction == "long":
        relevant = [(name, price) for name, price in barriers if price >= current_price]
    else:
        relevant = [(name, price) for name, price in barriers if price <= current_price]
    nearest = min(relevant, key=lambda item: abs(item[1] - current_price)) if relevant else None
    nearest_barrier = f"{nearest[0]} @ {nearest[1]:.5g}" if nearest else "None nearby"

    d1 = d2 = d3 = None
    if atr:
        d1 = abs((entry + sign * atr) - current_price)
        d2 = abs((entry + sign * 2 * atr) - current_price)
        d3 = abs((entry + sign * 3 * atr) - current_price)

    warnings: list[str] = []
    management_style = "hold"
    if nearest and atr and abs(nearest[1] - current_price) <= 0.5 * atr:
        management_style = "partial at next level"
        warnings.append(f"Price is approaching {nearest_barrier}.")
    if current_r >= 2:
        management_style = "tighten stop below higher low" if direction == "long" else "tighten stop above lower high"
    if current_r < 0.5:
        management_style = "avoid moving stop too early"
    if atr and abs(current_price - entry) >= 3 * atr:
        management_style = "trade is extended, consider reducing risk"
        warnings.append("Move may be stretched relative to ATR.")

    for fvg in fvgs[:4]:
        if direction == "long" and fvg.direction == "bearish" and fvg.lower >= current_price:
            warnings.append(f"Bearish FVG overhead: {fvg.lower:.5g}-{fvg.upper:.5g}.")
            break
        if direction == "short" and fvg.direction == "bullish" and fvg.upper <= current_price:
            warnings.append(f"Bullish FVG below: {fvg.lower:.5g}-{fvg.upper:.5g}.")
            break

    return ExitGuidance(
        direction=direction,
        current_r=current_r,
        distance_to_stop=abs(current_price - stop),
        distance_to_target=abs(target - current_price) if target is not None else None,
        distance_to_1x=d1,
        distance_to_2x=d2,
        distance_to_3x=d3,
        nearest_barrier=nearest_barrier,
        management_style=management_style,
        warnings=warnings,
    )
