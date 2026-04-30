from __future__ import annotations

from dataclasses import dataclass, field
from math import inf

import numpy as np


@dataclass(frozen=True)
class StrategyLeg:
    option_type: str
    position: str
    strike: float
    expiry: str
    quantity: int
    premium: float
    greeks: dict[str, float]
    contract_symbol: str = ""
    market_price: float | None = None
    fair_value: float | None = None
    bid: float | None = None
    ask: float | None = None
    open_interest: float | None = None
    volume: float | None = None


@dataclass
class StrategyAnalytics:
    max_profit: float | None
    max_loss: float | None
    breakevens: list[float]
    reward_risk: float | None
    payoff_curve: list[tuple[float, float]]
    heatmap_points: list[tuple[float, float, float]]
    warnings: list[str] = field(default_factory=list)


def _leg_multiplier(position: str, quantity: int) -> int:
    sign = 1 if position == "long" else -1
    return sign * quantity


def aggregate_greeks(legs: list[StrategyLeg]) -> dict[str, float]:
    totals = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    for leg in legs:
        mult = _leg_multiplier(leg.position, leg.quantity)
        if leg.option_type == "stock":
            totals["delta"] += float(mult)
            continue
        for key in totals:
            totals[key] += mult * float(leg.greeks.get(key, 0.0))
    return totals


def net_premium(legs: list[StrategyLeg]) -> float:
    total = 0.0
    for leg in legs:
        signed = _leg_multiplier(leg.position, leg.quantity)
        total += signed * float(leg.premium)
    return total


def payoff_at_expiry(legs: list[StrategyLeg], underlying_price: float) -> float:
    total = -net_premium(legs)
    for leg in legs:
        mult = _leg_multiplier(leg.position, leg.quantity)
        if leg.option_type == "call":
            total += mult * max(underlying_price - leg.strike, 0.0)
        elif leg.option_type == "put":
            total += mult * max(leg.strike - underlying_price, 0.0)
        elif leg.option_type == "stock":
            total += mult * (underlying_price - leg.strike)
    return float(total)


def payoff_curve(legs: list[StrategyLeg], spot: float, points: int = 181) -> list[tuple[float, float]]:
    low = max(spot * 0.5, 0.01)
    high = max(spot * 1.5, low + 1.0)
    grid = np.linspace(low, high, points)
    return [(float(price), payoff_at_expiry(legs, float(price))) for price in grid]


def breakevens_from_curve(curve: list[tuple[float, float]]) -> list[float]:
    if not curve:
        return []
    out: list[float] = []
    for left, right in zip(curve, curve[1:]):
        x1, y1 = left
        x2, y2 = right
        if y1 == 0:
            out.append(round(x1, 4))
        if y1 * y2 < 0:
            be = x1 - y1 * (x2 - x1) / (y2 - y1)
            out.append(round(float(be), 4))
    return sorted(set(out))


def max_profit_loss(curve: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if not curve:
        return None, None
    profits = np.array([point[1] for point in curve], dtype=float)
    max_profit = float(np.max(profits))
    max_loss = float(np.min(profits))
    left_flat = len(profits) >= 3 and np.allclose(profits[:3], profits[0], atol=1e-6)
    right_flat = len(profits) >= 3 and np.allclose(profits[-3:], profits[-1], atol=1e-6)
    bounded_profit = (not np.isclose(max_profit, profits[-1]) or right_flat) and (not np.isclose(max_profit, profits[0]) or left_flat)
    bounded_loss = (not np.isclose(max_loss, profits[-1]) or right_flat) and (not np.isclose(max_loss, profits[0]) or left_flat)
    return (max_profit if bounded_profit else None, max_loss if bounded_loss else None)


def _approx_mark_to_market(legs: list[StrategyLeg], spot_shift: float, vol_shift: float) -> float:
    total = 0.0
    for leg in legs:
        base = leg.fair_value if leg.fair_value is not None else leg.market_price if leg.market_price is not None else leg.premium
        delta = float(leg.greeks.get("delta", 0.0))
        gamma = float(leg.greeks.get("gamma", 0.0))
        vega = float(leg.greeks.get("vega", 0.0))
        theta = float(leg.greeks.get("theta", 0.0))
        mtm = float(base) + delta * spot_shift + 0.5 * gamma * spot_shift * spot_shift + vega * vol_shift + theta * (1.0 / 365.0)
        sign = _leg_multiplier(leg.position, leg.quantity)
        total += sign * mtm
    return total - net_premium(legs)


def pnl_heatmap(legs: list[StrategyLeg], spot: float) -> list[tuple[float, float, float]]:
    prices = np.linspace(max(spot * 0.85, 0.01), spot * 1.15, 9)
    vol_shifts = np.linspace(-0.10, 0.10, 7)
    points: list[tuple[float, float, float]] = []
    for price in prices:
        for vol_shift in vol_shifts:
            points.append((float(price), float(vol_shift), float(_approx_mark_to_market(legs, price - spot, vol_shift))))
    return points


def analyze_strategy(legs: list[StrategyLeg], spot: float) -> StrategyAnalytics:
    curve = payoff_curve(legs, spot)
    breakevens = breakevens_from_curve(curve)
    max_profit, max_loss = max_profit_loss(curve)
    warnings: list[str] = []
    reward_risk = None
    if max_profit is None:
        warnings.append("Profit is not fully bounded in the scanned payoff range.")
    if max_loss is None:
        warnings.append("Loss is not fully bounded in the scanned payoff range.")
    if max_profit is not None and max_loss is not None and max_loss < 0:
        reward_risk = max_profit / abs(max_loss)
    return StrategyAnalytics(
        max_profit=max_profit,
        max_loss=max_loss,
        breakevens=breakevens,
        reward_risk=reward_risk,
        payoff_curve=curve,
        heatmap_points=pnl_heatmap(legs, spot),
        warnings=warnings,
    )
