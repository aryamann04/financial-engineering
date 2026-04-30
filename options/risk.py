from __future__ import annotations

from dataclasses import dataclass

from options.strategies import StrategyAnalytics, StrategyLeg, aggregate_greeks


@dataclass(frozen=True)
class RiskSummary:
    max_profit: float | None
    max_loss: float | None
    breakevens: list[float]
    reward_risk: float | None
    margin_estimate: float | None
    probability_of_profit: float | None
    greeks: dict[str, float]
    exposure_text: str


def _margin_estimate(legs: list[StrategyLeg], analytics: StrategyAnalytics) -> float | None:
    if analytics.max_loss is not None:
        return abs(analytics.max_loss)
    naked_short = [leg for leg in legs if leg.position == "short" and leg.option_type in {"call", "put"}]
    if naked_short:
        return sum((leg.strike if leg.option_type == "put" else max(leg.strike, 1.0)) * leg.quantity * 0.2 for leg in naked_short)
    return None


def _pop_estimate(analytics: StrategyAnalytics) -> float | None:
    if not analytics.payoff_curve:
        return None
    positive = sum(1 for _, pnl in analytics.payoff_curve if pnl > 0)
    return positive / len(analytics.payoff_curve)


def _exposure_text(greeks: dict[str, float]) -> str:
    parts: list[str] = []
    delta = greeks.get("delta", 0.0)
    theta = greeks.get("theta", 0.0)
    vega = greeks.get("vega", 0.0)
    parts.append("bullish delta exposure" if delta > 0.15 else "bearish delta exposure" if delta < -0.15 else "low directional delta")
    parts.append("positive theta carry" if theta > 0 else "negative theta carry")
    parts.append("long volatility" if vega > 0 else "short volatility" if vega < 0 else "low vega exposure")
    return ", ".join(parts)


def summarize_risk(legs: list[StrategyLeg], analytics: StrategyAnalytics) -> RiskSummary:
    greeks = aggregate_greeks(legs)
    return RiskSummary(
        max_profit=analytics.max_profit,
        max_loss=analytics.max_loss,
        breakevens=analytics.breakevens,
        reward_risk=analytics.reward_risk,
        margin_estimate=_margin_estimate(legs, analytics),
        probability_of_profit=_pop_estimate(analytics),
        greeks=greeks,
        exposure_text=_exposure_text(greeks),
    )
