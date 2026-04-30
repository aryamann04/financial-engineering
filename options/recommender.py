from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd

from analyzer.data import get_ticker
from config.settings import load_settings
from options.models import (
    ModelOutput,
    PricingInputs,
    bs_greeks,
    binomial_model_price,
    black76_model_price,
    bs_model_price,
    ensemble_price,
    local_vol_adjusted_price,
    sabr_price,
    unreliable_heston_model,
)
from options.risk import summarize_risk
from options.strategies import StrategyLeg, analyze_strategy, aggregate_greeks, net_premium
from options.surface import ChainSnapshot, VolRegime, atm_iv, classify_vol_regime, load_chain_snapshot, nearest_expiries


@dataclass
class StrategyRecommendation:
    strategy_name: str
    underlying: str
    expiry: str
    view: str
    strategy_type: str
    legs: list[dict]
    net_debit_credit: float
    max_profit: float | None
    max_loss: float | None
    breakevens: list[float]
    greeks: dict[str, float]
    model_edge: dict[str, float | None]
    vol_regime_rationale: list[str]
    liquidity_warnings: list[str]
    warnings: list[str]
    final_score: float
    explanation: str
    why_this_strategy: str
    invalidation: str
    reward_risk: float | None
    probability_of_profit: float | None
    margin_estimate: float | None
    payoff_curve: list[tuple[float, float]]
    pnl_heatmap: list[tuple[float, float, float]]
    exposure_text: str
    model_comparison: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _contract_warnings(row: pd.Series) -> list[str]:
    warnings: list[str] = []
    spread_pct = float(row.get("spread_pct", np.nan))
    oi = float(row.get("openInterest", 0.0) or 0.0)
    vol = float(row.get("volume", 0.0) or 0.0)
    if np.isfinite(spread_pct) and spread_pct > 0.20:
        warnings.append("Wide bid/ask spread.")
    if oi <= 0:
        warnings.append("Zero open interest.")
    if vol <= 0:
        warnings.append("Zero volume.")
    return warnings


def _model_prices(snapshot: ChainSnapshot, row: pd.Series, option_type: str) -> list[ModelOutput]:
    bs_iv_value = float(row.get("bs_iv", np.nan))
    if not np.isfinite(bs_iv_value) or bs_iv_value <= 0:
        return [ModelOutput("black_scholes", None, None, 0.0, ["Missing valid implied volatility."])]
    inputs = PricingInputs(
        spot=snapshot.spot,
        strike=float(row["strike"]),
        maturity=snapshot.maturity,
        rate=snapshot.rate,
        dividend_yield=snapshot.dividend_yield,
        implied_vol=bs_iv_value,
        option_type=option_type,
        american=option_type == "put",
    )
    outputs = [bs_model_price(inputs), black76_model_price(inputs), binomial_model_price(inputs)]
    svi = snapshot.svi_call if option_type == "call" else snapshot.svi_put
    if svi is not None:
        svi_vol = svi.svi_vol(inputs.strike)
        if svi_vol is not None and svi_vol > 0:
            outputs.append(ModelOutput("svi", bs_model_price(PricingInputs(**{**inputs.__dict__, "implied_vol": float(svi_vol)})).price, float(svi_vol), 0.25))
        else:
            outputs.append(ModelOutput("svi", None, None, 0.0, ["SVI unavailable at strike."]))
    anchor_df = snapshot.calls if option_type == "call" else snapshot.puts
    outputs.append(local_vol_adjusted_price(inputs, anchor_df[["strike", "bs_iv"]].dropna().itertuples(index=False, name=None)))
    outputs.append(sabr_price(inputs, anchor_df["strike"].to_numpy(dtype=float), anchor_df["bs_iv"].fillna(np.nan).to_numpy(dtype=float)))
    outputs.append(unreliable_heston_model())
    return outputs


def _build_leg(snapshot: ChainSnapshot, row: pd.Series, option_type: str, position: str, quantity: int = 1) -> tuple[StrategyLeg, dict]:
    if option_type == "stock":
        leg = StrategyLeg(
            option_type="stock",
            position=position,
            strike=snapshot.spot,
            expiry=snapshot.expiry,
            quantity=quantity,
            premium=snapshot.spot,
            greeks={"delta": 1.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0},
            contract_symbol=f"{snapshot.symbol}-STOCK",
            market_price=snapshot.spot,
            fair_value=snapshot.spot,
            open_interest=None,
            volume=None,
        )
        return leg, {"ensemble_fair_value": snapshot.spot, "model_confidence": 1.0, "model_rows": [{"model": "spot", "price": snapshot.spot, "volatility": None, "warnings": []}], "warnings": []}
    models = _model_prices(snapshot, row, option_type)
    fair_value, model_confidence = ensemble_price(models)
    market = float(row.get("mid", 0.0) or 0.0)
    greeks = {
        "delta": float(row.get("delta", 0.0) or 0.0),
        "gamma": float(row.get("gamma", 0.0) or 0.0),
        "theta": float(row.get("theta", 0.0) or 0.0),
        "vega": float(row.get("vega", 0.0) or 0.0),
        "rho": float(row.get("rho", 0.0) or 0.0),
    }
    leg = StrategyLeg(
        option_type=option_type,
        position=position,
        strike=float(row["strike"]),
        expiry=snapshot.expiry,
        quantity=quantity,
        premium=market,
        greeks=greeks,
        contract_symbol=str(row.get("contractSymbol", "")),
        market_price=market,
        fair_value=fair_value,
        bid=float(row.get("bid", np.nan)) if pd.notna(row.get("bid")) else None,
        ask=float(row.get("ask", np.nan)) if pd.notna(row.get("ask")) else None,
        open_interest=float(row.get("openInterest", 0.0) or 0.0),
        volume=float(row.get("volume", 0.0) or 0.0),
    )
    model_rows = [
        {"model": item.name, "price": item.price, "volatility": item.volatility, "warnings": item.warnings}
        for item in models
    ]
    return leg, {"ensemble_fair_value": fair_value, "model_confidence": model_confidence, "model_rows": model_rows, "warnings": _contract_warnings(row)}


def _rows_near(snapshot: ChainSnapshot, frame: str, target: float, count: int = 6) -> pd.DataFrame:
    df = snapshot.calls if frame == "call" else snapshot.puts
    if df.empty:
        return pd.DataFrame()
    usable = df[df["mid"] > 0].copy()
    usable = usable.sort_values("liquidity_score", ascending=False).head(80)
    usable["distance"] = (usable["strike"] - target).abs()
    return usable.sort_values(["distance", "spread_pct", "liquidity_score"], ascending=[True, True, False]).head(count)


def _best_row(snapshot: ChainSnapshot, option_type: str, strike_target: float) -> pd.Series | None:
    frame = _rows_near(snapshot, option_type, strike_target, count=1)
    if frame.empty:
        return None
    return frame.iloc[0]


def _score_strategy(vol_regime: VolRegime, analytics, risk_summary, legs: list[StrategyLeg], edge_pct: float, model_confidence: float, warnings: list[str], strategy_name: str, view: str) -> float:
    liquidity = np.mean([max(0.0, min((leg.open_interest or 0.0) / 500.0, 1.0)) for leg in legs]) if legs else 0.0
    simplicity_penalty = max(0, len(legs) - 2) * 5.0
    risk_defined_bonus = 10.0 if risk_summary.max_loss is not None else -8.0
    vol_fit = 0.0
    if vol_regime.label == "high_iv" and any(name in strategy_name.lower() for name in ("iron", "condor", "bear call", "bull put", "covered", "cash-secured")):
        vol_fit = 12.0
    elif vol_regime.label == "low_iv" and any(name in strategy_name.lower() for name in ("bull call", "bear put", "calendar", "straddle", "strangle")):
        vol_fit = 12.0
    elif vol_regime.label == "realized_breakout" and any(name in strategy_name.lower() for name in ("straddle", "strangle", "debit", "calendar")):
        vol_fit = 10.0
    rr = (risk_summary.reward_risk or 0.0) * 8.0
    warning_penalty = len(warnings) * 4.0
    direction_fit = 0.0
    if view == "bullish" and any(name in strategy_name.lower() for name in ("bull", "covered", "cash-secured")):
        direction_fit = 8.0
    elif view == "bearish" and any(name in strategy_name.lower() for name in ("bear", "ratio")):
        direction_fit = 8.0
    elif view == "neutral" and any(name in strategy_name.lower() for name in ("condor", "butterfly", "straddle", "strangle")):
        direction_fit = 8.0
    return float(edge_pct * 20.0 + model_confidence * 20.0 + liquidity * 15.0 + rr + vol_fit + direction_fit + risk_defined_bonus - simplicity_penalty - warning_penalty)


def _recommendation_from_legs(
    strategy_name: str,
    underlying: str,
    expiry: str,
    strategy_type: str,
    view: str,
    vol_regime: VolRegime,
    legs: list[StrategyLeg],
    leg_meta: list[dict],
) -> StrategyRecommendation:
    analytics = analyze_strategy(legs, leg_meta[0]["spot"])
    risk = summarize_risk(legs, analytics)
    fair_value = sum(((1 if leg.position == "long" else -1) * (leg.fair_value or leg.premium)) for leg in legs)
    market_value = net_premium(legs)
    edge = fair_value - market_value if fair_value is not None else None
    edge_pct = (edge / max(abs(market_value), 0.01)) if edge is not None else 0.0
    liquidity_warnings = [warning for meta in leg_meta for warning in meta["warnings"]]
    model_confidence = float(np.mean([meta["model_confidence"] for meta in leg_meta])) if leg_meta else 0.0
    warnings = list(analytics.warnings) + liquidity_warnings
    if "ratio" in strategy_name.lower():
        warnings.append("Ratio spread carries asymmetric tail risk.")
    final_score = _score_strategy(vol_regime, analytics, risk, legs, edge_pct, model_confidence, warnings, strategy_name, view)
    return StrategyRecommendation(
        strategy_name=strategy_name,
        underlying=underlying,
        expiry=expiry,
        view=view,
        strategy_type=strategy_type,
        legs=[
            {
                "option_type": leg.option_type,
                "position": leg.position,
                "strike": leg.strike,
                "expiry": leg.expiry,
                "quantity": leg.quantity,
                "premium": leg.premium,
                "contract_symbol": leg.contract_symbol,
            }
            for leg in legs
        ],
        net_debit_credit=market_value,
        max_profit=risk.max_profit,
        max_loss=risk.max_loss,
        breakevens=risk.breakevens,
        greeks=risk.greeks,
        model_edge={
            "market_price": market_value,
            "ensemble_fair_value": fair_value,
            "edge_dollars": edge,
            "edge_pct": edge_pct * 100.0 if edge is not None else None,
            "confidence": model_confidence,
        },
        vol_regime_rationale=vol_regime.rationale,
        liquidity_warnings=liquidity_warnings,
        warnings=warnings,
        final_score=final_score,
        explanation=f"{strategy_name} matches a {vol_regime.label} regime with {risk.exposure_text}.",
        why_this_strategy=f"Selected because it offers {'defined risk' if risk.max_loss is not None else 'open-ended payout'} and an estimated edge of {edge_pct * 100.0:.1f}%.",
        invalidation="Invalid if the vol regime changes materially, liquidity deteriorates, or price moves beyond the intended structure zone.",
        reward_risk=risk.reward_risk,
        probability_of_profit=risk.probability_of_profit,
        margin_estimate=risk.margin_estimate,
        payoff_curve=analytics.payoff_curve,
        pnl_heatmap=analytics.heatmap_points,
        exposure_text=risk.exposure_text,
        model_comparison=[row for meta in leg_meta for row in meta["model_rows"]],
    )


def _greek_patch(snapshot: ChainSnapshot, row: pd.Series, option_type: str) -> pd.Series:
    if all(name in row and pd.notna(row.get(name)) for name in ("delta", "gamma", "theta", "vega", "rho")):
        return row
    sigma = float(row.get("bs_iv", np.nan))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = float(row.get("impliedVolatility", 0.25) or 0.25)
    for greek, value in bs_greeks(snapshot.spot, float(row["strike"]), snapshot.maturity, snapshot.rate, sigma, snapshot.dividend_yield, option_type).items():
        row[greek] = value
    return row


def _candidate_builders(snapshot: ChainSnapshot, vol_regime: VolRegime, view: str) -> list[StrategyRecommendation]:
    spot = snapshot.spot
    candidates: list[StrategyRecommendation] = []
    atm_call = _best_row(snapshot, "call", spot)
    atm_put = _best_row(snapshot, "put", spot)
    otm_call = _best_row(snapshot, "call", spot * 1.05)
    farther_call = _best_row(snapshot, "call", spot * 1.10)
    otm_put = _best_row(snapshot, "put", spot * 0.95)
    farther_put = _best_row(snapshot, "put", spot * 0.90)
    itm_call = _best_row(snapshot, "call", spot * 0.98)
    itm_put = _best_row(snapshot, "put", spot * 1.02)

    def _row(row: pd.Series | None, option_type: str) -> pd.Series | None:
        if row is None:
            return None
        return _greek_patch(snapshot, row.copy(), option_type)

    atm_call = _row(atm_call, "call")
    atm_put = _row(atm_put, "put")
    otm_call = _row(otm_call, "call")
    farther_call = _row(farther_call, "call")
    otm_put = _row(otm_put, "put")
    farther_put = _row(farther_put, "put")
    itm_call = _row(itm_call, "call")
    itm_put = _row(itm_put, "put")

    def _make(name: str, strategy_type: str, raw_legs: list[tuple[pd.Series, str, str, int]]) -> None:
        if any(item[0] is None for item in raw_legs):
            return
        built_legs: list[StrategyLeg] = []
        meta: list[dict] = []
        for row, option_type, position, quantity in raw_legs:
            assert row is not None
            leg, info = _build_leg(snapshot, row, option_type, position, quantity=quantity)
            info["spot"] = spot
            built_legs.append(leg)
            meta.append(info)
        candidates.append(_recommendation_from_legs(name, snapshot.symbol, snapshot.expiry, strategy_type, view, vol_regime, built_legs, meta))

    if view in {"bullish", "neutral"}:
        _make("Bull Call Spread", "vertical", [(itm_call, "call", "long", 1), (otm_call, "call", "short", 1)])
        _make("Bull Put Spread", "vertical", [(otm_put, "put", "short", 1), (farther_put, "put", "long", 1)])
        _make("Covered Call", "covered", [(pd.Series({"strike": spot}), "stock", "long", 1), (otm_call, "call", "short", 1)])
        _make("Cash-Secured Put", "cash_secured_put", [(otm_put, "put", "short", 1)])
    if view in {"bearish", "neutral"}:
        _make("Bear Put Spread", "vertical", [(itm_put, "put", "long", 1), (otm_put, "put", "short", 1)])
        _make("Bear Call Spread", "vertical", [(otm_call, "call", "short", 1), (farther_call, "call", "long", 1)])
        _make("Call Ratio Spread", "ratio", [(atm_call, "call", "long", 1), (otm_call, "call", "short", 2)])
    _make("Long Straddle", "volatility", [(atm_call, "call", "long", 1), (atm_put, "put", "long", 1)])
    _make("Long Strangle", "volatility", [(otm_call, "call", "long", 1), (otm_put, "put", "long", 1)])
    _make("Iron Condor", "neutral", [(farther_put, "put", "long", 1), (otm_put, "put", "short", 1), (otm_call, "call", "short", 1), (farther_call, "call", "long", 1)])
    _make("Iron Butterfly", "neutral", [(farther_put if farther_put is not None else otm_put, "put", "long", 1), (atm_put, "put", "short", 1), (atm_call, "call", "short", 1), (farther_call, "call", "long", 1)])
    _make("Call Butterfly", "butterfly", [(itm_call, "call", "long", 1), (atm_call, "call", "short", 2), (otm_call, "call", "long", 1)])
    _make("Call Condor", "condor", [(itm_call, "call", "long", 1), (atm_call, "call", "short", 1), (otm_call, "call", "short", 1), (farther_call, "call", "long", 1)])
    return candidates


def _calendar_and_diagonal(symbol: str, expiries: list[str], view: str) -> list[StrategyRecommendation]:
    if len(expiries) < 2:
        return []
    near_snapshot = load_chain_snapshot(symbol, expiries[0])
    far_snapshot = load_chain_snapshot(symbol, expiries[1])
    vol_regime = classify_vol_regime(symbol, near_snapshot)
    out: list[StrategyRecommendation] = []
    near_call = _best_row(near_snapshot, "call", near_snapshot.spot)
    far_call = _best_row(far_snapshot, "call", far_snapshot.spot)
    near_put = _best_row(near_snapshot, "put", near_snapshot.spot)
    far_put = _best_row(far_snapshot, "put", far_snapshot.spot)
    if near_call is not None and far_call is not None:
        near_call = _greek_patch(near_snapshot, near_call.copy(), "call")
        far_call = _greek_patch(far_snapshot, far_call.copy(), "call")
        legs = []
        meta = []
        for snap, row, option_type, position in ((far_snapshot, far_call, "call", "long"), (near_snapshot, near_call, "call", "short")):
            leg, info = _build_leg(snap, row, option_type, position)
            info["spot"] = far_snapshot.spot
            legs.append(leg)
            meta.append(info)
        out.append(_recommendation_from_legs("Call Calendar", symbol, far_snapshot.expiry, "calendar", view, vol_regime, legs, meta))
    if near_call is not None and far_put is not None and view in {"bearish", "neutral"}:
        far_put = _greek_patch(far_snapshot, far_put.copy(), "put")
        legs = []
        meta = []
        for snap, row, option_type, position in ((far_snapshot, far_put, "put", "long"), (near_snapshot, near_call, "call", "short")):
            leg, info = _build_leg(snap, row, option_type, position)
            info["spot"] = far_snapshot.spot
            legs.append(leg)
            meta.append(info)
        out.append(_recommendation_from_legs("Diagonal Hedge", symbol, far_snapshot.expiry, "diagonal", view, vol_regime, legs, meta))
    if near_put is not None and far_put is not None:
        near_put = _greek_patch(near_snapshot, near_put.copy(), "put")
        far_put = _greek_patch(far_snapshot, far_put.copy(), "put")
        legs = []
        meta = []
        for snap, row, option_type, position in ((far_snapshot, far_put, "put", "long"), (near_snapshot, near_put, "put", "short")):
            leg, info = _build_leg(snap, row, option_type, position)
            info["spot"] = far_snapshot.spot
            legs.append(leg)
            meta.append(info)
        out.append(_recommendation_from_legs("Put Calendar", symbol, far_snapshot.expiry, "calendar", view, vol_regime, legs, meta))
    return out


def recommend_strategies(
    symbol: str,
    *,
    view: str = "neutral",
    expiry: str | None = None,
    max_risk: float | None = None,
    strategy_type: str | None = None,
    min_score: float | None = None,
    limit: int = 12,
) -> list[StrategyRecommendation]:
    symbol = symbol.upper().strip()
    expiries = [expiry] if expiry else nearest_expiries(symbol, count=2)
    if not expiries:
        return []

    settings = load_settings()
    near_snapshot = load_chain_snapshot(symbol, expiries[0], ttl_seconds=settings.options_snapshot_ttl_seconds)
    vol_regime = classify_vol_regime(symbol, near_snapshot)
    recommendations = _candidate_builders(near_snapshot, vol_regime, view)
    recommendations.extend(_calendar_and_diagonal(symbol, expiries, view))

    filtered: list[StrategyRecommendation] = []
    for recommendation in recommendations:
        if strategy_type and recommendation.strategy_type != strategy_type:
            continue
        if max_risk is not None and recommendation.max_loss is not None and abs(recommendation.max_loss) > max_risk:
            continue
        if min_score is not None and recommendation.final_score < min_score:
            continue
        filtered.append(recommendation)
    filtered.sort(key=lambda item: item.final_score, reverse=True)
    return filtered[:limit]
