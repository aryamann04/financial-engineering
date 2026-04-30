from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from options.models import ModelOutput, PricingInputs, bs_model_price, ensemble_price
from options.recommender import recommend_strategies
from options.strategies import StrategyLeg, aggregate_greeks, analyze_strategy, payoff_at_expiry
from options.surface import ChainSnapshot, VolRegime


def _chain_row(strike: float, option_type: str, mid: float, bid: float, ask: float, iv: float, delta: float) -> dict:
    return {
        "contractSymbol": f"TST-{option_type.upper()}-{strike}",
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "lastPrice": mid,
        "mid": mid,
        "spread": ask - bid,
        "spread_pct": (ask - bid) / mid if mid else 0.0,
        "openInterest": 500,
        "volume": 200,
        "impliedVolatility": iv,
        "bs_iv": iv,
        "liquidity_score": 5.0,
        "delta": delta,
        "gamma": 0.02,
        "theta": -0.01,
        "vega": 0.12,
        "rho": 0.03,
        "option_type": option_type,
    }


def _snapshot(expiry: str = "2026-06-19") -> ChainSnapshot:
    calls = pd.DataFrame(
        [
            _chain_row(95, "call", 8.4, 8.2, 8.6, 0.26, 0.63),
            _chain_row(100, "call", 5.2, 5.0, 5.4, 0.24, 0.52),
            _chain_row(105, "call", 3.1, 3.0, 3.2, 0.23, 0.39),
            _chain_row(110, "call", 1.8, 1.7, 1.9, 0.22, 0.28),
        ]
    )
    puts = pd.DataFrame(
        [
            _chain_row(90, "put", 1.5, 1.4, 1.6, 0.25, -0.18),
            _chain_row(95, "put", 2.7, 2.6, 2.8, 0.26, -0.31),
            _chain_row(100, "put", 4.8, 4.6, 5.0, 0.27, -0.47),
            _chain_row(105, "put", 7.4, 7.2, 7.6, 0.29, -0.61),
        ]
    )
    return ChainSnapshot(
        symbol="TST",
        expiry=expiry,
        maturity=0.25,
        spot=100.0,
        rate=0.04,
        dividend_yield=0.0,
        calls=calls,
        puts=puts,
        svi_call=None,
        svi_put=None,
        warnings=[],
    )


def _vol_regime(label: str = "high_iv") -> VolRegime:
    return VolRegime(label=label, iv_rank=75.0, rv_iv_ratio=0.7, term_structure="contango", skew="put skew rich", rationale=["IV rank 75", "RV/IV 0.70", "put skew rich"])


def test_bull_call_spread_payoff_and_breakeven():
    legs = [
        StrategyLeg("call", "long", 100, "2026-06-19", 1, 5.0, {"delta": 0.5, "gamma": 0.02, "theta": -0.01, "vega": 0.1, "rho": 0.02}),
        StrategyLeg("call", "short", 110, "2026-06-19", 1, 2.0, {"delta": 0.25, "gamma": 0.01, "theta": -0.005, "vega": 0.06, "rho": 0.01}),
    ]
    analytics = analyze_strategy(legs, 100.0)
    assert payoff_at_expiry(legs, 100.0) == pytest.approx(-3.0)
    assert payoff_at_expiry(legs, 110.0) == pytest.approx(7.0)
    assert analytics.reward_risk is not None
    assert any(abs(point - 103.0) < 0.5 for point in analytics.breakevens)


def test_long_straddle_has_two_breakevens():
    legs = [
        StrategyLeg("call", "long", 100, "2026-06-19", 1, 4.0, {"delta": 0.5, "gamma": 0.03, "theta": -0.02, "vega": 0.11, "rho": 0.02}),
        StrategyLeg("put", "long", 100, "2026-06-19", 1, 4.0, {"delta": -0.5, "gamma": 0.03, "theta": -0.02, "vega": 0.11, "rho": -0.02}),
    ]
    analytics = analyze_strategy(legs, 100.0)
    assert len(analytics.breakevens) >= 2
    assert analytics.max_loss is not None


def test_aggregate_greeks_respects_short_legs():
    legs = [
        StrategyLeg("call", "long", 100, "2026-06-19", 1, 5.0, {"delta": 0.5, "gamma": 0.02, "theta": -0.01, "vega": 0.10, "rho": 0.02}),
        StrategyLeg("call", "short", 105, "2026-06-19", 1, 3.0, {"delta": 0.3, "gamma": 0.01, "theta": -0.005, "vega": 0.07, "rho": 0.01}),
    ]
    greeks = aggregate_greeks(legs)
    assert greeks["delta"] == pytest.approx(0.2)
    assert greeks["vega"] == pytest.approx(0.03)


def test_ensemble_fair_value_uses_weighted_average():
    outputs = [
        ModelOutput("a", 5.0, 0.2, 0.5),
        ModelOutput("b", 7.0, 0.22, 0.5),
        ModelOutput("bad", None, None, 0.0),
    ]
    price, confidence = ensemble_price(outputs)
    assert price == pytest.approx(6.0)
    assert 0.0 <= confidence <= 1.0


def test_recommendations_include_supported_strategy_types(monkeypatch):
    first = _snapshot("2026-06-19")
    second = _snapshot("2026-07-17")
    monkeypatch.setattr("options.recommender.nearest_expiries", lambda symbol, count=2: ["2026-06-19", "2026-07-17"][:count])
    monkeypatch.setattr("options.recommender.load_chain_snapshot", lambda symbol, expiry, ttl_seconds=180: first if expiry == "2026-06-19" else second)
    monkeypatch.setattr("options.recommender.classify_vol_regime", lambda symbol, snapshot: _vol_regime("high_iv"))

    recommendations = recommend_strategies("TST", view="neutral", limit=20)
    names = {item.strategy_name for item in recommendations}
    assert "Bull Call Spread" in names
    assert "Bear Put Spread" in names
    assert "Iron Condor" in names
    assert "Long Straddle" in names
    assert "Call Calendar" in names


def test_liquidity_warnings_surface_on_wide_spreads(monkeypatch):
    snapshot = _snapshot()
    snapshot.calls.loc[snapshot.calls["strike"] == 105, "spread_pct"] = 0.35
    snapshot.calls.loc[snapshot.calls["strike"] == 105, "volume"] = 0
    snapshot.calls.loc[snapshot.calls["strike"] == 105, "openInterest"] = 0

    monkeypatch.setattr("options.recommender.nearest_expiries", lambda symbol, count=2: ["2026-06-19"][:count])
    monkeypatch.setattr("options.recommender.load_chain_snapshot", lambda symbol, expiry, ttl_seconds=180: snapshot)
    monkeypatch.setattr("options.recommender.classify_vol_regime", lambda symbol, snapshot: _vol_regime("low_iv"))

    recommendations = recommend_strategies("TST", view="bullish", limit=10)
    assert recommendations
    warning_blob = " ".join(" ".join(item.liquidity_warnings + item.warnings) for item in recommendations)
    assert "Wide bid/ask spread." in warning_blob
    assert "Zero volume." in warning_blob or "Zero open interest." in warning_blob


def test_invalid_chain_data_returns_no_recommendations(monkeypatch):
    empty_snapshot = ChainSnapshot(
        symbol="TST",
        expiry="2026-06-19",
        maturity=0.25,
        spot=100.0,
        rate=0.04,
        dividend_yield=0.0,
        calls=pd.DataFrame(),
        puts=pd.DataFrame(),
        svi_call=None,
        svi_put=None,
        warnings=[],
    )
    monkeypatch.setattr("options.recommender.nearest_expiries", lambda symbol, count=2: ["2026-06-19"][:count])
    monkeypatch.setattr("options.recommender.load_chain_snapshot", lambda symbol, expiry, ttl_seconds=180: empty_snapshot)
    monkeypatch.setattr("options.recommender.classify_vol_regime", lambda symbol, snapshot: _vol_regime())
    assert recommend_strategies("TST") == []


def test_option_model_prices_are_finite():
    inputs = PricingInputs(spot=100.0, strike=100.0, maturity=0.5, rate=0.04, dividend_yield=0.0, implied_vol=0.25, option_type="call")
    output = bs_model_price(inputs)
    assert output.price is not None
    assert output.price > 0
