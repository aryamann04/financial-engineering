from __future__ import annotations

import pandas as pd

from config.settings import load_settings
from core.analysis import TradeEngine
from core.assistant import parse_intent
from core.data import MarketDataBundle
from core.liquidity import KeyLevel, detect_liquidity_sweeps
from core.structure import detect_swings


def _make_df(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None, freq: str = "5min") -> pd.DataFrame:
    n = len(closes)
    highs = highs or [price + 0.4 for price in closes]
    lows = lows or [price - 0.4 for price in closes]
    idx = pd.date_range("2024-01-10 09:30", periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": [1000] * n,
        },
        index=idx,
    )


def test_detect_swings_finds_highs_and_lows():
    closes = [100, 101, 103, 101, 99, 98, 100, 102, 101]
    highs = [100.5, 101.5, 103.6, 101.2, 99.6, 98.4, 100.6, 102.7, 101.4]
    lows = [99.5, 100.4, 102.2, 100.0, 98.6, 97.5, 99.3, 101.1, 100.5]
    swings = detect_swings(_make_df(closes, highs=highs, lows=lows), lookback=1)
    assert any(swing.kind == "high" for swing in swings)
    assert any(swing.kind == "low" for swing in swings)


def test_detect_liquidity_sweep_marks_bearish_rejection():
    df = _make_df(
        [100.0, 100.4, 100.1, 99.8],
        highs=[100.4, 101.8, 100.5, 100.1],
        lows=[99.7, 99.9, 99.8, 99.5],
    )
    levels = [KeyLevel("Previous Day High", 101.0, "previous_day_high", "1d", "daily bars")]
    events = detect_liquidity_sweeps(df, levels, atr=1.0, lookback_bars=4)
    assert any(event.direction == "bearish" for event in events)


def test_assistant_intent_router():
    assert parse_intent("What is the current bias on M6E?") == "bias"
    assert parse_intent("Where are the nearest downside liquidity levels?") == "liquidity"
    assert parse_intent("What are the relevant macro rates right now?") == "macro"


def test_settings_load_env(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fred-test")
    monkeypatch.setenv("DEFAULT_SYMBOL", "M6E=F")
    monkeypatch.setenv("DEFAULT_REFRESH_SECONDS", "12")
    settings = load_settings()
    assert settings.fred_api_key == "fred-test"
    assert settings.default_symbol == "M6E=F"
    assert settings.default_refresh_seconds == 12


def test_trade_engine_analysis_schema(monkeypatch):
    settings = load_settings()
    closes = [100 + i * 0.2 for i in range(80)]
    frame_5m = _make_df(closes)
    frame_1m = _make_df(closes[-60:], freq="1min")
    frame_15m = frame_5m.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    frame_1h = _make_df([100 + i * 0.4 for i in range(80)], freq="1h")
    daily = pd.DataFrame(
        {
            "Open": [99.0, 100.0, 101.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [98.0, 99.0, 100.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-08", periods=3, freq="1d", tz="America/New_York"),
    )

    bundle = MarketDataBundle(
        symbol="M6E=F",
        display_name="Micro Euro FX",
        spot=float(frame_5m["Close"].iloc[-1]),
        daily=daily,
        frames={"1m": frame_1m, "5m": frame_5m, "15m": frame_15m, "1h": frame_1h},
        contract_spec=None,
        errors=[],
    )

    monkeypatch.setattr("core.analysis.fetch_market_data", lambda symbol, settings: bundle)
    engine = TradeEngine(settings=settings)
    analysis = engine.analyze("M6E", timeframe="5m").to_dict()
    assert analysis["symbol"] == "M6E=F"
    assert analysis["setup"]["type"]
    assert "bias" in analysis
    assert "macro_context" in analysis
    assert "snapshots" in analysis
