from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analyzer.data import get_fast_price, get_history
from config.settings import Settings
from config.tickers import CONTRACT_SPECS, resolve_ticker
from core.resampling import recent_window, resample_ohlcv


@dataclass
class MarketDataBundle:
    symbol: str
    display_name: str
    spot: float | None
    daily: pd.DataFrame
    frames: dict[str, pd.DataFrame]
    contract_spec: dict | None
    errors: list[str] = field(default_factory=list)


def _trim(df: pd.DataFrame, bars: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return recent_window(df, bars=bars)


def fetch_market_data(symbol: str, settings: Settings) -> MarketDataBundle:
    yf_symbol, spec = resolve_ticker(symbol)
    display_name = (spec or CONTRACT_SPECS.get(yf_symbol) or {}).get("name", yf_symbol)
    errors: list[str] = []

    spot = get_fast_price(yf_symbol, ttl_seconds=max(settings.default_refresh_seconds, 10))

    daily = pd.DataFrame()
    frames: dict[str, pd.DataFrame] = {"1m": pd.DataFrame(), "5m": pd.DataFrame(), "15m": pd.DataFrame(), "1h": pd.DataFrame()}

    try:
        daily = get_history(yf_symbol, period="30d", interval="1d", auto_adjust=True, ttl_seconds=300)
        if spot is None and not daily.empty:
            spot = float(daily["Close"].iloc[-1])
    except Exception as exc:
        errors.append(f"Daily fetch failed: {exc}")

    try:
        frames["1m"] = _trim(
            get_history(yf_symbol, period="7d", interval="1m", auto_adjust=True, ttl_seconds=settings.default_refresh_seconds),
            settings.intraday_lookback_bars,
        )
    except Exception as exc:
        errors.append(f"1m fetch failed: {exc}")

    try:
        frames["5m"] = _trim(
            get_history(yf_symbol, period="10d", interval="5m", auto_adjust=True, ttl_seconds=settings.default_refresh_seconds),
            settings.intraday_lookback_bars,
        )
    except Exception as exc:
        errors.append(f"5m fetch failed: {exc}")

    if not frames["5m"].empty:
        frames["15m"] = _trim(resample_ohlcv(frames["5m"], "15min"), settings.intraday_lookback_bars)

    try:
        frames["1h"] = _trim(
            get_history(yf_symbol, period="90d", interval="1h", auto_adjust=True, ttl_seconds=max(settings.default_refresh_seconds * 2, 60)),
            settings.intraday_lookback_bars,
        )
    except Exception as exc:
        errors.append(f"1h fetch failed: {exc}")

    for timeframe in ("1m", "5m", "15m", "1h"):
        frame = frames.get(timeframe)
        if frame is not None and not frame.empty and spot is None:
            spot = float(frame["Close"].iloc[-1])

    return MarketDataBundle(
        symbol=yf_symbol,
        display_name=display_name,
        spot=spot,
        daily=daily,
        frames=frames,
        contract_spec=spec,
        errors=errors,
    )
