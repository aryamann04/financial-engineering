from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from analyzer.data import get_fast_price, get_history
from config.tickers import resolve_ticker


@dataclass
class FuturesData:
    yf_symbol: str
    display_name: str
    spot: float | None
    daily_df: pd.DataFrame      # last ~10 daily bars
    intraday_5m: pd.DataFrame   # last 5 days of 5m bars
    intraday_1h: pd.DataFrame   # last 60 days of 1h bars (for FVG on higher TF)
    contract_spec: dict | None
    errors: list[str] = field(default_factory=list)

def fetch_futures_data(user_input: str) -> FuturesData:
    """
    Resolve ticker, then fetch daily and 5m intraday OHLCV from yfinance.
    Gracefully degrades — errors are collected rather than raised.
    """
    yf_symbol, spec = resolve_ticker(user_input)
    display_name = spec["name"] if spec else yf_symbol
    errors: list[str] = []

    # --- Spot price ---
    spot: float | None = get_fast_price(yf_symbol)

    # --- Daily bars ---
    daily_df = pd.DataFrame()
    try:
        daily_df = get_history(yf_symbol, period="10d", interval="1d", auto_adjust=True, ttl_seconds=300)
        if daily_df.empty:
            errors.append("No daily OHLCV data — yfinance returned nothing.")
        elif spot is None and not daily_df.empty:
            spot = float(daily_df["Close"].iloc[-1])
    except Exception as exc:
        errors.append(f"Daily fetch failed: {exc}")

    # --- Intraday 5m ---
    intraday_5m = pd.DataFrame()
    try:
        intraday_5m = get_history(yf_symbol, period="5d", interval="5m", auto_adjust=True, ttl_seconds=90)
        if intraday_5m.empty:
            errors.append(
                "No 5m intraday data — yfinance may not support this symbol/interval. "
                "Session levels and ATR will be unavailable."
            )
        elif spot is None and not intraday_5m.empty:
            spot = float(intraday_5m["Close"].iloc[-1])
    except Exception as exc:
        errors.append(f"Intraday 5m fetch failed: {exc}")

    # --- Intraday 1h (for higher-timeframe FVG detection) ---
    intraday_1h = pd.DataFrame()
    try:
        intraday_1h = get_history(yf_symbol, period="60d", interval="1h", auto_adjust=True, ttl_seconds=180)
    except Exception as exc:
        errors.append(f"Intraday 1h fetch failed: {exc}")

    return FuturesData(
        yf_symbol=yf_symbol,
        display_name=display_name,
        spot=spot,
        daily_df=daily_df,
        intraday_5m=intraday_5m,
        intraday_1h=intraday_1h,
        contract_spec=spec,
        errors=errors,
    )
