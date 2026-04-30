from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests

    _HAS_CFFI = True
except ImportError:
    _HAS_CFFI = False

from fixed_income.core.fred_client import fred_latest, fred_series, fred_yoy, fred_mom


def _bucket(ttl_seconds: int) -> int:
    ttl = max(int(ttl_seconds), 1)
    return int(time.time() // ttl)


def _make_session():
    if _HAS_CFFI:
        return cffi_requests.Session(impersonate="chrome")
    return None


_SESSION = _make_session()


@lru_cache(maxsize=256)
def _ticker_cached(symbol: str) -> yf.Ticker:
    symbol = symbol.upper().strip()
    if _SESSION is not None:
        return yf.Ticker(symbol, session=_SESSION)
    return yf.Ticker(symbol)


def get_ticker(symbol: str) -> yf.Ticker:
    return _ticker_cached(symbol)


@lru_cache(maxsize=1024)
def _history_cached(
    symbol: str,
    period: str,
    interval: str,
    auto_adjust: bool,
    prepost: bool,
    bucket: int,
) -> pd.DataFrame:
    ticker = get_ticker(symbol)
    df = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        prepost=prepost,
    )
    return df


def get_history(
    symbol: str,
    *,
    period: str,
    interval: str,
    auto_adjust: bool = True,
    prepost: bool = False,
    ttl_seconds: int = 120,
) -> pd.DataFrame:
    df = _history_cached(symbol, period, interval, auto_adjust, prepost, _bucket(ttl_seconds))
    return df.copy() if hasattr(df, "copy") else df


def get_fast_price(symbol: str, ttl_seconds: int = 30) -> float | None:
    return _get_fast_price_cached(symbol.upper().strip(), _bucket(ttl_seconds))


@lru_cache(maxsize=512)
def _get_fast_price_cached(symbol: str, bucket: int) -> float | None:
    ticker = get_ticker(symbol)
    for attr in ("last_price", "previous_close"):
        try:
            value = getattr(ticker.fast_info, attr)
            if value:
                val = float(value)
                if val > 0:
                    return val
        except Exception:
            continue
    try:
        hist = get_history(symbol, period="5d", interval="1d", ttl_seconds=300)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class OptionChainSnapshot:
    expiry: str
    calls: pd.DataFrame
    puts: pd.DataFrame


@lru_cache(maxsize=256)
def _option_chain_cached(symbol: str, expiry: str, bucket: int) -> OptionChainSnapshot:
    ticker = get_ticker(symbol)
    chain = ticker.option_chain(expiry)
    return OptionChainSnapshot(expiry=expiry, calls=chain.calls.copy(), puts=chain.puts.copy())


def resolve_expiry_for_days(symbol: str, t_days: int) -> str:
    ticker = get_ticker(symbol)
    expiries = list(ticker.options or [])
    if not expiries:
        raise ValueError(f"No options expirations available for {symbol}.")
    target = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=max(int(t_days), 1))
    return min(expiries, key=lambda x: abs(pd.Timestamp(x) - target))


def get_option_chain(symbol: str, *, t_days: int, ttl_seconds: int = 180) -> OptionChainSnapshot:
    expiry = resolve_expiry_for_days(symbol, t_days)
    snapshot = _option_chain_cached(symbol.upper().strip(), expiry, _bucket(ttl_seconds))
    return OptionChainSnapshot(
        expiry=snapshot.expiry,
        calls=snapshot.calls.copy(),
        puts=snapshot.puts.copy(),
    )


@lru_cache(maxsize=512)
def _news_cached(symbol: str, bucket: int):
    ticker = get_ticker(symbol)
    try:
        return list(ticker.news or [])
    except Exception:
        return []


def get_news(symbol: str, ttl_seconds: int = 300) -> list[dict]:
    return list(_news_cached(symbol.upper().strip(), _bucket(ttl_seconds)))


def get_fred_latest(series_id: str, ttl_seconds: int = 3600) -> float | None:
    return fred_latest(series_id, ttl_seconds=ttl_seconds)


def get_fred_series(series_id: str, n: int = 13, ttl_seconds: int = 3600):
    return fred_series(series_id, n=n, ttl_seconds=ttl_seconds)


def get_fred_yoy(series_id: str, ttl_seconds: int = 3600) -> float | None:
    return fred_yoy(series_id, ttl_seconds=ttl_seconds)


def get_fred_mom(series_id: str, ttl_seconds: int = 3600) -> float | None:
    return fred_mom(series_id, ttl_seconds=ttl_seconds)


def cache_dir() -> Path:
    path = Path.home() / ".financial-engineering" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
