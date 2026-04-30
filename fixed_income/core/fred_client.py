"""
fred_client.py — Thin, cached wrapper around the FRED API.

Usage
-----
    from fixed_income.core.fred_client import get_fred, fred_latest, fred_series

    # one-time setup (or set FRED_API_KEY env var)
    get_fred(api_key="abc123")

    rate = fred_latest("SOFR")           # float in original FRED units
    cpi  = fred_series("CPIAUCSL", n=13) # pd.Series (latest 13 obs, sorted asc)

API key priority
----------------
  1. Explicit api_key argument to get_fred()
  2. FRED_API_KEY environment variable
  3. If neither is set, all calls return None silently (fallback path)

Caching
-------
  Each (series_id, ttl_bucket) is cached via lru_cache so the live API is
  hit at most once per TTL window regardless of how many callers ask for
  the same series.

  Default TTL windows:
    fred_latest()  → 3600s (1 hour)  — macro data rarely updates intraday
    fred_series()  → 3600s (1 hour)
"""

from __future__ import annotations

import functools
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global Fred instance — created lazily on first use
# ---------------------------------------------------------------------------

_fred_instance = None


def get_fred(api_key: str | None = None):
    """
    Return (and cache) a fredapi.Fred instance.

    Call this once at startup to register your API key.  After that, all
    fred_* helpers use the same instance automatically.
    """
    global _fred_instance
    if _fred_instance is not None and api_key is None:
        return _fred_instance

    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        return None

    try:
        from fredapi import Fred
        _fred_instance = Fred(api_key=key)
        return _fred_instance
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cached series fetchers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _fred_latest_cached(series_id: str, bucket: int) -> float | None:
    """Inner cached worker for fred_latest()."""
    fred = get_fred()
    if fred is None:
        return None
    try:
        s = fred.get_series(series_id)
        s = s.dropna()
        if s.empty:
            return None
        val = float(s.iloc[-1])
        return val if np.isfinite(val) else None
    except Exception:
        return None


@functools.lru_cache(maxsize=128)
def _fred_series_cached(series_id: str, n: int, bucket: int) -> bytes | None:
    """Inner cached worker for fred_series().  Serialised as pickle bytes."""
    import pickle
    fred = get_fred()
    if fred is None:
        return None
    try:
        s = fred.get_series(series_id)
        s = s.dropna()
        if s.empty:
            return None
        return pickle.dumps(s.iloc[-n:] if len(s) >= n else s)
    except Exception:
        return None


def fred_latest(series_id: str, ttl_seconds: int = 3600) -> float | None:
    """
    Return the most recent (non-NaN) observation for a FRED series.

    Returns None if no API key is configured or the fetch fails.
    Value is in FRED's native units (percent for rates/spreads, level for CPI, etc.).
    """
    bucket = int(time.time()) // ttl_seconds
    return _fred_latest_cached(series_id, bucket)


def fred_series(series_id: str, n: int = 13,
                ttl_seconds: int = 3600) -> pd.Series | None:
    """
    Return the latest n non-NaN observations as a pd.Series (date-indexed,
    sorted ascending).  Returns None if unavailable.
    """
    import pickle
    bucket = int(time.time()) // ttl_seconds
    raw = _fred_series_cached(series_id, n, bucket)
    if raw is None:
        return None
    try:
        return pickle.loads(raw)
    except Exception:
        return None


def fred_yoy(series_id: str, ttl_seconds: int = 3600) -> float | None:
    """
    Return the year-over-year percent change for a monthly level series
    (e.g. CPI, payrolls).  Fetches 13 months and computes (latest/year_ago - 1)*100.
    Returns None if data is insufficient.
    """
    s = fred_series(series_id, n=14, ttl_seconds=ttl_seconds)
    if s is None or len(s) < 13:
        return None
    try:
        latest    = float(s.iloc[-1])
        year_ago  = float(s.iloc[-13])
        if year_ago == 0:
            return None
        return (latest / year_ago - 1.0) * 100.0
    except Exception:
        return None


def fred_mom(series_id: str, ttl_seconds: int = 3600) -> float | None:
    """
    Return the month-over-month change (level units) for a series.
    Useful for payroll change, initial claims change, etc.
    """
    s = fred_series(series_id, n=3, ttl_seconds=ttl_seconds)
    if s is None or len(s) < 2:
        return None
    try:
        return float(s.iloc[-1]) - float(s.iloc[-2])
    except Exception:
        return None
