"""
regime.py — Intraday market regime detection.

Classifies the current market state from 5-minute OHLCV bars using:
  - VWAP position (price vs VWAP on the day)
  - EMA alignment (9 / 21 / 50 periods)
  - Intraday realized volatility vs ATM implied vol
  - Range compression relative to recent ATR

Regime labels
-------------
  TREND UP       price > VWAP, EMAs aligned up (9 > 21 > 50)
  TREND DOWN     price < VWAP, EMAs aligned down (9 < 21 < 50)
  CHOP           mixed EMA alignment or price crossing VWAP
  COMPRESSION    realized vol < 60% of ATM IV and EMA stack flat
  EXTENDED UP    TREND UP + price > 2σ above VWAP
  EXTENDED DOWN  TREND DOWN + price > 2σ below VWAP
  UNKNOWN        insufficient data (< 5 bars)
"""

import numpy as np
import pandas as pd

from analyzer.data import get_history

# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_intraday(ticker: str, interval: str = "5m", period: str = "1d") -> pd.DataFrame:
    """
    Fetch intraday OHLCV bars from yfinance.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Raises ValueError if no data is returned.
    """
    df = get_history(ticker, interval=interval, period=period, auto_adjust=True, ttl_seconds=90)
    if df.empty:
        raise ValueError(f"No intraday data returned for {ticker} "
                         f"(interval={interval}, period={period})")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP = cumsum(typical_price × volume) / cumsum(volume).
    typical_price = (H + L + C) / 3.
    Returns a Series aligned to df's index.
    """
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum_vol  = df["Volume"].cumsum().replace(0, np.nan)
    return (typical * df["Volume"]).cumsum() / cum_vol


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (adjusted=False for standard EMA)."""
    return series.ewm(span=span, adjust=False).mean()


def compute_realized_vol(df: pd.DataFrame, bars_per_day: int = 78) -> float:
    """
    Intraday annualized realized vol from 5-minute log returns.

    bars_per_day: 78 bars in a 6.5-hour regular US session at 5-minute intervals.
    Returns annualized vol as a decimal, or NaN if fewer than 3 returns.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if len(log_ret) < 3:
        return np.nan
    return float(log_ret.std()) * np.sqrt(252 * bars_per_day)


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def detect_regime(df: pd.DataFrame, atm_iv: float | None = None) -> dict:
    """
    Classify current market regime from intraday OHLCV bars.

    Parameters
    ----------
    df      : DataFrame with Open/High/Low/Close/Volume columns (≥ 5 bars)
    atm_iv  : ATM implied vol in decimal (e.g. 0.168 for 16.8%), used to
              compute the RV/IV ratio.  Pass None to skip that comparison.

    Returns
    -------
    dict with keys:
      regime        — regime label string (see module docstring)
      vwap          — current VWAP value
      price         — last close
      vwap_pct_diff — (price − vwap) / vwap × 100
      ema9          — last EMA-9
      ema21         — last EMA-21
      ema50         — last EMA-50
      ema_aligned   — True (up), False (down), None (mixed)
      realized_vol  — annualized intraday RV (decimal, or None if insufficient)
      rv_iv_ratio   — RV / ATM IV (None if atm_iv not supplied)
      n_bars        — number of bars in df
      error         — present only when regime = UNKNOWN
    """
    if df is None or len(df) < 5:
        return {"regime": "UNKNOWN", "error": "Insufficient intraday data (< 5 bars)"}

    vwap_series = compute_vwap(df)
    close       = df["Close"]
    price       = float(close.iloc[-1])
    vwap_now    = float(vwap_series.iloc[-1])
    if np.isnan(vwap_now):
        vwap_now = price

    ema9  = float(compute_ema(close, 9).iloc[-1])
    ema21 = float(compute_ema(close, 21).iloc[-1])
    ema50 = float(compute_ema(close, 50).iloc[-1])

    rv = compute_realized_vol(df)

    vwap_diff_pct = (price - vwap_now) / vwap_now * 100 if vwap_now else 0.0

    # EMA alignment
    aligned_up   = ema9 > ema21 > ema50
    aligned_down = ema9 < ema21 < ema50
    ema_aligned  = True if aligned_up else (False if aligned_down else None)

    # VWAP σ-distance (using intraday std of VWAP as scale proxy)
    vwap_std = float(vwap_series.std())
    sigma_dist = abs(price - vwap_now) / vwap_std if vwap_std > 0 else 0.0

    # RV/IV ratio
    rv_iv_ratio: float | None = None
    if atm_iv is not None and atm_iv > 0 and not np.isnan(rv):
        rv_iv_ratio = rv / atm_iv

    # Flat EMA stack: all three EMAs within 0.2% of each other
    ema_flat = abs(ema9 - ema50) / price < 0.002

    # ── Classify ──────────────────────────────────────────────────────────
    above_vwap = price > vwap_now

    if rv_iv_ratio is not None and rv_iv_ratio < 0.60 and ema_flat:
        regime = "COMPRESSION"
    elif aligned_up and above_vwap and sigma_dist >= 2.0:
        regime = "EXTENDED UP"
    elif aligned_down and not above_vwap and sigma_dist >= 2.0:
        regime = "EXTENDED DOWN"
    elif aligned_up and above_vwap:
        regime = "TREND UP"
    elif aligned_down and not above_vwap:
        regime = "TREND DOWN"
    else:
        regime = "CHOP"

    return {
        "regime":        regime,
        "vwap":          round(vwap_now, 4),
        "price":         round(price, 4),
        "vwap_pct_diff": round(vwap_diff_pct, 3),
        "ema9":          round(ema9, 4),
        "ema21":         round(ema21, 4),
        "ema50":         round(ema50, 4),
        "ema_aligned":   ema_aligned,
        "realized_vol":  round(rv, 4) if not np.isnan(rv) else None,
        "rv_iv_ratio":   round(rv_iv_ratio, 3) if rv_iv_ratio is not None else None,
        "n_bars":        len(df),
    }
