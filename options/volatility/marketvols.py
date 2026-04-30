"""
marketvols.py — option chain fetching, IV extraction, and vol surface helpers.

Key design: fetch_chain() resolves the expiry and pulls the option chain ONCE,
returning a (calls_df, puts_df, expiry_str) triple.  All downstream helpers
(get_bid_asks, compute_bs_ivs, plot_vol_skew) accept this cached triple so the
caller pays exactly one network round-trip per ticker/expiry, not one per strike.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from options.volatility.iv import bs_iv


# ---------------------------------------------------------------------------
# Expiry / strike resolution helpers
# ---------------------------------------------------------------------------

def _resolve_expiry(ticker_obj, T: float) -> str:
    """Return the expiry date string closest to T years from today."""
    exp_dates = ticker_obj.options
    if not exp_dates:
        raise ValueError(f"No options data available for {ticker_obj.ticker}")
    target = datetime.now() + timedelta(days=T * 365)
    return min(exp_dates, key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - target))


def closest_K_T(ticker_obj, K, T):
    """
    Return (closest_strike, expiry_str) for a given ticker, strike, and maturity.
    If K is None, returns just the expiry_str (used by callers that only need the date).
    Accepts a ticker string or a yfinance Ticker object.
    """
    if isinstance(ticker_obj, str):
        import yfinance as yf
        from curl_cffi import requests
        ticker_obj = yf.Ticker(ticker_obj, session=requests.Session(impersonate="chrome"))

    try:
        expiry = _resolve_expiry(ticker_obj, T)
    except Exception:
        return (K, T) if K is not None else T

    if K is None:
        return expiry

    try:
        chain = ticker_obj.option_chain(expiry)
        all_strikes = np.unique(
            np.concatenate([chain.calls["strike"].values, chain.puts["strike"].values])
        )
        if all_strikes.size == 0:
            return K, expiry
        closest = float(min(all_strikes, key=lambda s: abs(s - K)))
        return closest, expiry
    except Exception:
        return K, expiry


# ---------------------------------------------------------------------------
# Core: single-fetch chain helper
# ---------------------------------------------------------------------------

def fetch_chain(ticker_obj, T: float, min_strikes: int = 5):
    """
    Fetch the option chain closest to T years from today.

    Returns:
        calls_df  : pd.DataFrame of calls (from yfinance option_chain)
        puts_df   : pd.DataFrame of puts
        expiry_str: str  e.g. "2025-06-20"
        warnings  : list[str]  non-fatal data quality notes

    The data is filtered to strikes with openInterest > 0 OR volume > 0 (softer
    than the prior AND filter, which crashed on illiquid chains).  If fewer than
    min_strikes survive the filter we fall back to the unfiltered chain.
    """
    warnings_out = []

    expiry_str = _resolve_expiry(ticker_obj, T)
    chain = ticker_obj.option_chain(expiry_str)

    calls_raw = chain.calls.copy()
    puts_raw  = chain.puts.copy()

    def _filter(df, label):
        # Soft filter: keep strikes with any open interest or volume
        liquid = df[(df["openInterest"] > 0) | (df["volume"] > 0)]
        if len(liquid) >= min_strikes:
            return liquid.reset_index(drop=True)
        # Fall back to unfiltered if too few liquid strikes survive
        if len(liquid) < min_strikes and len(df) >= min_strikes:
            warnings_out.append(
                f"{label}: only {len(liquid)} liquid strike(s); using unfiltered chain"
            )
            return df.reset_index(drop=True)
        warnings_out.append(f"{label}: very few strikes available ({len(df)})")
        return df.reset_index(drop=True)

    calls_df = _filter(calls_raw, "calls")
    puts_df  = _filter(puts_raw,  "puts")

    return calls_df, puts_df, expiry_str, warnings_out


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Price extraction helper
# ---------------------------------------------------------------------------

def _mid_price(row) -> float:
    """
    Extract a usable mid price from an option chain row.

    Primary: (bid + ask) / 2  when both are positive.
    Fallback: lastPrice        when bid/ask are zero or null (common after market close).
    Returns 0.0 if no valid quote is available.
    """
    bid  = float(row.get("bid",       0) or 0)
    ask  = float(row.get("ask",       0) or 0)
    last = float(row.get("lastPrice", 0) or 0)

    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    # After-hours: bid=0 / ask=0 but lastPrice is still valid
    if last > 0:
        return last
    return 0.0


# ---------------------------------------------------------------------------
# IV extraction from a fetched chain
# ---------------------------------------------------------------------------

def compute_bs_ivs(df: pd.DataFrame, S_0: float, T: float, r: float, q: float,
                   option_type: str) -> tuple:
    """
    Compute Black-Scholes implied vols for all strikes in a chain dataframe.

    Reads mid prices from the dataframe (no additional API calls).
    Uses lastPrice as fallback when bid/ask are zero (after-hours).
    Returns (strikes_arr, bs_ivs_arr) — parallel arrays, NaN-free.
    """
    strikes, ivs = [], []
    for _, row in df.iterrows():
        K   = float(row["strike"])
        mid = _mid_price(row)

        if mid <= 0:
            continue

        iv = bs_iv(mid, S_0, K, T, r, q, option_type)
        if iv is not None and np.isfinite(iv) and iv > 0:
            strikes.append(K)
            ivs.append(iv)

    return np.array(strikes), np.array(ivs)


# ---------------------------------------------------------------------------
# Helpers used by the Analyzer / Option
# ---------------------------------------------------------------------------

def yf_strikes(ticker_obj, T: float):
    """Return available call strikes for the closest expiry to T."""
    expiry = _resolve_expiry(ticker_obj, T)
    chain  = ticker_obj.option_chain(expiry)
    return chain.calls["strike"].values


def get_yf_iv(ticker_obj, K: float, T: float, option_type: str = "call"):
    """Look up yfinance's own implied vol for a single strike/maturity."""
    try:
        strike, expiry = closest_K_T(ticker_obj, K, T)
        options = ticker_obj.option_chain(expiry)
        df = options.calls if option_type == "call" else options.puts
        row = df[df["strike"] == strike]
        return float(row["impliedVolatility"].values[0]) if not row.empty else None
    except Exception:
        return None


def yf_option_price(ticker_obj, K: float, T: float, option_type: str = "call",
                    bidask: bool = False):
    """
    Return the mid price (or bid/ask tuple) for the closest listed strike/expiry.
    Falls back to lastPrice when bid/ask are zero (after market close).
    """
    try:
        strike, expiry = closest_K_T(ticker_obj, K, T)
        options = ticker_obj.option_chain(expiry)
        df = options.calls if option_type == "call" else options.puts
        row = df[df["strike"] == strike]
        if not row.empty:
            r   = row.iloc[0]
            bid = float(r["bid"] or 0)
            ask = float(r["ask"] or 0)
            if bidask:
                return bid, ask
            mid = _mid_price(r)
            return (mid if mid > 0 else None), expiry
        return (None, expiry) if not bidask else (None, None)
    except Exception:
        return (None, None) if not bidask else (None, None)


def get_bid_asks(ticker_obj, T: float, strikes, option_type: str):
    """
    Return (bids, asks) for the given strikes using a single option_chain fetch.

    Previous implementation called yf_option_price (→ option_chain) once per
    strike = O(N) network round-trips.  Now: 1 fetch, O(1) dict lookups.
    """
    bids, asks = [], []
    try:
        expiry = _resolve_expiry(ticker_obj, T)
        chain  = ticker_obj.option_chain(expiry)
        df     = chain.calls if option_type == "call" else chain.puts
        # Index by strike for fast lookup
        indexed = df.set_index("strike")
    except Exception:
        # If the chain fetch itself fails, return Nones
        return [None] * len(strikes), [None] * len(strikes)

    for K in strikes:
        if K in indexed.index:
            row = indexed.loc[K]
        else:
            # Nearest listed strike as fallback
            nearest_K = float(
                df.iloc[(df["strike"] - K).abs().argsort().iloc[0]]["strike"]
            )
            row = indexed.loc[nearest_K] if nearest_K in indexed.index else None

        if row is not None:
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            # After-hours fallback: if bid/ask both zero, use lastPrice for both
            if bid == 0 and ask == 0:
                last = float(row.get("lastPrice", 0) or 0)
                bid = ask = last if last > 0 else None
        else:
            bid = ask = None

        bids.append(bid)
        asks.append(ask)

    return bids, asks


# ---------------------------------------------------------------------------
# Volatility surface plot (uses pre-fetched chain; no per-strike API calls)
# ---------------------------------------------------------------------------

def plot_vol_skew(ticker_obj, S_0: float, T: float, r: float, q: float,
                  option_type: str = "call", plot: bool = True):
    """
    Plot (or return) the implied vol skew for a given ticker and maturity.

    With plot=False returns (strikes_yf, vols_yf_pct, yf_prices) for use by
    SVI calibration — same signature as before, but now fetches the chain once.
    """
    # ---- resolve expiry ----
    try:
        expiry_str = _resolve_expiry(ticker_obj, T)
    except ValueError as e:
        if not plot:
            return [], [], []
        raise

    # ---- fetch chain once ----
    try:
        chain = ticker_obj.option_chain(date=expiry_str)
    except Exception:
        if not plot:
            return [], [], []
        raise RuntimeError(f"Failed to fetch option chain for {expiry_str}")

    options_df = chain.calls if option_type == "call" else chain.puts

    # Soft filter (OR condition; see fetch_chain docstring)
    liquid = options_df[(options_df["volume"] > 0) | (options_df["openInterest"] > 0)]
    if len(liquid) >= 5:
        options_df = liquid.reset_index(drop=True)

    # Remove clearly stale rows (zero implied vol from yfinance)
    options_df = options_df[options_df["impliedVolatility"] > 0].reset_index(drop=True)

    if options_df.empty:
        if not plot:
            return [], [], []
        raise ValueError(f"No option data with valid IV for {ticker_obj.ticker} at {expiry_str}")

    strikes          = options_df["strike"].values
    implied_vols_yf  = options_df["impliedVolatility"].values  # decimal

    # yfinance IVs (in percent for plot/SVI consumption)
    filtered_strikes_yf = strikes.tolist()
    filtered_vols_yf    = (implied_vols_yf * 100).tolist()

    # ---- BS IVs from mid prices (no extra API calls) ----
    # Use the already-fetched dataframe; read bid/ask directly.
    yf_prices         = []
    implied_vols_bs   = []
    filtered_strikes_bs = []

    options_indexed = options_df.set_index("strike")
    for K in strikes:
        if K not in options_indexed.index:
            continue
        row = options_indexed.loc[K]
        mid = _mid_price(row)
        if mid <= 0:
            continue
        iv = bs_iv(mid, S_0, K, T, r, q, option_type)
        if iv is not None and np.isfinite(iv) and iv > 0:
            yf_prices.append(mid)
            filtered_strikes_bs.append(K)
            implied_vols_bs.append(iv * 100)

    if not plot:
        return filtered_strikes_yf, filtered_vols_yf, yf_prices

    # ---- Plot ----
    _, ax = plt.subplots()
    ax.plot(filtered_strikes_yf, filtered_vols_yf,
            label="yfinance implied vol", marker="o", linestyle="-", color="blue")
    ax.plot(filtered_strikes_bs, implied_vols_bs,
            label="black-scholes implied vol", marker="o", linestyle="--", color="orange")
    ax.axvline(x=S_0, color="black", linestyle="--", label="current price")
    ax.set_xlabel("strikes")
    ax.set_ylabel("implied vols (%)")
    ax.set_title(f"Vol Skew — {ticker_obj.ticker} {option_type}s  {expiry_str}")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
