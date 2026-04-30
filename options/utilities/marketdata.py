import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analyzer.data import get_history, get_ticker
from options.volatility.marketvols import get_yf_iv, yf_option_price, plot_vol_skew, yf_strikes


class MarketDataFetcher:
    def __init__(self, ticker, T, creation_date=None):
        self.yfTicker   = get_ticker(ticker)
        self.ticker     = ticker.upper()
        self.T          = T
        self.creation_date = (
            pd.to_datetime(creation_date) if creation_date is not None else None
        )

        # Spot price: use fast_info (no history download required).
        # close_prices is populated lazily on first call to historical_volatility().
        self._spot_price: float | None  = None
        self._close_prices: pd.Series | None = None
        self._spot_price = self._fetch_spot()

    # ------------------------------------------------------------------
    # Spot price — fast path via fast_info, fallback to 5-day history
    # ------------------------------------------------------------------

    def _fetch_spot(self) -> float:
        """
        Return the current spot price as fast as possible.

        Priority:
          1. fast_info.last_price  — cached metadata, no OHLCV download
          2. fast_info.previous_close  — same source, day-old value
          3. 5-day daily history  — guaranteed to have at least one close
        """
        try:
            price = self.yfTicker.fast_info.last_price
            if price and np.isfinite(price) and price > 0:
                return float(price)
        except Exception:
            pass
        try:
            price = self.yfTicker.fast_info.previous_close
            if price and np.isfinite(price) and price > 0:
                return float(price)
        except Exception:
            pass
        # Minimal fallback: fetch only 5 days of daily data
        hist = get_history(self.ticker, period="5d", interval="1d", auto_adjust=True, ttl_seconds=300)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        raise ValueError(f"Cannot determine spot price for {self.ticker}")

    def current_price(self) -> float:
        return self._spot_price

    # ------------------------------------------------------------------
    # Historical close prices — fetched lazily
    # ------------------------------------------------------------------

    def get_close_prices(self) -> pd.Series:
        """
        Fetch T years of daily close prices.  Called lazily — only when
        historical_volatility() or the caller explicitly needs it.
        """
        if self._close_prices is not None:
            return self._close_prices

        num_days = max(int(self.T * 365), 30)   # at least 30 days for vol calc

        if self.creation_date is not None:
            end   = self.creation_date
            start = end - timedelta(days=num_days)
        else:
            end   = datetime.today()
            start = end - timedelta(days=num_days)

        hist = self.yfTicker.history(start=start, end=end)
        if hist.empty:
            raise ValueError(
                f"No price history for {self.ticker} between {start:%Y-%m-%d} "
                f"and {end:%Y-%m-%d}."
            )
        self._close_prices = hist["Close"]
        return self._close_prices

    @property
    def close_prices(self) -> pd.Series:
        """Back-compat property: triggers lazy fetch on first access."""
        return self.get_close_prices()

    def historical_volatility(self) -> float:
        closes  = self.get_close_prices()
        log_ret = np.log(closes / closes.shift(1))
        return float(log_ret.std() * np.sqrt(252))

    def dividend_yield(self) -> float:
        """
        Return the annual dividend yield as a decimal.

        yfinance info['dividendYield'] is already a decimal (e.g. 0.013 for 1.3%).
        Falls back to 0 gracefully.
        """
        try:
            val = self.yfTicker.info.get("dividendYield", 0) or 0
            # Sanity check: yf sometimes returns it in percent (> 0.2 is likely %)
            val = float(val)
            if val > 0.20:       # looks like a percent value, convert
                val /= 100.0
            return val if np.isfinite(val) else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Vol / price helpers (delegate to marketvols)
    # ------------------------------------------------------------------

    def market_iv(self, K, T, option_type="call"):
        return get_yf_iv(self.yfTicker, K, T, option_type)

    def actual_option_price(self, K, T, option_type="call"):
        return yf_option_price(self.yfTicker, K, T, option_type)

    def plot_implied_vols(self, r=0, option_type="call", plot=True):
        return plot_vol_skew(
            self.yfTicker, self.current_price(), self.T, r,
            self.dividend_yield(), option_type=option_type, plot=plot,
        )
