import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketDataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self._ticker_obj = yf.Ticker(self.ticker)
        self.close_prices = self.get_close_prices()

    def get_close_prices(self, reference_date=None):
        if reference_date is not None:
            reference_date = pd.to_datetime(reference_date)
            start_date = reference_date - pd.Timedelta(days=365)
            hist = self._ticker_obj.history(start=start_date, end=reference_date)
        else:
            hist = self._ticker_obj.history(period="1y")

        if hist.empty:
            raise ValueError(f"No data available for {self.ticker}.")

        return hist['Close']
    
    def current_price(self):
        return self.close_prices.iloc[-1]
    
    def historical_volatility(self):
        log_returns = np.log(self.close_prices / self.close_prices.shift(1))
        sigma = np.std(log_returns) * np.sqrt(252)
        return sigma

    def dividend_yield(self):
        try:
            yield_ = self._ticker_obj.info.get('dividendYield', 0)
            return yield_ if yield_ is not None else 0
        except Exception:
            return 0

    def implied_volatility(self, K, T, option_type="call"):
        K = 5 * round(K / 5)
        exp_dates = self._ticker_obj.options
        if not exp_dates:
            return None

        target_expiry = datetime.now() + timedelta(days=T * 365)
        closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))
        try:
            options = self._ticker_obj.option_chain(closest_expiry)
            option_df = options.calls if option_type == "call" else options.puts
            row = option_df[option_df['strike'] == K]
            return row['impliedVolatility'].values[0] if not row.empty else None
        except Exception:
            return None

    def actual_option_price(self, K, T, option_type="call"):
        K = 5 * round(K / 5)
        exp_dates = self._ticker_obj.options
        if not exp_dates:
            return None, None

        target_expiry = datetime.now() + timedelta(days=T * 365)
        closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))
        try:
            options = self._ticker_obj.option_chain(closest_expiry)
            option_df = options.calls if option_type == "call" else options.puts
            row = option_df[option_df['strike'] == K]
            if not row.empty:
                return row['lastPrice'].values[0], closest_expiry
            return None, closest_expiry
        except Exception:
            return None, closest_expiry