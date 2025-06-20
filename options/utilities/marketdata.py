import yfinance as yf
from curl_cffi import requests 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MarketDataFetcher:
    def __init__(self, ticker, T, creation_date=None):

        self.session = requests.Session(impersonate="chrome")
        self.yfTicker = yf.Ticker(ticker, session=self.session)
        self.ticker = ticker.upper()
        self.T = T
        self.creation_date = pd.to_datetime(creation_date) if creation_date is not None else None
        self.close_prices = self.get_close_prices()

    def get_close_prices(self):
        num_days = int(self.T * 365)  
        
        if self.creation_date is not None: 
            end = self.creation_date
            start = end - timedelta(days=num_days)  
        else:
            # default option is created today 
            end = datetime.today()
            start = end - timedelta(days=num_days)
        
        hist = self.yfTicker.history(start=start, end=end)

        if hist.empty:
            raise ValueError(f"No data available for {self.ticker} between {start} and {end}.")

        return hist['Close']
    
    def current_price(self):
        return self.close_prices.iloc[-1]
    
    def historical_volatility(self):
        log_returns = np.log(self.close_prices / self.close_prices.shift(1))
        sigma = np.std(log_returns) * np.sqrt(252)  # annualize
        return sigma

    def dividend_yield(self):
        try:
            yield_ = self.yfTicker.info.get('dividendYield', 0) / 100
            return yield_ if yield_ is not None else 0
        except Exception:
            return 0

    # yfinance implied volatility 
    def market_iv(self, K, T, option_type="call"):
        K = 5 * round(K / 5)
        exp_dates = self.yfTicker.options
        if not exp_dates:
            return None

        target_expiry = datetime.now() + timedelta(days=T * 365)
        closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))
        try:
            options = self.yfTicker.option_chain(closest_expiry)
            option_df = options.calls if option_type == "call" else options.puts
            row = option_df[option_df['strike'] == K]
            return row['impliedVolatility'].values[0] if not row.empty else None
        except Exception:
            return None

    def actual_option_price(self, K, T, option_type="call"):
        K = 5 * round(K / 5)
        exp_dates = self.yfTicker.options
        if not exp_dates:
            return None, None

        target_expiry = datetime.now() + timedelta(days=T * 365)
        closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))
        try:
            options = self.yfTicker.option_chain(closest_expiry)
            option_df = options.calls if option_type == "call" else options.puts
            row = option_df[option_df['strike'] == K]
            if not row.empty:
                return row['lastPrice'].values[0], closest_expiry
            return None, closest_expiry
        except Exception:
            return None, closest_expiry