import yfinance as yf
from curl_cffi import requests 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from options.volatility.marketvols import get_yf_iv, yf_option_price, plot_vol_skew

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
        return get_yf_iv(self.yfTicker, K, T, option_type)
    
    # yfinance option price
    def actual_option_price(self, K, T, option_type="call"):
        return yf_option_price(self.yfTicker, K, T, option_type)
    
    def plot_implied_vols(self, r=0): 
        plot_vol_skew(self.yfTicker, self.current_price(), self.T, r, self.dividend_yield(), option_type="call")