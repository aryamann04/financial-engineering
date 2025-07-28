import numpy as np
from options.utilities.marketdata import MarketDataFetcher
from options.core.pricing.montecarlo import monte_carlo_asian
from fixed_income.core.bootstrap import zc_yield

class AsianOption:
    def __init__(self, ticker, T, K, sigma=None, option_type="call"):
        self.ticker = ticker.upper()
        self.r = zc_yield(T)
        self.T = T
        self.K = float(K)
        self.option_type = option_type

        self.fetcher = MarketDataFetcher(ticker, T)
        self.S0 = self.fetcher.current_price()
        self.sigma = sigma if sigma else self.fetcher.historical_volatility()
        self.q = self.fetcher.dividend_yield()

        self.mc = monte_carlo_asian(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type)

    def price(self):
        print(f"{self.ticker} Asian {self.option_type.capitalize()} Option with strike {self.K}:")
        print(f"Monte Carlo: ${self.mc:.2f}")