import numpy as np
from options.utilities.marketdata import MarketDataFetcher
from options.core.pricing.pricing import monte_carlo_asian

class AsianOption:
    def __init__(self, ticker, r, T, K, option_type="call"):
        self.ticker = ticker.upper()
        self.r = r
        self.T = T
        self.K = float(K)
        self.option_type = option_type

        self.data = MarketDataFetcher(ticker)
        self.S0, self.sigma = self.data.current_price_and_vol()
        self.q = self.data.dividend_yield()

        self.mc = monte_carlo_asian(self.S0, self.K, self.T, self.r, self.q, self.sigma, self.option_type)

    def price(self):
        print(f"{self.ticker} Asian {self.option_type.capitalize()} Option with strike {self.K}:")
        print(f"Monte Carlo: ${self.mc:.2f}")