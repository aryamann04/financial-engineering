import numpy as np
from datetime import datetime
from scipy.stats import norm

from .pricing.pricing import bs_price, binom_price
from .pricing.montecarlo import monte_carlo_european
from options.utilities.marketdata import MarketDataFetcher
from options.volatility.iv import bs_iv
from options.utilities.printer import print_option_summary

class Option:
    def __init__(self, ticker, r, T, K, n, sigma=None, option_type="call", position="long", 
                 creation_date=None, fetcher=None):
        self.ticker = ticker
        self.r = r
        self.T = T
        self.K = K
        self.n = n

        self.option_type = option_type.lower()
        self.position = position 
        self.creation_date = creation_date

        # avoid repeatedly querying api 
        self.fetcher = fetcher if fetcher else MarketDataFetcher(ticker, T, creation_date)

        self.S_0 = self.fetcher.current_price()
        self.q = self.fetcher.dividend_yield()
        # self.sigma = sigma if sigma else self.fetcher.historical_volatility()
        self.sigma = self.fetcher.historical_volatility() 
        self.market_iv = self.fetcher.market_iv(K, T, option_type)

        self.greeks = self.calculate_greeks(self.S_0, self.K, self.T, self.r, self.sigma, self.q, self.option_type)

    def price_summary(self):
        prices = {}
        prices["Black-Scholes"] = self.bs_price
        prices["Binomial European"] = self.binom_european
        prices["Binomial American"] = self.binom_american
        prices["Monte Carlo"] = self.monte_carlo_price
        prices["Market Price"] = self.market

        # placeholder
        prices["Model Vol"] = self.sigma 
        prices["Implied Vol"] = self.bs_iv

        return prices

    def summary(self): 
        print_option_summary(self)
    
    @property
    def bs_price(self):
        return bs_price(self.S_0, self.K, self.T, self.r, self.sigma, self.q, self.option_type)

    @property
    def binom_european(self):
        return binom_price(self.S_0, self.K, self.T, self.r, self.sigma, self.q, self.n, self.option_type, american=False)

    @property
    def binom_american(self):
        return binom_price(self.S_0, self.K, self.T, self.r, self.sigma, self.q, self.n, self.option_type, american=True)

    @property
    def monte_carlo_price(self):
        return monte_carlo_european(self.S_0, self.K, self.T, self.r, self.q, self.sigma, self.option_type)

    @property
    def market(self):
        if self.creation_date is None:
            market, _ = self.fetcher.actual_option_price(self.K, self.T, self.option_type)
            return market
        return np.nan

    @property
    def bs_iv(self):
        if self.market and not np.isnan(self.market):
            return bs_iv(self.market, self.S_0, self.K, self.T, self.r, self.q, self.option_type)
        return "N/A"
    
    def calculate_greeks(self, S, K, T, r, sigma, q, option_type):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        greeks = {}

        if option_type == "call":
            greeks["delta"] = np.exp(-q * T) * norm.cdf(d1)
            greeks["rho"] = (K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
            greeks["theta"] = (
                - (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                - q * S * np.exp(-q * T) * norm.cdf(d1)
                + r * K * np.exp(-r * T) * norm.cdf(d2)
            ) / 365
        else:
            greeks["delta"] = -np.exp(-q * T) * norm.cdf(-d1)
            greeks["rho"] = (-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
            greeks["theta"] = (
                - (S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
                + q * S * np.exp(-q * T) * norm.cdf(-d1)
                - r * K * np.exp(-r * T) * norm.cdf(-d2)
            ) / 365

        greeks["gamma"] = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        greeks["vega"] = (S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)) / 100

        return greeks

# helper for strategies
def create_option(ticker, r, T, K, n, sigma=None, option_type="call", position="long", 
                 creation_date=None, fetcher=None):
    return Option(ticker, r, T, K, n, sigma, option_type, position, creation_date, fetcher)