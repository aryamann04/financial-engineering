from scipy.stats import norm
import numpy as np
from tabulate import tabulate
from optionspricing import (stock_data,
                            div_yield,
                            bs_price,
                            binom_price,
                            actual_option_price,
                            implied_volatility,
                            print_option_price)
from svi import SVIModel
from montecarlo import monte_carlo_european

class Option:
    def __init__(self, ticker, r, T, K, n, option_type="call", position="long", creation_date=None):

        self.ticker = ticker
        self.r = r
        self.T = T
        self.K = K
        self.n = n
        self.option_type = option_type
        self.position = position

        self.creation_date = creation_date
        self.S_0, _ = stock_data(ticker, creation_date)

        strikes, market_vols = self._fetch_market_vol_data()
        if strikes and market_vols:
            svi_model = SVIModel()
            svi_model.fit(strikes, market_vols, self.S_0)
            self.sigma = svi_model.svi_volatility(np.log(self.K / self.S_0))
        else:
            self.sigma = _ , = stock_data(ticker, creation_date)

        self.q = div_yield(ticker)
        self.price = bs_price(self.S_0, self.K, self.T, self.r, self.sigma, self.q, self.option_type)

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
            market, _ = actual_option_price(self.ticker, self.K, self.T, self.option_type)
            return market
        else:
            return np.nan  # historical options data unavailable

    @property
    def implied_volatility(self):
        actual_price, _ = self.actual
        if self.creation_date is not None:
            return np.nan  # historical options data unavailable
        if actual_price:
            return implied_volatility(actual_price, self.S_0, self.K, self.T, self.r, self.q, self.option_type)
        else:
            return "N/A"

    # greeks calculation (delta, gamma, theta, vega, rho)

    @property
    def delta(self):
        d1, _ = self._d1_d2()
        if self.option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(d1)
        else:
            return -np.exp(-self.q * self.T) * norm.cdf(-d1)

    @property
    def gamma(self):
        d1, _ = self._d1_d2()
        return (np.exp(-self.q * self.T) * norm.pdf(d1)) / (self.S_0 * self.sigma * np.sqrt(self.T))

    @property
    def theta(self):
        d1, d2 = self._d1_d2()
        if self.option_type == "call":
            theta = (- (self.S_0 * self.sigma * np.exp(-self.q * self.T) * norm.pdf(d1)) / (2 * np.sqrt(self.T))
                     - self.q * self.S_0 * np.exp(-self.q * self.T) * norm.cdf(d1)
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            theta = (- (self.S_0 * self.sigma * np.exp(-self.q * self.T) * norm.pdf(d1)) / (2 * np.sqrt(self.T))
                     + self.q * self.S_0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        return theta / 365  # convert to per day

    @property
    def vega(self):
        d1, _ = self._d1_d2()
        return (self.S_0 * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T)) / 100  # convert to percentage

    @property
    def rho(self):
        _, d2 = self._d1_d2()
        if self.option_type == "call":
            return (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)) / 100  # convert to percentage
        else:
            return (-self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)) / 100  # convert to percentage

    # get black-scholes parameters
    def _d1_d2(self):
        d1 = (np.log(self.S_0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    # print summary
    def summary(self):
        print_option_price(self.ticker, self.r, self.T, self.K, self.n, self.option_type)
        print("\n********** GREEKS **********\n")
        greeks_table = [
            ["Delta", f"{self.delta:.4f}"],
            ["Gamma", f"{self.gamma:.4f}"],
            ["Theta", f"{self.theta:.4f}"],
            ["Vega", f"{self.vega:.4f}"],
            ["Rho", f"{self.rho:.4f}"]
        ]
        print(tabulate(greeks_table, headers=["Greek", "Value"], tablefmt="grid"))
    
    def _fetch_market_vol_data(self):

        stock = yf.Ticker(self.ticker)
        options = stock.option_chain()

        strikes = options.calls['strike'].values
        market_vols = options.calls['impliedVolatility'].values

        valid_indices = ~np.isnan(market_vols)
        return strikes[valid_indices].tolist(), market_vols[valid_indices].tolist()

def create_option(ticker, r, T, K, n, option_type="call", position="long", creation_date=None):
    return Option(ticker, r, T, K, n, option_type, position, creation_date)
