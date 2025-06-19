import numpy as np
import matplotlib.pyplot as plt

from options.utilities.marketdata import MarketDataFetcher
from options.core.pricing.montecarlo import monte_carlo_range_accrual
from options.core.pricing.pricing import single_period_range_accrual_bs_price

class SinglePeriodRangeAccrual:
    def __init__(self, ticker, r, T, K_low, K_up, coupon, sigma=None):
        self.ticker = ticker.upper()
        self.r = r
        self.T = T
        self.K_low = float(K_low)
        self.K_up = float(K_up)
        self.coupon = float(coupon)

        self.fetcher = MarketDataFetcher(ticker, T)
        self.S0 = self.fetcher.current_price()
        self.sigma = sigma if sigma else self.fetcher.historical_volatility()
        self.q = self.fetcher.dividend_yield()

        self.black_scholes = self.black_scholes_range_price()
        self.monte_carlo = monte_carlo_range_accrual(self.S0, self.K_low, self.K_up, T, r, self.q, self.sigma, coupon)

    def black_scholes_range_price(self):
        return single_period_range_accrual_bs_price()

    def price(self):
        print(f"{self.ticker} Single-Period Range Accrual Option ({self.K_low}-{self.K_up}):")
        print(f"Black-Scholes: ${self.bs:.2f}")
        print(f"Monte Carlo:   ${self.mc:.2f}")

    def visualize_payoff(self):
        prices = np.linspace(self.K_low * 0.5, self.K_up * 1.5, 1000)
        payoff = np.where((prices > self.K_low) & (prices < self.K_up), self.coupon, 0)
        profit_loss = payoff - self.bs

        plt.figure(figsize=(10, 6))
        plt.plot(prices, profit_loss, label='Range Accrual P/L')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(self.K_low, color='red', linestyle='--')
        plt.axvline(self.K_up, color='red', linestyle='--')
        plt.text(self.K_low, max(profit_loss) * 0.5, f'K_low={self.K_low}', ha='right')
        plt.text(self.K_up, max(profit_loss) * 0.5, f'K_up={self.K_up}', ha='left')
        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.title(f'{self.ticker} Single-Period Range Accrual Option')
        plt.legend()
        plt.grid(True)
        plt.show()