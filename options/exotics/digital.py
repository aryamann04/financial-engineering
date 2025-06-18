import numpy as np
import matplotlib.pyplot as plt

from utilities.marketdata import MarketDataFetcher
from options.core.pricing.montecarlo import monte_carlo_digital
from options.core.pricing.pricing import digital_option_bs_price

class DigitalOption:
    def __init__(self, ticker, r, T, K, option_type="call", payoff_amount=1, position="long", creation_date=None):
        self.ticker = ticker
        self.r = r
        self.T = T
        self.K = K
        self.option_type = option_type
        self.position = position
        self.creation_date = creation_date
        self.payoff_amount = payoff_amount

        self.fetcher = MarketDataFetcher(ticker, creation_date)
        self.S_0, self.sigma = self.fetcher.current_price_and_vol()
        self.q = self.fetcher.dividend_yield()
        self.bs_price = self.bs_price()
        self.mc_price = self.monte_carlo_price()

    def bs_price(self):
        return digital_option_bs_price(self.S_0, self.K, self.T, self.r, self.q, self.sigma, self.option_type, self.payoff_amount)
    
    def monte_carlo_price(self):
        return monte_carlo_digital(self.S_0, self.K, self.T, self.r, self.q, self.sigma, self.option_type)

    def price(self):
        print(f"{self.ticker} Digital {self.option_type} with strike {self.K} (Black-Scholes): ${self.bs_price:.2f}")
        print(f"{self.ticker} Digital {self.option_type} with strike {self.K} (Monte Carlo): ${self.mc_price:.2f}")

    def visualize_payoff(self):
        stock_prices = np.linspace(self.S_0 * 0.5, self.S_0 * 1.5, 1000)
        if self.option_type == "call":
            payoff = np.where(stock_prices > self.K, self.payoff_amount, 0)
        else:
            payoff = np.where(stock_prices < self.K, self.payoff_amount, 0)

        profit_loss = payoff - self.bs_price

        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, profit_loss, label=f'{self.option_type.capitalize()} Profit/Loss')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(self.K, color='red', linestyle='--', linewidth=0.5)
        plt.text(self.K, max(profit_loss)/2, f'Strike={self.K}', verticalalignment='bottom', horizontalalignment='right')

        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.title(f'{self.ticker} Digital {self.option_type.capitalize()} Option Profit/Loss')
        plt.legend()
        plt.show()
