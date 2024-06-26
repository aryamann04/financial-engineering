import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm
from datetime import datetime, timedelta

from optionspricing import div_yield, stock_data, bs_price
from montecarlo import monte_carlo_digital, monte_carlo_range_accrual, monte_carlo_asian

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

        self.S_0, self.sigma = stock_data(ticker, creation_date)
        self.q = div_yield(ticker)
        self.bs_price = self.digital_option_bs_price()
        self.mc_price = self.monte_carlo_price()

    def _x(self):
        x = (np.log(self.K / self.S_0) - (self.r - self.q - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        return x

    def digital_option_bs_price(self):
        x = self._x()
        if self.option_type == "call":
            return np.exp(-self.r * self.T) * norm.cdf(-x) * self.payoff_amount
        else:
            return np.exp(-self.r * self.T) * norm.cdf(x) * self.payoff_amount

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

class SinglePeriodRangeAccrual:
    def __init__(self, ticker, r, T, K_low, K_up, coupon):
        self.ticker = ticker
        self.r = r
        self.T = T
        self.K_low = K_low
        self.K_up = K_up
        self.coupon = coupon

        self.S_0, self.sigma = stock_data(ticker)
        self.q = div_yield(ticker)
        self.bs_price = self.range_accrual_bs_price()
        self.mc_price = self.monte_carlo_price()

    def monte_carlo_price(self):
        return monte_carlo_range_accrual(self.S_0, self.K_low, self.K_up,
                                         self.T, self.r, self.q, self.sigma, self.coupon)

    def range_accrual_bs_price(self):
        digital_call_low = DigitalOption(self.ticker, self.r, self.T, self.K_low, option_type="call")
        digital_call_up = DigitalOption(self.ticker, self.r, self.T, self.K_up, option_type="call")

        price_low = digital_call_low.digital_option_bs_price()
        price_up = digital_call_up.digital_option_bs_price()

        return self.coupon * (price_low - price_up)

    def price(self):
        print(f"{self.ticker} Single-period range accrual {self.K_low}-{self.K_up} (Black-Scholes): ${self.bs_price:.2f}")
        print(f"{self.ticker} Single-period range accrual {self.K_low}-{self.K_up} (Monte Carlo): ${self.mc_price:.2f}")

    def visualize_payoff(self):
        stock_prices = np.linspace(self.K_low * 0.5, self.K_up * 1.5, 1000)
        payoff = np.where((stock_prices > self.K_low) & (stock_prices < self.K_up), self.coupon, 0)

        profit_loss = payoff - self.bs_price

        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, profit_loss, label='Range Accrual Profit/Loss')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(self.K_low, color='red', linestyle='--', linewidth=0.5)
        plt.text(self.K_low, max(profit_loss)/2, f'K_low={self.K_low}', verticalalignment='bottom', horizontalalignment='right')
        plt.axvline(self.K_up, color='red', linestyle='--', linewidth=0.5)
        plt.text(self.K_up, max(profit_loss)/2, f'K_up={self.K_up}', verticalalignment='bottom', horizontalalignment='right')

        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.title(f'{self.ticker} Single-period Range Accrual Profit/Loss')
        plt.legend()
        plt.show()

class AsianOption:
    def __init__(self, ticker, r, T, K, option_type="call"):
        self.ticker = ticker
        self.r = r
        self.T = T
        self.K = K
        self.option_type = option_type

        self.S_0, self.sigma = stock_data(ticker)
        self.q = div_yield(ticker)
        self.mc_price = self.monte_carlo_price()

    def monte_carlo_price(self):
            return monte_carlo_asian(self.S_0, self.K, self.T, self.r, self.q, self.sigma, self.option_type)

    def price(self):
        print(f"{self.ticker} Asian option with strike {self.K} (Monte Carlo): ${self.mc_price:.2f}")
