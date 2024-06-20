import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import math
from tabulate import tabulate

def stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    current_price = hist['Close'].iloc[-1]
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = np.std(log_returns) * np.sqrt(252)  # annualized

    return current_price, volatility

def div_yield(ticker):
    stock = yf.Ticker(ticker)
    try:
        dividend_yield = stock.info['dividendYield']
        return dividend_yield if dividend_yield is not None else 0
    except KeyError:
        return 0

def binom_price(S0, K, T, r, sigma, q, n, option_type="call", american=False):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    ST = np.zeros(n + 1)
    for i in range(n + 1):
        ST[i] = S0 * (u ** (n - i)) * (d ** i)

    if option_type == "call":
        option_values = np.maximum(0, ST - K)
    else:
        option_values = np.maximum(0, K - ST)

    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
            if american:
                ST = S0 * (u ** (j - i)) * (d ** i)
                if option_type == "call":
                    option_values[i] = np.maximum(option_values[i], ST - K)
                else:
                    option_values[i] = np.maximum(option_values[i], K - ST)

    return option_values[0]

def bs_price(S, K, T, r, sigma, q, option_type="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def actual_option_price(tic, K, T, option_type):
    ticker = yf.Ticker(tic)
    exp_dates = ticker.options
    today = datetime.now()
    target_expiry = today + timedelta(days=T * 365)
    closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))

    option_chain = ticker.option_chain(closest_expiry)
    if option_type == "call":
        options = option_chain.calls
    else:
        options = option_chain.puts

    option_row = options[options['strike'] == K]
    if option_row.empty:
        return None, closest_expiry
    else:
        return option_row['lastPrice'].values[0], closest_expiry

def implied_volatility(option_price, S, K, T, r, q, option_type):
    try:
        implied_vol = brentq(lambda x: bs_price(S, K, T, r, x, q, option_type) - option_price, 1e-10, 5)
        return implied_vol
    except ValueError:
        return np.nan

def print_option_price(ticker, r, T, K, n, option_type="call"):
    S_0, sigma = stock_data(ticker)
    q = div_yield(ticker)

    params_table = [
        ["Ticker", ticker],
        ["Risk-Free Rate", r],
        ["Dividend Yield", q],
        ["Time to Expiry (years)", T],
        ["Strike Price", K],
        ["Number of Periods (Binomial Model)", n],
        ["Option Type", option_type]
    ]
    print(tabulate(params_table, headers=["Parameter", "Value"], tablefmt="grid"))
    print("\n********** PRICES **********\n")

    european_price = binom_price(S_0, K, T, r, sigma, q, n, option_type=option_type, american=False)
    print(f"European {option_type} price (binomial model with {n} periods): {european_price:.2f}")

    american_price = binom_price(S_0, K, T, r, sigma, q, n, option_type=option_type, american=True)
    print(f"American {option_type} price (binomial model with {n} periods): {american_price:.2f}\n")

    black_scholes_price = bs_price(S_0, K, T, r, sigma, q, option_type=option_type)
    print(f"Black-Scholes European {option_type} price: {black_scholes_price:.2f}")

    actual, exp = actual_option_price(ticker, K, T, option_type)
    if actual:
        print(f"Actual {option_type} price (expiring on {exp}): {actual}\n")
    else:
        print(f"Actual {option_type} price not found for strike price {K} expiring on {exp}\n")

    print("********** VOLATILITY **********\n")
    print(f"Historical volatility (1yr std dev): {sigma:.5f}")
    if actual:
        iv = implied_volatility(actual, S_0, K, T, r, q, option_type)
        print(f"Implied volatility: {iv:.5f}")
    else:
        print("Implied volatility: N/A")
