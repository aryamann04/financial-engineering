import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import math

def stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    current_price = hist['Close'].iloc[-1]
    log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = np.std(log_returns) * np.sqrt(252)  # annualized

    return current_price, volatility

# binomial model price
def binom_price(S0, K, T, r, sigma, n, option_type="call", american=False):
    dt = T / n
    # risk-neutral parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # initialize asset prices at maturity
    ST = np.zeros(n + 1)
    for i in range(n + 1):
        ST[i] = S0 * (u ** (n - i)) * (d ** i)

    # initialize option values at maturity
    if option_type == "call":
        option_values = np.maximum(0, ST - K)
    else:
        option_values = np.maximum(0, K - ST)

    # backward induction
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])

            # for american options check if it should be exercised now
            if american:
                ST = S0 * (u ** (j - i)) * (d ** i)
                if option_type == "call":
                    option_values[i] = np.maximum(option_values[i], ST - K)
                else:
                    option_values[i] = np.maximum(option_values[i], K - ST)

    return option_values[0]

# black-scholes-merton model price
def bs_price(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# get actual current option price from yfinance
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
        return None
    else:
        return option_row['lastPrice'].values[0], closest_expiry
    
def option(ticker, r, T, K, n, option_type="call"):
    S_0, sigma = stock_data(ticker)

    european_price = binom_price(S_0, K, T, r, sigma, n, option_type=option_type, american=False)
    print(f"European {option_type} price (binomial model with {n} periods): {european_price:.2f}")

    black_scholes_price = bs_price(S_0, K, T, r, sigma, option_type=option_type)
    print(f"Black-Scholes {option_type} price: {black_scholes_price:.2f}")

    american_price = binom_price(S_0, K, T, r, sigma, n, option_type=option_type, american=True)
    # print(f"American {option_type} price (binomial model with {n} periods): {american_price}")

    actual, exp = actual_option_price(ticker, K, T, option_type)
    print(f"Actual {option_type} price (expiring on {exp}): {actual}")
