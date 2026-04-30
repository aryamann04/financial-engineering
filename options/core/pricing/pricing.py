import numpy as np
from scipy.stats import norm

# European option Black-Scholes pricing 

def bs_price(S, K, T, r, sigma, q=0, option_type="call"):
    # Guard: sigma=0 → return discounted intrinsic (avoids divide-by-zero in d1)
    if sigma < 1e-10:
        if option_type == "call":
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

# European/American option binomial pricing 

def binom_price(S0, K, T, r, sigma, q, n, option_type="call", american=False):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)

    ST = np.array([S0 * (u ** (n - i)) * (d ** i) for i in range(n + 1)])
    values = np.maximum(0, ST - K if option_type == "call" else K - ST)

    for j in range(n - 1, -1, -1):
        values = np.exp(-r * dt) * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            ST = np.array([S0 * (u ** (j - i)) * (d ** i) for i in range(j + 1)])
            intrinsic = ST - K if option_type == "call" else K - ST
            values = np.maximum(values, intrinsic)
    return values[0]

# Digial option Black-Scholes pricing

def x(S_0, K, T, r, q, sigma):
        x = (np.log(K / S_0) - (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return x

def digital_option_bs_price(S_0, K, T, r, q, sigma, option_type, payoff_amount):
        x = (np.log(K / S_0) - (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type == "call":
            return np.exp(-r * T) * norm.cdf(-x) * payoff_amount
        else:
            return np.exp(-r * T) * norm.cdf(x) * payoff_amount
        
# Single period range accrual Black-Scholes pricing

def single_period_range_accrual_bs_price(self):
        # use digital options for convenience 
        from options.exotics.digital import DigitalOption
        d_low = DigitalOption(self.ticker, self.r, self.T, self.K_low, option_type="call", payoff_amount=1)
        d_high = DigitalOption(self.ticker, self.r, self.T, self.K_up, option_type="call", payoff_amount=1)

        return self.coupon * (d_low.black_scholes_price() - d_high.black_scholes_price())
