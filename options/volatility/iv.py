import numpy as np 
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm 
from options.core.pricing.pricing import bs_price

def bs_iv(market, S_0, K, T, r, q, option_type="call"):
    if market is None or np.isnan(market) or T <= 0:
        return None

    def vega(sigma):
        if sigma <= 1e-8 or np.isnan(sigma):
            return 0
        d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S_0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    intrinsic = max(S_0 * np.exp(-q * T) - K * np.exp(-r * T), 0)
    eps = 1e-11
    if market <= intrinsic:
        market = intrinsic + eps

    sigma = np.sqrt(2 * abs(np.log(S_0 / K)) / T) if market - intrinsic < 1 else np.sqrt(2 * np.pi / T * (market - intrinsic) / (S_0 + K))

    max_iter = 1000
    for _ in range(max_iter):
        price = bs_price(S_0, K, T, r, sigma, q, option_type)
        if price is None:
            return None
        v = vega(sigma)
        if v < 1e-8:
            break
        increment = (price - market) / v
        sigma -= increment
        if sigma <= 0:
            sigma = 1e-4  
        if abs(increment) < 1e-9:
            return sigma
    
    return None


