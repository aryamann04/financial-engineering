import numpy as np 
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm 
from options.core.pricing.pricing import bs_price

def bs_iv_brentq(market, S_0, K, T, r, q, option_type="call"): 
    if market is None or np.isnan(market) or T <= 0:
        return None

    def objective(sigma):
        return bs_price(S_0, K, T, r, sigma, q, option_type) - market

    def loss(sigma):
        return abs(objective(sigma))

    try:
        low, high = 1e-6, 50
        if objective(low) * objective(high) < 0:
            return brentq(objective, low, high)
        else:
            res = minimize_scalar(loss, bounds=(low, high), method='bounded')
            if res.success:
                return res.x
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None


def bs_iv(market, S_0, K, T, r, q, option_type="call"):
    if market is None or np.isnan(market) or T <= 0:
        return None

    def vega(sigma):
        if sigma <= 1e-8 or np.isnan(sigma):
            return 0
        d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S_0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    intrinsic = max(S_0 - K, 0) if option_type == 'call' else max(K - S_0, 0)
    if market <= intrinsic:
        return None

    try:
        sigma = np.sqrt((2 * np.pi / T) * (market - intrinsic) / (S_0 + K))
    except:
        return None

    max_iter = 10000
    for _ in range(max_iter):
        price = bs_price(S_0, K, T, r, sigma, q, option_type)
        if price is None:
            return None
        v = vega(sigma)
        if v < 1e-8:
            break
        increment = (price - market) / v
        sigma -= increment
        if abs(increment) < 1e-8:
            return sigma

    print("insufficient convergence")
    return None

