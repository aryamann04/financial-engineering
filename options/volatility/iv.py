import numpy as np 
from scipy.optimize import brentq
import options.core.pricing.pricing as bs_price

def bs_iv(market, S_0, K, T, r, q, option_type="call"): 
    if market or np.isnan(market):
        return "N/A"

    def objective(sigma):
        return bs_price(S_0, K, T, r, sigma, q, option_type) - market

    try:
        implied_vol = brentq(objective, 1e-6, 5.0)  
        return implied_vol
    except ValueError:
        return "N/A"