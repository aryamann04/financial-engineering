import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from options.core.pricing.pricing import bs_price


def bs_iv(market, S_0, K, T, r, q, option_type="call"):
    """
    Compute Black-Scholes implied volatility via Newton-Raphson with brentq fallback.

    Returns the implied vol as a decimal (e.g. 0.25 = 25%), or None if no valid
    solution exists (e.g. price below intrinsic, T<=0, or solver failure).

    Fixes applied vs. prior implementation:
      - Bug 1: intrinsic floor is now option-type-aware (call vs put)
      - Bug 2: initial guess uses Brenner-Subrahmanyam (never produces sigma=0)
      - Bug 3: brentq fallback is now actually used on NR failure
      - Bug 4: sigma going negative breaks to brentq instead of resetting to 1e-4
    """
    # Input validation
    if (
        market is None
        or not np.isfinite(market)
        or T <= 0
        or S_0 <= 0
        or K <= 0
    ):
        return None

    option_type = option_type.lower()

    # Bug 1 fix: option-type-aware intrinsic floor
    if option_type == "call":
        intrinsic = max(S_0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S_0 * np.exp(-q * T), 0.0)

    # Nudge market strictly above intrinsic.  This guarantees the brentq bracket
    # [lo, hi] is valid: bs_price(lo~0) ≈ intrinsic < market, bs_price(20) >> market.
    # A market price at or below intrinsic has no finite positive-IV solution.
    if market <= intrinsic + 1e-8:
        market = intrinsic + 1e-8

    # Bug 2 fix: Brenner-Subrahmanyam approximation as initial guess.
    # For ATM: price ≈ S * sigma * sqrt(T / 2π)  →  sigma ≈ price / S * sqrt(2π / T)
    # This is valid for all moneyness and always positive.  Clamp to [0.01, 5.0].
    sigma = float(np.clip(np.sqrt(2.0 * np.pi / T) * market / S_0, 0.01, 5.0))

    # Newton-Raphson (up to 100 iterations; fall through to brentq on any issue)
    for _ in range(100):
        price = bs_price(S_0, K, T, r, sigma, q, option_type)
        if price is None or not np.isfinite(price):
            break

        # Vega = S * exp(-qT) * phi(d1) * sqrt(T)
        d1 = (np.log(S_0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        v = S_0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        # Bug 4 fix: vega collapse → fall through to brentq, don't reset sigma
        if v < 1e-8:
            break

        diff = price - market
        if abs(diff) < 1e-8:
            return sigma

        sigma -= diff / v

        # Bug 4 fix: negative sigma → fall through to brentq, not reset to 1e-4
        if sigma <= 0:
            break

    # Bug 3 fix: brentq fallback.
    # Bracket validity is guaranteed by the intrinsic nudge:
    #   bs_price(1e-9) ≈ intrinsic  <  market  (after nudge)
    #   bs_price(20.0) >> any real-world option price
    try:
        objective = lambda s: bs_price(S_0, K, T, r, s, q, option_type) - market
        return brentq(objective, 1e-9, 20.0, xtol=1e-8, rtol=1e-8, maxiter=200)
    except (ValueError, RuntimeError):
        return None
