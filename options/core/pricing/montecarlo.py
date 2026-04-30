"""
montecarlo.py — vectorized Monte Carlo option pricing.

Changes vs. prior version:
  - European: samples terminal distribution directly (no step loop) — ~250× faster
  - Path-dependent (Asian, range accrual): pre-generates full Gaussian matrix then
    uses np.cumsum — no Python loop over time steps
  - plot_price_paths: same vectorized approach
"""

import numpy as np
import matplotlib.pyplot as plt


def monte_carlo_european(S0, K, T, r, q, sigma, option_type="call",
                         simulations=10_000, n=None):
    """
    Price a European option via Monte Carlo.

    Uses the exact GBM terminal distribution — no time-step loop needed.
    `n` is ignored (kept for API compatibility with callers that pass it).
    """
    Z = np.random.standard_normal(simulations)
    S_T = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type.lower() == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - S_T, 0.0)
    elif option_type == "stock":
        return None
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(np.exp(-r * T) * payoffs.mean())


def monte_carlo_digital(S0, K, T, r, q, sigma, option_type="call",
                        simulations=10_000, n=None):
    """Price a cash-or-nothing digital option via Monte Carlo (vectorized)."""
    Z = np.random.standard_normal(simulations)
    S_T = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    if option_type.lower() == "call":
        payoffs = (S_T > K).astype(float)
    elif option_type.lower() == "put":
        payoffs = (S_T < K).astype(float)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(np.exp(-r * T) * payoffs.mean())


def monte_carlo_asian(S0, K, T, r, q, sigma, option_type="call",
                      simulations=10_000, n=252):
    """Price an arithmetic-average Asian option via vectorized path simulation."""
    dt = T / n
    # (simulations, n) Gaussian matrix — no Python loop over time steps
    Z = np.random.standard_normal((simulations, n))
    log_steps = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    # paths[:, t] = S0 * exp(sum of log_steps up to step t)
    paths = S0 * np.exp(np.cumsum(log_steps, axis=1))   # shape: (simulations, n)

    S_avg = paths.mean(axis=1)

    if option_type.lower() == "call":
        payoffs = np.maximum(S_avg - K, 0.0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - S_avg, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(np.exp(-r * T) * payoffs.mean())


def monte_carlo_range_accrual(S0, K_low, K_up, T, r, q, sigma, coupon,
                               simulations=10_000, n=252):
    """Price a single-period range accrual via vectorized path simulation."""
    dt = T / n
    Z = np.random.standard_normal((simulations, n))
    log_steps = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_steps, axis=1))

    S_T = paths[:, -1]
    in_range = (S_T > K_low) & (S_T < K_up)
    payoffs  = in_range.astype(float) * coupon
    return float(np.exp(-r * T) * payoffs.mean())


def plot_price_paths(S0, K, T, r, sigma, simulations=10, n=252):
    """Plot a handful of GBM price paths (vectorized)."""
    dt = T / n
    Z = np.random.standard_normal((simulations, n))
    log_steps = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    paths = S0 * np.exp(np.cumsum(log_steps, axis=1))

    plt.figure(figsize=(10, 6))
    for i in range(simulations):
        plt.plot(paths[i], lw=0.5)
    plt.axhline(y=K, color="r", linestyle="--", label="Strike Price")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")
    plt.title("Simulated Stock Price Paths")
    plt.legend()
    plt.show()
