import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_european(S0, K, T, r, q, sigma, option_type='call', simulations=10000, n=252):
    dt = T / n
    S = np.zeros((simulations, n))
    S[:, 0] = S0

    for t in range(1, n):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    if option_type == 'call':
        payoffs = np.maximum(S[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - S[:, -1], 0)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def monte_carlo_digital(S0, K, T, r, q, sigma, option_type='call', simulations=10000, n=252):
    dt = T / n
    S = np.zeros((simulations, n))
    S[:, 0] = S0

    for t in range(1, n):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    if option_type == 'call':
        payoffs = np.where(S[:, -1] > K, 1, 0)
    elif option_type == 'put':
        payoffs = np.where(S[:, -1] < K, 1, 0)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def monte_carlo_range_accrual(S0, K_low, K_up, T, r, q, sigma, coupon, simulations=10000, n=252):
    dt = T / n
    S = np.zeros((simulations, n))
    S[:, 0] = S0

    for t in range(1, n):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    in_range = np.logical_and(S[:, -1] > K_low, S[:, -1] < K_up)
    payoffs = in_range * coupon
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def monte_carlo_asian(S0, K, T, r, q, sigma, option_type='call', simulations=10000, n=252):
    dt = T / n
    S = np.zeros((simulations, n))
    S[:, 0] = S0

    for t in range(1, n):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    S_avg = np.mean(S, axis=1)

    if option_type == 'call':
        payoffs = np.maximum(S_avg - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - S_avg, 0)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

def plot_price_paths(S0, K, T, r, sigma, simulations=10, n=252):
    dt = T / n
    S = np.zeros((simulations, n))
    S[:, 0] = S0

    for t in range(1, n):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    plt.figure(figsize=(10, 6))
    for i in range(simulations):
        plt.plot(S[i, :], lw=0.5)

    plt.axhline(y=K, color='r', linestyle='--', label='Strike Price')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title('Simulated Stock Price Paths')
    plt.legend()
    plt.show()
