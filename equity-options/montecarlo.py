import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_european(S0, K, T, r, sigma, option_type='call', simulations=10000, n=252):
    dt = T / n
    S = np.zeros((simulations, n))
    S[:, 0] = S0

    for t in range(1, n):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    if option_type == 'call':
        payoffs = np.maximum(S[:, -1] - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - S[:, -1], 0)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price


S0 = 100
K = 105
T = 1
r = 0.05
sigma = 0.2
simulations = 10000
n = 252

call_price = monte_carlo_option_price(S0, K, T, r, sigma, option_type='call',
                                      simulations=simulations, n=n)
put_price = monte_carlo_option_price(S0, K, T, r, sigma, option_type='put',
                                     simulations=simulations, n=n)

print(f"Monte Carlo Call Option Price: {call_price:.2f}")
print(f"Monte Carlo Put Option Price: {put_price:.2f}")

def plot_price_paths(S0, T, r, sigma, simulations=10, n=252):
    dt = T / n
    S = np.zeros((simulations, n))
    S[:, 0] = S0

    for t in range(1, n):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    plt.figure(figsize=(10, 6))
    for i in range(simulations):
        plt.plot(S[i, :], lw=0.5)

    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title('Simulated Stock Price Paths')
    plt.show()


plot_price_paths(S0, T, r, sigma)
