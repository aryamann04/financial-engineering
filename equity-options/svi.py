import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf

class SVIModel:
    def __init__(self, a=0, b=0, rho=0, m=0, sigma=0):
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def svi_volatility(self, k):
        return np.sqrt(
            self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m)**2 + self.sigma**2))
        )

    def fit(self, strikes, market_vols, forward_price):
        log_moneyness = np.log(np.array(strikes) / forward_price)

        def objective(params):
            a, b, rho, m, sigma = params
            self.a, self.b, self.rho, self.m, self.sigma = a, b, rho, m, sigma
            model_vols = np.array([self.svi_volatility(k) for k in log_moneyness])
            return np.sum((market_vols - model_vols)**2)

        initial_guess = [0.1, 0.1, 0, 0, 0.1]
        bounds = [(0, None), (0, None), (-1, 1), (None, None), (0, None)]

        result = minimize(objective, initial_guess, bounds=bounds)
        if not result.success:
            raise ValueError("SVI model fitting failed")

        self.a, self.b, self.rho, self.m, self.sigma = result.x

    def plot_volatility_surface(self, strikes, forward_price, market_vols):
        log_moneyness = np.log(np.array(strikes) / forward_price)
        model_vols = [self.svi_volatility(k) for k in log_moneyness]

        plt.figure(figsize=(10, 6))
        plt.plot(strikes, market_vols, 'o', label='Market Volatilities', color='blue')
        plt.plot(strikes, model_vols, '-', label='SVI Model Fit', color='orange')
        plt.title('SVI Model Volatility Curve')
        plt.xlabel('Strike Prices')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid()
        plt.show()