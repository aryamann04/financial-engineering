import numpy as np
import scipy.optimize as opt
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class SABRModel:
    def __init__(self, alpha, beta, rho, nu):
        self.alpha = alpha  # Initial volatility
        self.beta = beta    # Elasticity of variance
        self.rho = rho      # Correlation between F and volatility
        self.nu = nu        # Volatility of volatility
    
    def sabr_vol(self, F, K, T):
        if K == F:
            ATM_vol = self.alpha / (F**(1 - self.beta))
            return ATM_vol
        
        z = self.nu / self.alpha * (F * K) ** ((1 - self.beta) / 2) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho))
        
        vol = (self.alpha / ((F * K) ** ((1 - self.beta) / 2))) * (z / x)
        return vol

    def calibrate_sabr(self, market_strikes, market_vols, F, T):
        def error(params):
            alpha, beta, rho, nu = params
            self.alpha, self.beta, self.rho, self.nu = alpha, beta, rho, nu
            model_vols = [self.sabr_vol(F, K, T) for K in market_strikes]
            return np.sum((np.array(model_vols) - np.array(market_vols))**2)
        
        initial_guess = [self.alpha, self.beta, self.rho, self.nu]
        bounds = [(0, None), (0, 1), (-1, 1), (0, None)]
        opt_params = opt.minimize(error, initial_guess, bounds=bounds).x
        self.alpha, self.beta, self.rho, self.nu = opt_params

        return opt_params

def get_iv(tic, K, T, option_type):
    ticker = yf.Ticker(tic)
    K = 5 * round(K / 5)  # Round strike to nearest 5 for finding market prices
    exp_dates = ticker.options
    target_expiry = datetime.now() + timedelta(days=T * 365)
    closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))
    
    option_chain = ticker.option_chain(date=closest_expiry)
    options = option_chain.calls if option_type == "call" else option_chain.puts
    
    option_row = options[options['strike'] == K]
    return option_row['impliedVolatility'].values[0] if not option_row.empty else None

def plot_sabr_vol_smile(ticker, expiry_years, F, sabr_model):
    today = datetime.today()
    expiry_date = today + timedelta(days=expiry_years * 365)
    ticker_obj = yf.Ticker(ticker)
    options = ticker_obj.options

    if not options:
        raise ValueError(f"No options data available for ticker {ticker}")

    option_dates = [datetime.strptime(exp_date, '%Y-%m-%d') for exp_date in options]
    expiry_date_str = min(option_dates, key=lambda x: abs(x - expiry_date)).strftime('%Y-%m-%d')

    option_chain = ticker_obj.option_chain(date=expiry_date_str)
    calls = option_chain.calls

    strikes = calls['strike'].values
    implied_vols = calls['impliedVolatility'].values

    valid_indices = implied_vols != 0
    strikes = strikes[valid_indices]
    implied_vols = implied_vols[valid_indices]

    model_vols = [sabr_model.sabr_vol(F, K, expiry_years) for K in strikes]
    
    plt.figure(figsize=(8, 5))
    plt.plot(strikes, implied_vols, label='Market IV', marker='o', linestyle='-', color='blue')
    plt.plot(strikes, model_vols, label='SABR Model IV', marker='s', linestyle='--', color='red')
    plt.xlabel('Strike Prices')
    plt.ylabel('Implied Volatility')
    plt.title(f'SABR Volatility Smile for {ticker} on {expiry_date_str}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example Usage
sabr = SABRModel(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
ticker = input("Enter ticker: ")
T = int(input("Enter expiry (years): "))
F = int(input("Enter F: "))
plot_sabr_vol_smile(ticker, T, F, sabr)
