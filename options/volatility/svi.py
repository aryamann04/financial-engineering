import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class SVI: 
    def __init__(self, S_0, T, r, q, fetcher): 
        self.S_0 = S_0
        self.T = T
        self.r = r
        self.q = q 
        self.ticker = fetcher.ticker

        self.strikes, self.implied_vols = fetcher.plot_implied_vols(r=r, plot=False)
        self.implied_vols = np.array(self.implied_vols, dtype=float) / 100.0
        self.strikes = np.array(self.strikes, dtype=float)

        self.a = None
        self.b = None
        self.rho = None
        self.m = None
        self.sigma = None

        self.calibrate()
    
    def calibrate(self): 
        k_i = self.log_moneyness()
        w_i = self.implied_vols ** 2 * self.T
        p0 = [
            np.min(w_i),
            (np.max(w_i)-np.min(w_i)) / (np.max(k_i)-np.min(k_i)+1e-6),
            0.0,
            np.mean(k_i),
            np.std(k_i)
        ]

        lower = [-np.inf, 0.0, -0.999, np.min(k_i), 0.0]
        upper = [ np.inf, np.inf,  0.999, np.max(k_i), np.inf]

        params, _ = curve_fit(self.svi_variance, k_i, w_i, p0=p0, bounds=(lower, upper), maxfev=20000)        
        
        self.a, self.b, self.rho, self.m, self.sigma = params

    def svi_vol(self, K): 
        w = self.svi_variance(np.log(K / self.forward_price(self.S_0, self.T, self.r, self.q)), 
                         self.a, self.b, self.rho, self.m, self.sigma)
        return np.sqrt(w / self.T)

    def svi_variance(self, k, a, b, rho, m, sigma): 
        return a + b * (rho * (k - m) + np.sqrt((k-m) ** 2 + sigma ** 2))

    def forward_price(self, S_0, T, r, q): 
        return S_0 * np.exp((r - q) * T)

    def log_moneyness(self): 
        return np.log(self.strikes / (self.S_0 * np.exp((self.r - self.q) * self.T)))
    
    def plot_svi(self): 
        svi_vols = []
        for strike in self.strikes: 
            svi_vols.append(self.svi_vol(strike))
        
        _, ax = plt.subplots()
        ax.plot(self.strikes, self.implied_vols, label='black-scholes implied vol', marker='o', linestyle='-', color="blue")
        ax.plot(self.strikes, svi_vols, label='SVI-calibrated vols', marker='o', linestyle='--', color="orange")
        ax.axvline(x=self.S_0, color='black', linestyle='--', label=f'current price: ${self.S_0:.2f}')
        
        ax.set_xlabel('strikes')
        ax.set_ylabel(f'implied vols (%)')
        ax.set_title(f'SVI Vol Surface {self.ticker} (expiry {self.T:.2f} years)')
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()