import datetime
import numpy as np
from scipy.optimize import fsolve
from tabulate import tabulate

from bootstrap import get_disc_factors, get_zc_yields

class Bond:
    def __init__(self, face_value, coupon_rate, maturity, coupon_freq=2):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon_freq = coupon_freq
        self.freq_type = "annual" if coupon_freq == 1 else "semi-annual" if coupon_freq == 2 else "quarterly" if coupon_freq == 4 else "monthly"
        self.maturity = maturity
        self.maturity_date = datetime.datetime.today() + datetime.timedelta(days=int(maturity * 365))
        self.coupons_notl = face_value * coupon_rate / coupon_freq

        self.zc_yields = get_zc_yields()
        self.disc_factors = get_disc_factors()
    
    @property
    def price(self):
        periods = int(self.maturity * self.coupon_freq)
        cash_flows = np.array([self.coupons_notl] * periods)
        cash_flows[-1] += self.face_value  
        times = np.arange(0.5, self.maturity + 0.1, 0.5)
        times = [float(t) for t in times]  
        pv = sum(cf * self.disc_factors[t] for cf, t in zip(cash_flows, times))
        return pv

    @property
    def ytm(self):
        price = self.price
        periods = int(self.maturity * self.coupon_freq)
        cash_flows = np.array([self.coupons_notl] * periods)
        cash_flows[-1] += self.face_value 
        times = np.arange(1, periods + 1)

        def func(y):
            pv = sum(cf / (1 + y / 2) ** n for cf, n in zip(cash_flows, times))
            return pv - price

        ytm = fsolve(func, 0.05)[0]
        return ytm * 100
    
    def summary(self):
        print("\n********** BOND SUMMARY **********")
        print(f"{self.maturity}Y {self.freq_type} bond with {self.coupon_rate * 100:.2f}% coupon and face value ${self.face_value:.2f}")
        print(tabulate([
            ["Price", f"{self.price:.2f}"],
            ["Yield to Maturity (%)", f"{self.ytm:.2f}%"]
            ["PV01", 0],
            ["Convexity", 0],
        ], headers=["Parameter", "Value"], tablefmt="grid"))
        print("**********************************")

bond1 = Bond(face_value=1000, coupon_rate=0.05, maturity=5, coupon_freq=4)
bond2 = Bond(face_value=1000, coupon_rate=0.08, maturity=7, coupon_freq=2)
bond3 = Bond(face_value=1000, coupon_rate=0.03, maturity=0.5, coupon_freq=1)

bond1.summary()
bond2.summary()
bond3.summary()
