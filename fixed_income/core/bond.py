import numpy as np
from scipy.optimize import newton
import datetime

class Bond:
    def __init__(self, face_value, coupon_rate, maturity=None,
                 coupon_freq=2):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon_freq = coupon_freq
        self.maturity = maturity
        self.coupons_notl = face_value * coupon_rate / coupon_freq

    def price(self, yield_to_maturity):
        periods = int(self.coupon_freq * self.time_to_maturity)
        cash_flows = np.array([self.coupons_notl] * periods)
        
        discount_rate = yield_to_maturity / self.periods_per_year
        cash_flows = [self.coupon_payment] * int(periods)
        cash_flows[-1] += self.face_value

        times = np.arrange(0.5, self.maturity, 1 / self.coupon_freq)
        present_values = [cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows)]
        return sum(present_values)

    def yield_to_maturity(self, price):
        def bond_price_func(yield_rate):
            return self.price(yield_rate) - price
        return newton(bond_price_func, 0.05)  # initial guess of 5%