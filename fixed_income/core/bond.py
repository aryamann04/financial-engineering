import numpy as np
from scipy.optimize import newton
import datetime

class Bond:
    def __init__(self, face_value, coupon_rate, maturity=None, time_to_maturity=None,
                 coupon_payment_frequency='semi-annual'):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupon_payment_frequency = coupon_payment_frequency.lower()
        self.periods_per_year = {'annual': 1, 'semi-annual': 2, 'quarterly': 4, 'monthly': 12}[
            self.coupon_payment_frequency]
        self.coupon_payment = self.face_value * self.coupon_rate / self.periods_per_year

        if maturity:
            self.maturity_date = maturity
            self.time_to_maturity = self.calculate_time_to_maturity()
        elif time_to_maturity:
            self.time_to_maturity = time_to_maturity
            self.maturity_date = None
        else:
            raise ValueError("Either maturity date or time to maturity in years must be provided.")

    def calculate_time_to_maturity(self):
        today = datetime.date.today()
        delta = self.maturity_date - today
        return delta.days / 365.0

    def price(self, yield_to_maturity):
        periods = self.periods_per_year * self.time_to_maturity
        discount_rate = yield_to_maturity / self.periods_per_year
        cash_flows = [self.coupon_payment] * int(periods)
        cash_flows[-1] += self.face_value
        present_values = [cf / (1 + discount_rate) ** (i + 1) for i, cf in enumerate(cash_flows)]
        return sum(present_values)

    def yield_to_maturity(self, price):
        def bond_price_func(yield_rate):
            return self.price(yield_rate) - price
        return newton(bond_price_func, 0.05)  # initial guess of 5%