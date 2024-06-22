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

        return newton(bond_price_func, 0.05)  # Initial guess of 5%

def accrued_interest(face_value, coupon_rate, last_coupon_date, current_date, day_count_convention='30/360'):
    if day_count_convention == '30/360':
        days_in_period = 360 / 2  # Assuming semi-annual payments
        days_accrued = (current_date - last_coupon_date).days
    elif day_count_convention == 'actual/360':
        days_in_period = 360 / 2  # Assuming semi-annual payments
        days_accrued = (current_date - last_coupon_date).days
    elif day_count_convention == 'actual/365':
        days_in_period = 365 / 2  # Assuming semi-annual payments
        days_accrued = (current_date - last_coupon_date).days
    elif day_count_convention == 'actual/actual':
        days_in_period = (current_date.year - last_coupon_date.year) * 365 + (
                    current_date - last_coupon_date.replace(year=current_date.year)).days
        days_accrued = (current_date - last_coupon_date).days
    else:
        raise ValueError("Unsupported day count convention")

    accrued_interest_value = face_value * coupon_rate / 2 * (days_accrued / days_in_period)
    return accrued_interest_value

def dirty_price(clean_price, accrued_interest):
    return clean_price + accrued_interest

def zcb(fv, r, T):
    return fv / (1 + r) ** T

def bond(fv, r, T, c):
    pv_coupon = 0
    for i in range(1, T + 1):
        pv_coupon += c * fv / (1 + r) ** i
    return pv_coupon + fv / (1 + r) ** T
