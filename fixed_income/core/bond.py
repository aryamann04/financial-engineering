import numpy as np
import pandas as pd
import textwrap
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import fsolve
from tabulate import tabulate

from bootstrap import get_disc_factors, get_zc_yields, interpolate_d

class Bond:
    def __init__(self, face_value, coupon_rate, maturity=None, coupon_freq=2, issue_date=None, maturity_date=None, purchase_date=None):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.coupons_notl = face_value * coupon_rate / coupon_freq
        self.coupon_freq = coupon_freq
        self.freq_type = (
            "annual" if coupon_freq == 1 else
            "semi-annual" if coupon_freq == 2 else
            "quarterly" if coupon_freq == 4 else
            "monthly"
        )

        if isinstance(purchase_date, str): 
            self.purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')
        else: 
            self.purchase_date = datetime.today()

        if issue_date is None:
            self.issue_date = datetime.today()
        elif isinstance(issue_date, str):
            self.issue_date = datetime.strptime(issue_date, '%Y-%m-%d')
        else:
            self.issue_date = issue_date

        if maturity_date is None: 
            self.maturity = maturity
            self.maturity_date = self.issue_date + relativedelta(
                years=int(maturity),
                days=int((maturity % 1) * 365)
            )
        else: 
            self.maturity_date = datetime.strptime(maturity_date, '%Y-%m-%d')
            self.maturity = (self.maturity_date - self.issue_date).days / 365.0
        
        self.purchase_date = datetime.today()

        past = [d for d in self.coupon_schedule() if d <= datetime.today()]
        self.last_coupon_date = past[-1] if past else self.issue_date

        future = [d for d in self.coupon_schedule() if d > datetime.today()]
        self.next_coupon_date = future[0] if future else self.maturity_date
        
        self.zc_yields = get_zc_yields()
        self.disc_factors = get_disc_factors()
        self.accrued_interest = self.accr_int()

    def coupon_schedule(self):
        schedule = []
        dt_months = int(12 / self.coupon_freq)
        for i in range(1, int(self.maturity * self.coupon_freq) + 1):
            schedule.append(
                self.issue_date + relativedelta(months=dt_months * i)
            )
        return schedule

    def previous_coupon_dates(self):
        today = datetime.today()
        prev = [d.strftime('%Y-%m-%d') for d in self.coupon_schedule() if d < today]
        text = ', '.join(prev) if prev else 'N/A'
        return textwrap.fill(text, width=40)

    def future_coupon_dates(self):
        today = datetime.today()
        fut = [d.strftime('%Y-%m-%d') for d in self.coupon_schedule() if d > today]
        text = ', '.join(fut) if fut else 'N/A'
        return textwrap.fill(text, width=40)

    def accr_int(self, on_date=None):
        if on_date is None:
            on_date = datetime.today()
        schedule = self.coupon_schedule()

        past = [d for d in schedule if d <= on_date]
        last_coupon = past[-1] if past else self.issue_date

        future = [d for d in schedule if d > on_date]
        next_coupon = future[0] if future else self.maturity_date

        days_between = (next_coupon - last_coupon).days
        days_elapsed = (on_date - last_coupon).days

        if days_between <= 0:
            return 0
        
        w = days_elapsed / days_between
        return self.coupons_notl * w

    def build_price_lists(self):
        dates = pd.date_range(start=self.issue_date, end=self.maturity_date, freq='D')

        dirty_prices = []
        clean_prices = []

        for date in dates:
            dp = self.dirty_price(on_date=date)
            ai = self.accr_int(on_date=date)
            dirty_prices.append(dp)
            clean_prices.append(dp - ai) 

        self.dates = dates
        self.dirty_prices = dirty_prices
        self.clean_prices = clean_prices

        return dates, dirty_prices, clean_prices

    def dirty_price(self, on_date=None):
        if on_date is None:
            on_date = self.purchase_date

        y = self.ytm / 100.0
        m = self.coupon_freq
        C = self.coupons_notl
        sched = self.coupon_schedule()

        past = [d for d in sched if d <= on_date]
        last = past[-1] if past else self.issue_date
        future = [d for d in sched if d > on_date]
        next_c = future[0] if future else sched[-1]

        dt = (next_c - last).days
        de = (on_date - last).days
        
        if dt == 0:
            k = 0
        else:
            k = (dt - de) / dt   

        N = sum(d > next_c for d in sched)

        pv = 0.0
        pv += C / (1 + y/m)**k

        for i in range(1, N+1):
            pv += C / (1 + y/m)**(k + i)
        pv += self.face_value / (1 + y/m)**(k + N)
        return pv
    
    def clean_price(self, on_date=None):
        return self.dirty_price(on_date) - self.accr_int(on_date)
    
    @property
    def price(self):
        periods = int(self.maturity * self.coupon_freq)
        dt = 1 / self.coupon_freq
        cash_flows = np.array([self.coupons_notl] * periods)
        cash_flows[-1] += self.face_value  
        times = [round(dt * i, 8) for i in range(1, periods + 1)]

        pv = 0.0
        for cf, t in zip(cash_flows, times):
            D_t = self.disc_factors.get(t)
            if D_t is None:
                D_t = interpolate_d(t, self.disc_factors, method='log-linear')
            pv += cf * D_t

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

    @property
    def modified_duration(self):
        pv = self.dirty_price(datetime.today())
        periods = int(self.maturity * self.coupon_freq)
        dt = 1 / self.coupon_freq
        cf = np.array([self.coupons_notl] * periods)
        cf[-1] += self.face_value
        times = [i * dt for i in range(1, periods + 1)]

        d_vals = []
        for t in times:
            key = round(t, 6)
            D_t = self.disc_factors.get(key) or interpolate_d(key, self.disc_factors, method='log-linear')
            d_vals.append(D_t)

        macaulay = sum(t * c * d for t, c, d in zip(times, cf, d_vals)) / pv
        y = self.ytm / 100.0
        return macaulay / (1 + y / self.coupon_freq)
    
    @property
    def pv01(self):
        y = self.ytm / 100.0
        modified_duration = self.modified_duration
        return modified_duration * 0.0001 * self.dirty_price(datetime.today())
    
    @property
    def convexity(self): 
        y = self.ytm / 100.0
        m = self.coupon_freq
        n = int(self.maturity * m)
        cfm = self.coupons_notl  
        P = self.dirty_price(datetime.today())

        conv_sum = 0.0
        for i in range(1, n+1):
            Ci = cfm if i < n else cfm + self.face_value
            conv_sum += Ci * i * (i + 1) / (1 + y/m)**(i + 2)

        return conv_sum / (P * m**2)
    
    def summary(self):
        today = datetime.today()
        print("\n********** BOND SUMMARY **********")
        print(f"{self.maturity}Y {self.freq_type} bond issued on {datetime.strftime(self.issue_date, '%Y-%m-%d')} with {self.coupon_rate * 100:.2f}% coupon and face value ${self.face_value:.2f}")
        ai = self.accrued_interest
        print(tabulate([
            ["Issue date", self.issue_date.strftime("%Y-%m-%d")],
            ["Purchase date", self.purchase_date.strftime("%Y-%m-%d")],
            ["Maturity date", self.maturity_date.strftime("%Y-%m-%d")],
            ["Coupon frequency", self.freq_type],
            ["Last coupon date", self.last_coupon_date.strftime("%Y-%m-%d")],
            ["Next coupon date", self.next_coupon_date.strftime("%Y-%m-%d")],
            ["Face value", f"${self.face_value:.2f}"],
            ["Coupon rate (%)", f"{self.coupon_rate * 100:.2f}%"],
            ["Previous coupon dates", self.previous_coupon_dates()],
            ["Future coupon dates", self.future_coupon_dates()],
            ["Accrued interest", f"${ai:.2f}"],
        ], headers=["Parameter", "Value"], tablefmt="grid"))
        today_str = datetime.strftime(today, '%Y-%m-%d')
        print(tabulate([
            [f"Dirty price ({today_str})", f"${self.dirty_price(today):.3f}"],
            [f"Clean price ({today_str})", f"${self.clean_price(today):.3f}"],
            [f"Price ({today_str})", f"${self.price:.3f}"],
            ["Yield to maturity (%)", f"{self.ytm:.3f}%"],
            ["Modified duration", f"{self.modified_duration:.3f}"],
            ["PV01", 0],
            ["Convexity", 0],
        ], tablefmt="grid"))
        print("**********************************")
    def plot_price_trajectory(self):
        dates, dirty_prices, clean_prices = self.build_price_lists()
        plt.figure(figsize=(12, 6))
        plt.plot(dates, dirty_prices, label=f'dirty price')
        plt.plot(dates, clean_prices, label=f'clean price')
        plt.title(f'bond prices (yield constant at {self.ytm:.2f}%)')
        plt.xlabel('date')
        plt.ylabel('price ($)')
        plt.legend()
        plt.grid()
        plt.show()

bond1 = Bond(face_value=1000, coupon_rate=0.05, maturity=5, coupon_freq=4, issue_date='2023-01-01')
bond2 = Bond(face_value=1000, coupon_rate=0.08, maturity=7, coupon_freq=2, issue_date='2022-06-01')
bond3 = Bond(face_value=1000, coupon_rate=0.03, maturity=0.5, coupon_freq=2, issue_date='2023-10-01')
bond4 = Bond(face_value=1000, coupon_rate=0.04, maturity=0.5, coupon_freq=4)
bond5 = Bond(face_value=1000, coupon_rate=0.02, coupon_freq=2, maturity_date='2055-07-26')

bond1.summary()
bond2.summary()
bond3.summary()
bond4.summary()
bond5.summary()

bond1.plot_price_trajectory()
bond2.plot_price_trajectory()
bond3.plot_price_trajectory()
bond4.plot_price_trajectory()
bond5.plot_price_trajectory()
