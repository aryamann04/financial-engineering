import numpy as np
from datetime import datetime
from tabulate import tabulate 

from fixed_income.core.bond import Bond
from fixed_income.core.bootstrap import interpolate_d

class ZeroCouponBond:
    def __init__(self, face_value, u, d, maturity=None, issue_date=None, maturity_date=None, purchase_date=None):
        self.face_value = face_value
        self.u = u
        self.d = d
        self.interest_tree = None
        self.bond_tree = None
        self.bond = Bond(face_value=face_value, coupon_rate=0, maturity=maturity, coupon_freq=1, 
                         issue_date=issue_date, maturity_date=maturity_date, purchase_date=purchase_date)
    
    def price(self, on_date=None): 
        return self.bond.dirty_price(on_date=on_date)
    
    @property
    def ytm(self):
        return self.bond.ytm
    
    @property
    def modified_duration(self): 
        return self.bond.modified_duration
    
    @property 
    def pv01(self): 
        return self.bond.pv01

    @property
    def convexity(self):
        return self.bond.convexity
    
    def n(self, as_of=None): 
        if as_of is None: 
            as_of = self.bond.purchase_date
        T = (self.bond.maturity_date - as_of).days / 365.0
        return int(round(T))
    
    def r_0(self, as_of=None): 
        if as_of is None: 
            as_of = self.bond.purchase_date
        T = (self.bond.maturity_date - as_of).days / 365.0
        dt = T / self.n(as_of=as_of)
        D_dt = self.bond.disc_factors.get(dt,
             interpolate_d(dt, self.bond.disc_factors, method='log-linear'))
        return D_dt**(-1/dt) - 1

    def binomial_price(self, as_of=None):
        n = self.n(as_of=as_of)
        r_0 = self.r_0(as_of=as_of)

        interest_tree = np.zeros((n + 1, n + 1))
        interest_tree[0, 0] = r_0

        for i in range(1, n + 1):
            for j in range(i + 1):
                if j == 0:
                    interest_tree[j, i] = interest_tree[j, i - 1] * self.u
                else:
                    interest_tree[j, i] = interest_tree[j - 1, i - 1] * self.d

        bond_tree = np.zeros((n + 1, n + 1))
        bond_tree[:, n] = self.face_value

        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                up_value = bond_tree[j, i + 1]
                down_value = bond_tree[j + 1, i + 1]
                rate = interest_tree[j, i]
                bond_tree[j, i] = (up_value + down_value) / 2 / (1 + rate)

        self.interest_tree = interest_tree
        self.bond_tree = bond_tree

        return bond_tree[0, 0]

    def summary(self):
        today = datetime.today()
        print("\n********** BOND SUMMARY **********")
        print(f"{self.bond.maturity:.3f}Y zero coupon bond issued on {datetime.strftime(self.bond.issue_date, '%Y-%m-%d')} ${self.face_value:.2f}")
        print(tabulate([
            ["Issue date", self.bond.issue_date.strftime("%Y-%m-%d")],
            ["Purchase date", self.bond.purchase_date.strftime("%Y-%m-%d")],
            ["Maturity date", self.bond.maturity_date.strftime("%Y-%m-%d")],
            ["Face value", f"${self.face_value:.2f}"]
        ], headers=["Parameter", "Value"], tablefmt="grid"))
        today_str = datetime.strftime(today, '%Y-%m-%d')
        print(tabulate([
            [f"Price ({today_str})", f"${self.price(today):.3f}"],
            ["Yield to maturity (%)", f"{self.ytm:.3f}%"],
            ["Modified duration", f"{self.modified_duration:.3f}"],
            ["PV01", f"{self.pv01:.3f}"],
            ["Convexity", f"{self.convexity:.3f}"],
        ], tablefmt="grid"))
        print("\n********** BINOMIAL MODEL FOR ZCB **********")
        print(tabulate([
            [f"Binomial price ({today_str})", f"${self.binomial_price(today):.3f}"],
            [f"r_0 ({today_str})", f"{self.r_0(today) * 100:.3f}%"],
            ["u", self.u],
            ["d", self.d]
        ], headers=["Parameter", "Value"], tablefmt="grid"))
        self.print_interest_tree()
        self.print_bond_tree()
        print("**********************************")
    
    def plot_price_trajectory(self): 
        self.bond.plot_price_trajectory()

    def print_tree(self, tree, rate=False):
        max_width = tree.shape[0]
        for i in range(tree.shape[1]):
            for j in range(max_width - i):
                print("    ", end="") 
            for j in range(i + 1):
                if rate:
                    print(f"{tree[j, i]*100:.3f}%", end="    ")
                else:
                    print(f"{tree[j, i]:.4f}", end="    ")
            print()
        print()

    def print_interest_tree(self):
        print("Interest Rate Tree:")
        self.print_tree(self.interest_tree, rate=True)

    def print_bond_tree(self):
        print("Bond Price Tree:")
        self.print_tree(self.bond_tree)