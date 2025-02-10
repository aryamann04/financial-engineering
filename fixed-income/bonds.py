import numpy as np
from scipy.optimize import newton
import datetime
from currentbonds import treasury_yield

class ZeroCouponBond:
    def __init__(self, face_value, n, r_0, u, d):
        self.face_value = face_value
        self.n = int(n)
        self.r_0 = r_0
        self.u = u
        self.d = d
        self.interest_tree = None
        self.bond_tree = None

    def binomial_price(self):
        interest_tree = np.zeros((self.n + 1, self.n + 1))
        interest_tree[0, 0] = self.r_0

        for i in range(1, self.n + 1):
            for j in range(i + 1):
                if j == 0:
                    interest_tree[j, i] = interest_tree[j, i - 1] * self.u
                else:
                    interest_tree[j, i] = interest_tree[j - 1, i - 1] * self.d

        bond_tree = np.zeros((self.n + 1, self.n + 1))
        bond_tree[:, self.n] = self.face_value

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                up_value = bond_tree[j, i + 1]
                down_value = bond_tree[j + 1, i + 1]
                rate = interest_tree[j, i]
                bond_tree[j, i] = (up_value + down_value) / 2 / (1 + rate)

        self.interest_tree = interest_tree
        self.bond_tree = bond_tree

        return bond_tree[0, 0]

    def price(self):
        print(f"Zero coupon bond binomial price: ${self.binomial_price():.2f}")
        print()

    def print_r0(self):
        print(f"t=0 interest rate: {self.r_0*100:.3f}%")
        print()

    def print_tree(self, tree, rate=False):
        max_width = tree.shape[0]
        for i in range(tree.shape[1]):
            for j in range(max_width - i):
                print("    ", end="")  # Padding for alignment
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

class ZeroCouponBondOption:
    def __init__(self, zcb, strike, expiry):
        self.zcb = zcb
        self.strike = strike
        self.expiry = int(expiry)
        self.option_tree = None

    def binomial_price(self):
        n = self.zcb.n
        interest_tree = self.zcb.interest_tree
        bond_tree = self.zcb.bond_tree
        option_tree = np.zeros((n + 1, n + 1))

        for j in range(self.expiry + 1):
            option_tree[j, self.expiry] = max(0, bond_tree[j, self.expiry] - self.strike)

        for i in range(self.expiry - 1, -1, -1):
            for j in range(i + 1):
                up_value = option_tree[j, i + 1]
                down_value = option_tree[j + 1, i + 1]
                rate = interest_tree[j, i]
                option_tree[j, i] = (up_value + down_value) / 2 / (1 + rate)

        self.option_tree = option_tree

        return option_tree[0, 0]

    def price(self):
        print(f"Option binomial price: ${self.binomial_price():.2f}")
        print()

    def print_option_tree(self):
        print("Option Price Tree:")
        self.zcb.print_tree(self.option_tree)

class Caplet:
    def __init__(self, r_0, strike, expiry, u, d, notional):
        self.r_0 = r_0
        self.u = u
        self.d = d
        self.strike = strike
        self.expiry = expiry
        self.notional = notional

        self.interest_tree = None
        self.caplet_tree = None

    def build_interest_tree(self):
        interest_tree = np.zeros((self.expiry, self.expiry))
        interest_tree[0, 0] = self.r_0

        for i in range(1, self.expiry):
            for j in range(i + 1):
                if j == 0:
                    interest_tree[j, i] = interest_tree[j, i - 1] * self.u
                else:
                    interest_tree[j, i] = interest_tree[j - 1, i - 1] * self.d

        self.interest_tree = interest_tree

    def binomial_price(self):
        self.build_interest_tree()
        caplet_tree = np.zeros((self.expiry, self.expiry))

        for j in range(self.expiry):
            caplet_tree[j, self.expiry - 1] = max(0, self.interest_tree[j, self.expiry - 1] - self.strike) / (1 + self.interest_tree[j, self.expiry - 1])

        for i in range(self.expiry - 2, -1, -1):
            for j in range(i + 1):
                up_value = caplet_tree[j, i + 1]
                down_value = caplet_tree[j + 1, i + 1]
                rate = self.interest_tree[j, i]
                caplet_tree[j, i] = (up_value + down_value) / 2 / (1 + rate)

        self.caplet_tree = caplet_tree
        return caplet_tree[0, 0]

    def price(self):
        print(f"Caplet binomial price (notional: ${self.notional}): ${self.binomial_price()*self.notional:.2f}")
        print()

    def print_interest_tree(self):
        print("Interest Rate Tree:")
        ZeroCouponBond.print_tree(self, self.interest_tree, rate=True)

    def print_caplet_tree(self):
        print("Caplet Price Tree:")
        ZeroCouponBond.print_tree(self, self.caplet_tree)

class Floorlet:
    def __init__(self, r_0, strike, expiry, u, d, notional):
        self.r_0 = r_0
        self.u = u
        self.d = d
        self.strike = strike
        self.expiry = expiry
        self.notional = notional

        self.interest_tree = None
        self.floorlet_tree = None

    def build_interest_tree(self):
        interest_tree = np.zeros((self.expiry, self.expiry))
        interest_tree[0, 0] = self.r_0

        for i in range(1, self.expiry):
            for j in range(i + 1):
                if j == 0:
                    interest_tree[j, i] = interest_tree[j, i - 1] * self.u
                else:
                    interest_tree[j, i] = interest_tree[j - 1, i - 1] * self.d

        self.interest_tree = interest_tree

    def binomial_price(self):
        self.build_interest_tree()

        floorlet_tree = np.zeros((self.expiry, self.expiry))

        for j in range(self.expiry):
            floorlet_tree[j, self.expiry - 1] = max(0, self.strike - self.interest_tree[j, self.expiry - 1]) / (1 + self.interest_tree[j, self.expiry - 1])

        for i in range(self.expiry - 2, -1, -1):
            for j in range(i + 1):
                up_value = floorlet_tree[j, i + 1]
                down_value = floorlet_tree[j + 1, i + 1]
                rate = self.interest_tree[j, i]
                floorlet_tree[j, i] = (up_value + down_value) / 2 / (1 + rate)

        self.floorlet_tree = floorlet_tree

        return floorlet_tree[0, 0]

    def price(self):
        print(f"Floorlet binomial price (notional: ${self.notional}): ${self.binomial_price()*self.notional:.2f}")
        print()

    def print_interest_tree(self):
        print("Interest Rate Tree:")
        ZeroCouponBond.print_tree(self, self.interest_tree, rate=True)

    def print_floorlet_tree(self):
        print("Floorlet Price Tree:")
        ZeroCouponBond.print_tree(self, self.floorlet_tree)

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
