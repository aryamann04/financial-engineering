import numpy as np
from fixed_income.core.zcb import ZeroCouponBond

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