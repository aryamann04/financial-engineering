import numpy as np

class ZeroCouponBondOption:
    def __init__(self, zcb, strike, expiry):
        self.zcb = zcb
        self.zcb_price = zcb.binomial_price()
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
        print(f"\nOption binomial price: ${self.binomial_price():.2f}")
        print()

    def print_option_tree(self):
        print("\nOption Price Tree:")
        self.zcb.print_tree(self.option_tree)