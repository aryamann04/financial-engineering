import numpy as np

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
        print(f"Option binomial price: ${self.binomial_price():.2f}")
        print()

    def print_option_tree(self):
        print("Option Price Tree:")
        self.zcb.print_tree(self.option_tree)