from bonds import Bond, ZeroCouponBond, ZeroCouponBondOption, Caplet, Floorlet
from currentbonds import treasury_yield, plot_yield_curve

# creating a zero coupon bond
#-----------------------------------------------------------#
face_value = 100
T = 4  # bond maturity in years
r_0 = 0.06
u = 1.25
d = 0.9
#-----------------------------------------------------------#

# create a zero coupon bond, print the price and associated binomial trees
zcb_4y = ZeroCouponBond(face_value, T, r_0, u, d)
zcb_4y.price()
zcb_4y.print_bond_tree()
zcb_4y.print_interest_tree()

# creating an option on a zero coupon bond
#-----------------------------------------------------------#
zcb_option_expiry = 2
zcb_option_strike = 84
#-----------------------------------------------------------#

# create an option on the above zero coupon bond, print the price and associated binomial trees
zcb_4y_2yoption = ZeroCouponBondOption(zcb_4y, zcb_option_strike, zcb_option_expiry)
zcb_4y_2yoption.price()
zcb_4y_2yoption.print_option_tree()

# create a caplet and a floorlet
#-----------------------------------------------------------#
cf_expiry = 6
cf_notional = 1000  # notional amount in dollars
caplet_strike = 0.02
floorlet_strike = 0.08
#-----------------------------------------------------------#

# create a caplet and floorlet, print their prices and associated binomial trees
caplet_6y = Caplet(r_0, caplet_strike, cf_expiry, u, d, cf_notional)
caplet_6y.price()
caplet_6y.print_caplet_tree()
caplet_6y.print_interest_tree()

floorlet_6y = Floorlet(r_0, floorlet_strike, cf_expiry, u, d, cf_notional)
floorlet_6y.price()
floorlet_6y.print_floorlet_tree()
floorlet_6y.print_interest_tree()
