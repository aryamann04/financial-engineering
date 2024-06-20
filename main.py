from optionspricing import print_option_price

#---------[parameters]---------#
ticker = "AAPL"
r = 0.05 # risk-free annual rate
T = 1  # years
K = 150
n = 100 # number of periods in the binomial model
option_type = "call" # "call" or "put"

print_option_price(ticker, r, T, K, n, option_type)
