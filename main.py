from option import create_option

#---------[parameters]---------#
ticker = "PG"
r = 0.0509 # risk-free rate (annual)
T = 1.5  # years
K = 200 # strike 
n = 10 # number of periods in the binomial model
option_type = "call" # "call" or "put"

PnG_call_200 = create_option(ticker, r, T, K, n, option_type)
PnG_call_200.summary()
