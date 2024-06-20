from optionspricing import option

#---------[parameters]---------#
ticker = "AAPL"
r = 0.05 # risk-free annual rate
T = 1  # years
K = 150
n = 100 # number of periods in the binomial model
option_type = "call" # "call" or "put"

option(ticker, r, T, K, n, option_type)
