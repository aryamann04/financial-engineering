from optionstrategies import OptionStrategy

#-----------------------------------------------------------#
ticker = "AAPL"
r = 0.05 # risk-free rate (annual)
T = 1.5  # years
n = 10 # number of periods in the binomial model
percent_itm_otm = 0.1 # for option strategies (0, 1)
#-----------------------------------------------------------#

# create a strategy object and call the relevant strategy function
AAPL_strategy = OptionStrategy(ticker, percent_itm_otm, T, r, n)
AAPL_strategy.long_strangle()

AAPL_strategy.strategy_price() # print Black-Scholes and market price
AAPL_strategy.greeks() # print strategy greeks
AAPL_strategy.visualize_payoff() # view payoff graph and break-even points

'''
available option strategies: 
1.  covered_call()
2.  married_put()
3.  bull_call_spread()
4.  bear_put_spread()
5.  protective_collar()
6.  long_straddle()
7.  long_strangle()
8.  long_call_butterfly_spread()
9.  long_put_butterfly_spread()
'''
