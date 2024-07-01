from optionstrategies import OptionStrategy
from currentbonds import treasury_yield
from exotics import DigitalOption, SinglePeriodRangeAccrual, AsianOption

#-----------------------------------------------------------#
ticker = "AAPL"
T = 5  # years
r = treasury_yield(T)  # risk-free rate (annual)
n = 10  # number of periods in the binomial model
percent_itm_otm = 0.2  # for option strategies
#-----------------------------------------------------------#

# create a strategy object and call the relevant strategy function
strategy = OptionStrategy(ticker, percent_itm_otm, T, r, n)
strategy.bull_call_spread()

strategy.strategy_price()  # print Black-Scholes and market price
strategy.greeks()  # print strategy greeks
strategy.visualize_payoff(mp_payoff)  # view payoff graph and break-even points

#-----------------------------------------------------------#
#                         EXOTICS                           #
#-----------------------------------------------------------#

digital_option_strike = 250
option_type = "call"
payoff_amount = 1
#-----------------------------------------------------------#

digital_call_option = DigitalOption(ticker, r, T, digital_option_strike,
                                    option_type, payoff_amount)
digital_call_option.price()
digital_call_option.visualize_payoff()


#-----------------------------------------------------------#
ra_strike_low = 225
ra_strike_high = 275
#-----------------------------------------------------------#

single_period_range_accrual = SinglePeriodRangeAccrual(ticker, r, T,
                                                       ra_strike_low,
                                                       ra_strike_high,
                                                       payoff_amount)
single_period_range_accrual.price()
single_period_range_accrual.visualize_payoff()

#-----------------------------------------------------------#
asian_option_strike = 250
#-----------------------------------------------------------#

asian_call_option = AsianOption(ticker, r, T, asian_option_strike, option_type)
asian_call_option.price()

'''
available option strategies: 

*   atm_call()
*   itm_call()
*   otm_call()
*   short_atm_call()
*   short_itm_call()
*   short_otm_call()

*   atm_put()
*   itm_put()
*   otm_put()
*   short_atm_put()
*   short_itm_put()
*   short_otm_put()

*   covered_call()
*   married_put()

*   bull_call_spread()
*   bear_put_spread()
*   credit_call_spread()
*   credit_put_spread() 

*   protective_collar()
*   long_straddle()
*   long_strangle()
*   short_straddle()
*   short_strangle()

*   long_call_butterfly_spread()
*   short_call_butterfly_spread()
*   iron_condor()

available exotics: 

*   digital call/put (Black-Scholes & Monte Carlo)
*   single-period range accrual (Black-Scholes & Monte Carlo)
*   asian call/put (Monte Carlo)
'''
