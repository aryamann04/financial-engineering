# financial-engineering

- [Installation Guide](#installation-guide)
- [Equity Options](#equity-options)
  - [Option Strategies](#option-strategies)
  - [Exotics](#exotics)
  - [Volatility Analysis](#volatility-analysis)
  - [Options Pricing](#options-pricing)
- [Fixed Income](#fixed-income)
  - [Zero Coupon Bonds](#zero-coupon-bonds-and-options-on-zcbs)
  - [Caplets and Floorlets](#caplets-and-floorlets)
  - [Current Bonds & Yield Curve](#current-bonds-and-yield-curve)

- ```optionstrategies.py``` Price and visualize various option strategies on a ticker of your choice. Will output the options made, along with key information such as their Black-Scholes price, market price, and greeks. Enter a percent OTM/ITM the strategy should be. For instance, if you would like to place a long strangle with a long call 10% OTM and a long put 10% OTM, enter 0.1 in the ```percent_itm_otm``` field. Both the Black-Scholes price and market price of the strategy are printed as well as the breakeven points on the profit & loss plot. The greeks of the overall strategy are also printed. Current strategies available include:
  
  - ```atm_call()```
  - ```itm_call()```
  - ```otm_call()```
  - ```short_atm_call()```
  - ```short_itm_call()```
  - ```short_otm_call()```
    
  - ```atm_put()```
  - ```itm_put()```
  - ```otm_put()```
  - ```short_atm_put()```
  - ```short_itm_put()```
  - ```short_otm_put()```
    
  -  ```covered_call()```
  -  ```married_put()```

  -  ```bull_call_spread()```
  -  ```bear_put_spread()```
  - ```call_credit_spread()```
  - ```put_credit_spread()```
    
  -  ```protective_collar()```
  -  ```long_straddle()```
  -  ```long_strangle()```
  -  ```short_straddle()```
  -  ```short_strangle()```
    
  -  ```long_call_butterfly_spread()```
  -  ```long_put_butterfly_spread()```
  -  ```iron_condor()```

```python
ticker = "AAPL"
T = 0.25  # years
r = treasury_yield(T)  # risk-free rate (annual)
n = 10  # number of periods in the binomial model
percent_itm_otm = 0.1  # for option strategies

# create a strategy object and call the relevant strategy function
strategy = OptionStrategy(ticker, percent_itm_otm, T, r, n)
strategy.iron_condor()

strategy.strategy_price()  # print Black-Scholes and market price
strategy.greeks()  # print strategy greeks
strategy.visualize_payoff()  # view payoff graph and break-even points
```

<img width="600" alt="Screenshot 2024-06-22 at 9 04 23 PM" src="https://github.com/aryamann04/options/assets/140534650/3ce31b2b-0b1c-440d-82e8-82dcf3ad3724">

- ```exotics.py```  Price digital call/put options, single period range accruals, and Asian options with the Black-Scholes model and Monte Carlo simulation.
<img width="600" alt="Screenshot 2024-06-25 at 9 24 02 PM" src="https://github.com/aryamann04/financial-engineering/assets/140534650/7d51d623-9359-480c-bfa6-4795e1982620">

- ```volatility.py``` Calculate the volatility skew of an option at a given strike price and plot the current real-time volatility surface.
  
- ```optionspricing.py``` Prices options with the binomial model as well as the Black Scholes model. Given a ticker, the current stock price and dividend yield are retreived via the yfinance library. The user enters the strike price, time to expiry, and option type ("call" or "put") as well as the number of periods for the binomial model. The volatility parameter is proxied by a 1 year historical volatility (standard deviation) of the stock's price. The program outputs the price calcualed by the bimonial model (both European and American), the Black-Scholes price, the current actual market price of the option, and the implied volatility. 

### Fixed Income
- ```bonds.py``` The main classes include ZeroCouponBond, ZeroCouponBondOption, Caplet, and Floorlet. Each class offers methods to construct interest rate trees, calculate instrument prices using the binomial model, and print the trees for visualization.

#### Zero Coupon Bonds and Options on ZCBs
```python
face_value = 100
T = 4  # bond maturity in years
r_0 = 0.06
u = 1.25
d = 0.9

zcb_4y = ZeroCouponBond(face_value, T, r_0, u, d)
zcb_4y.price()
zcb_4y.print_bond_tree()
zcb_4y.print_interest_tree()

zcb_option_expiry = 2
zcb_option_strike = 84

zcb_4y_2yoption = ZeroCouponBondOption(zcb_4y, zcb_option_strike, zcb_option_expiry)
zcb_4y_2yoption.price()
zcb_4y_2yoption.print_option_tree()
```

#### Caplets and Floorlets
```python
cf_expiry = 6
cf_notional = 1000  # notional amount in dollars
caplet_strike = 0.02
floorlet_strike = 0.08

caplet_6y = Caplet(r_0, caplet_strike, cf_expiry, u, d, cf_notional)
caplet_6y.price()
caplet_6y.print_caplet_tree()
caplet_6y.print_interest_tree()

floorlet_6y = Floorlet(r_0, floorlet_strike, cf_expiry, u, d, cf_notional)
floorlet_6y.price()
floorlet_6y.print_floorlet_tree()
floorlet_6y.print_interest_tree()
```

The following is an example of the binomial price tree output for the zero coupon bond. 

<img width="480" alt="Screenshot 2024-06-23 at 1 55 48 AM" src="https://github.com/aryamann04/options/assets/140534650/178e6eea-5221-4a83-81c2-9251069961c9">

- ```currentbonds.py``` Get live information on U.S. Treasury Yields (1, 2, 3, 4, 6 months; 1, 2, 3, 5, 7, 10, 20, 30 years) and plot the current yield curve with ```plot_yield_curve()```. 
