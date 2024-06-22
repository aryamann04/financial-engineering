# financial-engineering

[in progress]

## Functionalities 
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
    
- ```optionspricing.py``` Prices options with the binomial model as well as the Black Scholes model. Given a ticker, the current stock price and dividend yield are retreived via the yfinance library. The user enters the strike price, time to expiry, and option type ("call" or "put") as well as the number of periods for the binomial model. The volatility parameter is proxied by a 1 year historical volatility (standard deviation) of the stock's price. The program outputs the price calcualed by the bimonial model (both European and American), the Black-Scholes price, the current actual market price of the option, and the implied volatility. 
