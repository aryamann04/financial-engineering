# financial-engineering

[in progress]

## Functionalities 
### Equity Options
<div style="text-align: center;">
    <img width="500" alt="Screenshot 2024-06-22 at 9 04 23 PM" src="https://github.com/aryamann04/options/assets/140534650/3ce31b2b-0b1c-440d-82e8-82dcf3ad3724">
</div>

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

### Fixed Income
- ```bonds.py``` Calculate the price of a bond with or without coupons and calculate yield to maturity.
- ```currentbonds.py``` Get live information on U.S. Treasury Yields (1, 2, 3, 4, 6 months; 1, 2, 3, 5, 7, 10, 20, 30 years)
