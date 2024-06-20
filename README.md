# financial-engineering

[in progress]

## Functionalities 
- ```optionspricing.py``` Prices options with the binomial model as well as the Black Scholes model. Given a ticker, the current stock price and dividend yield are retreived via the yfinance library. The user enters the strike price, time to expiry, and option type ("call" or "put") as well as the number of periods for the binomial model. The volatility parameter is proxied by a 1 year historical volatility (standard deviation) of the stock's price. The program outputs the price calcualed by the bimonial model (both European and American), the Black-Scholes price, the current actual market price of the option, and the implied volatility. 
