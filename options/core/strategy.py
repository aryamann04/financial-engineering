import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime

from options.core.option import create_option
from options.utilities.marketdata import MarketDataFetcher
from options.utilities.printer import print_option_summary
from options.volatility.marketvols import closest_K_T

class OptionStrategy:
    def __init__(self, ticker, T, rf, sigma=None, n=100, creation_date=None):
        self.ticker = ticker

        _, self.exp = closest_K_T(ticker, 0, T) if isinstance(T, (int, float)) else (_, T)
        self.T = (datetime.strptime(self.exp, "%Y-%m-%d") - datetime.today()).days / 365.0 if isinstance(self.exp, str) else T

        self.rf = rf
        self.n = n
        self.creation_date = creation_date

        self.fetcher = MarketDataFetcher(ticker, T, creation_date)
        self.S_0 = self.fetcher.current_price()
        self.q = self.fetcher.dividend_yield()
        self.sigma = sigma if sigma else self.fetcher.historical_volatility()

        self.options = []
        self.strategy_name = ""
        self.total_price = 0
        self.total_market_price = 0

    # helper for ease of option creation with same parameters 
    def create_option(self, option_type, strike_price, position='long'):

        if option_type == 'stock':
            market_match = False
        else:
            market_match = True
        
        option = create_option(self.ticker, self.rf, self.T, strike_price, self.n, option_type=option_type, position=position, 
                               creation_date=self.creation_date, fetcher=self.fetcher, market_match=market_match)
        itm_otm = ""
        percent_itm_otm = abs((strike_price - self.S_0) / self.S_0)

        print("\n+----------------------------------------------------------+")
        if option_type == "call":
            if strike_price < self.S_0:
                itm_otm = "ITM"
                print(f"created ITM call option with strike price {strike_price:.3f}")
            elif strike_price > self.S_0:
                itm_otm = "OTM"
                print(f"created OTM call option with strike price {strike_price:.3f}")
            else:
                itm_otm = "ATM"
                print(f"created ATM call option with strike price {strike_price:.3f}")
        elif option_type == "put":
            if strike_price > self.S_0:
                itm_otm = "ITM"
                print(f"created ITM put option with strike price {strike_price:.3f}")
            elif strike_price < self.S_0:
                itm_otm = "OTM"
                print(f"created OTM put option with strike price {strike_price:.3f}")
            else:
                itm_otm = "ATM"
                print(f"created ATM put option with strike price {strike_price:.3f}")
        elif option_type == "stock":
            itm_otm = "N/A"
            print(f"Created stock position at price {strike_price:.3f}")

        print(f"Option Type: {option_type.capitalize()}")
        print(f"Position: {position.capitalize()}")

        if option_type != 'stock':
            print(f"Strike Price: {strike_price:.3f}")
            print(f"Black-Scholes Price: {option.bs_price:.3f}")

            if option.market is not None:
                print(f"Market Price: {option.market}")
            else:
                print(f"Market Price: does not exist")

            print(f"Moneyness: {itm_otm}")
            print(f"Percent ITM/OTM: {percent_itm_otm*100:.2f}%")

            print_option_summary(option)
        print("+----------------------------------------------------------+")
        return option

    def strategy_price(self):
        print("\n\n+==========================================================+")
        print(f"\n******************** STRATEGY SUMMARY ********************")
        print(f"{self.ticker} {self.strategy_name} expiring {self.exp}")
        print(f"***********************************************************\n")
        self.total_price = sum(option.bs_price if option.position == 'long' else -option.bs_price for option in self.options)
        self.total_market_price = sum(
            option.market if option.position == 'long' else -1 * option.market
            for option in self.options
            if option.market is not None
        )
        print(f"+---------------- STRATEGY PRICE ----------------+\n")
        price_table = [
            ["Black-Scholes", f"${self.total_price:.3f}"],
            ["Market", f"${self.total_market_price:.3f}"],
        ]
        print(tabulate(price_table, headers=["Type", "Price"], tablefmt="grid"))

        if self.total_market_price < self.total_price: 
            msg = f"strategy is cheap! recommendation: execute the {self.strategy_name} on {self.ticker}"
        elif self.total_market_price > self.total_price: 
            msg = f"strategy is expensive! recommendation: short the {self.strategy_name} on {self.ticker}"
        else: 
            msg = "fairly priced! no recommendation"

        print("+" + "-"*(len(msg)+2) + "+" + f"\n| {msg} |\n" + "+" + "-"*(len(msg)+2) + "+")

        return self.total_price, self.total_market_price

    def greeks(self):
        delta = 0
        gamma = 0
        theta = 0
        vega = 0
        rho = 0

        for option in self.options:
            if option.option_type != 'stock':
                if option.position == 'long':
                    delta += option.greeks["delta"]
                    gamma += option.greeks["gamma"]
                    theta += option.greeks["theta"]
                    vega += option.greeks["vega"]
                    rho += option.greeks["rho"]
                else:  # short position
                    delta -= option.greeks["delta"]
                    gamma -= option.greeks["gamma"]
                    theta -= option.greeks["theta"]
                    vega -= option.greeks["vega"]
                    rho -= option.greeks["rho"]
            else:  # for stock
                if option.position == 'long':
                    delta += 1
                else:  # short stock position
                    delta -= 1

        print(f"\n+---------------- STRATEGY GREEKS ----------------+\n")
        greeks_table = [
            ["delta", f"{delta:.4f}"],
            ["gamma", f"{gamma:.4f}"],
            ["theta", f"{theta:.4f}"],
            ["vega", f"{vega:.4f}"],
            ["rho", f"{rho:.4f}"]
        ]
        print(tabulate(greeks_table, headers=["Greek", "Value"], tablefmt="grid"))

        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

    def visualize_payoff(self, market_price=False):
        stock_prices = np.linspace(self.S_0 * 0.5, self.S_0 * 1.5, 1000)
        total_payoff = np.zeros_like(stock_prices)

        for option in self.options:
            if option.option_type == "call":
                payoff = np.maximum(stock_prices - option.K, 0)
            elif option.option_type == "put":
                payoff = np.maximum(option.K - stock_prices, 0)
            else:
                payoff = stock_prices - option.K

            if option.position == 'long':
                total_payoff += payoff
            elif option.position == 'short':
                total_payoff -= payoff

        if market_price:
            total_profit_loss = total_payoff - self.total_market_price
        else:
            total_profit_loss = total_payoff - self.total_price

        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, total_profit_loss, label=f'{self.strategy_name} P/L')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.axvline(self.S_0, color='black', linestyle='--', linewidth=0.5, label=f"current price: ${self.S_0:.2f}")

        break_even_points = []
        for i in range(1, len(stock_prices)):
            if total_profit_loss[i - 1] * total_profit_loss[i] < 0:  # sign change 
                # linear interpolation 
                x1, x2 = stock_prices[i - 1], stock_prices[i]
                y1, y2 = total_profit_loss[i - 1], total_profit_loss[i]
                break_even_point = x1 - y1 * (x2 - x1) / (y2 - y1)
                break_even_points.append(break_even_point)

        break_even_points = np.unique(np.round(break_even_points, 2))

        print(f"\n+---------------- BREAKEVEN POINTS ----------------+")
        if len(break_even_points) == 0:
            print("N/A")
        else:
            print(f"{', '.join([f'${point:.2f}' for point in break_even_points])}")
            for point in break_even_points:
                plt.axvline(point, color='red', linestyle='--', linewidth=0.5, label = f"breakeven point: ${point}")

        plt.xlabel('stock price')
        plt.ylabel('profit/loss')
        plt.title(f'{self.ticker} {self.strategy_name} PnL diagram')
        plt.legend()
        plt.show()
        print("+==========================================================+")
        return break_even_points

    def atm_call(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        call = self.create_option('call', self.S_0, 'long')
        self.options.append(call)
        self.strategy_name = "ATM Call"

        return self.options

    def itm_call(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"ITM call strike (< {self.S_0:.2f}): "))
        call = self.create_option('call', K, 'long')
        self.options.append(call)
        self.strategy_name = "ITM Call"

        return self.options

    def otm_call(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"OTM call strike (> {self.S_0:.2f}): "))
        call = self.create_option('call', K, 'long')
        self.options.append(call)
        self.strategy_name = "OTM Call"

        return self.options

    def short_atm_call(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        call = self.create_option('call', self.S_0, 'short')
        self.options.append(call)
        self.strategy_name = "Short ATM Call"

        return self.options

    def short_itm_call(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"ITM call strike (< {self.S_0:.2f}): "))
        call = self.create_option('call', K, 'short')
        self.options.append(call)
        self.strategy_name = "Short ITM Call"

        return self.options

    def short_otm_call(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"OTM call strike (> {self.S_0:.2f}): "))
        call = self.create_option('call', K, 'short')
        self.options.append(call)
        self.strategy_name = "Short OTM Call"

        return self.options

    def atm_put(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        put = self.create_option('put', self.S_0, 'long')
        self.options.append(put)
        self.strategy_name = "ATM Put"

        return self.options

    def itm_put(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"ITM put strike (> {self.S_0:.2f}): "))
        put = self.create_option('put', K, 'long')
        self.options.append(put)
        self.strategy_name = "ITM Put"

        return self.options

    def otm_put(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"OTM put strike (< {self.S_0:.2f}): "))
        put = self.create_option('put', K, 'long')
        self.options.append(put)
        self.strategy_name = "OTM Put"

        return self.options

    def short_atm_put(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        put = self.create_option('put', self.S_0, 'short')
        self.options.append(put)
        self.strategy_name = "Short ATM Put"

        return self.options

    def short_itm_put(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"ITM put strike (> {self.S_0:.2f}): "))
        put = self.create_option('put', K, 'short')
        self.options.append(put)
        self.strategy_name = "Short ITM Put"

        return self.options

    def short_otm_put(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"OTM put strike (< {self.S_0:.2f}): "))
        put = self.create_option('put', K, 'short')
        self.options.append(put)
        self.strategy_name = "Short OTM Put"

        return self.options

    def covered_call(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"covered call strike (> {self.S_0:.2f}): "))
        call = self.create_option('call', K, 'short')
        stock = self.create_option('stock', self.S_0, 'long')

        self.options.append(call)
        self.options.append(stock)
        self.strategy_name = "Covered Call"

        return self.options

    def married_put(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"married put strike (< {self.S_0:.2f}): "))
        put = self.create_option('put', K, 'long')
        stock = self.create_option('stock', self.S_0, 'long')

        self.options.append(put)
        self.options.append(stock)
        self.strategy_name = "Married Put"

        return self.options

    def bull_call_spread(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K1 = float(input(f"[bull call spread] lower strike (< {self.S_0:.2f}): "))
        K2 = float(input(f"[bull call spread] upper strike (> {K1}): "))
        call1 = self.create_option('call', K1, 'long')
        call2 = self.create_option('call', K2, 'short')

        self.options.append(call1)
        self.options.append(call2)
        self.strategy_name = "Bull Call Spread"

        return self.options

    def bear_put_spread(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K1 = float(input(f"[bear put spread] upper strike (> {self.S_0:.2f}): "))
        K2 = float(input(f"[bear put spread] lower strike (< {K1}): "))
        put1 = self.create_option('put', K1, 'long')
        put2 = self.create_option('put', K2, 'short')

        self.options.append(put1)
        self.options.append(put2)
        self.strategy_name = "Bear Put Spread"

        return self.options

    def credit_call_spread(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K1 = float(input(f"[credit call spread] upper strike (> {self.S_0:.2f}): "))
        K2 = float(input(f"[credit call spread] lower strike (< {K1}): "))
        call1 = self.create_option('call', K1, 'long')
        call2 = self.create_option('call', K2, 'short')

        self.options.append(call1)
        self.options.append(call2)
        self.strategy_name = "Credit Call Spread (Bearish)"

        return self.options

    def credit_put_spread(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K1 = float(input(f"[credit put spread] lower strike (< {self.S_0:.2f}): "))
        K2 = float(input(f"[credit put spread] upper strike (> {K1}): "))
        put1 = self.create_option('put', K1, 'long')
        put2 = self.create_option('put', K2, 'short')

        self.options.append(put1)
        self.options.append(put2)
        self.strategy_name = "Credit Put Spread (Bullish)"

        return self.options

    def protective_collar(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K1 = float(input(f"[protective collar] OTM put strike (< {self.S_0:.2f}): "))
        K2 = float(input(f"[protective collar] OTM call strike (> {self.S_0:.2f}): "))
        put = self.create_option('put', K1, 'long')
        call = self.create_option('call', K2, 'short')
        stock = self.create_option('stock', self.S_0, 'long')

        self.options.extend([put, call, stock])
        self.strategy_name = "Protective Collar"

        return self.options

    def long_straddle(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"[long straddle] straddle strike: "))
        call = self.create_option('call', K, 'long')
        put  = self.create_option('put',  K, 'long')

        self.options.extend([call, put])
        self.strategy_name = "Long Straddle"
        return self.options

    def long_strangle(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        Kp = float(input(f"[long strangle] OTM put strike (< {self.S_0:.2f}): "))
        Kc = float(input(f"[long strangle] OTM call strike (> {self.S_0:.2f}): "))
        call = self.create_option('call', Kc, 'long')
        put  = self.create_option('put',  Kp, 'long')

        self.options.extend([call, put])
        self.strategy_name = "Long Strangle"
        return self.options

    def short_straddle(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K = float(input(f"[short straddle] straddle strike: "))
        call = self.create_option('call', K, 'short')
        put  = self.create_option('put',  K, 'short')

        self.options.extend([call, put])
        self.strategy_name = "Short Straddle"
        return self.options

    def short_strangle(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        Kp = float(input(f"[short strangle] OTM put strike (< {self.S_0:.2f}): "))
        Kc = float(input(f"[short strangle] OTM call strike (> {self.S_0:.2f}): "))
        call = self.create_option('call', Kc, 'short')
        put  = self.create_option('put',  Kp, 'short')

        self.options.extend([call, put])
        self.strategy_name = "Short Strangle"
        return self.options

    def long_call_butterfly_spread(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K1 = float(input(f"[butterfly] low strike (long call): "))
        K2 = float(input(f"[butterfly] mid strike (short 2×): "))
        K3 = float(input(f"[butterfly] high strike (long call): "))
        c1 = self.create_option('call', K1, 'long')
        c2a = self.create_option('call', K2, 'short')
        c2b = self.create_option('call', K2, 'short')
        c3 = self.create_option('call', K3, 'long')
        
        self.options.extend([c1, c2a, c2b, c3])
        self.strategy_name = "Long Call Butterfly Spread"
        return self.options

    def short_call_butterfly_spread(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        K1 = float(input(f"[short butterfly] low strike (short call): "))
        K2 = float(input(f"[short butterfly] mid strike (long 2×): "))
        K3 = float(input(f"[short butterfly] high strike (short call): "))
        c1 = self.create_option('call', K1, 'short')
        c2a = self.create_option('call', K2, 'long')
        c2b = self.create_option('call', K2, 'long')
        c3 = self.create_option('call', K3, 'short')

        self.options.extend([c1, c2a, c2b, c3])
        self.strategy_name = "Short Call Butterfly Spread"
        return self.options

    def iron_condor(self):
        print(f"+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+\n|{self.ticker} current price: {self.S_0:.2f}|\n+{'-'*len(f'{self.ticker} current price: {self.S_0:.2f}')}+")
        self.options = []
        Kp1 = float(input(f"[iron condor] long put strike (lowest): "))
        Kp2 = float(input(f"[iron condor] short put strike: "))
        Kc1 = float(input(f"[iron condor] short call strike: "))
        Kc2 = float(input(f"[iron condor] long call strike (highest): "))
        p1 = self.create_option('put',  Kp1, 'long')
        p2 = self.create_option('put',  Kp2, 'short')
        c1 = self.create_option('call', Kc1, 'short')
        c2 = self.create_option('call', Kc2, 'long')

        self.options.extend([p1, p2, c1, c2])
        self.strategy_name = "Iron Condor"
        return self.options
