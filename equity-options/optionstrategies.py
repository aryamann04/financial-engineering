from option import create_option
from optionspricing import stock_data
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

class OptionStrategy:
    def __init__(self, ticker, percent_otm_itm, expiry_date, rf, n=100):
        self.ticker = ticker
        self.stock_price, self.sigma = stock_data(ticker)
        self.percent_otm_itm = percent_otm_itm
        self.expiry_date = expiry_date
        self.rf = rf
        self.n = n
        self.options = []
        self.strategy_name = ""
        self.total_price = 0
        self.total_market_price = 0

    def create_option(self, option_type, strike_price, position='long'):
        option = create_option(self.ticker, self.rf, self.expiry_date, strike_price, self.n, option_type, position)
        itm_otm = ""
        percent_itm_otm = abs((strike_price - self.stock_price) / self.stock_price)

        if option_type == "call":
            if strike_price < self.stock_price:
                itm_otm = "ITM"
                print(f"Created ITM call option with strike price {strike_price:.3f}")
            elif strike_price > self.stock_price:
                itm_otm = "OTM"
                print(f"Created OTM call option with strike price {strike_price:.3f}")
            else:
                itm_otm = "ATM"
                print(f"Created ATM call option with strike price {strike_price:.3f}")
        elif option_type == "put":
            if strike_price > self.stock_price:
                itm_otm = "ITM"
                print(f"Created ITM put option with strike price {strike_price:.3f}")
            elif strike_price < self.stock_price:
                itm_otm = "OTM"
                print(f"Created OTM put option with strike price {strike_price:.3f}")
            else:
                itm_otm = "ATM"
                print(f"Created ATM put option with strike price {strike_price:.3f}")
        elif option_type == "stock":
            itm_otm = "N/A"
            print(f"Created stock position at price {strike_price:.3f}")

        # Print option details
        print(f"Option Type: {option_type.capitalize()}")
        print(f"Position: {position.capitalize()}")

        if option_type != 'stock':
            print(f"Strike Price: {strike_price:.3f}")
            print(f"Black-Scholes Price: {option.price:.3f}")

            if option.market is not None:
                print(f"Market Price: {option.market}")
            else:
                print(f"Market Price: does not exist")

            print(f"Moneyness: {itm_otm}")
            print(f"Percent ITM/OTM: {percent_itm_otm:.3f}")

            option.summary()

        print()

        return option

    def strategy_price(self):
        print(f"\n********** STRATEGY **********")
        print(f"{self.ticker} {self.percent_otm_itm*100}% {self.strategy_name}")
        print(f"******************************\n")
        self.total_price = sum(option.price if option.position == 'long' else -option.price for option in self.options)
        self.total_market_price = sum(option.market if option.position == 'long' else -option.market for option in self.options)

        print(f"\n********** STRATEGY PRICE **********\n")
        price_table = [
            ["Black-Scholes", f"${self.total_price:.2f}"],
            ["Market", f"${self.total_market_price:.2f}"],
        ]
        print(tabulate(price_table, headers=["Type", "Price"], tablefmt="grid"))
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
                    delta += option.delta
                    gamma += option.gamma
                    theta += option.theta
                    vega += option.vega
                    rho += option.rho
                else:  # short position
                    delta -= option.delta
                    gamma -= option.gamma
                    theta -= option.theta
                    vega -= option.vega
                    rho -= option.rho
            else:  # for stock
                if option.position == 'long':
                    delta += 1
                else:  # short stock position
                    delta -= 1

        print(f"\n********** STRATEGY GREEKS **********\n")
        greeks_table = [
            ["Delta", f"{delta:.4f}"],
            ["Gamma", f"{gamma:.4f}"],
            ["Theta", f"{theta:.4f}"],
            ["Vega", f"{vega:.4f}"],
            ["Rho", f"{rho:.4f}"]
        ]
        print(tabulate(greeks_table, headers=["Greek", "Value"], tablefmt="grid"))

        return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}

    def visualize_payoff(self):
        stock_prices = np.linspace(self.stock_price * 0.5, self.stock_price * 1.5, 1000)
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

        total_profit_loss = total_payoff - self.total_price

        plt.figure(figsize=(10, 6))
        plt.plot(stock_prices, total_profit_loss, label=f'{self.strategy_name} P/L')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)

        break_even_points = stock_prices[np.isclose(total_profit_loss, 0, atol=0.1)]
        break_even_points = np.unique(np.round(break_even_points, 2))

        if len(break_even_points) == 0:
            print("\nBreak-even points: N/A\n")
        else:
            print(f"\nBreak-even points: {', '.join([f'{point:.2f}' for point in break_even_points])}\n")
            for point in break_even_points:
                plt.axvline(point, color='red', linestyle='--', linewidth=0.5)
                plt.text(point, 0, f'{point:.2f}', verticalalignment='bottom')

        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.title(f'{self.ticker} {self.percent_otm_itm*100}% {self.strategy_name} Profit/Loss Diagram')
        plt.legend()
        plt.show()

    def covered_call(self):
        self.options = []
        call = self.create_option('call', self.stock_price * (1 + self.percent_otm_itm), 'short')
        stock = self.create_option('stock', self.stock_price, 'long')

        self.options.append(call)
        self.options.append(stock)
        self.strategy_name = "Covered Call"

        return self.options

    def married_put(self):
        self.options = []
        put = self.create_option('put', self.stock_price * (1 - self.percent_otm_itm), 'long')
        stock = self.create_option('stock', self.stock_price, 'long')

        self.options.append(put)
        self.options.append(stock)
        self.strategy_name = "Married Put"

        return self.options

    def bull_call_spread(self):
        self.options = []
        call1 = self.create_option('call', self.stock_price * (1 - self.percent_otm_itm), 'long')
        call2 = self.create_option('call', self.stock_price * (1 + self.percent_otm_itm), 'short')

        self.options.append(call1)
        self.options.append(call2)
        self.strategy_name = "Bull Call Spread"

        return self.options

    def bear_put_spread(self):
        self.options = []
        put1 = self.create_option('put', self.stock_price * (1 + self.percent_otm_itm), 'long')
        put2 = self.create_option('put', self.stock_price * (1 - self.percent_otm_itm), 'short')

        self.options.append(put1)
        self.options.append(put2)
        self.strategy_name = "Bear Put Spread"

        return self.options

    def protective_collar(self):
        self.options = []
        put = self.create_option('put', self.stock_price * (1 - self.percent_otm_itm), 'long')
        call = self.create_option('call', self.stock_price * (1 + self.percent_otm_itm), 'short')
        stock = self.create_option('stock', self.stock_price, 'long')

        self.options.append(put)
        self.options.append(call)
        self.options.append(stock)
        self.strategy_name = "Protective Collar"

        return self.options

    def long_straddle(self):
        self.options = []
        call = self.create_option('call', self.stock_price, 'long')
        put = self.create_option('put', self.stock_price, 'long')

        self.options.append(call)
        self.options.append(put)
        self.strategy_name = "Long Straddle"

        return self.options

    def long_strangle(self):
        self.options = []
        call = self.create_option('call', self.stock_price * (1 + self.percent_otm_itm), 'long')
        put = self.create_option('put', self.stock_price * (1 - self.percent_otm_itm), 'long')

        self.options.append(call)
        self.options.append(put)
        self.strategy_name = "Long Strangle"

        return self.options

    def long_call_butterfly_spread(self):
        self.options = []
        call1 = self.create_option('call', self.stock_price * (1 - self.percent_otm_itm), 'long')
        call2a = self.create_option('call', self.stock_price, 'short')
        call2b = self.create_option('call', self.stock_price, 'short')
        call3 = self.create_option('call', self.stock_price * (1 + self.percent_otm_itm), 'long')

        self.options.append(call1)
        self.options.append(call2a)
        self.options.append(call2b)
        self.options.append(call3)
        self.strategy_name = "Long Call Butterfly Spread"

        return self.options

    def long_put_butterfly_spread(self):
        self.options = []
        put1 = self.create_option('put', self.stock_price * (1 + self.percent_otm_itm), 'long')
        put2a = self.create_option('put', self.stock_price, 'short')
        put2b = self.create_option('put', self.stock_price, 'short')
        put3 = self.create_option('put', self.stock_price * (1 - self.percent_otm_itm), 'long')

        self.options.append(put1)
        self.options.append(put2a)
        self.options.append(put2b)
        self.options.append(put3)
        self.strategy_name = "Long Put Butterfly Spread"

        return self.options
