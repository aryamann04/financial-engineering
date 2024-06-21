from option import create_option
import option
from optionspricing import stock_data
import numpy as np
import matplotlib.pyplot as plt

class OptionStrategy:
    def __init__(self, ticker, strike_price, expiry_date, rf, n):
        self.ticker = ticker
        self.stock_price, self.sigma = stock_data(ticker)
        self.strike_price = strike_price
        self.expiry_date = expiry_date
        self.rf = rf
        self.n = n
        self.options = []
        self.strategy_name = ""
        self.total_price = 0

    def create_option(self, option_type, strike_price, position='long'):
        return option.create_option(self.ticker, self.rf, self.expiry_date, strike_price, self.n, option_type, position)

    def calculate_strategy_price(self):
        self.total_price = sum(option.price if option.position == 'long' else -option.price for option in self.options)
        print(f'Total price of the {self.strategy_name}: {self.total_price}')

    def visualize_payoff(self):
        stock_prices = np.linspace(self.stock_price * 0.5, self.stock_price * 1.5, 100)
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
        for point in break_even_points:
            plt.axvline(point, color='red', linestyle='--', linewidth=0.5)
            plt.text(point, 0, f'{point:.2f}', verticalalignment='bottom')

        plt.xlabel('Stock Price')
        plt.ylabel('Profit/Loss')
        plt.title(f'{self.strategy_name} Profit/Loss Diagram')
        plt.legend()
        plt.show()

    def covered_call(self, p=0.1):
        call = self.create_option('call', self.strike_price, 'short')
        stock = self.create_option('stock', self.stock_price, 'long')

        self.options.append(call)
        self.options.append(stock)
        self.strategy_name = "Covered Call"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def married_put(self):
        put = self.create_option('put', self.strike_price, 'long')
        stock = self.create_option('stock', self.stock_price, 'long')

        self.options.append(put)
        self.options.append(stock)
        self.strategy_name = "Married Put"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def bull_call_spread(self, p=0.1):
        call1 = self.create_option('call', self.strike_price, 'long')
        call2 = self.create_option('call', self.strike_price * (1 + p), 'short')

        self.options.append(call1)
        self.options.append(call2)
        self.strategy_name = "Bull Call Spread"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def bear_put_spread(self, p=0.1):
        put1 = self.create_option('put', self.strike_price, 'long')
        put2 = self.create_option('put', self.strike_price * (1 - p), 'short')

        self.options.append(put1)
        self.options.append(put2)
        self.strategy_name = "Bear Put Spread"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def protective_collar(self, p=0.1):
        put = self.create_option('put', self.stock_price * (1 - p), 'long')
        call = self.create_option('call', self.stock_price * (1 + p), 'short')
        stock = self.create_option('stock', self.stock_price, 'long')

        self.options.append(put)
        self.options.append(call)
        self.options.append(stock)
        self.strategy_name = "Protective Collar"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def long_straddle(self):
        call = self.create_option('call', self.strike_price, 'long')
        put = self.create_option('put', self.strike_price, 'long')

        self.options.append(call)
        self.options.append(put)
        self.strategy_name = "Long Straddle"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def long_strangle(self, p=0.1):
        call = self.create_option('call', self.strike_price * (1 + p), 'long')
        put = self.create_option('put', self.strike_price * (1 - p), 'long')
        
        self.options.append(call)
        self.options.append(put)
        self.strategy_name = "Long Strangle"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def long_call_butterfly_spread(self, p=0.1):
        call1 = self.create_option('call', self.strike_price * (1 - p), 'long')
        call2 = self.create_option('call', self.strike_price, 'short')
        call3 = self.create_option('call', self.strike_price * (1 + p), 'long')
        
        self.options.append(call1)
        self.options.append(call2)
        self.options.append(call2)  # Selling two at-the-money calls
        self.options.append(call3)
        self.strategy_name = "Long Call Butterfly Spread"
        self.calculate_strategy_price()
        self.visualize_payoff()

    def long_put_butterfly_spread(self, p=0.1):
        put1 = self.create_option('put', self.strike_price * (1 + p), 'long')
        put2 = self.create_option('put', self.strike_price, 'short')
        put3 = self.create_option('put', self.strike_price * (1 - p), 'long')
        
        self.options.append(put1)
        self.options.append(put2)
        self.options.append(put2)  # Selling two at-the-money puts
        self.options.append(put3)
        self.strategy_name = "Long Put Butterfly Spread"
        self.calculate_strategy_price()
        self.visualize_payoff()

# Example usage
strat = OptionStrategy("AAPL", 250, 2, 0.05, 100)
strat.long_call_butterfly_spread()
