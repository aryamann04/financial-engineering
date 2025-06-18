from tabulate import tabulate

def print_option_summary(option):
    print("\n********** OPTION PARAMETERS **********")
    print(tabulate([
        ["Ticker", option.ticker],
        ["Risk-Free Rate", f"{option.r*100:.2f}%"],
        ["Dividend Yield", f"{option.q*100:.2f}%"],
        ["Time to Expiry", f"{option.T:.2f} years"],
        ["Strike Price", f"{option.K}"],
        ["Volatility", f"{option.sigma*100:.2f}%"],
        ["Option Type", option.option_type.capitalize()],
        ["Position", option.position.capitalize()]
    ], headers=["Parameter", "Value"], tablefmt="grid"))

    print("\n********** PRICES **********")
    prices = option.price_summary()
    price_table = [
        ["Binomial", "European", prices["Binomial European"]],
        ["Binomial", "American", prices["Binomial American"]],
        ["Black-Scholes", "European", prices["Black-Scholes"]],
        ["Monte Carlo", "European", prices["Monte Carlo"]],
        ["Actual Market", "(yfinance API)", prices["Market Price"] or "N/A"]
    ]
    print(tabulate(price_table, headers=["Model", "Option Type", "Price"], tablefmt="grid"))

    print("\n********** VOLATILITY **********")
    vol_table = [
        ["Historical (Model Input)", f"{prices['Model Vol']*100:.2f}%"],
        ["Market Implied Vol", f"{(prices['Implied Vol']*100 if prices['Implied Vol'] else 'N/A')}"]
    ]
    print(tabulate(vol_table, headers=["Type", "Value"], tablefmt="grid"))

    print("\n********** GREEKS **********")
    greeks = option.greeks_dict()
    greek_table = [[k, f"{v:.4f}"] for k, v in greeks.items()]
    print(tabulate(greek_table, headers=["Greek", "Value"], tablefmt="grid"))
