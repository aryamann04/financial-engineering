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
        ["Binomial", "European", f"${prices["Binomial European"]:.3f}"],
        ["Binomial", "American", f"${prices["Binomial American"]:.3f}"],
        ["Black-Scholes", "European", f"${prices["Black-Scholes"]:.3f}"],
        ["Monte Carlo", "European", f"${prices["Monte Carlo"]:.3f}"],
        ["Actual Market", "(yfinance API)", f"${prices["Market Price"]:.3f}" or "N/A"]
    ]
    print(tabulate(price_table, headers=["Model", "Option Type", "Price"], tablefmt="grid"))

    print("\n********** VOLATILITY **********")
    vol_table = [
        [f"Historical Vol (over {option.T:.2f} years)", fmt_pct(prices['Historical Vol'])],
        ["Model Vol (SVI)", fmt_pct(prices['Model Vol'])],
        ["Market Implied Vol (via BS)", fmt_pct(prices['Implied Vol'])],
        ["Market Implied Vol (via yfinance)", fmt_pct(prices['Implied Vol (yf)'])]
    ]   
    print(tabulate(vol_table, headers=["Type", "Value"], tablefmt="grid"))

    print("\n********** GREEKS **********")
    greek_table = [[k, f"{v:.4f}"] for k, v in option.greeks.items()]
    print(tabulate(greek_table, headers=["Greek", "Value"], tablefmt="grid"))

def fmt_pct(x):
    return f"{x*100:.2f}%" if isinstance(x, (int, float)) else "N/A"