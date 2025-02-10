import sys
import os
import datetime

sys.path.append(os.path.abspath("equity-options"))
sys.path.append(os.path.abspath("fixed-income"))

from optionstrategies import OptionStrategy
from exotics import DigitalOption, SinglePeriodRangeAccrual, AsianOption
from bonds import Bond, ZeroCouponBond, ZeroCouponBondOption, Caplet, Floorlet
from currentbonds import treasury_yield, plot_yield_curve

def main():
    while True:
        print("\nWhat would you like to price?")
        print("1. Equity Options")
        print("2. Fixed Income")
        print("3. Exit")
        choice = input("Select an option (1-3): ")
        
        if choice == '1':
            handle_equity_options()
        elif choice == '2':
            handle_fixed_income()
        elif choice == '3':
            print("exit")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

def handle_equity_options():
    print("\nEquity Options Selected.")
    ticker = input("Enter the stock ticker: ")
    T = float(input("Enter time to maturity (in years): "))
    r = treasury_yield(T)
    n = int(input("Enter the number of periods in the binomial model: "))
    percent_itm_otm = float(input("Enter the percent in-the-money/out-the-money (e.g., 0.2): "))
    exotic = input("Do you want an exotic option? (yes/no): ").strip().lower()

    if exotic == "yes":
        print("\nAvailable Exotic Options:")
        exotic_strategies = [
            "Digital Call Option",
            "Single Period Range Accrual",
            "Asian Call Option"
        ]
    
        for i, strat in enumerate(exotic_strategies, 1):
            print(f"{i}. {strat}")
    
        exotic_choice = int(input(f"Select an exotic option (1-{len(exotic_strategies)}) : "))

        if exotic_choice == 1:  # Digital Call Option
            digital_option_strike = 250
            option_type = "call"
            payoff_amount = 1

            digital_call_option = DigitalOption(ticker, r, T, digital_option_strike, option_type, payoff_amount)
            digital_call_option.price()
            digital_call_option.visualize_payoff()

        elif exotic_choice == 2:  # Single Period Range Accrual
            ra_strike_low = 225
            ra_strike_high = 275

            single_period_range_accrual = SinglePeriodRangeAccrual(ticker, r, T, ra_strike_low, ra_strike_high, payoff_amount)
            single_period_range_accrual.price()
            single_period_range_accrual.visualize_payoff()

        elif exotic_choice == 3:  # Asian Call Option
            asian_option_strike = 250

            asian_call_option = AsianOption(ticker, r, T, asian_option_strike, option_type)
            asian_call_option.price()

        else:
            print("Invalid choice. Please restart and select a valid option.")

    else:
        strategy = OptionStrategy(ticker, percent_itm_otm, T, r, n)
        print("\nAvailable Standard Option Strategies:")
    
        strategies = [
            "atm_call", "itm_call", "otm_call", "short_atm_call", "short_itm_call", "short_otm_call",
            "atm_put", "itm_put", "otm_put", "short_atm_put", "short_itm_put", "short_otm_put",
            "covered_call", "married_put", "bull_call_spread", "bear_put_spread", "credit_call_spread", 
            "credit_put_spread", "protective_collar", "long_straddle", "long_strangle", "short_straddle", 
            "short_strangle", "long_call_butterfly_spread", "short_call_butterfly_spread", "iron_condor"
        ]
    for i, strat in enumerate(strategies, 1):
        print(f"{i}. {strat}")
    strat_choice = int(input("Select a strategy (1-{}) : ".format(len(strategies))))
    
    if 1 <= strat_choice <= len(strategies):
        selected_strategy = strategies[strat_choice - 1]
        getattr(strategy, selected_strategy)()
        strategy.strategy_price()
        strategy.greeks()
        strategy.visualize_payoff(False)
    else:
        print("Invalid choice. Returning to main menu.")

def handle_fixed_income():
    print("\nFixed Income Selected.")
    print("1. Zero Coupon Bond")
    print("2. Option on Zero Coupon Bond")
    print("3. Caplet/Floorlet")
    print("4. Plot Yield Curve")
    choice = input("Select an option (1-4): ")
    
    if choice == '1':
        face_value = float(input("Enter face value: "))
        T = float(input("Enter maturity (years): "))
        r_0 = float(input("Enter initial interest rate: "))
        u = float(input("Enter up-factor: "))
        d = float(input("Enter down-factor: "))
        
        zcb = ZeroCouponBond(face_value, T, r_0, u, d)
        zcb.price()
        zcb.print_bond_tree()
        zcb.print_interest_tree()
    
    elif choice == '2':
        face_value = float(input("Enter face value of bond: "))
        n = float(input("Enter maturity of bond (integer): "))
        r_0 = float(input("Enter initial interest rate: "))
        u = float(input("Enter up-factor: "))
        d = float(input("Enter down-factor: "))
        zcb_option_expiry = float(input("Enter option expiry (years): "))
        zcb_option_strike = float(input("Enter option strike price: "))
        
        zcb = ZeroCouponBond(face_value, n, r_0, u, d)
        option = ZeroCouponBondOption(zcb, zcb_option_strike, zcb_option_expiry)
        option.price()
        option.print_option_tree()
    
    elif choice == '3':
        cf_expiry = int(input("Enter expiry (years): "))
        cf_notional = float(input("Enter notional amount: "))
        caplet_strike = float(input("Enter caplet strike rate: "))
        floorlet_strike = float(input("Enter floorlet strike rate: "))
        r_0 = float(input("Enter initial interest rate: "))
        u = float(input("Enter up-factor: "))
        d = float(input("Enter down-factor: "))
        
        caplet = Caplet(r_0, caplet_strike, cf_expiry, u, d, cf_notional)
        caplet.price()
        caplet.print_caplet_tree()
        caplet.print_interest_tree()

        floorlet = Floorlet(r_0, floorlet_strike, cf_expiry, u, d, cf_notional)
        floorlet.price()
        floorlet.print_floorlet_tree()
        floorlet.print_interest_tree()
    
    elif choice == '4':
        print(f"Plotting yield curve... (Date: {datetime.datetime.today().strftime('%Y-%m-%d')})")
        plot_yield_curve()
    
    else:
        print("Invalid choice. Returning to main menu.")

if __name__ == "__main__":
    main()