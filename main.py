import datetime
import warnings
warnings.filterwarnings("ignore")

from options.core.option import Option
from options.core.strategy import OptionStrategy
from options.core.analyzer import Analyzer
from options.exotics.asian import AsianOption
from options.exotics.digital import DigitalOption
from options.exotics.range import SinglePeriodRangeAccrual
from options.utilities.marketdata import MarketDataFetcher

from fixed_income.core.bond import Bond
from fixed_income.core.zcb import ZeroCouponBond
from fixed_income.derivatives.zcb_option import ZeroCouponBondOption
from fixed_income.derivatives.caplet import Caplet
from fixed_income.derivatives.floorlet import Floorlet
from fixed_income.core.marketyields import plot_yield_curve
from fixed_income.core.bootstrap import plot_zc_yields

def main():
    while True:
        print("\n-----------------------------------------------")
        print("\t\t\tMENU\t\t\t")
        print("-----------------------------------------------")
        print("1. equity options")
        print("2. fixed income")
        print("3. exit")
        choice = input("enter selection (1-3): ")
        
        if choice == '1':
            handle_equity_options()
        elif choice == '2':
            handle_fixed_income()
        elif choice == '3':
            print("exit")
            break
        else:
            print("invalid choice: must be 1, 2, or 3.")

def handle_equity_options():
    print("\n-----------------------------------------------")
    print("\t\tEQUITY OPTIONS\t\t")
    print("-----------------------------------------------")

    choice = input("enter 1 for vanilla options, 2 for exotics, 3 for analyzer: ")

    if choice == '1':
        print("\n-----------------------------------------------")
        print("\t\tVANILLA OPTIONS\t\t")
        print("-----------------------------------------------")
        ticker = input("stock ticker: ").upper().strip()
        T = get_yrs()
        n = max(int(input("desired number of periods in the binomial model: ")), 10)

        strategy = OptionStrategy(ticker, T, n)
        print("\noption strategies available:")
    
        strategies = [
            "atm_call", "itm_call", "otm_call", "short_atm_call", "short_itm_call", "short_otm_call",
            "atm_put", "itm_put", "otm_put", "short_atm_put", "short_itm_put", "short_otm_put",
            "covered_call", "married_put", "bull_call_spread", "bear_put_spread", "credit_call_spread", 
            "credit_put_spread", "protective_collar", "long_straddle", "long_strangle", "short_straddle", 
            "short_strangle", "long_call_butterfly_spread", "short_call_butterfly_spread", "iron_condor"
        ]
        for i, strat in enumerate(strategies, 1):
            print(f"{i}. {strat}")
        strat_choice = int(input("select a strategy (1-{}) : ".format(len(strategies))))

        if 1 <= strat_choice <= len(strategies):
            print()
            selected_strategy = strategies[strat_choice - 1]
            getattr(strategy, selected_strategy)()
            strategy.strategy_price()
            strategy.greeks()
            strategy.visualize_payoff(False)
        else:
            print("invalid choice: must be between 1 and {}".format(len(strategies)))
            return
        
        # dummy option to generate plots
        option = Option(ticker, T, 0, n, option_type='call')
        plotsvi = input(f"plot SVI calibrated vols for {ticker} (yes/no)? ").strip().lower()
        if plotsvi == 'yes': 
            try: 
                option.plot_svi_calibration()
            except:
                pass
         
        plotskew = input(f"plot vol skew for {ticker} (yes/no)? ").strip().lower()
        if plotskew == 'yes':
            try:
                option.plot_implied_vols()
            except: 
                pass

    elif choice == '2': 
        print("\n-----------------------------------------------")
        print("\t\tEXOTIC OPTIONS\t\t")
        print("-----------------------------------------------")
        ticker = input("stock ticker: ").upper().strip()
        T = get_yrs()
        
        print("\navailable exotics:")
        exotic_strategies = [
            "digital option",
            "single period range accrual",
            "asian call option"
        ]
    
        for i, strat in enumerate(exotic_strategies, 1):
            print(f"{i}. {strat}")
    
        exotic_choice = int(input(f"select an exotic (1-{len(exotic_strategies)}) : "))

        if exotic_choice == 1:  
            print("\n**** DIGITAL OPTION ****")
            option_type = input("enter option type (call/put): ").strip().lower()
            if option_type not in ["call", "put"]:
                print("invalid option type: must be 'call' or 'put', using call as default")
                option_type = 'call'

            digital_option_strike = input("strike : ")
            payoff_amount = input("payoff (notional): ")

            digital_call_option = DigitalOption(ticker, T, digital_option_strike, option_type, payoff_amount)
            digital_call_option.price()
            digital_call_option.visualize_payoff()

        elif exotic_choice == 2:  
            print("\n**** SINGLE-PERIOD RANGE ACCRUAL ****")
            ra_strike_low = input("low strike: ")
            ra_strike_high = input("high strike: ")
            payoff_amount = input("payoff (notional): ")

            single_period_range_accrual = SinglePeriodRangeAccrual(ticker, T, ra_strike_low, ra_strike_high, payoff_amount)
            single_period_range_accrual.price()
            single_period_range_accrual.visualize_payoff()

        elif exotic_choice == 3: 
            print("\n**** ASIAN OPTION ****")
            asian_option_strike = input("strike: ")

            asian_call_option = AsianOption(ticker, T, asian_option_strike)
            asian_call_option.price()

        else:
            print("invalid choice: must be between 1 and {}".format(len(exotic_strategies)))

    elif choice == '3':
        ticker = input("stock ticker: ").upper().strip()
        T = get_yrs()
        fetcher = MarketDataFetcher(ticker, T, creation_date=None)   

        print(f"\nlaunching {ticker} options analyzer...\n")
        analyzer = Analyzer(ticker, T, fetcher=fetcher)
        analyzer.launch_analyzer()
        analyzer.plot_vols()

    else:
        print("invalid choice: returning to menu.")

def handle_fixed_income():
    print("\n-----------------------------------------------")
    print("\t\tFIXED INCOME\t\t")
    print("-----------------------------------------------")
    print("1. bonds")
    print("2. zero coupon bond")
    print("3. option on zero coupon bond")
    print("4. caplet")
    print("5. floorlet")
    print("6. plot the current yield curve")
    choice = input("select an option (1-5): ")

    if choice == '1':
        print("\n**** BOND ****")
        issue_date = input("issue date (YYYY-MM-DD) (enter 1 to use today's date): ")
        if issue_date == '1':
            issue_date = datetime.datetime.today().strftime('%Y-%m-%d')
        else:
            try:
                issue_date = datetime.datetime.strptime(issue_date, '%Y-%m-%d')
            except ValueError:
                print("invalid date format: must be YYYY-MM-DD")
                return

        selection = input("enter maturity as number of years (enter 1) or date (enter 2): ")
        if selection == 1: 
            maturity = get_yrs()
            maturity_date = None
        elif selection == 2:
            maturity_date = input("maturity date (YYYY-MM-DD): ")
            maturity = None
        else: 
            print("invalid selection: must be 1 or 2")
            return
        
        purchase_date = input("purchase date (YYYY-MM-DD) (enter 1 to use today's date): ")
        if purchase_date == '1':
            purchase_date = datetime.datetime.today().strftime('%Y-%m-%d')
        else:
            try:
                purchase_date = datetime.datetime.strptime(purchase_date, '%Y-%m-%d')
            except ValueError:
                print("invalid date format: must be YYYY-MM-DD")
                return
        
        face_value = float(input("face value: "))
        coupon_rate = float(input("coupon rate (as decimal, e.g. 0.05 for 5%): "))
        freq_type = input("coupons paid per year (e.g. enter 2 for semiannual, 12 for monthly, etc.): ").strip().lower()
        
        if freq_type.isdigit():
            freq_type = int(freq_type)
        else:
            print("invalid frequency type: must be an integer")
            return
        
        bond = Bond(face_value, coupon_rate, maturity, freq_type, issue_date, maturity_date, purchase_date)
        bond.summary()
        
        plot = input("plot clean vs. dirty prices over time? (yes/no): ").strip().lower()
        if plot == 'yes':
            bond.plot_price_trajectory()

    elif choice == '2':
        print("\n**** ZERO COUPON BOND ****")
        issue_date = input("issue date (YYYY-MM-DD) (enter 1 to use today's date): ")
        if issue_date == '1':
            issue_date = datetime.datetime.today().strftime('%Y-%m-%d')
        else:
            try:
                issue_date = datetime.datetime.strptime(issue_date, '%Y-%m-%d')
            except ValueError:
                print("invalid date format: must be YYYY-MM-DD")
                return

        selection = input("enter maturity as number of years (enter 1) or date (enter 2): ")
        if selection == 1: 
            maturity = get_yrs()
            maturity_date = None
        elif selection == 2:
            maturity_date = input("maturity date (YYYY-MM-DD): ")
            maturity = None
        else: 
            print("invalid selection: must be 1 or 2")
            return
        
        purchase_date = input("purchase date (YYYY-MM-DD) (enter 1 to use today's date): ")
        if purchase_date == '1':
            purchase_date = datetime.datetime.today().strftime('%Y-%m-%d')
        else:
            try:
                purchase_date = datetime.datetime.strptime(purchase_date, '%Y-%m-%d')
            except ValueError:
                print("invalid date format: must be YYYY-MM-DD")
                return
        
        face_value = float(input("face value: "))
        u = float(input("[binomial model] up-factor (u): "))
        d = float(input("[binomial model] down-factor (d): "))

        zcb = ZeroCouponBond(face_value, u, d, maturity, issue_date, maturity_date, purchase_date)
        zcb.summary()
        plot = input("plot zcb prices over time? (yes/no): ").strip().lower()
        if plot == 'yes':
            zcb.plot_price_trajectory()
    
    elif choice == '3':
        print("\n**** OPTION ON ZERO COUPON BOND ****")
        face_value = float(input("face value of underlying zero coupon bond: "))
        n = float(input("maturity of bond (integer years): "))
        r_0 = float(input("initial interest rate: "))
        u = float(input("up-factor (u): "))
        d = float(input("down-factor (d): "))
        zcb_option_expiry = float(input("option expiry (years): "))
        zcb_option_strike = float(input("enter option strike price: "))
        
        zcb = ZeroCouponBond(face_value, n, r_0, u, d)
        option = ZeroCouponBondOption(zcb, zcb_option_strike, zcb_option_expiry)
        option.price()
        option.print_option_tree()
    
    elif choice == '4':
        print("\n**** CAPLET ****")
        cf_expiry = int(input("expiry (years): "))
        cf_notional = float(input("notional amount: "))
        caplet_strike = float(input("strike rate: "))
        r_0 = float(input("initial interest rate: "))
        u = float(input("up-factor (u): "))
        d = float(input("down-factor (d): "))

        if not (0 < d < 1 + r_0 < u):
            print("invalid: we require 0 < d < 1 + r_0 < u")
            return
        
        caplet = Caplet(r_0, caplet_strike, cf_expiry, u, d, cf_notional)
        caplet.price()
        caplet.print_caplet_tree()
        caplet.print_interest_tree()

        floorlet = Floorlet(r_0, floorlet_strike, cf_expiry, u, d, cf_notional)
        floorlet.price()
        floorlet.print_floorlet_tree()
        floorlet.print_interest_tree()

    elif choice == '5':
        print("\n**** FLOORLET ****")
        cf_expiry = int(input("expiry (years): "))
        cf_notional = float(input("notional amount: "))
        floorlet_strike = float(input("strike rate: "))
        r_0 = float(input("initial interest rate: "))
        u = float(input("up-factor (u): "))
        d = float(input("down-factor (d): "))

        if not (0 < d < 1 + r_0 < u): 
            print("invalid: we require 0 < d < 1 + r_0 < u")
            return

        floorlet = Floorlet(r_0, floorlet_strike, cf_expiry, u, d, cf_notional)
        floorlet.price()
        floorlet.print_floorlet_tree()
        floorlet.print_interest_tree()
    
    elif choice == '6':
        print(f"plotting yield curve... (date: {datetime.datetime.today().strftime('%Y-%m-%d')})")
        plot_yield_curve()
        zc_plot = input("plot the bootstrapped zero coupon yield curve? (yes/no): ").strip().lower()
        if zc_plot == 'yes':
            print(f"plotting bootstrapped zero coupon yields... (date: {datetime.datetime.today().strftime('%Y-%m-%d')})")
            plot_zc_yields()
    else:
        print("invalid choice: returning to menu")

def get_yrs(): 
    T = input("maturity (enter with trailing 'y', 'm', or 'd', e.g. 10y, 1m, 5d, etc.): ").strip().lower()
    if T.endswith('y'):
        return float(T[:-1])
    elif T.endswith('m'):
        return float(T[:-1]) / 12
    elif T.endswith('d'):
        return float(T[:-1]) / 365
    else:
        print("invalid: use 'y', 'm', or 'd', try again")
        return get_yrs()
    
if __name__ == "__main__":
    main()