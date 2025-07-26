import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import fsolve

from fixed_income.core.marketyields import treasury_yield

def par_price(y, c, T): 
    # zero coupon convention for short-dated
    if T < 1: 
        return 100 / (1 + y * T)
    
    periods = int(T * 2) 
    c_rate = c / 2 
    
    # assume face value of 100
    cash_flows = np.full(periods, c_rate * 100)
    cash_flows[-1] += 100 
    
    discounted = np.array([(1 + y/2) ** (-1 * n) for n in range(1, periods + 1)])
    return (cash_flows * discounted).sum()

def build_instruments(): 
    # use live treasury yields to build hypothetical collection of instruments
    T = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    instruments = []

    for t in T: 
        y = treasury_yield(t) 
        instruments.append({
            'maturity': t,
            'coupon': y, 
            'market price': par_price(y, y, t)
        })
    
    return instruments

def print_instruments(instruments): 
    for i in instruments: 
        print(f"Maturity: {i['maturity']:.2f}y, Coupon: {i['coupon'] * 100:.3f}%, Market Price: {i['market price']:.3f}")

def init_disc_factors(instruments): 
    short_t = [1/12, 2/12, 3/12, 4/12, 6/12]

    D = {0.0: 1.0}
    for t in short_t:
        D[t] = 1 / (1 + treasury_yield(t) * (t * 365 / 360)) # ACT/360
    
    return D

def bootstrap(instruments, interpolation='linear'):
    d_factors = init_disc_factors(instruments)
    known_times = sorted(d_factors.keys())

    for i in instruments:
        maturity = float(i['maturity'])
        if maturity in d_factors:
            continue

        c_rate = i['coupon'] 
        price = i['market price']
        periods = int(maturity * 2)

        cash_flows = np.array([c_rate / 2 * 100] * periods)
        cash_flows[-1] += 100  
        times = np.arange(0.5, maturity + 0.1, 0.5)
        times = [float(t) for t in times]

        def equation(d_tn):
            d_factors_temp = d_factors.copy()
            d_factors_temp[maturity] = d_tn[0]
            pv = 0
            for t, cf in zip(times, cash_flows):
                D_t = interpolate_d(t, d_factors_temp, interpolation)
                pv += cf * D_t
            return pv - price

        d_tn_initial = [0.9]
        d_tn_solution = fsolve(equation, d_tn_initial)
        d_factors[maturity] = d_tn_solution[0]
        known_times.append(maturity)
        known_times.sort()
    return d_factors

def interpolate_d(t, d_factors, method='log-linear'):
    t = float(t)
    if t <= 0:
        return 1.0
    if t in d_factors:
        return d_factors[t]
    else:
        known_times_temp = sorted(d_factors.keys())
        t1 = max([k for k in known_times_temp if k < t], default=None)
        t2 = min([k for k in known_times_temp if k > t], default=None)
        if t1 is None or t2 is None:
            raise ValueError(f"cannot interpolate for t={t}: insufficient known discount factors.")
        D1 = d_factors[t1]
        D2 = d_factors[t2]
        if method == 'linear':
            return D1 + (t - t1) / (t2 - t1) * (D2 - D1)
        elif method == 'log-linear':
            return D1 * (D2 / D1) ** ((t - t1) / (t2 - t1))

def full_disc_factors(d_factors, method='log-linear'):
    times = np.arange(1/12, 30 + 1/12, 1/12)
    full_d = {}
    for t in times:
        if t in d_factors:
            full_d[t] = d_factors[t]
        else:
            full_d[t] = interpolate_d(t, d_factors, method)
    return full_d

def zero_coupon_yields(d_factors):
    yields = {}
    for t in d_factors:
        if t > 0:
            s_t = 2 * (d_factors[t] ** (-1 / (2 * t)) - 1)
            yields[t] = s_t  
    return yields

def plot(yields):
    fig, ax = plt.subplots(figsize=(10, 6))
    today = datetime.today()
    maturities = sorted(yields.keys())
    zc_dates = [ today + timedelta(days=int(t*365)) for t in maturities ]
    zc_vals = [ yields[t] * 100 for t in maturities ]
    
    T = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    t_dates = [ today + timedelta(days=int(t*365)) for t in T ]
    t_vals = [ treasury_yield(t) * 100 for t in T ]
    
    ax.plot(zc_dates, zc_vals, '-o', color='orange', label='bootstrapped zero-coupon yields')
    ax.plot(t_dates,  t_vals,  '-o', color='blue', label='treasury yields')
    ax.set_xticks(t_dates)

    today = datetime.today()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=today.month, day=today.day))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('maturity dates')
    ax.set_ylabel('yield (%)')
    ax.set_title(f'bootstrapped zero-coupon vs treasury yields as of {today.strftime("%Y-%m-%d")}')
    ax.legend()
    plt.tight_layout()
    plt.show()
 
 # ----------------------------------------------------------------------------- #
def plot_zc_yields():
    plot(get_zc_yields())

def get_zc_yields():
    return zero_coupon_yields(full_disc_factors(bootstrap(build_instruments())))

def zc_yield(t):
    zc_yields = get_zc_yields()
    maturities = list(zc_yields.keys())
    closest = min(maturities, key=lambda tau: abs(tau - t))
    return zc_yields[closest]
    
def get_disc_factors(): 
    return full_disc_factors(bootstrap(build_instruments()))
