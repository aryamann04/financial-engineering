import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from options.volatility.iv import bs_iv

def closest_K_T(ticker_obj, K, T):
    if isinstance(ticker_obj, str): 
        import yfinance as yf
        from curl_cffi import requests 
        ticker_obj = yf.Ticker(ticker_obj, session=requests.Session(impersonate="chrome"))
    
    exp_dates = ticker_obj.options
    if not exp_dates:
        return K, T

    target_expiry = datetime.now() + timedelta(days=T * 365)
    closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))

    chain = ticker_obj.option_chain(closest_expiry)
    all_strikes = np.unique(
        np.concatenate([chain.calls["strike"].values,
                        chain.puts["strike"].values])
    )
    if all_strikes.size == 0:
        return K, closest_expiry

    closest_strike = float(min(all_strikes, key=lambda s: abs(s - K)))
    return closest_strike, closest_expiry

def yf_option_price(ticker_obj, K, T, option_type="call"):
    try:
        strike, expiry = closest_K_T(ticker_obj, K, T)
        if strike is None:
            return None, expiry

        options = ticker_obj.option_chain(expiry)
        df = options.calls if option_type == "call" else options.puts
        row = df[df["strike"] == strike]

        if not row.empty:
            return row["lastPrice"].values[0], expiry
        return None, expiry

    except Exception:
        return None, None


def get_yf_iv(tic, K, T, option_type="call"):
    try:
        strike, expiry = closest_K_T(tic, K, T)
        if strike is None:
            return None

        options = tic.option_chain(expiry)
        df = options.calls if option_type == "call" else options.puts
        row = df[df["strike"] == strike]

        return row["impliedVolatility"].values[0] if not row.empty else None

    except Exception:
        return None

def vol_skew(ticker_obj, expiry_years, strike):
    if not ticker_obj.options:
        raise ValueError(f"No options data available for ticker {ticker_obj.ticker}")

    vol_mid = get_yf_iv(ticker_obj, strike, expiry_years, "call")
    if vol_mid is None:
        raise ValueError(f"No implied volatility data available for strike {strike}")

    vol_lo = get_yf_iv(ticker_obj, strike - 5, expiry_years, "call")
    if vol_lo is None:
        raise ValueError(f"No implied volatility data available for strike {strike-5}")

    vol_hi = get_yf_iv(ticker_obj, strike + 5, expiry_years, "call")
    if vol_hi is None:
        raise ValueError(f"No implied volatility data available for strike {strike+5}")

    return (vol_hi - vol_lo) / ((strike + 5) - (strike - 5))


def plot_vol_skew(ticker_obj, S_0, T, r, q, option_type="call"):
    today = datetime.today()
    expiry_date = today + timedelta(days=T * 365)

    options = ticker_obj.options

    if not options:
        raise ValueError(f"No options data available for ticker {ticker_obj.ticker}")

    option_dates = [datetime.strptime(exp_date, '%Y-%m-%d') for exp_date in options]
    expiry_date_str = min(option_dates, key=lambda x: abs(x - expiry_date)).strftime('%Y-%m-%d')

    option_chain = ticker_obj.option_chain(date=expiry_date_str)
    calls = option_chain.calls

    strikes = calls['strike'].values

    # yf implied vols 

    implied_vols_yf = calls['impliedVolatility'].values
    valid_indices = implied_vols_yf != 0
    strikes = strikes[valid_indices]
    implied_vols_yf = implied_vols_yf[valid_indices]

    filtered_strikes_yf = []
    filtered_vols_yf = []
    for i in range(len(strikes)):
        if implied_vols_yf[i] != 0:
            filtered_strikes_yf.append(strikes[i])
            filtered_vols_yf.append(implied_vols_yf[i] * 100)

    # bs implied vols 

    implied_vols_bs = []
    filtered_strikes_bs = []
    for strike in strikes: 
        yf_price, _ = yf_option_price(ticker_obj, strike, T, 'call')
        if yf_price is None:
            pass
        else:
            iv = bs_iv(yf_price, S_0, strike, T, r, q, option_type)
            if iv is not None and not np.isnan(iv):
                filtered_strikes_bs.append(strike)
                implied_vols_bs.append(iv * 100)

    _, ax = plt.subplots()
    ax.plot(filtered_strikes_yf, filtered_vols_yf, label='yfinance implied vol', marker='o', linestyle='-', color="blue")
    ax.plot(filtered_strikes_bs, implied_vols_bs, label='black-scholes implied vol', marker='o', linestyle='--', color="orange")
    ax.axvline(x=S_0, color='black', linestyle='--', label='current price')

    ax.set_xlabel('strikes')
    ax.set_ylabel(f'implied vols (%)')
    ax.set_title(f'Vol Skew for {ticker_obj.ticker} on {expiry_date_str}')
    ax.legend()
    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()