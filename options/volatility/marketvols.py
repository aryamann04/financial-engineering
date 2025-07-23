import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from options.volatility.iv import bs_iv

def yf_option_price(ticker_obj, K, T, option_type="call"):
        K = 5 * round(K / 5)
        exp_dates = ticker_obj.options
        if not exp_dates:
            return None, None

        target_expiry = datetime.now() + timedelta(days=T * 365)
        closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))
        try:
            options = ticker_obj.option_chain(closest_expiry)
            option_df = options.calls if option_type == "call" else options.puts
            row = option_df[option_df['strike'] == K]
            if not row.empty:
                return row['lastPrice'].values[0], closest_expiry
            return None, closest_expiry
        except Exception:
            return None, closest_expiry

def get_yf_iv(tic, K, T, option_type):
    K = 5 * round(K / 5)
    exp_dates = tic.options
    if not exp_dates:
        return None

    target_expiry = datetime.now() + timedelta(days=T * 365)
    closest_expiry = min(exp_dates, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - target_expiry))
    try:
        options = tic.option_chain(closest_expiry)
        option_df = options.calls if option_type == "call" else options.puts
        row = option_df[option_df['strike'] == K]
        return row['impliedVolatility'].values[0] if not row.empty else None
    except Exception:
        return None

def vol_skew(ticker_obj, expiry_years, strike):
    options = ticker_obj.options

    if not options:
        raise ValueError(f"No options data available for ticker {ticker_obj.ticker}")

    vol_at_strike = get_yf_iv(ticker_obj, strike, expiry_years, 'call')
    if vol_at_strike is None:
        raise ValueError(f"No implied volatility data available for strike {strike}")

    lower_strike = strike - 5
    vol_lower = get_yf_iv(ticker_obj, lower_strike, expiry_years, 'call')
    if vol_lower is None:
        raise ValueError(f"No implied volatility data available for strike {lower_strike}")

    upper_strike = strike + 5
    vol_upper = get_yf_iv(ticker_obj, upper_strike, expiry_years, 'call')
    if vol_upper is None:
        raise ValueError(f"No implied volatility data available for strike {upper_strike}")
    
    skew = (vol_upper - vol_lower) / (upper_strike - lower_strike)
    return skew

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
            filtered_vols_yf.append(implied_vols_yf[i])
        elif filtered_vols_yf:
            filtered_strikes_yf.append(None)
            filtered_vols_yf.append(None)

    # bs implied vols 

    implied_vols_bs = []
    filtered_strikes_bs = []
    for strike in strikes: 
        yf_price = yf_option_price(ticker_obj, strike, T, 'call')
        if yf_price is None:
            filtered_strikes_bs.append(None)
            implied_vols_bs.append(None)
        else:
            filtered_strikes_bs.append(strike)
            implied_vols_bs.append(bs_iv(yf_price, S_0, strike, T, r, q, option_type) * 100)

    ax = plt.subplots()
    ax.plot(filtered_strikes_yf, filtered_vols_yf, label='YFinance Implied Vol', marker='o', linestyle='-', color="blue")
    ax.plot(filtered_strikes_bs, implied_vols_bs, label='BS Implied Vol', marker='o', linestyle='--', color="orange")
    
    ax.set_xlabel('Strike Prices')
    ax.set_ylabel(f'Implied Vol')
    ax.set_title(f'Vol Skew for {ticker_obj.ticker} on {expiry_date_str}')
    ax.legend()
    ax.grid(True)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()