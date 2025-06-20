import time
import os
import finnhub as finn
import pandas as pd
from datetime import datetime, timedelta

client = finn.Client(api_key=os.getenv('FINNHUB_API_KEY'))

def get_stock_close_prices(symbol: str, years: int) -> pd.Series:
    end_time = int(time.time())
    start_time = int((datetime.now() - timedelta(days=365 * years)).timestamp())

    candles = client.stock_candles(
        symbol=symbol,
        resolution='D',
        _from=start_time,
        to=end_time
    )

    if candles['s'] != 'ok':
        raise ValueError(f"Finnhub error: {candles}")

    df = pd.DataFrame({
        'timestamp': pd.to_datetime(candles['t'], unit='s'),
        'close': candles['c']
    })

    df.set_index('timestamp', inplace=True)
    return df['close']

closes = get_stock_close_prices("AAPL", years=0.5)
print(closes.tail())