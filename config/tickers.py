from __future__ import annotations

FUTURES_TICKER_MAP: dict[str, str] = {
    # Micro Bitcoin
    "MBT": "MBT=F",
    "MICRO BTC": "MBT=F",
    "MICRO BITCOIN": "MBT=F",
    "BTC MICRO": "MBT=F",
    # Micro Euro FX
    "M6E": "M6E=F",
    "MICRO EURO": "M6E=F",
    "MICRO EUR": "M6E=F",
    "MICRO FX EUR": "M6E=F",
    # Micro E-mini S&P 500
    "MES": "MES=F",
    "MICRO ES": "MES=F",
    "MICRO SP": "MES=F",
    "MICRO SP500": "MES=F",
    # Micro Nasdaq-100
    "MNQ": "MNQ=F",
    "MICRO NQ": "MNQ=F",
    "MICRO NASDAQ": "MNQ=F",
    # Micro Dow Jones
    "MYM": "MYM=F",
    "MICRO DOW": "MYM=F",
    "MICRO YM": "MYM=F",
    # Micro Russell 2000
    "M2K": "M2K=F",
    "MICRO RUSSELL": "M2K=F",
    "MICRO RTY": "M2K=F",
    # E-mini S&P 500
    "ES": "ES=F",
    "EMINI SP": "ES=F",
    "EMINI S&P": "ES=F",
    # E-mini Nasdaq-100
    "NQ": "NQ=F",
    "EMINI NQ": "NQ=F",
    "EMINI NASDAQ": "NQ=F",
    # Micro Gold
    "MGC": "MGC=F",
    "MICRO GOLD": "MGC=F",
    # Gold
    "GC": "GC=F",
    "GOLD": "GC=F",
    # Micro Crude Oil
    "MCL": "MCL=F",
    "MICRO CRUDE": "MCL=F",
    "MICRO OIL": "MCL=F",
    # Crude Oil
    "CL": "CL=F",
    "CRUDE": "CL=F",
    "WTI": "CL=F",
    # Natural Gas
    "NG": "NG=F",
    "NAT GAS": "NG=F",
    "NATURAL GAS": "NG=F",
    # Bitcoin (large)
    "BTC": "BTC=F",
    "BITCOIN": "BTC=F",
    # Silver
    "SI": "SI=F",
    "SILVER": "SI=F",
}

CONTRACT_SPECS: dict[str, dict] = {
    "MBT=F": {
        "name": "Micro Bitcoin",
        "multiplier": 0.1,
        "tick_size": 5.0,
        "tick_value": 0.50,
        "currency": "USD",
        "exchange": "CME",
    },
    "BTC=F": {
        "name": "Bitcoin",
        "multiplier": 5.0,
        "tick_size": 5.0,
        "tick_value": 25.00,
        "currency": "USD",
        "exchange": "CME",
    },
    "M6E=F": {
        "name": "Micro Euro FX",
        "multiplier": 12_500,
        "tick_size": 0.00005,
        "tick_value": 0.625,
        "currency": "USD",
        "exchange": "CME",
    },
    "MES=F": {
        "name": "Micro E-mini S&P 500",
        "multiplier": 5,
        "tick_size": 0.25,
        "tick_value": 1.25,
        "currency": "USD",
        "exchange": "CME",
    },
    "MNQ=F": {
        "name": "Micro E-mini Nasdaq-100",
        "multiplier": 2,
        "tick_size": 0.25,
        "tick_value": 0.50,
        "currency": "USD",
        "exchange": "CME",
    },
    "MYM=F": {
        "name": "Micro E-mini Dow Jones",
        "multiplier": 0.5,
        "tick_size": 1.0,
        "tick_value": 0.50,
        "currency": "USD",
        "exchange": "CME",
    },
    "M2K=F": {
        "name": "Micro Russell 2000",
        "multiplier": 5,
        "tick_size": 0.10,
        "tick_value": 0.50,
        "currency": "USD",
        "exchange": "CME",
    },
    "ES=F": {
        "name": "E-mini S&P 500",
        "multiplier": 50,
        "tick_size": 0.25,
        "tick_value": 12.50,
        "currency": "USD",
        "exchange": "CME",
    },
    "NQ=F": {
        "name": "E-mini Nasdaq-100",
        "multiplier": 20,
        "tick_size": 0.25,
        "tick_value": 5.00,
        "currency": "USD",
        "exchange": "CME",
    },
    "MGC=F": {
        "name": "Micro Gold",
        "multiplier": 10,
        "tick_size": 0.10,
        "tick_value": 1.00,
        "currency": "USD",
        "exchange": "COMEX",
    },
    "GC=F": {
        "name": "Gold",
        "multiplier": 100,
        "tick_size": 0.10,
        "tick_value": 10.00,
        "currency": "USD",
        "exchange": "COMEX",
    },
    "MCL=F": {
        "name": "Micro Crude Oil",
        "multiplier": 100,
        "tick_size": 0.01,
        "tick_value": 1.00,
        "currency": "USD",
        "exchange": "NYMEX",
    },
    "CL=F": {
        "name": "Crude Oil (WTI)",
        "multiplier": 1000,
        "tick_size": 0.01,
        "tick_value": 10.00,
        "currency": "USD",
        "exchange": "NYMEX",
    },
    "NG=F": {
        "name": "Natural Gas",
        "multiplier": 10_000,
        "tick_size": 0.001,
        "tick_value": 10.00,
        "currency": "USD",
        "exchange": "NYMEX",
    },
    "SI=F": {
        "name": "Silver",
        "multiplier": 5000,
        "tick_size": 0.005,
        "tick_value": 25.00,
        "currency": "USD",
        "exchange": "COMEX",
    },
}


def resolve_ticker(user_input: str) -> tuple[str, dict | None]:
    """
    Map a user-supplied string to a (yfinance_symbol, contract_spec) pair.
    Tries alias map → =F suffix → direct lookup → pass-through.
    """
    raw = user_input.strip().upper()

    if raw in FUTURES_TICKER_MAP:
        yf_sym = FUTURES_TICKER_MAP[raw]
        return yf_sym, CONTRACT_SPECS.get(yf_sym)

    if raw.endswith("=F"):
        return raw, CONTRACT_SPECS.get(raw)

    with_suffix = raw + "=F"
    if with_suffix in CONTRACT_SPECS:
        return with_suffix, CONTRACT_SPECS[with_suffix]

    return raw, CONTRACT_SPECS.get(raw)


def list_known_tickers() -> list[tuple[str, str, str]]:
    """Return [(alias, yf_symbol, name)] for all known contracts."""
    seen: set[str] = set()
    rows = []
    for alias, sym in FUTURES_TICKER_MAP.items():
        if sym not in seen:
            seen.add(sym)
            spec = CONTRACT_SPECS.get(sym, {})
            rows.append((alias, sym, spec.get("name", sym)))
    return rows
