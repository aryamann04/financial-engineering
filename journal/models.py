from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

FUTURES_SETUPS: list[str] = [
    "Asia High Breakout",
    "Asia Low Breakdown",
    "London High Breakout",
    "London Low Breakdown",
    "NY Open Range Breakout",
    "NY Open Range Breakdown",
    "VWAP Reclaim",
    "Liquidity Sweep & Reclaim",
    "Shallow Pullback Continuation",
    "Prior Day High Breakout",
    "Prior Day Low Breakdown",
    "Manual/Other",
]

OPTIONS_STRATEGIES: list[str] = [
    "Long Call",
    "Long Put",
    "Bull Call Spread",
    "Bear Put Spread",
    "Credit Call Spread",
    "Credit Put Spread",
    "Iron Condor",
    "Long Straddle",
    "Long Strangle",
    "Short Straddle",
    "Short Strangle",
    "Gamma Scalp",
    "Vol Trade",
    "Earnings Trade",
    "Manual/Other",
]

MISTAKE_TAGS: list[str] = [
    "chased",
    "early_exit",
    "late_entry",
    "moved_stop",
    "revenge_trade",
    "no_setup",
    "sized_wrong",
    "ignored_news",
    "over-managed",
    "FOMO",
    "against_trend",
]

TIMEFRAMES: list[str] = ["1m", "5m", "15m", "30m", "1h"]


@dataclass
class FuturesTrade:
    ticker: str
    direction: str              # 'long' | 'short'
    entry_time: str             # ISO datetime string (ET)
    exit_time: str
    entry_price: float
    exit_price: float
    stop_price: float
    setup_type: str
    timeframe: str
    quantity: float             # number of contracts
    # Optional inputs
    planned_target: float | None = None
    fees: float = 0.0
    notes: str = ""
    screenshot_path: str | None = None
    mistake_tags: list[str] = field(default_factory=list)
    did_follow_plan: bool | None = None
    contract_multiplier: float | None = None
    atr_5m: float | None = None
    atr_15m: float | None = None
    # Calculated fields (set by compute_trade_metrics)
    pnl_points: float = 0.0
    pnl_dollars: float | None = None
    r_multiple: float | None = None
    holding_period_minutes: float = 0.0
    session_bucket: str = ""
    time_of_day_bucket: str = ""
    reached_1x_atr: bool | None = None
    reached_2x_atr: bool | None = None
    reached_3x_atr: bool | None = None
    # Trade lifecycle state
    state: str = "closed"   # 'planned' | 'open' | 'closed' | 'cancelled'
    analyzer_idea: str = ""
    confluence_score: int | None = None
    bias_at_entry: str = ""
    confidence_at_entry: str = ""
    atr_at_entry: float | None = None
    fvg_involved: bool | None = None
    volume_node_involved: bool | None = None
    session_level_involved: bool | None = None
    planned_vs_impulsive: str = ""   # 'planned' | 'impulsive' | ''
    reason_for_entry: str = ""
    # DB id (set after save)
    id: int | None = None


@dataclass
class OptionsTrade:
    underlying: str
    option_type: str            # 'call' | 'put'
    strike: float
    expiration: str             # YYYY-MM-DD
    entry_premium: float
    exit_premium: float
    quantity: int               # number of contracts (positive = long)
    entry_time: str
    exit_time: str
    setup_type: str             # strategy label from OPTIONS_STRATEGIES
    # Optional inputs
    stop_price: float | None = None
    notes: str = ""
    screenshot_path: str | None = None
    mistake_tags: list[str] = field(default_factory=list)
    did_follow_plan: bool | None = None
    iv_at_entry: float | None = None
    delta_at_entry: float | None = None
    gamma_at_entry: float | None = None
    theta_at_entry: float | None = None
    vega_at_entry: float | None = None
    # Calculated fields
    pnl_dollars: float = 0.0
    return_pct: float = 0.0
    r_multiple: float | None = None
    holding_period_minutes: float = 0.0
    underlying_change_pct: float | None = None
    # DB id
    id: int | None = None


@dataclass
class DailyReview:
    review_date: str                   # YYYY-MM-DD
    market_notes: str = ""
    psychological_notes: str = ""
    rule_violations: str = ""
    what_to_improve: str = ""
    best_setup: str = ""
    worst_setup: str = ""
    mistake_tags: list[str] = field(default_factory=list)
    screenshot_paths: list[str] = field(default_factory=list)
    notes: str = ""
    id: int | None = None


def compute_futures_metrics(trade: FuturesTrade) -> FuturesTrade:
    """Fill in calculated fields on a FuturesTrade in-place."""
    from zoneinfo import ZoneInfo
    from datetime import datetime

    sign = 1 if trade.direction == "long" else -1
    trade.pnl_points = sign * (trade.exit_price - trade.entry_price) * trade.quantity

    if trade.contract_multiplier is not None and trade.contract_multiplier > 0:
        raw_dollar_pnl = sign * (trade.exit_price - trade.entry_price) * trade.quantity * trade.contract_multiplier
        trade.pnl_dollars = raw_dollar_pnl - trade.fees
    else:
        trade.pnl_dollars = None

    risk_pts = abs(trade.entry_price - trade.stop_price)
    if risk_pts > 1e-10:
        trade.r_multiple = (sign * (trade.exit_price - trade.entry_price)) / risk_pts

    # Holding period
    try:
        fmt = "%Y-%m-%d %H:%M"
        entry_dt = datetime.strptime(trade.entry_time[:16], fmt)
        exit_dt  = datetime.strptime(trade.exit_time[:16],  fmt)
        trade.holding_period_minutes = (exit_dt - entry_dt).total_seconds() / 60
    except Exception:
        trade.holding_period_minutes = 0.0

    # Session & time-of-day buckets (from entry time)
    from config.sessions import get_session_label, get_time_of_day_label
    try:
        entry_h, entry_m = int(trade.entry_time[11:13]), int(trade.entry_time[14:16])
        trade.session_bucket    = get_session_label(entry_h, entry_m)
        trade.time_of_day_bucket = get_time_of_day_label(entry_h, entry_m)
    except Exception:
        trade.session_bucket = ""
        trade.time_of_day_bucket = ""

    # ATR target checks
    if trade.atr_5m or trade.atr_15m:
        ref_atr = trade.atr_15m or trade.atr_5m
        excursion = sign * (trade.exit_price - trade.entry_price)
        trade.reached_1x_atr = excursion >= 1.0 * ref_atr if ref_atr else None
        trade.reached_2x_atr = excursion >= 2.0 * ref_atr if ref_atr else None
        trade.reached_3x_atr = excursion >= 3.0 * ref_atr if ref_atr else None

    return trade


def compute_options_metrics(trade: OptionsTrade) -> OptionsTrade:
    """Fill in calculated fields on an OptionsTrade in-place."""
    from datetime import datetime

    contract_size = 100
    sign = 1 if trade.quantity > 0 else -1
    trade.pnl_dollars = (
        (trade.exit_premium - trade.entry_premium) * trade.quantity * contract_size
    )

    if trade.entry_premium > 0:
        trade.return_pct = (trade.exit_premium - trade.entry_premium) / trade.entry_premium * 100

    if trade.stop_price is not None:
        risk = abs(trade.entry_premium - trade.stop_price) * abs(trade.quantity) * contract_size
        if risk > 1e-10:
            trade.r_multiple = trade.pnl_dollars / risk

    try:
        fmt = "%Y-%m-%d %H:%M"
        entry_dt = datetime.strptime(trade.entry_time[:16], fmt)
        exit_dt  = datetime.strptime(trade.exit_time[:16],  fmt)
        trade.holding_period_minutes = (exit_dt - entry_dt).total_seconds() / 60
    except Exception:
        trade.holding_period_minutes = 0.0

    return trade
