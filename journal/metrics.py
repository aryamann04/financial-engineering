from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import statistics


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_r: float = 0.0
    median_r: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    max_drawdown: float = 0.0
    avg_holding_minutes: float = 0.0
    # ATR hit rates (futures only)
    atr_hit_1x_pct: float | None = None
    atr_hit_2x_pct: float | None = None
    atr_hit_3x_pct: float | None = None
    # Breakdowns
    by_ticker:       dict[str, dict] = field(default_factory=dict)
    by_setup:        dict[str, dict] = field(default_factory=dict)
    by_timeframe:    dict[str, dict] = field(default_factory=dict)
    by_session:      dict[str, dict] = field(default_factory=dict)
    by_time_of_day:  dict[str, dict] = field(default_factory=dict)
    by_asset_class:  dict[str, dict] = field(default_factory=dict)
    by_confluence_score: dict[str, dict] = field(default_factory=dict)
    by_fvg_involved: dict[str, dict] = field(default_factory=dict)
    by_volume_node_involved: dict[str, dict] = field(default_factory=dict)
    by_bias_alignment: dict[str, dict] = field(default_factory=dict)
    by_planned_vs_impulsive: dict[str, dict] = field(default_factory=dict)
    by_followed_plan: dict[str, dict] = field(default_factory=dict)
    by_reason_for_entry: dict[str, dict] = field(default_factory=dict)
    # Trade count by direction
    long_trades:  int = 0
    short_trades: int = 0


def _pnl_key(trade: dict) -> float:
    """Get the primary P&L value from a trade dict (dollars preferred, else points)."""
    pnl_d = trade.get("pnl_dollars")
    pnl_p = trade.get("pnl_points")
    if pnl_d is not None:
        return float(pnl_d)
    if pnl_p is not None:
        return float(pnl_p)
    return 0.0


def _compute_group_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0.0, "win_rate": 0.0, "avg_r": None}
    pnls = [_pnl_key(t) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    rs = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    return {
        "n": len(trades),
        "pnl": sum(pnls),
        "win_rate": len(wins) / len(trades) * 100,
        "avg_r": statistics.mean(rs) if rs else None,
        "profit_factor": (
            abs(sum(wins)) / abs(sum(losses)) if losses and sum(losses) != 0 else None
        ),
    }


def _compute_drawdown(pnl_series: list[float]) -> float:
    """Max peak-to-trough drawdown on cumulative P&L."""
    if not pnl_series:
        return 0.0
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnl_series:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_metrics(trades: list[dict]) -> PerformanceMetrics:
    """Compute aggregate performance metrics from a list of trade dicts."""
    if not trades:
        return PerformanceMetrics()

    pnls = [_pnl_key(t) for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    rs = [t["r_multiple"] for t in trades if t.get("r_multiple") is not None]
    hold = [t.get("holding_period_minutes", 0) or 0 for t in trades]

    total_pnl = sum(pnls)
    profit_factor = (
        abs(sum(wins)) / abs(sum(losses))
        if losses and abs(sum(losses)) > 0
        else float("inf") if wins else 0.0
    )
    expectancy = (
        (len(wins) / len(trades) * statistics.mean(wins) if wins else 0)
        + (len(losses) / len(trades) * statistics.mean(losses) if losses else 0)
    )

    # ATR hit rates (futures trades that have the fields)
    def _hit_rate(field: str) -> float | None:
        eligible = [t for t in trades if t.get(field) is not None]
        if not eligible:
            return None
        return sum(1 for t in eligible if t[field]) / len(eligible) * 100

    # Breakdowns by various dimensions
    def _group_by(key: str, trades: list[dict]) -> dict[str, dict]:
        groups: dict[str, list[dict]] = {}
        for t in trades:
            val = str(t.get(key) or "Unknown")
            groups.setdefault(val, []).append(t)
        return {k: _compute_group_stats(v) for k, v in groups.items()}

    # Asset class key (present when get_all_trades used, else infer)
    def _asset_class_key(t: dict) -> str:
        if "asset_class" in t:
            return t["asset_class"]
        return "futures" if "pnl_points" in t else "options"

    ticker_key = lambda t: t.get("ticker") or t.get("underlying") or "Unknown"
    grouped_by_ticker: dict[str, list[dict]] = {}
    for t in trades:
        k = ticker_key(t)
        grouped_by_ticker.setdefault(k, []).append(t)
    by_ticker = {k: _compute_group_stats(v) for k, v in grouped_by_ticker.items()}

    asset_groups: dict[str, list[dict]] = {}
    for t in trades:
        k = _asset_class_key(t)
        asset_groups.setdefault(k, []).append(t)
    by_asset_class = {k: _compute_group_stats(v) for k, v in asset_groups.items()}

    return PerformanceMetrics(
        total_trades=len(trades),
        total_pnl=total_pnl,
        avg_pnl=total_pnl / len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades) * 100,
        avg_winner=statistics.mean(wins) if wins else 0.0,
        avg_loser=statistics.mean(losses) if losses else 0.0,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_r=statistics.mean(rs) if rs else 0.0,
        median_r=statistics.median(rs) if rs else 0.0,
        best_trade=max(pnls),
        worst_trade=min(pnls),
        max_drawdown=_compute_drawdown(pnls),
        avg_holding_minutes=statistics.mean(hold) if hold else 0.0,
        atr_hit_1x_pct=_hit_rate("reached_1x_atr"),
        atr_hit_2x_pct=_hit_rate("reached_2x_atr"),
        atr_hit_3x_pct=_hit_rate("reached_3x_atr"),
        by_ticker=by_ticker,
        by_setup=_group_by("setup_type", trades),
        by_timeframe=_group_by("timeframe", trades),
        by_session=_group_by("session_bucket", trades),
        by_time_of_day=_group_by("time_of_day_bucket", trades),
        by_asset_class=by_asset_class,
        by_confluence_score=_group_by("confluence_score", trades),
        by_fvg_involved=_group_by("fvg_involved", trades),
        by_volume_node_involved=_group_by("volume_node_involved", trades),
        by_bias_alignment=_group_by("bias_at_entry", trades),
        by_planned_vs_impulsive=_group_by("planned_vs_impulsive", trades),
        by_followed_plan=_group_by("did_follow_plan", trades),
        by_reason_for_entry=_group_by("reason_for_entry", trades),
        long_trades=sum(1 for t in trades if t.get("direction") == "long"),
        short_trades=sum(1 for t in trades if t.get("direction") == "short"),
    )
