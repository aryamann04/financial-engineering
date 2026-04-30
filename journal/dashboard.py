from __future__ import annotations

from typing import Any

from tabulate import tabulate

from journal.metrics import PerformanceMetrics


def _pct(val: float | None, dec: int = 1) -> str:
    return f"{val:.{dec}f}%" if val is not None else "N/A"


def _flt(val: float | None, dec: int = 2) -> str:
    return f"{val:,.{dec}f}" if val is not None else "N/A"


def _pnl_str(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:,.2f}"


def build_overview_text(m: PerformanceMetrics, title: str = "PERFORMANCE OVERVIEW") -> str:
    sep = "═" * 60
    thin = "─" * 60

    rows_summary = [
        ["Total Trades", str(m.total_trades)],
        ["  Wins / Losses", f"{m.wins} / {m.losses}"],
        ["  Long / Short", f"{m.long_trades} / {m.short_trades}"],
        ["Total P&L", _pnl_str(m.total_pnl)],
        ["Average P&L", _pnl_str(m.avg_pnl)],
        ["Win Rate", _pct(m.win_rate)],
        ["Profit Factor", _flt(m.profit_factor if m.profit_factor != float("inf") else None)],
        ["Expectancy", _flt(m.expectancy)],
        ["Average Winner", _flt(m.avg_winner)],
        ["Average Loser", _flt(m.avg_loser)],
        ["Best Trade", _flt(m.best_trade)],
        ["Worst Trade", _flt(m.worst_trade)],
        ["Max Drawdown", _flt(m.max_drawdown)],
        ["Average R", _flt(m.avg_r)],
        ["Median R", _flt(m.median_r)],
        ["Avg Hold (min)", _flt(m.avg_holding_minutes, 1)],
    ]

    lines = [
        f"  {sep}",
        f"  {title}",
        f"  {sep}",
        tabulate(rows_summary, tablefmt="plain", colalign=("left", "right")),
    ]

    if any(v is not None for v in [m.atr_hit_1x_pct, m.atr_hit_2x_pct, m.atr_hit_3x_pct]):
        atr_rows = [
            ["Reached 1x ATR", _pct(m.atr_hit_1x_pct)],
            ["Reached 2x ATR", _pct(m.atr_hit_2x_pct)],
            ["Reached 3x ATR", _pct(m.atr_hit_3x_pct)],
        ]
        lines.extend([
            "",
            f"  {thin}",
            "  ATR TARGET HIT RATES (futures only)",
            f"  {thin}",
            tabulate(atr_rows, tablefmt="plain", colalign=("left", "right")),
        ])

    return "\n".join(lines)


def render_overview(m: PerformanceMetrics, title: str = "PERFORMANCE OVERVIEW") -> str:
    text = build_overview_text(m, title)
    print(text)
    return text


def build_breakdown_text(breakdown: dict[str, dict], title: str) -> str:
    if not breakdown:
        return ""
    sep = "─" * 60
    rows = []
    for key, stats in sorted(breakdown.items(), key=lambda x: x[1].get("pnl", 0), reverse=True):
        pf = stats.get("profit_factor")
        rows.append([
            key,
            stats.get("n", 0),
            _pnl_str(stats.get("pnl", 0.0)),
            _pct(stats.get("win_rate")),
            _flt(stats.get("avg_r")) if stats.get("avg_r") is not None else "N/A",
            _flt(pf) if pf is not None else "N/A",
        ])
    table = tabulate(
        rows,
        headers=["", "Trades", "P&L", "Win%", "Avg R", "PF"],
        tablefmt="simple",
        colalign=("left", "right", "right", "right", "right", "right"),
    )
    return "\n".join([f"  {sep}", f"  {title}", f"  {sep}", table])


def render_breakdown(breakdown: dict[str, dict], title: str) -> str:
    text = build_breakdown_text(breakdown, title)
    if text:
        print(f"\n{text}")
    return text


def build_full_dashboard_text(m: PerformanceMetrics) -> str:
    sections = [build_overview_text(m)]
    for breakdown, title in [
        (m.by_asset_class, "BY ASSET CLASS"),
        (m.by_ticker, "BY TICKER"),
        (m.by_setup, "BY SETUP TYPE"),
        (m.by_timeframe, "BY TIMEFRAME"),
        (m.by_session, "BY SESSION"),
        (m.by_time_of_day, "BY TIME OF DAY"),
        (m.by_confluence_score, "BY CONFLUENCE SCORE"),
        (m.by_fvg_involved, "BY FVG INVOLVEMENT"),
        (m.by_volume_node_involved, "BY VOLUME NODE INVOLVEMENT"),
        (m.by_bias_alignment, "BY BIAS ALIGNMENT"),
        (m.by_planned_vs_impulsive, "BY PLANNED VS IMPULSIVE"),
        (m.by_followed_plan, "BY FOLLOWED PLAN"),
        (m.by_reason_for_entry, "BY REASON FOR ENTRY"),
    ]:
        text = build_breakdown_text(breakdown, title)
        if text:
            sections.append(text)
    return "\n\n".join(sections)


def render_full_dashboard(m: PerformanceMetrics) -> str:
    text = build_full_dashboard_text(m)
    print(text)
    return text


def build_trades_table_text(trades: list[dict], max_rows: int = 50) -> str:
    if not trades:
        return "\n  No trades to display."

    rows = []
    for t in trades[:max_rows]:
        ac = t.get("asset_class", "futures" if "pnl_points" in t else "options")
        ticker = t.get("ticker") or t.get("underlying", "?")
        direction = (t.get("direction", "") or "").upper()
        entry = t.get("entry_time", "")[:16]
        pnl = t.get("pnl_dollars") if t.get("pnl_dollars") is not None else t.get("pnl_points", 0)
        r = t.get("r_multiple")
        setup = (t.get("setup_type") or "")[:22]
        state = t.get("state", "")
        rows.append([
            t.get("id", ""),
            ac[:3].upper(),
            ticker,
            direction,
            state,
            entry,
            _pnl_str(pnl) if pnl is not None else "N/A",
            f"{r:.2f}R" if r is not None else "N/A",
            setup,
        ])

    text = tabulate(
        rows,
        headers=["ID", "AC", "Ticker", "Dir", "State", "Entry Time", "P&L", "R", "Setup"],
        tablefmt="simple",
    )
    if len(trades) > max_rows:
        text += f"\n\n  ... showing {max_rows} of {len(trades)} trades."
    return text


def render_trades_table(trades: list[dict], max_rows: int = 50) -> str:
    text = build_trades_table_text(trades, max_rows=max_rows)
    print(text)
    return text
