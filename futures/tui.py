from __future__ import annotations

"""
Futures TUI — 8-tab textual dashboard.

Tabs: Overview | Levels | Volume | FVGs | Bias | Ideas | Monitor | Journal
Keys: r=refresh  l=log  p=plan  j=journal  ?=help  q=quit  ←/→=tabs
"""

from typing import TYPE_CHECKING

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import (
        Footer,
        Header,
        Static,
        TabbedContent,
        TabPane,
    )
    from textual import work

    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

if TYPE_CHECKING:
    from futures.data import FuturesData
    from futures.levels import FuturesLevels
    from futures.volume import VolumeAnalytics, VolumeProfile
    from futures.fvg import FairValueGap
    from futures.bias import BiasResult, ConfluenceZone, PullbackZone
    from futures.ideas import TradeIdea
    from futures.monitor import MonitorEntry
    from futures.signals import AlertMessage, ExitGuidance

_TAB_NAMES = [
    "overview", "levels", "volume", "fvgs",
    "bias", "ideas", "monitor", "journal",
]


# ---------------------------------------------------------------------------
# Rich markup helpers
# ---------------------------------------------------------------------------

def _fmt(val: float | None, decimals: int = 4, na: str = "N/A") -> str:
    return f"{val:.{decimals}f}" if val is not None else na


def _color_trend(t: str) -> str:
    return {"bullish": "[green]bullish[/green]",
            "bearish": "[red]bearish[/red]",
            "neutral": "[dim]neutral[/dim]",
            "mixed":   "[yellow]mixed[/yellow]"}.get(t, t)


def _color_bias(b: str) -> str:
    return {"bullish": "[green]BULLISH[/green]",
            "bearish": "[red]BEARISH[/red]",
            "neutral": "[dim]NEUTRAL[/dim]",
            "mixed":   "[yellow]MIXED[/yellow]"}.get(b, b)


def _color_conf(c: str) -> str:
    return {"high":   "[green]HIGH[/green]",
            "medium": "[yellow]MEDIUM[/yellow]",
            "low":    "[dim]LOW[/dim]"}.get(c, c)


# ---------------------------------------------------------------------------
# Per-tab renderers (return Rich markup strings)
# ---------------------------------------------------------------------------

def _render_overview(
    data: "FuturesData",
    levels: "FuturesLevels | None",
    bias: "BiasResult | None",
    alerts: list["AlertMessage"],
) -> str:
    lines = ["[bold]MARKET OVERVIEW[/bold]", ""]
    lines.append(f"  [cyan]Symbol[/cyan]      : {data.display_name}  ({data.yf_symbol})")
    spot = data.spot
    lines.append(f"  [cyan]Spot Price[/cyan]  : {_fmt(spot, 4)}")

    if levels:
        atr5  = _fmt(levels.atr_5m,  4)
        atr15 = _fmt(levels.atr_15m, 4)
        lines.append(f"  [cyan]5m  ATR[/cyan]     : {atr5}")
        lines.append(f"  [cyan]15m ATR[/cyan]     : {atr15}")
        lines.append("")
        lines.append(f"  [cyan]5m  Trend[/cyan]   : {_color_trend(levels.trend_5m)}")
        lines.append(f"  [cyan]15m Trend[/cyan]   : {_color_trend(levels.trend_15m)}")
        lines.append(f"  [cyan]Alignment[/cyan]   : {_color_trend(levels.trend_align)}")
        lines.append("")
        lines.append(f"  [cyan]VWAP[/cyan]        : {_fmt(levels.vwap, 4)}")
        lines.append(f"  [cyan]Prev Day High[/cyan]: {_fmt(levels.prev_day_high, 4)}")
        lines.append(f"  [cyan]Prev Day Low[/cyan] : {_fmt(levels.prev_day_low,  4)}")
        lines.append("")
        lines.append("  [bold]CONTEXT[/bold]")
        for cl in levels.context_lines:
            lines.append(f"  [dim]•[/dim] {cl}")
    if bias:
        lines.append("")
        lines.append(f"  [bold]Bias[/bold]        : {_color_bias(bias.bias)}  {_color_conf(bias.confidence)}")
        if bias.cautions:
            lines.append(f"  [yellow]Main caution[/yellow]: {bias.cautions[0]}")

    if alerts:
        lines.append("")
        lines.append("  [bold]REFRESH ALERTS[/bold]")
        for alert in alerts[:5]:
            color = {
                "bullish": "green",
                "bearish": "red",
                "caution": "yellow",
                "info": "cyan",
            }.get(alert.level, "white")
            lines.append(f"  [{color}]•[/{color}] {alert.message}")

    if data.errors:
        lines.append("")
        lines.append("  [yellow]WARNINGS[/yellow]")
        for e in data.errors:
            lines.append(f"  [yellow]![/yellow] {e}")

    return "\n".join(lines)


def _render_levels(levels: "FuturesLevels | None") -> str:
    if levels is None:
        return "[dim]No levels data.[/dim]"

    lines = ["[bold]KEY LEVELS[/bold]", ""]
    rows = [
        ("Previous Day High",   levels.prev_day_high),
        ("Previous Day Low",    levels.prev_day_low),
        ("Previous Day Close",  levels.prev_day_close),
        ("Today High",          levels.today_high),
        ("Today Low",           levels.today_low),
        ("Asia High",           levels.asia.high),
        ("Asia Low",            levels.asia.low),
        ("London High",         levels.london.high),
        ("London Low",          levels.london.low),
        ("NY Open Range High",  levels.ny_open_range.high),
        ("NY Open Range Low",   levels.ny_open_range.low),
        ("NY Session High",     levels.new_york.high),
        ("NY Session Low",      levels.new_york.low),
        ("VWAP",                levels.vwap),
    ]
    for name, val in rows:
        val_str = f"[cyan]{val:.5g}[/cyan]" if val is not None else "[dim]N/A[/dim]"
        lines.append(f"  {name:<26}: {val_str}")

    spot = levels.current_price
    ref_atr = levels.atr_15m or levels.atr_5m
    if spot is not None and ref_atr:
        atr_label = "15m" if levels.atr_15m else "5m"
        lines.append("")
        lines.append(f"  [bold]ATR TARGETS[/bold]  ({atr_label} ATR = {ref_atr:.5g})")
        lines.append("")
        for label, mult, sign in [
            ("Long  1x", 1, +1), ("Long  2x", 2, +1), ("Long  3x", 3, +1),
            ("Short 1x", 1, -1), ("Short 2x", 2, -1), ("Short 3x", 3, -1),
        ]:
            tgt = spot + sign * mult * ref_atr
            col = "green" if sign > 0 else "red"
            lines.append(f"  {label} ATR : [{col}]{tgt:.5g}[/{col}]")

    if levels.errors:
        lines.append("")
        for e in levels.errors:
            lines.append(f"  [yellow]! {e}[/yellow]")

    return "\n".join(lines)


def _render_volume(analytics: "VolumeAnalytics | None", spot: float | None) -> str:
    if analytics is None:
        return "[dim]No volume data.[/dim]"
    from futures.volume import render_volume_analytics
    return render_volume_analytics(analytics, spot)


def _render_fvgs(
    fvgs_5m: list["FairValueGap"],
    fvgs_15m: list["FairValueGap"],
    fvgs_1h: list["FairValueGap"],
) -> str:
    lines = ["[bold]FAIR VALUE GAPS[/bold]", ""]

    def _render_tf_fvgs(fvgs: list, tf: str) -> None:
        lines.append(f"  [cyan underline]{tf} FVGs[/cyan underline]")
        if not fvgs:
            lines.append("  [dim]None detected.[/dim]")
            return
        for f in fvgs[:8]:
            dir_col = "green" if f.direction == "bullish" else "red"
            fill_col = "dim" if f.is_filled else ("yellow" if f.fill_pct > 30 else "white")
            near_tag = " [cyan](near)[/cyan]" if f.is_near else ""
            filled_tag = " [dim](filled)[/dim]" if f.is_filled else ""
            atr_tag = f" {f.atr_mult:.2f}x ATR" if f.atr_mult else ""
            lines.append(
                f"  [{dir_col}]{'▲' if f.direction == 'bullish' else '▼'}[/{dir_col}] "
                f"[{fill_col}]{f.lower:.5g} – {f.upper:.5g}[/{fill_col}]{atr_tag}"
                f"  fill={f.fill_pct:.0f}%  age={f.age_bars}b{near_tag}{filled_tag}"
            )

    _render_tf_fvgs(fvgs_5m,  "5m")
    lines.append("")
    _render_tf_fvgs(fvgs_15m, "15m")
    lines.append("")
    _render_tf_fvgs(fvgs_1h,  "1h")

    return "\n".join(lines)


def _render_bias(
    bias: "BiasResult | None",
    confluence: list["ConfluenceZone"],
    pullbacks: list["PullbackZone"],
) -> str:
    if bias is None:
        return "[dim]No bias data.[/dim]"

    lines = ["[bold]BIAS & CONFLUENCE[/bold]", ""]
    lines.append(f"  Bias       : {_color_bias(bias.bias)}")
    lines.append(f"  Confidence : {_color_conf(bias.confidence)}")
    lines.append(f"  Score      : {bias.score:+d}")
    lines.append("")

    if bias.bull_signals:
        lines.append("  [green]Bull Signals[/green]")
        for s in bias.bull_signals:
            lines.append(f"  [green]▲[/green] {s}")
        lines.append("")

    if bias.bear_signals:
        lines.append("  [red]Bear Signals[/red]")
        for s in bias.bear_signals:
            lines.append(f"  [red]▼[/red] {s}")
        lines.append("")

    if bias.cautions:
        lines.append("  [yellow]Cautions[/yellow]")
        for c in bias.cautions:
            lines.append(f"  [yellow]![/yellow] {c}")
        lines.append("")

    if confluence:
        lines.append("  [bold]CONFLUENCE ZONES[/bold]")
        for z in confluence[:5]:
            col = "green" if z.zone_type == "support" else ("red" if z.zone_type == "resistance" else "yellow")
            lvl_names = ", ".join(n for n, _ in z.levels)
            lines.append(
                f"  [{col}]{z.lower:.5g} – {z.upper:.5g}[/{col}]  "
                f"[dim]({z.strength} lvls: {lvl_names})[/dim]"
            )
            suggestion = "support retest / possible long continuation" if z.zone_type == "support" else (
                "resistance retest / possible short continuation" if z.zone_type == "resistance" else "mixed zone / wait for confirmation"
            )
            lines.append(f"    [dim]{suggestion}[/dim]")
        lines.append("")

    if pullbacks:
        lines.append("  [bold]PULLBACK WATCH ZONES[/bold]")
        for pz in pullbacks[:3]:
            col = "green" if pz.direction == "long" else "red"
            lines.append(f"  [{col}]{pz.lower:.5g} – {pz.upper:.5g}[/{col}]  inv: {pz.invalidation:.5g}")
            for r in pz.reasons[:2]:
                lines.append(f"    [dim]• {r}[/dim]")

    return "\n".join(lines)


def _render_ideas(
    ideas: list["TradeIdea"],
    selected_idx: int,
    exit_guidance: "ExitGuidance | None",
) -> str:
    from futures.ideas import format_idea_rich
    if not ideas:
        return "[dim]No trade ideas — bias may be neutral or data insufficient.[/dim]"

    lines = ["[bold]TRADE IDEAS[/bold]", ""]
    lines.append(
        "[dim]These are educational setup observations, not buy/sell signals.[/dim]"
    )
    lines.append("[dim]Use ↑/↓ to choose an idea and Enter or p to plan from it.[/dim]")
    lines.append("")
    for i, idea in enumerate(ideas, 1):
        prefix = "[cyan]>[/cyan] " if i - 1 == selected_idx else "  "
        lines.append(prefix + ("[bold]Selected[/bold]" if i - 1 == selected_idx else "[dim]Idea[/dim]"))
        lines.append(format_idea_rich(idea, i, None))
        lines.append("")

    if exit_guidance:
        lines.append("[bold]EXIT HELPER[/bold]")
        lines.append(f"  Style        : [yellow]{exit_guidance.management_style}[/yellow]")
        if exit_guidance.current_r is not None:
            color = "green" if exit_guidance.current_r >= 0 else "red"
            lines.append(f"  Unrealized R : [{color}]{exit_guidance.current_r:.2f}R[/{color}]")
        lines.append(f"  Stop dist    : {exit_guidance.distance_to_stop:.5g}")
        if exit_guidance.distance_to_target is not None:
            lines.append(f"  Target dist  : {exit_guidance.distance_to_target:.5g}")
        lines.append(f"  Next barrier : {exit_guidance.nearest_barrier}")
        for label, dist in [("1x ATR", exit_guidance.distance_to_1x), ("2x ATR", exit_guidance.distance_to_2x), ("3x ATR", exit_guidance.distance_to_3x)]:
            if dist is not None:
                lines.append(f"  {label:<12}: {dist:.5g}")
        for warning in exit_guidance.warnings[:3]:
            lines.append(f"  [yellow]![/yellow] {warning}")

    return "\n".join(lines)


def _render_monitor(entries: list["MonitorEntry"]):
    if not entries:
        return "[dim]No monitor data.[/dim]"

    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    table = Table(
        title="MULTI-TICKER MONITOR",
        expand=True,
        show_lines=False,
        header_style="bold",
        pad_edge=False,
        box=None,
    )
    table.add_column("Symbol", style="cyan", min_width=12, no_wrap=True)
    table.add_column("Price", justify="right", min_width=9, no_wrap=True)
    table.add_column("Chg%", justify="right", min_width=8, no_wrap=True)
    table.add_column("5m", justify="center", min_width=4, no_wrap=True)
    table.add_column("15m", justify="center", min_width=5, no_wrap=True)
    table.add_column("1h", justify="center", min_width=4, no_wrap=True)
    table.add_column("Bias", min_width=9, no_wrap=True)
    table.add_column("Conf", min_width=6, no_wrap=True)
    table.add_column("VWAP", justify="center", min_width=5, no_wrap=True)
    table.add_column("Spk", justify="center", min_width=4, no_wrap=True)
    table.add_column("ATR", min_width=10, no_wrap=True)
    table.add_column("Idea", overflow="fold")

    detail_lines: list[Text] = []

    def _tc(t: str) -> Text:
        return {
            "bullish": Text("B", style="green"),
            "bearish": Text("S", style="red"),
            "neutral": Text("N", style="dim"),
            "mixed": Text("M", style="yellow"),
            "?": Text("?"),
        }.get(t, Text(t))

    for e in entries:
        price_str = f"{e.price:.4g}" if e.price is not None else "N/A"
        chg_text = Text(
            f"{e.daily_chg_pct:+.2f}%" if e.daily_chg_pct is not None else "N/A",
            style=("green" if (e.daily_chg_pct or 0) >= 0 else "red") if e.daily_chg_pct is not None else "dim",
        )
        bias_text = Text(
            e.bias,
            style={
                "bullish": "green",
                "bearish": "red",
                "neutral": "dim",
                "mixed": "yellow",
                "?": "white",
            }.get(e.bias, "white"),
        )
        conf_text = Text(
            e.confidence,
            style={"high": "green", "medium": "yellow", "low": "dim", "?": "white"}.get(e.confidence, "white"),
        )
        vwap_text = Text("Above" if e.above_vwap else "Below", style="cyan") if e.above_vwap is not None else Text("N/A", style="dim")
        spike_text = Text("Yes", style="yellow") if e.vol_spike else Text("No", style="dim")
        idea_text = Text(e.best_idea or "-", overflow="fold")

        table.add_row(
            e.display_name,
            price_str,
            chg_text,
            _tc(e.trend_5m),
            _tc(e.trend_15m),
            _tc(e.trend_1h),
            bias_text,
            conf_text,
            vwap_text,
            spike_text,
            e.atr_regime,
            idea_text,
        )

        detail = Text("  ", style="default")
        if e.nearest_level:
            detail.append(f"{e.display_name}: ", style="cyan")
            detail.append(f"session {e.session or '?'} | ", style="dim")
            detail.append("near " if e.near_key_level else "watch ", style="cyan" if e.near_key_level else "dim")
            detail.append(e.nearest_level, style="dim")
        if e.nearest_fvg_dir:
            if detail.plain.strip():
                detail.append(" | ", style="dim")
            detail.append("FVG ", style="dim")
            detail.append(e.nearest_fvg_dir, style="green" if e.nearest_fvg_dir == "bullish" else "red")
            detail.append(f" {e.nearest_fvg_tf}", style="dim")
        if detail.plain.strip():
            detail_lines.append(detail)

    if detail_lines:
        return Group(table, Text(""), *detail_lines)
    return table


def _render_journal(db_path: str) -> str:
    try:
        from journal.storage import TradeDatabase
        from journal.dashboard import build_trades_table_text, build_overview_text, build_breakdown_text
        from journal.metrics import compute_metrics

        db = TradeDatabase(db_path)
        futures_trades = db.get_all_futures()
        if not futures_trades:
            return "[dim]No futures trades logged yet. Press [bold]l[/bold] to log a trade.[/dim]"

        metrics = compute_metrics(futures_trades)
        overview = build_overview_text(metrics, "Recent Journal Summary")
        recent = build_trades_table_text(futures_trades, max_rows=8)
        reviews = db.get_daily_reviews()
        review_block = ""
        if reviews:
            review = reviews[0]
            review_block = (
                "\n\n[bold]DAILY REVIEW[/bold]\n"
                f"  Date           : {review.get('review_date', '')}\n"
                f"  Best Setup     : {review.get('best_setup', '') or 'N/A'}\n"
                f"  Worst Setup    : {review.get('worst_setup', '') or 'N/A'}\n"
                f"  Improve        : {review.get('what_to_improve', '') or 'N/A'}\n"
                f"  Mistake Tags   : {', '.join(review.get('mistake_tags', [])) or 'None'}\n"
                f"  Psych Notes    : {review.get('psychological_notes', '') or 'N/A'}"
            )
        by_plan = build_breakdown_text(metrics.by_planned_vs_impulsive, "PLANNED VS IMPULSIVE")
        return f"{overview}\n\n{recent}{review_block}\n\n{by_plan}".strip()
    except Exception as exc:
        return f"[red]Journal error: {exc}[/red]"


# ---------------------------------------------------------------------------
# Loading / error placeholders
# ---------------------------------------------------------------------------

_LOADING = "[dim]Loading…[/dim]"
_NO_DATA = "[dim]No data.[/dim]"


# ---------------------------------------------------------------------------
# Textual TUI App
# ---------------------------------------------------------------------------

if _TEXTUAL_AVAILABLE:

    class FuturesTUI(App):  # type: ignore[misc]
        """8-tab futures dashboard powered by textual."""

        CSS = """
        Screen {
            background: $surface;
        }
        TabbedContent {
            height: 1fr;
        }
        TabPane {
            padding: 0 1;
            overflow-y: auto;
        }
        Static {
            padding: 0 1;
        }
        #loading {
            color: $text-muted;
        }
        """

        BINDINGS = [
            Binding("r",            "refresh",   "Refresh"),
            Binding("q",            "quit_app",  "Quit"),
            Binding("l",            "log_trade", "Log Trade"),
            Binding("p",            "plan_trade","Plan"),
            Binding("j",            "open_journal","Journal"),
            Binding("question_mark","show_help",  "Help"),
            Binding("left",         "prev_tab",  "Prev Tab"),
            Binding("right",        "next_tab",  "Next Tab"),
            Binding("up",           "move_up",   "Up"),
            Binding("down",         "move_down", "Down"),
            Binding("enter",        "select_row","Select"),
            Binding("left_square_bracket", "prev_symbol", "Prev Symbol"),
            Binding("right_square_bracket", "next_symbol", "Next Symbol"),
        ]

        def __init__(self, symbol: str, watchlist: list[str], db_path: str) -> None:
            super().__init__()
            self.symbol    = symbol
            self.watchlist = watchlist
            self.db_path   = db_path

            # Cached analytics state
            self._data: "FuturesData | None"       = None
            self._levels: "FuturesLevels | None"   = None
            self._volume_analytics: "VolumeAnalytics | None" = None
            self._fvgs_5m:  list = []
            self._fvgs_15m: list = []
            self._fvgs_1h:  list = []
            self._bias: "BiasResult | None"        = None
            self._confluence: list = []
            self._pullbacks:  list = []
            self._ideas: list                      = []
            self._monitor: list["MonitorEntry"]    = []
            self._alerts: list["AlertMessage"]     = []
            self._previous_bias: str | None        = None
            self._selected_idea_idx: int           = 0
            self._exit_guidance: "ExitGuidance | None" = None

        # ------------------------------------------------------------------
        # Compose
        # ------------------------------------------------------------------

        def compose(self) -> "ComposeResult":
            yield Header(show_clock=True)
            with TabbedContent(
                "Overview", "Levels", "Volume", "FVGs",
                "Bias", "Ideas", "Monitor", "Journal",
                id="tabs",
            ):
                yield TabPane("Overview", Static(_LOADING, id="pane-overview"),  id="tab-overview")
                yield TabPane("Levels",   Static(_LOADING, id="pane-levels"),    id="tab-levels")
                yield TabPane("Volume",   Static(_LOADING, id="pane-volume"),    id="tab-volume")
                yield TabPane("FVGs",     Static(_LOADING, id="pane-fvgs"),      id="tab-fvgs")
                yield TabPane("Bias",     Static(_LOADING, id="pane-bias"),      id="tab-bias")
                yield TabPane("Ideas",    Static(_LOADING, id="pane-ideas"),     id="tab-ideas")
                yield TabPane("Monitor",  Static(_LOADING, id="pane-monitor"),   id="tab-monitor")
                yield TabPane("Journal",  Static(_LOADING, id="pane-journal"),   id="tab-journal")
            yield Footer()

        def on_mount(self) -> None:
            self.title = f"Futures Dashboard — {self.symbol}"
            self._load_data()
            self._load_monitor()
            self._update_journal()

        # ------------------------------------------------------------------
        # Worker — main symbol analytics
        # ------------------------------------------------------------------

        @work(thread=True)
        def _load_data(self) -> None:
            try:
                from futures.data import fetch_futures_data
                from futures.levels import compute_levels
                from futures.volume import analyze_volume
                from futures.fvg import detect_fvgs
                from futures.bias import compute_bias, find_confluence_zones, suggest_pullback_zones
                from futures.ideas import generate_ideas
                from futures.atr import resample_to_15m, compute_ema
                from futures.signals import build_alerts, build_exit_guidance
                from journal.storage import TradeDatabase
                from config.settings import load_settings

                data = fetch_futures_data(self.symbol)
                settings = load_settings()
                self._data = data
                levels = compute_levels(data)
                self._levels = levels

                spot    = data.spot
                atr_5m  = levels.atr_5m
                atr_15m = levels.atr_15m

                # Volume profile
                volume_analytics = None
                if not data.intraday_5m.empty:
                    try:
                        volume_analytics = analyze_volume(data.intraday_5m, spot)
                    except Exception:
                        pass
                self._volume_analytics = volume_analytics

                # FVGs
                fvgs_5m = fvgs_15m = fvgs_1h = []
                if spot is not None:
                    try:
                        if not data.intraday_5m.empty:
                            fvgs_5m  = detect_fvgs(
                                data.intraday_5m, "5m", spot, atr_5m,
                                min_size_atr=settings.fvg_min_size_atr,
                            )
                        if not data.intraday_5m.empty:
                            df15 = resample_to_15m(data.intraday_5m)
                            fvgs_15m = detect_fvgs(
                                df15, "15m", spot, atr_15m,
                                min_size_atr=settings.fvg_min_size_atr,
                            )
                        if not data.intraday_1h.empty:
                            fvgs_1h  = detect_fvgs(
                                data.intraday_1h, "1h", spot, atr_15m,
                                min_size_atr=settings.fvg_min_size_atr,
                            )
                    except Exception:
                        pass
                self._fvgs_5m  = fvgs_5m
                self._fvgs_15m = fvgs_15m
                self._fvgs_1h  = fvgs_1h

                all_fvgs = fvgs_5m + fvgs_15m + fvgs_1h

                # 1h trend
                trend_1h = "neutral"
                if not data.intraday_1h.empty:
                    try:
                        close = data.intraday_1h["Close"]
                        ema9  = compute_ema(close, 9).iloc[-1]
                        ema21 = compute_ema(close, 21).iloc[-1]
                        p1h   = close.iloc[-1]
                        if p1h > ema9 > ema21:
                            trend_1h = "bullish"
                        elif p1h < ema9 < ema21:
                            trend_1h = "bearish"
                    except Exception:
                        pass

                # Bias
                bias = None
                try:
                    bias = compute_bias(
                        spot=spot, trend_5m=levels.trend_5m,
                        trend_15m=levels.trend_15m, trend_1h=trend_1h,
                        vwap=levels.vwap, levels=levels,
                        volume_profile=volume_analytics.volume_profile if volume_analytics else None,
                        fvgs_5m=fvgs_5m, fvgs_15m=fvgs_15m, fvgs_1h=fvgs_1h,
                        atr_15m=atr_15m,
                    )
                except Exception:
                    pass
                self._bias = bias

                # Confluence & pullbacks
                confluence: list = []
                pullbacks:  list = []
                if bias and spot is not None and atr_15m:
                    named = {
                        "Prev Day High":    levels.prev_day_high,
                        "Prev Day Low":     levels.prev_day_low,
                        "Asia High":        levels.asia.high,
                        "Asia Low":         levels.asia.low,
                        "London High":      levels.london.high,
                        "London Low":       levels.london.low,
                        "VWAP":             levels.vwap,
                        "NY OR High":       levels.ny_open_range.high,
                        "NY OR Low":        levels.ny_open_range.low,
                        "POC":              volume_analytics.volume_profile.poc if volume_analytics and volume_analytics.volume_profile else None,
                        "VAH":              volume_analytics.volume_profile.vah if volume_analytics and volume_analytics.volume_profile else None,
                        "VAL":              volume_analytics.volume_profile.val if volume_analytics and volume_analytics.volume_profile else None,
                    }
                    try:
                        confluence = find_confluence_zones(named, atr=atr_15m)
                        pullbacks  = suggest_pullback_zones(
                            bias, confluence, spot, atr_15m, all_fvgs,
                            volume_analytics.volume_profile if volume_analytics else None
                        )
                    except Exception:
                        pass
                self._confluence = confluence
                self._pullbacks  = pullbacks

                # Ideas
                ideas: list = []
                if bias and spot is not None:
                    try:
                        ideas = generate_ideas(
                            bias=bias, levels=levels,
                            fvgs=all_fvgs,
                            volume_profile=volume_analytics.volume_profile if volume_analytics else None,
                            confluence_zones=confluence,
                            pullback_zones=pullbacks,
                            spot=spot, atr_5m=atr_5m, atr_15m=atr_15m,
                        )
                    except Exception:
                        pass
                self._ideas = ideas
                self._selected_idea_idx = min(self._selected_idea_idx, max(len(ideas) - 1, 0))

                self._alerts = build_alerts(
                    spot, levels, bias, self._previous_bias, confluence, all_fvgs,
                    volume_analytics.volume_profile if volume_analytics else None,
                    atr_15m, bool(volume_analytics and volume_analytics.relative_volume and volume_analytics.relative_volume >= 1.8),
                )
                self._previous_bias = bias.bias if bias else self._previous_bias

                self._exit_guidance = None
                try:
                    db = TradeDatabase(self.db_path)
                    open_trades = db.get_futures_trades(state="open", ticker=data.yf_symbol)
                    if open_trades and spot is not None:
                        t = open_trades[0]
                        self._exit_guidance = build_exit_guidance(
                            t["direction"], float(t["entry_price"]), float(t["stop_price"]), spot,
                            t.get("planned_target"), atr_15m or atr_5m, levels,
                            volume_analytics.volume_profile if volume_analytics else None,
                            all_fvgs,
                        )
                except Exception:
                    pass

            except Exception as exc:
                self.call_from_thread(self._show_error, str(exc))
                return

            self.call_from_thread(self._refresh_panes)

        # ------------------------------------------------------------------
        # Worker — monitor (parallel watchlist)
        # ------------------------------------------------------------------

        @work(thread=True)
        def _load_monitor(self) -> None:
            try:
                from futures.monitor import build_watchlist
                self._monitor = build_watchlist(self.watchlist)
            except Exception:
                self._monitor = []
            self.call_from_thread(self._update_monitor_pane)

        # ------------------------------------------------------------------
        # UI update helpers (must be called from the main thread)
        # ------------------------------------------------------------------

        def _refresh_panes(self) -> None:
            data   = self._data
            levels = self._levels

            if data is None:
                return

            self.query_one("#pane-overview", Static).update(
                _render_overview(data, levels, self._bias, self._alerts)
            )
            self.query_one("#pane-levels", Static).update(
                _render_levels(levels)
            )
            self.query_one("#pane-volume", Static).update(
                _render_volume(self._volume_analytics, data.spot)
            )
            self.query_one("#pane-fvgs", Static).update(
                _render_fvgs(self._fvgs_5m, self._fvgs_15m, self._fvgs_1h)
            )
            self.query_one("#pane-bias", Static).update(
                _render_bias(self._bias, self._confluence, self._pullbacks)
            )
            self.query_one("#pane-ideas", Static).update(
                _render_ideas(self._ideas, self._selected_idea_idx, self._exit_guidance)
            )

        def _update_monitor_pane(self) -> None:
            self.query_one("#pane-monitor", Static).update(
                _render_monitor(self._monitor)
            )

        def _update_journal(self) -> None:
            self.query_one("#pane-journal", Static).update(
                _render_journal(self.db_path)
            )

        def _show_error(self, msg: str) -> None:
            for pane_id in [
                "#pane-overview", "#pane-levels", "#pane-volume",
                "#pane-fvgs", "#pane-bias", "#pane-ideas",
            ]:
                self.query_one(pane_id, Static).update(f"[red]Error: {msg}[/red]")

        # ------------------------------------------------------------------
        # Actions
        # ------------------------------------------------------------------

        def action_refresh(self) -> None:
            for pane_id in [
                "#pane-overview", "#pane-levels", "#pane-volume",
                "#pane-fvgs", "#pane-bias", "#pane-ideas",
            ]:
                self.query_one(pane_id, Static).update(_LOADING)
            self.query_one("#pane-monitor", Static).update(_LOADING)
            self._load_data()
            self._load_monitor()
            self._update_journal()
            self.notify("Refreshing data…", severity="information")

        def action_quit_app(self) -> None:
            self.exit()

        def action_log_trade(self) -> None:
            self.exit(result="log")

        def action_plan_trade(self) -> None:
            payload = {"action": "plan"}
            if self._ideas:
                idea = self._ideas[self._selected_idea_idx]
                payload["idea"] = {
                    "direction": idea.direction,
                    "entry": round(idea.entry_mid, 8),
                    "stop": idea.invalidation,
                    "target": idea.target_2x or idea.target_1x,
                    "setup_type": idea.setup_type if idea.setup_type in {
                        "Asia High Breakout", "Asia Low Breakdown", "London High Breakout",
                        "London Low Breakdown", "NY Open Range Breakout", "NY Open Range Breakdown",
                        "VWAP Reclaim", "Liquidity Sweep & Reclaim", "Shallow Pullback Continuation",
                        "Prior Day High Breakout", "Prior Day Low Breakdown", "Manual/Other",
                    } else "Manual/Other",
                    "timeframe": "5m",
                    "notes": "Possible setup from futures dashboard. Wait for confirmation.",
                    "reason_for_entry": "; ".join(idea.reasons[:3]),
                    "analyzer_idea": idea.setup_type,
                    "bias_at_entry": self._bias.bias if self._bias else "",
                    "confidence_at_entry": self._bias.confidence if self._bias else "",
                    "confluence_score": self._pullbacks[self._selected_idea_idx].strength if self._pullbacks and self._selected_idea_idx < len(self._pullbacks) else None,
                    "fvg_involved": any("FVG" in reason for reason in idea.reasons),
                    "volume_node_involved": any("HVN" in reason or "LVN" in reason for reason in idea.reasons),
                    "session_level_involved": any(term in " ".join(idea.reasons) for term in ("London", "Asia", "Prev Day", "NY")),
                }
            self.exit(result=payload)

        def action_open_journal(self) -> None:
            self.exit(result="journal")

        def action_show_help(self) -> None:
            help_text = (
                "[bold]KEYBOARD SHORTCUTS[/bold]\n\n"
                "  [cyan]r[/cyan]  Refresh all data\n"
                "  [cyan]l[/cyan]  Log a trade\n"
                "  [cyan]p[/cyan]  Trade planner\n"
                "  [cyan]j[/cyan]  Open journal menu\n"
                "  [cyan]←/→[/cyan] Switch tabs\n"
                "  [cyan][ / ][/cyan] Switch watchlist ticker\n"
                "  [cyan]↑/↓[/cyan] Move through idea rows\n"
                "  [cyan]Enter[/cyan] Plan from selected idea\n"
                "  [cyan]?[/cyan]  This help screen\n"
                "  [cyan]q[/cyan]  Quit\n\n"
                "  Navigate tabs with the tab bar at the top.\n\n"
                "[dim]This tool provides educational context only.\n"
                "Nothing here is a buy/sell signal.[/dim]"
            )
            self.notify(help_text, title="Help", severity="information", timeout=10)

        def action_prev_tab(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            idx = max(_TAB_NAMES.index(tabs.active[4:]) - 1, 0) if tabs.active else 0
            tabs.active = f"tab-{_TAB_NAMES[idx]}"

        def action_next_tab(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            idx = min(_TAB_NAMES.index(tabs.active[4:]) + 1, len(_TAB_NAMES) - 1) if tabs.active else 0
            tabs.active = f"tab-{_TAB_NAMES[idx]}"

        def action_move_up(self) -> None:
            if self._ideas:
                self._selected_idea_idx = max(0, self._selected_idea_idx - 1)
                self.query_one("#pane-ideas", Static).update(
                    _render_ideas(self._ideas, self._selected_idea_idx, self._exit_guidance)
                )

        def action_move_down(self) -> None:
            if self._ideas:
                self._selected_idea_idx = min(len(self._ideas) - 1, self._selected_idea_idx + 1)
                self.query_one("#pane-ideas", Static).update(
                    _render_ideas(self._ideas, self._selected_idea_idx, self._exit_guidance)
                )

        def action_select_row(self) -> None:
            if self._ideas:
                self.action_plan_trade()

        def _switch_symbol(self, step: int) -> None:
            if not self.watchlist:
                return
            idx = self.watchlist.index(self.symbol) if self.symbol in self.watchlist else 0
            self.symbol = self.watchlist[(idx + step) % len(self.watchlist)]
            self.title = f"Futures Dashboard — {self.symbol}"
            for pane_id in [
                "#pane-overview", "#pane-levels", "#pane-volume",
                "#pane-fvgs", "#pane-bias", "#pane-ideas",
            ]:
                self.query_one(pane_id, Static).update(_LOADING)
            self._load_data()

        def action_prev_symbol(self) -> None:
            self._switch_symbol(-1)

        def action_next_symbol(self) -> None:
            self._switch_symbol(1)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_tui(symbol: str, watchlist: list[str] | None = None, db_path: str | None = None) -> str | None:
    """
    Launch the textual TUI for the given symbol.
    Returns an action string ('log', 'plan', 'journal') if the user
    pressed a hotkey, or None if they just quit.
    Raises ImportError if textual is not installed.
    """
    if not _TEXTUAL_AVAILABLE:
        raise ImportError("textual is not installed — run: pip install textual")

    from config.settings import load_settings
    settings = load_settings()
    wl  = watchlist or settings.watchlist
    dbp = db_path   or settings.db_path

    app = FuturesTUI(symbol=symbol, watchlist=wl, db_path=dbp)
    result = app.run()
    return result  # type: ignore[return-value]
