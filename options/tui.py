from __future__ import annotations

from typing import TYPE_CHECKING

from analyzer.formatting import format_percent, format_price
from config.settings import load_settings
from options.snapshot import OptionsSnapshot, get_options_snapshot

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

if TYPE_CHECKING:
    from options.core.analyzer import Analyzer
    from options.monitor import OptionsMonitorEntry


_TAB_NAMES = [
    "overview", "summary", "calls", "puts", "straddle", "gamma",
    "regime", "macro", "surface", "diagnostics", "monitor", "journal",
]
_LOADING = "[dim]Loading…[/dim]"


def _df_to_text(df) -> str:
    from tabulate import tabulate

    if df is None or (hasattr(df, "empty") and df.empty):
        return "[dim]No data.[/dim]"
    display_df = df.copy()
    for col in display_df.columns:
        if getattr(display_df[col], "dtype", None) is not None and str(display_df[col].dtype).startswith(("float", "int")):
            display_df[col] = display_df[col].map(lambda v: format_price(v) if v is not None and v == v else "N/A")
    return f"[bold]{getattr(df, 'name', '')}[/bold]\n\n" + tabulate(display_df, headers="keys", tablefmt="simple", showindex=False)


def _render_snapshot(snapshot: OptionsSnapshot | None, news_lines: list[str]) -> str:
    if snapshot is None:
        return _LOADING
    lines = [
        "[bold]OPTIONS OVERVIEW[/bold]",
        "",
        f"[cyan]Symbol[/cyan]: {snapshot.symbol}",
        f"[cyan]Price[/cyan]: {format_price(snapshot.price)}",
        f"[cyan]Regime[/cyan]: {snapshot.regime}",
        f"[cyan]ATM IV[/cyan]: {format_percent(snapshot.atm_iv * 100 if snapshot.atm_iv is not None else None, decimals=1)}",
        f"[cyan]Approx RR[/cyan]: {format_percent(snapshot.rr_25d * 100 if snapshot.rr_25d is not None else None, decimals=1, signed=True, suffix='pp')}",
        f"[cyan]Dealer Read[/cyan]: {snapshot.gex_regime}",
        f"[cyan]Signal[/cyan]: {snapshot.signal}",
        f"[cyan]Confidence[/cyan]: {snapshot.confidence}",
        f"[cyan]Expiry[/cyan]: {snapshot.expiry or 'N/A'}",
        f"[dim]Snapshot updated: {snapshot.updated_at}[/dim]",
        "[dim]Dashboard values are lightweight cached snapshots. Deeper tabs render the full analytics pipeline immediately when available.[/dim]",
    ]
    if snapshot.errors:
        lines.extend(["", "[yellow]Cautions[/yellow]"])
        lines.extend(f"- {msg}" for msg in snapshot.errors[:3])
    if news_lines:
        lines.extend(["", "[bold]NEWS / SENTIMENT[/bold]"])
        lines.extend(news_lines[:8])
    return "\n".join(lines)


def _render_macro(analyzer: "Analyzer | None") -> str:
    if analyzer is None:
        return _LOADING
    return "\n".join(analyzer._macro_lines())


def _render_monitor(entries: list["OptionsMonitorEntry"]):
    if not entries:
        return "[dim]No options watchlist data.[/dim]"
    from rich.table import Table
    from rich.text import Text

    table = Table(title="OPTIONS WATCHLIST", expand=True, box=None, pad_edge=False)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Price", justify="right", no_wrap=True)
    table.add_column("Regime", no_wrap=True)
    table.add_column("ATM IV", justify="right", no_wrap=True)
    table.add_column("RR", justify="right", no_wrap=True)
    table.add_column("GEX", no_wrap=True)
    table.add_column("Conf", no_wrap=True)
    table.add_column("Sent", no_wrap=True)
    table.add_column("Signal")
    conf_rank = {"low": "dim", "medium": "yellow", "high": "green"}
    sent_rank = {"bearish": "red", "neutral": "dim", "bullish": "green"}
    for e in entries:
        table.add_row(
            e.symbol,
            format_price(e.price),
            e.regime,
            format_percent(e.atm_iv * 100 if e.atm_iv is not None else None, decimals=1),
            format_percent(e.rr_25d * 100 if e.rr_25d is not None else None, decimals=1, signed=True, suffix="pp"),
            e.gex_regime,
            Text(e.confidence, style=conf_rank.get(e.confidence, "white")),
            Text(e.sentiment, style=sent_rank.get(e.sentiment, "white")),
            e.signal,
        )
    return table


def _render_journal(db_path: str) -> str:
    from futures.tui import _render_journal as _futures_render_journal

    return _futures_render_journal(db_path)


if _TEXTUAL_AVAILABLE:
    class OptionsTUI(App):  # type: ignore[misc]
        CSS = """
        Screen { background: $surface; }
        TabbedContent { height: 1fr; }
        TabPane { padding: 0 1; overflow-y: auto; }
        Static { padding: 0 1; }
        """

        BINDINGS = [
            Binding("r", "refresh", "Refresh"),
            Binding("q", "quit_app", "Quit"),
            Binding("l", "log_trade", "Log Trade"),
            Binding("p", "plan_trade", "Plan"),
            Binding("j", "open_journal", "Journal"),
            Binding("question_mark", "show_help", "Help"),
            Binding("left", "prev_tab", "Prev Tab"),
            Binding("right", "next_tab", "Next Tab"),
            Binding("left_square_bracket", "prev_symbol", "Prev Symbol"),
            Binding("right_square_bracket", "next_symbol", "Next Symbol"),
        ]

        def __init__(self, symbol: str, watchlist: list[str], db_path: str, t_days: int = 30) -> None:
            super().__init__()
            self.symbol = symbol.upper()
            self.watchlist = [s.upper() for s in watchlist] or [self.symbol]
            if self.symbol not in self.watchlist:
                self.watchlist.insert(0, self.symbol)
            self.db_path = db_path
            self.t_days = t_days
            self._snapshot: OptionsSnapshot | None = None
            self._analyzer: Analyzer | None = None
            self._analyzer_symbol: str | None = None
            self._monitor = []
            self._news_lines: list[str] = []

        def compose(self) -> "ComposeResult":
            yield Header(show_clock=True)
            with TabbedContent("Overview", "Summary", "Calls", "Puts", "Straddle", "Gamma", "Regime", "Macro", "Surface", "Diagnostics", "Monitor", "Journal", id="tabs"):
                yield TabPane("Overview", Static(_LOADING, id="pane-overview"), id="tab-overview")
                yield TabPane("Summary", Static(_LOADING, id="pane-summary"), id="tab-summary")
                yield TabPane("Calls", Static(_LOADING, id="pane-calls"), id="tab-calls")
                yield TabPane("Puts", Static(_LOADING, id="pane-puts"), id="tab-puts")
                yield TabPane("Straddle", Static(_LOADING, id="pane-straddle"), id="tab-straddle")
                yield TabPane("Gamma", Static(_LOADING, id="pane-gamma"), id="tab-gamma")
                yield TabPane("Regime", Static(_LOADING, id="pane-regime"), id="tab-regime")
                yield TabPane("Macro", Static(_LOADING, id="pane-macro"), id="tab-macro")
                yield TabPane("Surface", Static(_LOADING, id="pane-surface"), id="tab-surface")
                yield TabPane("Diagnostics", Static(_LOADING, id="pane-diagnostics"), id="tab-diagnostics")
                yield TabPane("Monitor", Static(_LOADING, id="pane-monitor"), id="tab-monitor")
                yield TabPane("Journal", Static(_LOADING, id="pane-journal"), id="tab-journal")
            yield Footer()

        def on_mount(self) -> None:
            self.title = f"Options Dashboard — {self.symbol}"
            self._load_snapshot()
            self._load_monitor()
            self._load_analyzer()
            self._update_journal()

        @work(thread=True)
        def _load_snapshot(self, force_refresh: bool = False) -> None:
            try:
                settings = load_settings()
                self._snapshot = get_options_snapshot(
                    self.symbol,
                    t_days=self.t_days,
                    ttl_seconds=settings.options_snapshot_ttl_seconds,
                    force_refresh=force_refresh,
                )
                from analyzer.news import aggregate_sentiment, fetch_news

                news = fetch_news(self.symbol, max_items=5)
                self._news_lines = []
                if news:
                    sentiment, score = aggregate_sentiment(news)
                    sentiment_color = {"bullish": "green", "bearish": "red", "neutral": "yellow"}[sentiment]
                    self._news_lines.append(
                        f"[cyan]Aggregate sentiment[/cyan]: [{sentiment_color}]{sentiment}[/{sentiment_color}] ({score:+.1f})"
                    )
                    for item in news[:4]:
                        col = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(item.sentiment_label, "white")
                        self._news_lines.append(f"[{col}]•[/{col}] {item.title[:90]}")
                self.call_from_thread(self._update_overview)
            except Exception as exc:
                self.call_from_thread(self._show_error, str(exc), only_overview=True)

        @work(thread=True)
        def _load_monitor(self) -> None:
            try:
                from options.monitor import build_options_watchlist

                self._monitor = build_options_watchlist(self.watchlist, t_days=self.t_days)
            except Exception:
                self._monitor = []
            self.call_from_thread(self._update_monitor)

        @work(thread=True)
        def _load_analyzer(self) -> None:
            if self._analyzer_symbol == self.symbol and self._analyzer is not None:
                self.call_from_thread(self._refresh_detail_panes)
                return
            try:
                from options.core.analyzer import Analyzer

                analyzer = Analyzer(self.symbol, self.t_days / 365.0)
                self._analyzer = analyzer
                self._analyzer_symbol = self.symbol
                self.call_from_thread(self._refresh_detail_panes)
            except Exception as exc:
                self.call_from_thread(self._show_error, str(exc))

        def _update_overview(self) -> None:
            self.query_one("#pane-overview", Static).update(_render_snapshot(self._snapshot, self._news_lines))
            self.title = f"Options Dashboard — {self.symbol}"

        def _refresh_detail_panes(self) -> None:
            a = self._analyzer
            self.query_one("#pane-summary", Static).update("\n".join(a._summary_lines()) if a else _LOADING)
            self.query_one("#pane-calls", Static).update(_df_to_text(a.calls_df if a else None))
            self.query_one("#pane-puts", Static).update(_df_to_text(a.puts_df if a else None))
            self.query_one("#pane-straddle", Static).update(_df_to_text(a.straddle_df if a else None))
            self.query_one("#pane-gamma", Static).update(_df_to_text(a._gamma_gex_df() if a else None))
            self.query_one("#pane-regime", Static).update(_df_to_text(a._regime_df() if a else None))
            self.query_one("#pane-macro", Static).update(_render_macro(a))
            self.query_one("#pane-surface", Static).update(_df_to_text(a._surface_df() if a else None))
            self.query_one("#pane-diagnostics", Static).update(_df_to_text(a._diagnostics_df() if a else None))

        def _update_monitor(self) -> None:
            self.query_one("#pane-monitor", Static).update(_render_monitor(self._monitor))

        def _update_journal(self) -> None:
            self.query_one("#pane-journal", Static).update(_render_journal(self.db_path))

        def _show_error(self, msg: str, only_overview: bool = False) -> None:
            targets = ["#pane-overview"] if only_overview else [
                "#pane-summary", "#pane-calls", "#pane-puts", "#pane-straddle",
                "#pane-gamma", "#pane-regime", "#pane-macro", "#pane-surface", "#pane-diagnostics",
            ]
            for pane_id in targets:
                self.query_one(pane_id, Static).update(f"[red]Unavailable[/red]\n{msg}")

        def _switch_symbol(self, step: int) -> None:
            if not self.watchlist:
                return
            idx = self.watchlist.index(self.symbol)
            self.symbol = self.watchlist[(idx + step) % len(self.watchlist)]
            self._snapshot = None
            self._analyzer = None
            self._analyzer_symbol = None
            self.query_one("#pane-overview", Static).update(_LOADING)
            for pane_id in ["#pane-summary", "#pane-calls", "#pane-puts", "#pane-straddle", "#pane-gamma", "#pane-regime", "#pane-macro", "#pane-surface", "#pane-diagnostics"]:
                self.query_one(pane_id, Static).update(_LOADING)
            self._load_snapshot()
            self._load_analyzer()

        def action_refresh(self) -> None:
            self._load_snapshot(force_refresh=True)
            self._load_monitor()
            self._update_journal()
            if self._analyzer is not None:
                self._analyzer = None
                self._analyzer_symbol = None
                self._load_analyzer()

        def action_prev_symbol(self) -> None:
            self._switch_symbol(-1)

        def action_next_symbol(self) -> None:
            self._switch_symbol(1)

        def action_prev_tab(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            idx = max(_TAB_NAMES.index(tabs.active[4:]) - 1, 0) if tabs.active else 0
            tabs.active = f"tab-{_TAB_NAMES[idx]}"

        def action_next_tab(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            idx = min(_TAB_NAMES.index(tabs.active[4:]) + 1, len(_TAB_NAMES) - 1) if tabs.active else 0
            tabs.active = f"tab-{_TAB_NAMES[idx]}"

        def action_quit_app(self) -> None:
            self.exit()

        def action_log_trade(self) -> None:
            self.exit(result={"action": "log", "symbol": self.symbol})

        def action_plan_trade(self) -> None:
            self.exit(result={"action": "plan", "symbol": self.symbol})

        def action_open_journal(self) -> None:
            self.exit(result={"action": "journal"})

        def action_show_help(self) -> None:
            self.notify(
                "[bold]KEYBOARD SHORTCUTS[/bold]\n\n"
                "  [cyan]r[/cyan] refresh\n"
                "  [cyan][ / ][/cyan] previous/next watchlist ticker\n"
                "  [cyan]← / →[/cyan] switch tabs\n"
                "  [cyan]l[/cyan] log options trade\n"
                "  [cyan]p[/cyan] plan trade\n"
                "  [cyan]j[/cyan] journal\n"
                "  [cyan]q[/cyan] quit",
                title="Help",
                timeout=10,
            )


def run_options_tui(symbol: str, watchlist: list[str], db_path: str, t_days: int = 30):
    if not _TEXTUAL_AVAILABLE:
        raise ImportError("textual is not installed")
    app = OptionsTUI(symbol=symbol, watchlist=watchlist, db_path=db_path, t_days=t_days)
    return app.run()
