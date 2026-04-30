from __future__ import annotations

import os
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from analyzer.formatting import format_percent, format_price
from analyzer.macro import fetch_macro_context
from analyzer.news import aggregate_sentiment, fetch_news
from analyzer.watchlists import load_watchlists, save_watchlists
from config.settings import load_settings
from fixed_income.core.marketyields import last_treasury_curve_status
from journal.dashboard import build_full_dashboard_text, build_trades_table_text
from journal.metrics import compute_metrics
from journal.storage import TradeDatabase
from options.monitor import build_options_watchlist

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import DataTable, Footer, Header, Input, Static, TabbedContent, TabPane

    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False


_DB = TradeDatabase()
_TAB_NAMES = ["dashboard", "options", "futures", "watchlists", "journal", "performance", "help"]


def _clear() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def _render_macro_panel() -> Panel:
    table = Table(box=None, pad_edge=False, expand=True)
    table.add_column("Macro", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", no_wrap=True)
    table.add_column("Chg", justify="right", no_wrap=True)
    table.add_column("Read")
    for row in fetch_macro_context():
        table.add_row(row.label, row.value, row.change, row.interpretation)
    snapshot = last_treasury_curve_status()
    footer = ""
    if snapshot.error_message:
        footer = f"\n[yellow]{snapshot.error_message}[/yellow]"
    else:
        footer = f"\n[dim]Treasury curve source: {snapshot.source} @ {snapshot.fetched_at}[/dim]"
    return Panel(Group(table, Text.from_markup(footer)), title="Macro Context", border_style="cyan")


def _bucket_line(label: str, bucket) -> Text:
    color = {"bullish": "green", "bearish": "red", "neutral": "grey70"}.get(getattr(bucket, "bias", "neutral"), "grey70")
    return Text.assemble((f"{label}: ", "bold"), (getattr(bucket, "summary", "Unavailable"), color))


def _render_options_entry_panel(entry) -> Panel:
    header = Text.assemble(
        (entry.symbol, "bold cyan"),
        ("  "),
        (format_price(entry.price), "white"),
        ("  "),
        (entry.signal, "green" if entry.confidence == "high" else "yellow" if entry.confidence == "medium" else "white"),
    )
    rows: list[Text] = [
        header,
        Text(f"ATM IV {format_percent(entry.atm_iv * 100 if entry.atm_iv is not None else None, decimals=1)}  |  RR {format_percent(entry.rr_25d * 100 if entry.rr_25d is not None else None, decimals=1, signed=True, suffix='pp')}  |  {entry.gex_regime}"),
        _bucket_line("Price Action", entry.price_action),
        _bucket_line("Macro", entry.macro),
        _bucket_line("Correlated", entry.correlated_assets),
        _bucket_line("Indicators", entry.indicators),
        _bucket_line("Flow", entry.volume_flow),
        Text(f"News: {entry.news_snippet}", style="magenta"),
    ]
    if entry.errors:
        rows.append(Text(f"Data note: {entry.errors[0]}", style="yellow"))
    return Panel(Group(*rows), title=f"Options • {entry.confidence}", border_style="green", padding=(1, 2))


def _render_futures_entry_panel(entry) -> Panel:
    price_action = Text(
        f"{entry.bias} bias; 5m {entry.trend_5m}, 15m {entry.trend_15m}, 1h {entry.trend_1h}; "
        f"{'above' if entry.above_vwap else 'below' if entry.above_vwap is not None else 'vs'} VWAP",
        style="green" if entry.bias == "bullish" else "red" if entry.bias == "bearish" else "grey70",
    )
    indicators = Text(
        f"ATR {format_price(entry.atr_15m)} ({entry.atr_regime}); nearest level {entry.nearest_level or 'N/A'}",
        style="grey70",
    )
    flow = Text(
        f"Vol spike {'yes' if entry.vol_spike else 'no'}; nearest FVG {entry.nearest_fvg_tf or 'N/A'} {entry.nearest_fvg_dir or ''}; idea {entry.best_idea or 'none'}",
        style="green" if entry.vol_spike else "grey70",
    )
    macro = Text(
        f"Session {entry.session or 'N/A'}; daily change {format_percent(entry.daily_chg_pct, decimals=2, signed=True)}",
        style="grey70",
    )
    return Panel(
        Group(
            Text.assemble((entry.symbol, "bold cyan"), ("  "), (format_price(entry.price), "white")),
            Text(f"Confidence {entry.confidence}", style="yellow" if entry.confidence == "medium" else "green" if entry.confidence == "high" else "white"),
            Text.assemble(("Price Action: ", "bold"), price_action),
            Text.assemble(("Macro: ", "bold"), macro),
            Text.assemble(("Indicators: ", "bold"), indicators),
            Text.assemble(("Flow: ", "bold"), flow),
        ),
        title="Futures",
        border_style="blue",
        padding=(1, 2),
    )


def _render_options_watchlist_panel():
    settings = load_settings()
    watchlists = load_watchlists()
    entries = build_options_watchlist(watchlists.options, t_days=settings.options_default_t_days)
    panels = [_render_options_entry_panel(e) for e in entries]
    return Panel(Columns(panels, equal=True, expand=True), title="Options Watchlist", border_style="green", padding=(0, 1)), entries


def _render_news_panel() -> Panel:
    items = fetch_news("SPY", max_items=5)
    sentiment, score = aggregate_sentiment(items)
    body = [
        Text(
            f"Aggregate sentiment: {sentiment} ({score:+.1f})",
            style={"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(sentiment, "white"),
        )
    ]
    for item in items[:4]:
        style = {"bullish": "green", "bearish": "red", "neutral": "yellow"}.get(item.sentiment_label, "white")
        body.append(Text(f"• {item.title[:100]}", style=style))
    return Panel.fit(Text("\n").join(body) if body else Text("No news available", style="dim"), title="News / Sentiment", border_style="magenta")


def _render_futures_watchlist_panel():
    from futures.monitor import build_watchlist

    entries = build_watchlist(load_watchlists().futures)
    conf_rank = {"low": 0, "medium": 1, "high": 2}
    entries.sort(key=lambda e: (conf_rank.get(e.confidence, 0), e.symbol))
    panels = [_render_futures_entry_panel(e) for e in entries]
    return Panel(Columns(panels, equal=True, expand=True), title="Futures Watchlist", border_style="blue", padding=(0, 1)), entries


def show_unified_dashboard() -> None:
    _clear()
    console = Console()
    options_panel, _ = _render_options_watchlist_panel()
    futures_panel, _ = _render_futures_watchlist_panel()
    console.print(Panel.fit("[bold]UNIFIED ANALYZER DASHBOARD[/bold]", border_style="white"))
    console.print(Group(_render_macro_panel(), _render_news_panel(), options_panel, futures_panel))
    input("\nPress Enter to continue...")


def launch_options_workspace(symbol: str | None = None) -> None:
    from options.tui import _TEXTUAL_AVAILABLE as _OPTIONS_TEXTUAL_AVAILABLE
    from options.tui import run_options_tui

    settings = load_settings()
    watchlists = load_watchlists()
    symbol = (symbol or settings.default_options_symbol or (watchlists.options[0] if watchlists.options else "SPY")).upper()
    if _OPTIONS_TEXTUAL_AVAILABLE and settings.use_tui and settings.tui_enabled:
        result = run_options_tui(symbol=symbol, watchlist=watchlists.options, db_path=settings.db_path, t_days=settings.options_default_t_days)
        if isinstance(result, dict) and result.get("action") == "journal":
            from journal.cli import run_journal_menu

            run_journal_menu()
        elif isinstance(result, dict) and result.get("action") in {"log", "plan"}:
            from journal.cli import log_options_trade

            log_options_trade()
        return

    from options.core.analyzer import Analyzer

    analyzer = Analyzer(symbol, settings.options_default_t_days / 365.0)
    analyzer.launch_analyzer()


def launch_futures_workspace(symbol: str | None = None) -> None:
    from futures.cli import run_tui_for_symbol

    settings = load_settings()
    run_tui_for_symbol((symbol or settings.default_symbol or "MES=F").upper())


if _TEXTUAL_AVAILABLE:
    class UnifiedAnalyzerApp(App):  # type: ignore[misc]
        CSS = """
        Screen { background: $surface; }
        TabbedContent { height: 1fr; }
        TabPane { padding: 0 1; overflow-y: auto; }
        Static { padding: 0 1; }
        DataTable { height: 1fr; }
        #watchlist-editor { height: auto; }
        """

        BINDINGS = [
            Binding("left", "prev_tab", "Prev Tab"),
            Binding("right", "next_tab", "Next Tab"),
            Binding("enter", "open_selected", "Open"),
            Binding("r", "refresh", "Refresh"),
            Binding("s", "save_watchlists", "Save Watchlists"),
            Binding("o", "open_options", "Options"),
            Binding("f", "open_futures", "Futures"),
            Binding("j", "open_journal", "Journal"),
            Binding("d", "open_performance", "Performance"),
            Binding("question_mark", "show_help", "Help"),
            Binding("q", "quit_app", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.watchlists = load_watchlists()
            self.settings = load_settings()
            self._options_entries = []
            self._futures_entries = []

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with TabbedContent("Dashboard", "Options", "Futures", "Watchlists", "Journal", "Performance", "Help", id="tabs"):
                yield TabPane("Dashboard", Static("[dim]Loading dashboard…[/dim]", id="pane-dashboard"), id="tab-dashboard")
                yield TabPane("Options", DataTable(id="table-options"), id="tab-options")
                yield TabPane("Futures", DataTable(id="table-futures"), id="tab-futures")
                with TabPane("Watchlists", id="tab-watchlists"):
                    yield Static(
                        "[bold]Watchlists[/bold]\nEdit comma-separated symbols and press [cyan]s[/cyan] to save.",
                        id="watchlist-help",
                    )
                    yield Input(value=", ".join(self.watchlists.options), placeholder="Options watchlist", id="input-options")
                    yield Input(value=", ".join(self.watchlists.futures), placeholder="Futures watchlist", id="input-futures")
                    yield Static("", id="watchlist-status")
                yield TabPane("Journal", Static("[dim]Loading journal…[/dim]", id="pane-journal"), id="tab-journal")
                yield TabPane("Performance", Static("[dim]Loading performance…[/dim]", id="pane-performance"), id="tab-performance")
                yield TabPane("Help", Static(self._help_text(), id="pane-help"), id="tab-help")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "Unified Analyzer"
            self._init_tables()
            self._refresh_all()

        def _init_tables(self) -> None:
            options_table = self.query_one("#table-options", DataTable)
            futures_table = self.query_one("#table-futures", DataTable)
            options_table.cursor_type = "row"
            futures_table.cursor_type = "row"
            options_table.add_columns("Symbol", "Price", "Regime", "ATM IV", "Conf", "Signal")
            futures_table.add_columns("Ticker", "Price", "Bias", "Conf", "VWAP", "Idea")

        def _help_text(self) -> str:
            return (
                "[bold]Keyboard Guide[/bold]\n\n"
                "[cyan]← / →[/cyan] switch tabs\n"
                "[cyan]↑ / ↓[/cyan] move through watchlist rows\n"
                "[cyan]Enter[/cyan] open selected options/futures instrument\n"
                "[cyan]o[/cyan] open first options watchlist symbol\n"
                "[cyan]f[/cyan] open first futures watchlist symbol\n"
                "[cyan]s[/cyan] save watchlists from the Watchlists tab\n"
                "[cyan]r[/cyan] refresh snapshots, macro, journal, and performance\n"
                "[cyan]j[/cyan] open the journal workflow\n"
                "[cyan]d[/cyan] open the performance dashboard workflow\n"
                "[cyan]q[/cyan] quit\n"
            )

        @work(thread=True)
        def _refresh_all(self) -> None:
            dashboard = self._build_dashboard_renderable()
            journal = build_trades_table_text(_DB.get_all_trades(), max_rows=20)
            performance = build_full_dashboard_text(compute_metrics(_DB.get_all_trades()))
            self.call_from_thread(self._apply_dashboard, dashboard)
            self.call_from_thread(self._apply_journal, journal)
            self.call_from_thread(self._apply_performance, performance)
            self.call_from_thread(self._apply_tables)

        def _build_dashboard_renderable(self):
            options_panel, options_entries = _render_options_watchlist_panel()
            futures_panel, futures_entries = _render_futures_watchlist_panel()
            self._options_entries = options_entries
            self._futures_entries = futures_entries
            return Group(_render_macro_panel(), _render_news_panel(), options_panel, futures_panel)

        def _apply_dashboard(self, renderable) -> None:
            self.query_one("#pane-dashboard", Static).update(renderable)

        def _apply_journal(self, text: str) -> None:
            self.query_one("#pane-journal", Static).update(text)

        def _apply_performance(self, text: str) -> None:
            self.query_one("#pane-performance", Static).update(text)

        def _apply_tables(self) -> None:
            options_table = self.query_one("#table-options", DataTable)
            futures_table = self.query_one("#table-futures", DataTable)
            options_table.clear(columns=False)
            futures_table.clear(columns=False)
            for entry in self._options_entries:
                options_table.add_row(
                    entry.symbol,
                    format_price(entry.price),
                    entry.regime,
                    format_percent(entry.atm_iv * 100 if entry.atm_iv is not None else None, decimals=1),
                    entry.confidence,
                    entry.signal,
                )
            for entry in self._futures_entries:
                futures_table.add_row(
                    entry.symbol,
                    format_price(entry.price),
                    entry.bias,
                    entry.confidence,
                    "Above" if entry.above_vwap else ("Below" if entry.above_vwap is not None else "N/A"),
                    entry.best_idea or "-",
                )

        def _selected_symbol(self, table_id: str) -> str | None:
            table = self.query_one(table_id, DataTable)
            if table.row_count == 0 or table.cursor_row is None:
                return None
            row = table.get_row_at(table.cursor_row)
            return str(row[0]) if row else None

        def action_prev_tab(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            idx = max(_TAB_NAMES.index(tabs.active[4:]) - 1, 0) if tabs.active else 0
            tabs.active = f"tab-{_TAB_NAMES[idx]}"

        def action_next_tab(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            idx = min(_TAB_NAMES.index(tabs.active[4:]) + 1, len(_TAB_NAMES) - 1) if tabs.active else 0
            tabs.active = f"tab-{_TAB_NAMES[idx]}"

        def action_refresh(self) -> None:
            self.watchlists = load_watchlists()
            self.settings = load_settings()
            self._refresh_all()
            self.query_one("#watchlist-status", Static).update("[green]Refreshed.[/green]")

        def action_save_watchlists(self) -> None:
            options_raw = self.query_one("#input-options", Input).value
            futures_raw = self.query_one("#input-futures", Input).value
            options = [s.strip().upper() for s in options_raw.split(",") if s.strip()]
            futures = [s.strip().upper() for s in futures_raw.split(",") if s.strip()]
            save_watchlists(options=options, futures=futures)
            self.watchlists = load_watchlists()
            self.query_one("#watchlist-status", Static).update("[green]Watchlists saved.[/green]")
            self._refresh_all()

        def action_open_selected(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            active = tabs.active[4:] if tabs.active else "dashboard"
            if active == "options":
                symbol = self._selected_symbol("#table-options")
                if symbol:
                    self.exit(result={"action": "options", "symbol": symbol})
            elif active == "futures":
                symbol = self._selected_symbol("#table-futures")
                if symbol:
                    self.exit(result={"action": "futures", "symbol": symbol})

        def action_open_options(self) -> None:
            symbol = (self.watchlists.options[0] if self.watchlists.options else self.settings.default_options_symbol or "SPY").upper()
            self.exit(result={"action": "options", "symbol": symbol})

        def action_open_futures(self) -> None:
            symbol = (self.watchlists.futures[0] if self.watchlists.futures else self.settings.default_symbol or "MES=F").upper()
            self.exit(result={"action": "futures", "symbol": symbol})

        def action_open_journal(self) -> None:
            self.exit(result={"action": "journal"})

        def action_open_performance(self) -> None:
            self.exit(result={"action": "performance"})

        def action_show_help(self) -> None:
            tabs = self.query_one("#tabs", TabbedContent)
            tabs.active = "tab-help"

        def action_quit_app(self) -> None:
            self.exit(result={"action": "quit"})


def _fallback_prompt() -> str:
    print("\nUnified analyzer actions: [options] [futures] [journal] [performance] [dashboard] [quit]")
    return input("Action: ").strip().lower()


def run_unified_analyzer() -> None:
    while True:
        try:
            if _TEXTUAL_AVAILABLE:
                result = UnifiedAnalyzerApp().run()  # type: ignore[name-defined]
                action = result.get("action") if isinstance(result, dict) else "quit"
            else:
                action = _fallback_prompt()
                result = {"action": action}
        except Exception as exc:
            _clear()
            print(f"Unified analyzer unavailable: {exc}")
            input("Press Enter to return...")
            return

        if action == "options":
            try:
                launch_options_workspace(result.get("symbol") if isinstance(result, dict) else None)
            except Exception as exc:
                _clear()
                print(f"Options analyzer unavailable: {exc}")
                input("Press Enter to continue...")
        elif action == "futures":
            try:
                launch_futures_workspace(result.get("symbol") if isinstance(result, dict) else None)
            except Exception as exc:
                _clear()
                print(f"Futures analyzer unavailable: {exc}")
                input("Press Enter to continue...")
        elif action == "journal":
            from journal.cli import run_journal_menu

            run_journal_menu()
        elif action == "performance":
            from journal.cli import run_dashboard

            run_dashboard()
        elif action == "dashboard":
            show_unified_dashboard()
        else:
            return
