from __future__ import annotations

from typing import TYPE_CHECKING

from analyzer.formatting import format_percent, format_price
from config.settings import load_settings
from options.snapshot import OptionsSnapshot, get_options_snapshot

try:
    from textual import work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane

    _TEXTUAL_AVAILABLE = True
except ImportError:
    _TEXTUAL_AVAILABLE = False

if TYPE_CHECKING:
    from options.core.analyzer import Analyzer
    from options.monitor import OptionsMonitorEntry


_TAB_NAMES = [
    "overview", "summary", "calls", "puts", "straddle", "gamma", "volume",
    "regime", "macro", "surface", "diagnostics", "monitor", "journal",
]
_LOADING = "[dim]Loading…[/dim]"
_CHAIN_TABLE_IDS = ("#table-calls", "#table-puts", "#table-straddle", "#table-gamma")


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


def _style_signal(text: str):
    from rich.text import Text

    lowered = text.lower()
    if any(word in lowered for word in ("buy", "bull", "long", "high")):
        return Text(text, style="green")
    if any(word in lowered for word in ("sell", "bear", "short", "low")):
        return Text(text, style="red")
    if any(word in lowered for word in ("review", "warn", "caution", "mixed", "neutral")):
        return Text(text, style="yellow")
    return Text(text, style="white")


def _strike_window(df, spot: float | None, max_rows: int = 29):
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df, 0
    working = df.copy()
    strike_col = next((col for col in working.columns if str(col).lower().startswith("strike")), None)
    if strike_col is None:
        return working, 0
    working[strike_col] = pd.to_numeric(working[strike_col], errors="coerce")
    working = working.dropna(subset=[strike_col]).reset_index(drop=True)
    if working.empty or spot is None:
        return working.head(max_rows), 0
    bounds = max(abs(spot) * 0.12, 5.0)
    filtered = working[(working[strike_col] >= spot - bounds) & (working[strike_col] <= spot + bounds)].copy()
    if filtered.empty:
        filtered = working.copy()
    filtered = filtered.sort_values(strike_col).reset_index(drop=True)
    atm_idx = int((filtered[strike_col] - spot).abs().idxmin())
    half = max_rows // 2
    start = max(0, atm_idx - half)
    end = min(len(filtered), start + max_rows)
    start = max(0, end - max_rows)
    sliced = filtered.iloc[start:end].reset_index(drop=True)
    local_atm = int((sliced[strike_col] - spot).abs().idxmin()) if not sliced.empty else 0
    return sliced, local_atm


def _populate_chain_table(table: "DataTable", df, spot: float | None, mode: str) -> None:
    from rich.text import Text

    table.clear(columns=False)
    window_df, atm_idx = _strike_window(df, spot)
    if window_df is None or window_df.empty:
        table.add_row("No data", "", "", "", "", "", "")
        return

    for _, row in window_df.iterrows():
        strike = float(row.get("Strike", row.get("strike", 0)) or 0)
        atm = abs(strike - (spot or strike)) <= max((spot or strike) * 0.0025, 0.5)
        atm_mark = Text("●", style="bold magenta") if atm else Text("·", style="grey50")
        strike_text = Text(format_price(strike), style="bold magenta" if atm else "cyan")

        if mode in {"calls", "puts"}:
            table.add_row(
                atm_mark,
                strike_text,
                row.get("Market IV", "N/A"),
                row.get("Model IV", "N/A"),
                format_price(row.get("Bid")) if row.get("Bid") != "N/A" else "N/A",
                format_price(row.get("Ask")) if row.get("Ask") != "N/A" else "N/A",
                _style_signal(str(row.get("Action", "-"))),
            )
        elif mode == "straddle":
            table.add_row(
                atm_mark,
                strike_text,
                format_price(row.get("Bid_Call")) if row.get("Bid_Call") != "N/A" else "N/A",
                format_price(row.get("Ask_Call")) if row.get("Ask_Call") != "N/A" else "N/A",
                format_price(row.get("Bid_Put")) if row.get("Bid_Put") != "N/A" else "N/A",
                format_price(row.get("Ask_Put")) if row.get("Ask_Put") != "N/A" else "N/A",
                "",
            )
        else:
            zone = str(row.get("Zone", ""))
            table.add_row(
                atm_mark,
                strike_text,
                format_price(row.get("Call OI")) if row.get("Call OI", "") not in ("", "N/A") else str(row.get("Call OI", "")),
                str(row.get("Call GEX($M)", "")),
                format_price(row.get("Put OI")) if row.get("Put OI", "") not in ("", "N/A") else str(row.get("Put OI", "")),
                str(row.get("Net GEX($M)", row.get("Put GEX($M)", ""))),
                _style_signal(zone if zone else "near spot"),
            )
    table.cursor_row = atm_idx


def _volume_analytics_text(analyzer: "Analyzer | None") -> str:
    import pandas as pd
    from tabulate import tabulate

    if analyzer is None:
        return _LOADING
    calls = analyzer._raw_calls_df.copy()
    puts = analyzer._raw_puts_df.copy()
    expiry = analyzer.expiry_str
    spot = analyzer.S_0

    for df in (calls, puts):
        df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce")
        df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
        df["openInterest"] = pd.to_numeric(df.get("openInterest"), errors="coerce").fillna(0.0)
        df["vol_oi"] = df["volume"] / df["openInterest"].replace(0, pd.NA)
        df["vol_oi"] = df["vol_oi"].fillna(0.0)

    band = max(abs(spot) * 0.12, 5.0)
    calls = calls[(calls["strike"] >= spot - band) & (calls["strike"] <= spot + band)].copy()
    puts = puts[(puts["strike"] >= spot - band) & (puts["strike"] <= spot + band)].copy()

    call_vol = float(calls["volume"].sum())
    put_vol = float(puts["volume"].sum())
    ratio = put_vol / max(call_vol, 1.0)

    total_oi = float(calls["openInterest"].sum() + puts["openInterest"].sum())
    total_vol = call_vol + put_vol
    weighted_signal = "bullish call-led flow" if call_vol > put_vol * 1.15 else "bearish put-led flow" if put_vol > call_vol * 1.15 else "balanced flow"
    trend = "rising put pressure" if ratio > 1.1 else "rising call demand" if ratio < 0.9 else "stable put/call mix"

    joined = pd.concat(
        [
            calls.assign(side="CALL")[["side", "strike", "volume", "openInterest", "vol_oi"]],
            puts.assign(side="PUT")[["side", "strike", "volume", "openInterest", "vol_oi"]],
        ],
        ignore_index=True,
    )
    unusual = joined[joined["vol_oi"] >= 1.25].sort_values(["vol_oi", "volume"], ascending=False).head(10)
    leaders = joined.sort_values(["volume", "vol_oi"], ascending=False).head(12).copy()
    leaders["Strike"] = leaders["strike"].map(format_price)
    leaders["Vol"] = leaders["volume"].map(format_price)
    leaders["OI"] = leaders["openInterest"].map(format_price)
    leaders["Vol/OI"] = leaders["vol_oi"].map(lambda v: f"{v:.2f}x")

    unusual_lines = [f"- {row.side} {format_price(row.strike)} at {row.vol_oi:.2f}x vol/OI" for row in unusual.itertuples()]
    if not unusual_lines:
        unusual_lines = ["- No unusually aggressive volume near spot"]

    leader_table = tabulate(
        leaders[["side", "Strike", "Vol", "OI", "Vol/OI"]],
        headers=["Side", "Strike", "Vol", "OI", "Vol/OI"],
        tablefmt="simple",
        showindex=False,
    ) if not leaders.empty else "No strike-level volume table available"

    lines = [
        "[bold]VOLUME ANALYTICS[/bold]",
        "",
        f"[cyan]Expiry[/cyan]: {expiry}",
        f"[cyan]Spot Window[/cyan]: {format_price(spot - band)} to {format_price(spot + band)}",
        f"[cyan]Put/Call Volume Ratio[/cyan]: {ratio:.2f} ({trend})",
        f"[cyan]Total Volume vs OI[/cyan]: {total_vol / max(total_oi, 1.0):.2f}x",
        f"[cyan]Volume-Weighted Read[/cyan]: {weighted_signal}",
        "",
        "[bold]Unusual Volume Flags[/bold]",
        *unusual_lines,
        "",
        "[bold]Top Strikes by Volume / OI[/bold]",
        leader_table,
    ]
    return "\n".join(lines)


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
        TabPane { padding: 1 2; overflow: hidden auto; }
        Static { padding: 0 1; width: 1fr; }
        DataTable { height: 1fr; width: 1fr; }
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
            self._volume_text = _LOADING

        def compose(self) -> "ComposeResult":
            yield Header(show_clock=True)
            with TabbedContent("Overview", "Summary", "Calls", "Puts", "Straddle", "Gamma", "Volume", "Regime", "Macro", "Surface", "Diagnostics", "Monitor", "Journal", id="tabs"):
                yield TabPane("Overview", Static(_LOADING, id="pane-overview"), id="tab-overview")
                yield TabPane("Summary", Static(_LOADING, id="pane-summary"), id="tab-summary")
                yield TabPane("Calls", DataTable(id="table-calls"), id="tab-calls")
                yield TabPane("Puts", DataTable(id="table-puts"), id="tab-puts")
                yield TabPane("Straddle", DataTable(id="table-straddle"), id="tab-straddle")
                yield TabPane("Gamma", DataTable(id="table-gamma"), id="tab-gamma")
                yield TabPane("Volume", Static(_LOADING, id="pane-volume"), id="tab-volume")
                yield TabPane("Regime", Static(_LOADING, id="pane-regime"), id="tab-regime")
                yield TabPane("Macro", Static(_LOADING, id="pane-macro"), id="tab-macro")
                yield TabPane("Surface", Static(_LOADING, id="pane-surface"), id="tab-surface")
                yield TabPane("Diagnostics", Static(_LOADING, id="pane-diagnostics"), id="tab-diagnostics")
                yield TabPane("Monitor", Static(_LOADING, id="pane-monitor"), id="tab-monitor")
                yield TabPane("Journal", Static(_LOADING, id="pane-journal"), id="tab-journal")
            yield Footer()

        def on_mount(self) -> None:
            self.title = f"Options Dashboard — {self.symbol}"
            self._init_tables()
            self._load_snapshot()
            self._load_monitor()
            self._load_analyzer()
            self._update_journal()

        def _init_tables(self) -> None:
            specs = {
                "#table-calls": ["ATM", "Strike", "Mkt IV", "Model IV", "Bid", "Ask", "Action"],
                "#table-puts": ["ATM", "Strike", "Mkt IV", "Model IV", "Bid", "Ask", "Action"],
                "#table-straddle": ["ATM", "Strike", "Call Bid", "Call Ask", "Put Bid", "Put Ask", ""],
                "#table-gamma": ["ATM", "Strike", "Call OI", "Call GEX", "Put OI", "Net GEX", "Zone"],
            }
            for table_id, columns in specs.items():
                table = self.query_one(table_id, DataTable)
                table.cursor_type = "row"
                table.zebra_stripes = True
                table.add_columns(*columns)

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
            if a:
                _populate_chain_table(self.query_one("#table-calls", DataTable), a.calls_df, a.S_0, "calls")
                _populate_chain_table(self.query_one("#table-puts", DataTable), a.puts_df, a.S_0, "puts")
                _populate_chain_table(self.query_one("#table-straddle", DataTable), a.straddle_df, a.S_0, "straddle")
                _populate_chain_table(self.query_one("#table-gamma", DataTable), a._gamma_gex_df(), a.S_0, "gamma")
            self.query_one("#pane-regime", Static).update(_df_to_text(a._regime_df() if a else None))
            self.query_one("#pane-macro", Static).update(_render_macro(a))
            self.query_one("#pane-surface", Static).update(_df_to_text(a._surface_df() if a else None))
            self.query_one("#pane-diagnostics", Static).update(_df_to_text(a._diagnostics_df() if a else None))
            self._volume_text = _volume_analytics_text(a)
            self.query_one("#pane-volume", Static).update(self._volume_text)

        def _update_monitor(self) -> None:
            self.query_one("#pane-monitor", Static).update(_render_monitor(self._monitor))

        def _update_journal(self) -> None:
            self.query_one("#pane-journal", Static).update(_render_journal(self.db_path))

        def _show_error(self, msg: str, only_overview: bool = False) -> None:
            targets = ["#pane-overview"] if only_overview else [
                "#pane-summary", "#pane-volume", "#pane-regime", "#pane-macro", "#pane-surface", "#pane-diagnostics",
            ]
            for pane_id in targets:
                self.query_one(pane_id, Static).update(f"[red]Unavailable[/red]\n{msg}")
            if not only_overview:
                for table_id in _CHAIN_TABLE_IDS:
                    table = self.query_one(table_id, DataTable)
                    table.clear(columns=False)
                    table.add_row("Unavailable", "", "", "", "", "", "")

        def _switch_symbol(self, step: int) -> None:
            if not self.watchlist:
                return
            idx = self.watchlist.index(self.symbol)
            self.symbol = self.watchlist[(idx + step) % len(self.watchlist)]
            self._snapshot = None
            self._analyzer = None
            self._analyzer_symbol = None
            self.query_one("#pane-overview", Static).update(_LOADING)
            for pane_id in ["#pane-summary", "#pane-volume", "#pane-regime", "#pane-macro", "#pane-surface", "#pane-diagnostics"]:
                self.query_one(pane_id, Static).update(_LOADING)
            for table_id in _CHAIN_TABLE_IDS:
                self.query_one(table_id, DataTable).clear(columns=False)
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
