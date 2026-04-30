from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import load_settings
from core.analysis import TradeEngine
from options.recommender import recommend_strategies


console = Console()


def _levels_table(analysis: dict) -> Table:
    table = Table(title="Nearby Levels")
    table.add_column("Level")
    table.add_column("Price", justify="right")
    table.add_column("Source")
    for level in analysis["key_levels"][:8]:
        table.add_row(level["label"], f"{level['price']:.5g}", level["source"])
    return table


def _alignment_table(analysis: dict) -> Table:
    table = Table(title="MTF Snapshot")
    table.add_column("TF")
    table.add_column("Trend")
    table.add_column("Structure")
    table.add_column("Position")
    table.add_column("ATR", justify="right")
    for tf, snap in analysis["snapshots"].items():
        atr = snap["atr"]
        table.add_row(tf, snap["trend"], snap["structure"], snap["price_position"], f"{atr:.5g}" if atr is not None else "N/A")
    return table


def _print_analysis(analysis: dict) -> None:
    header = f"{analysis['display_name']} ({analysis['symbol']})"
    console.print(
        Panel.fit(
            f"{header}\nBias: {analysis['bias']} ({analysis['confidence']}%)\nRegime: {analysis['regime']}\nAlignment: {analysis['alignment']}",
            title="Trade Terminal",
        )
    )
    console.print(Panel.fit(json.dumps(analysis["setup"], indent=2), title="Current Setup"))
    if analysis["risk_plan"]:
        console.print(Panel.fit(json.dumps(analysis["risk_plan"], indent=2), title="Risk / Reward"))
    console.print(_alignment_table(analysis))
    console.print(_levels_table(analysis))
    if analysis["liquidity_events"]:
        event_table = Table(title="Recent Liquidity Events")
        event_table.add_column("Direction")
        event_table.add_column("Level")
        event_table.add_column("Description")
        for event in analysis["liquidity_events"][:5]:
            event_table.add_row(event["direction"], event["level"], event["description"])
        console.print(event_table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trade Terminal speed-first CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze one symbol")
    analyze.add_argument("symbol")
    analyze.add_argument("--timeframe", default="5m")
    analyze.add_argument("--json", action="store_true")

    scan = sub.add_parser("scan", help="Scan watchlist symbols")
    scan.add_argument("symbols", nargs="*")
    scan.add_argument("--timeframe", default="5m")

    levels = sub.add_parser("levels", help="Show nearby levels")
    levels.add_argument("symbol")
    levels.add_argument("--timeframe", default="5m")

    setup = sub.add_parser("setup", help="Show current setup framing")
    setup.add_argument("symbol")
    setup.add_argument("--timeframe", default="5m")

    ask = sub.add_parser("ask", help="Query the local assistant")
    ask.add_argument("symbol")
    ask.add_argument("query")
    ask.add_argument("--timeframe", default="5m")

    export = sub.add_parser("export", help="Export latest analysis to disk")
    export.add_argument("symbol")
    export.add_argument("--timeframe", default="5m")
    export.add_argument("--output", default=None)

    backtest = sub.add_parser("backtest", help="Run the heuristic backtest summary")
    backtest.add_argument("symbol")
    backtest.add_argument("--timeframe", default="5m")

    options = sub.add_parser("options", help="Recommend options strategies")
    options.add_argument("symbol")
    options.add_argument("--view", default="neutral", choices=["bullish", "bearish", "neutral"])
    options.add_argument("--expiry", default=None)
    options.add_argument("--max-risk", type=float, default=None)
    options.add_argument("--strategy-type", default=None)
    options.add_argument("--min-score", type=float, default=None)
    options.add_argument("--limit", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    engine = TradeEngine(settings=settings)

    if args.command == "scan":
        symbols = args.symbols or settings.futures_watchlist or [settings.default_symbol]
        table = Table(title=f"Scan ({args.timeframe})")
        table.add_column("Symbol")
        table.add_column("Bias")
        table.add_column("Regime")
        table.add_column("Setup")
        table.add_column("Confidence", justify="right")
        for symbol in symbols:
            analysis = engine.analyze(symbol, timeframe=args.timeframe).to_dict()
            table.add_row(analysis["symbol"], analysis["bias"], analysis["regime"], analysis["setup"]["type"], str(analysis["confidence"]))
        console.print(table)
        return

    if args.command == "ask":
        response = engine.query(args.symbol, args.query, timeframe=args.timeframe)
        console.print(Panel.fit(response.answer, title=f"Assistant • {response.intent}"))
        return

    if args.command == "options":
        recommendations = recommend_strategies(
            args.symbol,
            view=args.view,
            expiry=args.expiry,
            max_risk=args.max_risk,
            strategy_type=args.strategy_type,
            min_score=args.min_score,
            limit=args.limit,
        )
        if not recommendations:
            console.print("No strategy recommendations matched the current filters.")
            return
        table = Table(title=f"Options Strategies • {args.symbol.upper()}")
        table.add_column("Strategy")
        table.add_column("Expiry")
        table.add_column("Score", justify="right")
        table.add_column("Edge %", justify="right")
        table.add_column("Max Loss", justify="right")
        table.add_column("Why")
        for recommendation in recommendations:
            edge_pct = recommendation.model_edge.get("edge_pct")
            max_loss = recommendation.max_loss
            table.add_row(
                recommendation.strategy_name,
                recommendation.expiry,
                f"{recommendation.final_score:.1f}",
                f"{edge_pct:.1f}%" if edge_pct is not None else "N/A",
                f"{max_loss:.2f}" if max_loss is not None else "Open",
                recommendation.why_this_strategy,
            )
        console.print(table)
        top = recommendations[0]
        console.print(Panel.fit(json.dumps(top.to_dict(), indent=2), title="Top Recommendation"))
        return

    analysis = engine.analyze(args.symbol, timeframe=args.timeframe).to_dict()

    if args.command == "analyze":
        if args.json:
            console.print_json(data=analysis)
        else:
            _print_analysis(analysis)
        return

    if args.command == "levels":
        console.print(_levels_table(analysis))
        return

    if args.command == "setup":
        console.print(Panel.fit(json.dumps(analysis["setup"], indent=2), title="Setup"))
        if analysis["risk_plan"]:
            console.print(Panel.fit(json.dumps(analysis["risk_plan"], indent=2), title="Risk"))
        return

    if args.command == "export":
        output = Path(args.output) if args.output else settings.data_path / f"{analysis['symbol'].replace('=','_').lower()}_{args.timeframe}_analysis.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(analysis, indent=2))
        console.print(f"Exported to {output}")
        return

    if args.command == "backtest":
        summary = analysis.get("backtest_summary")
        if not summary:
            console.print("Backtest summary unavailable.")
            return
        console.print(Panel.fit(json.dumps(summary, indent=2), title="Backtest Summary"))
        return
