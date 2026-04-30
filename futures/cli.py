from __future__ import annotations

import os
import sys
from datetime import datetime

from config.tickers import list_known_tickers, resolve_ticker, CONTRACT_SPECS
from futures.data import fetch_futures_data
from futures.levels import compute_levels, format_levels_display
from futures.planner import SETUP_TYPES, TIMEFRAMES, assess_trade, format_plan_display
from config.settings import load_settings

try:
    from futures.tui import run_tui, _TEXTUAL_AVAILABLE
except ImportError:
    _TEXTUAL_AVAILABLE = False
    run_tui = None  # type: ignore[assignment]


def _clear() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {msg}{suffix}: ").strip()
    return val if val else default


def _prompt_float(msg: str, default: float | None = None) -> float | None:
    suffix = f" [{default}]" if default is not None else " (leave blank to skip)"
    raw = input(f"  {msg}{suffix}: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("  Invalid number — skipping.")
        return default


def _prompt_choice(options: list[str], title: str = "Select") -> int:
    """Display a numbered list and return the 0-based index chosen."""
    print(f"\n  {title}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i:>2}. {opt}")
    while True:
        raw = input(f"  Choice (1-{len(options)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


# ---------------------------------------------------------------------------
# Levels screen
# ---------------------------------------------------------------------------

def run_levels_screen(symbol: str | None = None) -> None:
    _clear()
    print("\n  FUTURES LEVELS ANALYZER")
    print("  " + "─" * 42)

    if symbol is None:
        symbol = _prompt("Ticker (e.g. MES, MNQ, MBT, or yfinance symbol like ES=F)").upper()
    if not symbol:
        return

    print(f"\n  Fetching data for {symbol}... (this may take a few seconds)")
    try:
        data = fetch_futures_data(symbol)
    except Exception as exc:
        print(f"\n  ERROR: Could not fetch data — {exc}")
        input("\n  Press Enter to continue...")
        return

    levels = compute_levels(data)
    _clear()
    print(format_levels_display(levels))

    print("\n  Options:")
    print("    1. Plan a trade on this instrument")
    print("    2. Log a trade on this instrument")
    print("    3. Refresh")
    print("    0. Back")
    choice = input("  Choice: ").strip()

    if choice == "1":
        run_planner(symbol=symbol, prefetched_data=data)
    elif choice == "2":
        from journal.cli import log_futures_trade
        log_futures_trade(prefill_ticker=symbol)
    elif choice == "3":
        run_levels_screen(symbol)


# ---------------------------------------------------------------------------
# Trade planner
# ---------------------------------------------------------------------------

def run_planner(symbol: str | None = None, prefetched_data=None, idea_defaults: dict | None = None) -> None:
    _clear()
    print("\n  FUTURES TRADE PLANNER")
    print("  " + "─" * 42)

    if symbol is None:
        symbol = _prompt("Ticker (e.g. MES, M6E, MBT)").upper()
    if not symbol:
        return

    # Fetch ATR if not already available
    atr_5m = atr_15m = None
    if prefetched_data is not None:
        from futures.levels import compute_levels
        lvl = compute_levels(prefetched_data)
        atr_5m  = lvl.atr_5m
        atr_15m = lvl.atr_15m
        print(f"\n  5m ATR: {atr_5m:.5g}" if atr_5m else "\n  5m ATR: N/A")
        print(f"  15m ATR: {atr_15m:.5g}" if atr_15m else "  15m ATR: N/A")
    else:
        print(f"\n  Fetching ATR for {symbol}...")
        try:
            data = fetch_futures_data(symbol)
            from futures.levels import compute_levels
            lvl = compute_levels(data)
            atr_5m  = lvl.atr_5m
            atr_15m = lvl.atr_15m
            print(f"  5m ATR: {atr_5m:.5g}" if atr_5m else "  5m ATR: N/A")
            print(f"  15m ATR: {atr_15m:.5g}" if atr_15m else "  15m ATR: N/A")
        except Exception as exc:
            print(f"  Could not fetch ATR ({exc}). You can enter it manually.")
            atr_5m  = _prompt_float("5m ATR (optional)")
            atr_15m = _prompt_float("15m ATR (optional)")

    print()
    # Direction
    default_direction = (idea_defaults or {}).get("direction", "long")
    while True:
        direction = _prompt("Direction (long/short)", default_direction).lower()
        if direction in ("long", "short"):
            break
        print("  Please enter 'long' or 'short'.")

    entry  = _prompt_float("Entry price", (idea_defaults or {}).get("entry"))
    if entry is None:
        print("  Entry price required.")
        input("  Press Enter to continue...")
        return

    stop = _prompt_float("Stop price", (idea_defaults or {}).get("stop"))
    if stop is None:
        print("  Stop price required.")
        input("  Press Enter to continue...")
        return

    target = _prompt_float("Target price (leave blank to skip)", (idea_defaults or {}).get("target"))

    default_setup = (idea_defaults or {}).get("setup_type")
    if default_setup in SETUP_TYPES:
        setup_type = default_setup
    else:
        setup_idx = _prompt_choice(SETUP_TYPES, "Setup type")
        setup_type = SETUP_TYPES[setup_idx]

    default_timeframe = (idea_defaults or {}).get("timeframe")
    if default_timeframe in TIMEFRAMES:
        timeframe = default_timeframe
    else:
        tf_idx = _prompt_choice(TIMEFRAMES, "Timeframe")
        timeframe = TIMEFRAMES[tf_idx]

    notes = _prompt("Notes (optional)", (idea_defaults or {}).get("notes", ""))

    plan = assess_trade(
        symbol=resolve_ticker(symbol)[0],
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        setup_type=setup_type,
        timeframe=timeframe,
        notes=notes,
        atr_5m=atr_5m,
        atr_15m=atr_15m,
    )

    _clear()
    print(format_plan_display(plan))

    print("\n  Options:")
    print("    1. Save as planned trade")
    print("    2. Log this trade when complete")
    print("    3. Plan another trade")
    print("    0. Back")
    choice = input("  Choice: ").strip()

    if choice == "1":
        from journal.cli import plan_futures_trade
        plan_futures_trade(
            prefill={
                "ticker": symbol,
                "direction": direction,
                "entry_price": entry,
                "stop_price": stop,
                "planned_target": target,
                "setup_type": setup_type,
                "timeframe": timeframe,
                "notes": notes,
                "atr_5m": atr_5m,
                "atr_15m": atr_15m,
                "reason_for_entry": (idea_defaults or {}).get("reason_for_entry", ""),
                "analyzer_idea": (idea_defaults or {}).get("analyzer_idea", ""),
                "bias_at_entry": (idea_defaults or {}).get("bias_at_entry", ""),
                "confidence_at_entry": (idea_defaults or {}).get("confidence_at_entry", ""),
                "confluence_score": (idea_defaults or {}).get("confluence_score"),
                "fvg_involved": (idea_defaults or {}).get("fvg_involved"),
                "volume_node_involved": (idea_defaults or {}).get("volume_node_involved"),
                "session_level_involved": (idea_defaults or {}).get("session_level_involved"),
            }
        )
    elif choice == "2":
        from journal.cli import log_futures_trade
        log_futures_trade(prefill_ticker=symbol)
    elif choice == "3":
        run_planner()


# ---------------------------------------------------------------------------
# Futures Analyzer main menu
# ---------------------------------------------------------------------------

def run_tui_for_symbol(symbol: str) -> None:
    """Launch the textual TUI, falling back to the levels screen on error."""
    if not _TEXTUAL_AVAILABLE or run_tui is None:
        print("\n  [textual not installed] Falling back to levels screen.")
        print("  Install with: pip install textual")
        input("  Press Enter to continue...")
        run_levels_screen(symbol)
        return

    try:
        result = run_tui(symbol)
    except Exception as exc:
        print(f"\n  TUI error: {exc}")
        print("  Falling back to levels screen.")
        input("  Press Enter to continue...")
        run_levels_screen(symbol)
        return

    # Handle hotkey exits from the TUI
    if result == "log":
        from journal.cli import log_futures_trade
        log_futures_trade(prefill_ticker=symbol)
    elif result == "plan":
        run_planner(symbol=symbol)
    elif result == "journal":
        from journal.cli import run_journal_menu
        run_journal_menu()
    elif isinstance(result, dict) and result.get("action") == "plan":
        run_planner(symbol=symbol, idea_defaults=result.get("idea"))


def launch_futures_workspace() -> None:
    """
    Preferred futures entrypoint from main.py.
    Uses the richer textual dashboard first when available, with the older
    analyzer menu kept as a secondary fallback path.
    """
    settings = load_settings()
    default_symbol = (settings.default_symbol or "MES=F").upper()

    while True:
        _clear()
        print("\n  FUTURES WORKSPACE")
        print("  " + "═" * 42)
        print(f"    Default symbol: {default_symbol}")
        if _TEXTUAL_AVAILABLE and settings.use_tui and settings.tui_enabled:
            print("    1. Open dashboard for default symbol")
            print("    2. Open dashboard for another symbol")
        else:
            print("    1. Open analyzer for default symbol")
            print("    2. Open analyzer for another symbol")
        print("    3. Trade planner")
        print("    4. Trade journal")
        print("    5. Legacy futures tools")
        print("    0. Back to main menu")
        print("  " + "─" * 42)
        choice = input("  Choice: ").strip()

        if choice == "1":
            if _TEXTUAL_AVAILABLE and settings.use_tui and settings.tui_enabled:
                run_tui_for_symbol(default_symbol)
            else:
                run_levels_screen(default_symbol)
        elif choice == "2":
            symbol = _prompt("Ticker (e.g. MES, MNQ, MBT, ES=F)", default_symbol).upper()
            if not symbol:
                continue
            if _TEXTUAL_AVAILABLE and settings.use_tui and settings.tui_enabled:
                run_tui_for_symbol(symbol)
            else:
                run_levels_screen(symbol)
        elif choice == "3":
            run_planner(symbol=default_symbol)
        elif choice == "4":
            from journal.cli import run_journal_menu
            run_journal_menu()
        elif choice == "5":
            run_futures_analyzer()
        elif choice == "0":
            return
        else:
            print("  Invalid choice.")
            input("  Press Enter to continue...")


def run_futures_analyzer() -> None:
    while True:
        _clear()
        print("\n  FUTURES ANALYZER")
        print("  " + "═" * 42)
        if _TEXTUAL_AVAILABLE:
            print("    1. Open TUI dashboard (textual)")
        else:
            print("    1. Analyze futures levels")
        print("    2. Plan a futures trade")
        print("    3. Log a futures trade")
        print("    4. List supported tickers")
        print("    0. Back to main menu")
        print("  " + "─" * 42)
        choice = input("  Choice: ").strip()

        if choice == "1":
            symbol = _prompt(
                "Ticker (e.g. MES, MNQ, MBT, ES=F)",
                "MES",
            ).upper()
            if not symbol:
                continue
            if _TEXTUAL_AVAILABLE:
                run_tui_for_symbol(symbol)
            else:
                run_levels_screen(symbol)
        elif choice == "2":
            run_planner()
        elif choice == "3":
            from journal.cli import log_futures_trade
            log_futures_trade()
        elif choice == "4":
            _show_supported_tickers()
        elif choice == "0":
            return
        else:
            print("  Invalid choice.")
            input("  Press Enter to continue...")


def _show_supported_tickers() -> None:
    _clear()
    rows = list_known_tickers()
    from tabulate import tabulate
    print("\n  SUPPORTED FUTURES TICKERS\n")
    print(tabulate(rows, headers=["Alias", "yfinance Symbol", "Name"], tablefmt="simple"))
    print("\n  You can also type any yfinance symbol directly (e.g. MES=F, NQ=F).")
    input("\n  Press Enter to continue...")
