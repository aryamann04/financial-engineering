from __future__ import annotations

import os
from datetime import datetime

from journal.models import (
    DailyReview, FuturesTrade, OptionsTrade,
    FUTURES_SETUPS, OPTIONS_STRATEGIES, MISTAKE_TAGS, TIMEFRAMES,
    compute_futures_metrics, compute_options_metrics,
)
from journal.storage import TradeDatabase
from journal.metrics import compute_metrics
from journal.dashboard import render_full_dashboard, render_trades_table, render_overview, render_breakdown

_DB = TradeDatabase()


def _clear() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def _p(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {msg}{suffix}: ").strip()
    return val if val else default


def _pf(msg: str, required: bool = False, default: float | None = None) -> float | None:
    suffix = f" [{default}]" if default is not None else (" (required)" if required else " (blank to skip)")
    while True:
        raw = input(f"  {msg}{suffix}: ").strip()
        if not raw:
            if required:
                print("  This field is required.")
                continue
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Invalid number, please try again.")


def _pick(options: list[str], title: str = "Select") -> int:
    print(f"\n  {title}:")
    for i, o in enumerate(options, 1):
        print(f"    {i:>2}. {o}")
    while True:
        raw = input(f"  Choice (1-{len(options)}): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(options)}.")


def _pick_datetime(prompt: str, default: str = "") -> str:
    fmt_hint = "YYYY-MM-DD HH:MM"
    if not default:
        default = datetime.now().strftime("%Y-%m-%d %H:%M")
    while True:
        raw = input(f"  {prompt} ({fmt_hint}) [{default}]: ").strip()
        val = raw if raw else default
        try:
            datetime.strptime(val[:16], "%Y-%m-%d %H:%M")
            return val[:16]
        except ValueError:
            print(f"  Invalid format. Use {fmt_hint}.")


def _pick_mistake_tags() -> list[str]:
    print("\n  Mistake tags (press Enter to skip, or type numbers separated by space):")
    for i, tag in enumerate(MISTAKE_TAGS, 1):
        print(f"    {i:>2}. {tag}")
    raw = input("  Tags: ").strip()
    if not raw:
        return []
    selected = []
    for tok in raw.split():
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(MISTAKE_TAGS):
                selected.append(MISTAKE_TAGS[idx])
        except ValueError:
            pass
    return selected


def _pick_bool(prompt: str) -> bool | None:
    raw = input(f"  {prompt} (y/n/blank): ").strip().lower()
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    return None


# ---------------------------------------------------------------------------
# Log futures trade
# ---------------------------------------------------------------------------

def log_futures_trade(prefill_ticker: str = "") -> None:
    _clear()
    print("\n  LOG FUTURES TRADE")
    print("  " + "─" * 42)

    ticker = _p("Ticker (e.g. MES, M6E, MBT)", prefill_ticker).upper()
    if not ticker:
        return

    # Resolve display name and contract spec
    from config.tickers import resolve_ticker, CONTRACT_SPECS
    yf_sym, spec = resolve_ticker(ticker)
    if spec:
        print(f"  Contract: {spec['name']}  |  Multiplier: {spec['multiplier']}  |  Tick: {spec['tick_size']}")

    direction = ""
    while direction not in ("long", "short"):
        direction = _p("Direction (long/short)", "long").lower()

    entry_time = _pick_datetime("Entry time (ET)")
    exit_time  = _pick_datetime("Exit time  (ET)", default=entry_time)

    entry_price = _pf("Entry price", required=True)
    exit_price  = _pf("Exit price",  required=True)
    stop_price  = _pf("Stop price",  required=True)

    planned_target = _pf("Planned target (blank to skip)")
    quantity = _pf("Contracts", default=1.0) or 1.0

    # Contract multiplier — prefer spec, else ask
    multiplier: float | None = None
    if spec:
        multiplier = float(spec["multiplier"])
        print(f"  Using contract multiplier: {multiplier}")
    else:
        multiplier = _pf("Contract multiplier (blank to skip, needed for $ P&L)")

    fees = _pf("Fees/commission (optional)", default=0.0) or 0.0

    setup_idx  = _pick(FUTURES_SETUPS, "Setup type")
    setup_type = FUTURES_SETUPS[setup_idx]

    tf_idx    = _pick(TIMEFRAMES, "Timeframe")
    timeframe = TIMEFRAMES[tf_idx]

    # ATR
    print("\n  ATR at time of trade (optional, used for performance analysis):")
    atr_5m  = _pf("5m ATR")
    atr_15m = _pf("15m ATR")

    notes = _p("Notes (optional)")
    mistake_tags = _pick_mistake_tags()
    did_follow_plan = _pick_bool("Did you follow the plan?")
    screenshot = _p("Screenshot path (optional)")

    trade = FuturesTrade(
        ticker=yf_sym,
        direction=direction,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=stop_price,
        planned_target=planned_target,
        setup_type=setup_type,
        timeframe=timeframe,
        quantity=quantity,
        fees=fees,
        notes=notes,
        screenshot_path=screenshot or None,
        mistake_tags=mistake_tags,
        did_follow_plan=did_follow_plan,
        contract_multiplier=multiplier,
        atr_5m=atr_5m,
        atr_15m=atr_15m,
    )
    compute_futures_metrics(trade)

    # Preview
    _clear()
    print("\n  TRADE PREVIEW")
    print("  " + "─" * 42)
    sign = 1 if direction == "long" else -1
    pnl_pts = sign * (exit_price - entry_price) * quantity
    print(f"  Ticker      : {yf_sym}  ({direction.upper()})")
    print(f"  Entry/Exit  : {entry_price} → {exit_price}")
    print(f"  P&L (pts)   : {pnl_pts:+.4f}")
    if trade.pnl_dollars is not None:
        print(f"  P&L ($)     : {trade.pnl_dollars:+.2f}")
    if trade.r_multiple is not None:
        print(f"  R multiple  : {trade.r_multiple:.2f}R")
    print(f"  Setup       : {setup_type}")
    print(f"  Session     : {trade.session_bucket}")
    print(f"  Hold (min)  : {trade.holding_period_minutes:.0f}")

    confirm = _p("\n  Save this trade? (y/n)", "y").lower()
    if confirm == "y":
        trade_id = _DB.save_futures_trade(trade)
        print(f"\n  Saved — trade ID: {trade_id}")
    else:
        print("\n  Trade not saved.")

    input("\n  Press Enter to continue...")


def plan_futures_trade(prefill: dict | None = None) -> None:
    _clear()
    print("\n  PLAN FUTURES TRADE")
    print("  " + "─" * 42)

    prefill = prefill or {}
    ticker = _p("Ticker (e.g. MES, M6E, MBT)", str(prefill.get("ticker", ""))).upper()
    if not ticker:
        return

    from config.tickers import resolve_ticker
    yf_sym, spec = resolve_ticker(ticker)

    direction = ""
    while direction not in ("long", "short"):
        direction = _p("Direction (long/short)", str(prefill.get("direction", "long"))).lower()

    plan_time = _pick_datetime("Planned entry time (ET)")
    entry_price = _pf("Planned entry price", required=True, default=prefill.get("entry_price"))
    stop_price = _pf("Invalidation / stop price", required=True, default=prefill.get("stop_price"))
    planned_target = _pf("Planned target", default=prefill.get("planned_target"))
    quantity = _pf("Contracts", default=1.0) or 1.0

    setup_default = str(prefill.get("setup_type", "Manual/Other"))
    if setup_default in FUTURES_SETUPS:
        setup_type = setup_default
    else:
        setup_type = FUTURES_SETUPS[_pick(FUTURES_SETUPS, "Setup type")]

    timeframe_default = str(prefill.get("timeframe", "5m"))
    if timeframe_default in TIMEFRAMES:
        timeframe = timeframe_default
    else:
        timeframe = TIMEFRAMES[_pick(TIMEFRAMES, "Timeframe")]

    notes = _p("Notes", str(prefill.get("notes", "")))
    reason_for_entry = _p("Reason for entry", str(prefill.get("reason_for_entry", prefill.get("analyzer_idea", ""))))
    did_follow_plan = True

    trade = FuturesTrade(
        ticker=yf_sym,
        direction=direction,
        entry_time=plan_time,
        exit_time=plan_time,
        entry_price=float(entry_price),
        exit_price=float(entry_price),
        stop_price=float(stop_price),
        planned_target=planned_target,
        setup_type=setup_type,
        timeframe=timeframe,
        quantity=float(quantity),
        notes=notes,
        did_follow_plan=did_follow_plan,
        contract_multiplier=float(spec["multiplier"]) if spec else None,
        atr_5m=prefill.get("atr_5m"),
        atr_15m=prefill.get("atr_15m"),
        state="planned",
        analyzer_idea=str(prefill.get("analyzer_idea", "")),
        confluence_score=prefill.get("confluence_score"),
        bias_at_entry=str(prefill.get("bias_at_entry", "")),
        confidence_at_entry=str(prefill.get("confidence_at_entry", "")),
        atr_at_entry=prefill.get("atr_15m") or prefill.get("atr_5m"),
        fvg_involved=prefill.get("fvg_involved"),
        volume_node_involved=prefill.get("volume_node_involved"),
        session_level_involved=prefill.get("session_level_involved"),
        planned_vs_impulsive="planned",
        reason_for_entry=reason_for_entry,
    )
    compute_futures_metrics(trade)

    _clear()
    print("\n  PLANNED TRADE PREVIEW")
    print("  " + "─" * 42)
    print(f"  Ticker           : {trade.ticker}")
    print(f"  Direction        : {trade.direction.upper()}")
    print(f"  Entry / Stop     : {trade.entry_price} / {trade.stop_price}")
    print(f"  Planned Target   : {trade.planned_target if trade.planned_target is not None else 'N/A'}")
    print(f"  Analyzer Idea    : {trade.analyzer_idea or 'N/A'}")
    print(f"  Bias / Confidence: {trade.bias_at_entry or 'N/A'} / {trade.confidence_at_entry or 'N/A'}")
    print(f"  Confluence Score : {trade.confluence_score if trade.confluence_score is not None else 'N/A'}")

    confirm = _p("\n  Save this planned trade? (y/n)", "y").lower()
    if confirm == "y":
        trade_id = _DB.save_futures_trade(trade)
        print(f"\n  Saved planned trade — ID: {trade_id}")
    else:
        print("\n  Planned trade not saved.")
    input("\n  Press Enter to continue...")


# ---------------------------------------------------------------------------
# Log options trade
# ---------------------------------------------------------------------------

def log_options_trade() -> None:
    _clear()
    print("\n  LOG OPTIONS TRADE")
    print("  " + "─" * 42)

    underlying = _p("Underlying ticker (e.g. SPY, AAPL)").upper()
    if not underlying:
        return

    option_type = ""
    while option_type not in ("call", "put"):
        option_type = _p("Type (call/put)", "call").lower()

    strike      = _pf("Strike price", required=True)
    expiration  = _p("Expiration (YYYY-MM-DD)")
    entry_prem  = _pf("Entry premium (per share)", required=True)
    exit_prem   = _pf("Exit premium (per share)", required=True)
    quantity    = int(_pf("Contracts (negative = short)", default=1.0) or 1)

    entry_time = _pick_datetime("Entry time")
    exit_time  = _pick_datetime("Exit time", default=entry_time)

    stop_price = _pf("Stop premium (blank to skip)")

    setup_idx  = _pick(OPTIONS_STRATEGIES, "Strategy type")
    setup_type = OPTIONS_STRATEGIES[setup_idx]

    notes = _p("Notes (optional)")
    mistake_tags = _pick_mistake_tags()
    did_follow_plan = _pick_bool("Did you follow the plan?")

    print("\n  Greeks at entry (optional — press Enter to skip):")
    iv    = _pf("IV at entry (as decimal, e.g. 0.35)")
    delta = _pf("Delta")
    gamma = _pf("Gamma")
    theta = _pf("Theta")
    vega  = _pf("Vega")

    screenshot = _p("Screenshot path (optional)")

    trade = OptionsTrade(
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiration=expiration,
        entry_premium=entry_prem,
        exit_premium=exit_prem,
        quantity=quantity,
        entry_time=entry_time,
        exit_time=exit_time,
        setup_type=setup_type,
        stop_price=stop_price,
        notes=notes,
        screenshot_path=screenshot or None,
        mistake_tags=mistake_tags,
        did_follow_plan=did_follow_plan,
        iv_at_entry=iv,
        delta_at_entry=delta,
        gamma_at_entry=gamma,
        theta_at_entry=theta,
        vega_at_entry=vega,
    )
    compute_options_metrics(trade)

    # Preview
    _clear()
    print("\n  TRADE PREVIEW")
    print("  " + "─" * 42)
    print(f"  Underlying  : {underlying}")
    print(f"  Contract    : {option_type.upper()} {strike} exp {expiration}")
    print(f"  Qty         : {quantity}")
    print(f"  Entry/Exit  : {entry_prem} → {exit_prem}")
    print(f"  P&L ($)     : {trade.pnl_dollars:+.2f}")
    print(f"  Return %    : {trade.return_pct:+.1f}%")
    if trade.r_multiple is not None:
        print(f"  R multiple  : {trade.r_multiple:.2f}R")
    print(f"  Strategy    : {setup_type}")

    confirm = _p("\n  Save this trade? (y/n)", "y").lower()
    if confirm == "y":
        trade_id = _DB.save_options_trade(trade)
        print(f"\n  Saved — trade ID: {trade_id}")
    else:
        print("\n  Trade not saved.")

    input("\n  Press Enter to continue...")


# ---------------------------------------------------------------------------
# View trades
# ---------------------------------------------------------------------------

def view_trades() -> None:
    _clear()
    print("\n  VIEW TRADES")
    print("  " + "─" * 42)
    print("    1. All trades")
    print("    2. Futures trades only")
    print("    3. Options trades only")
    print("    4. Filter by ticker")
    print("    5. Filter by date range")
    print("    6. Futures by state")
    print("    0. Back")
    choice = input("  Choice: ").strip()

    filters: dict = {}
    trades: list[dict] = []

    if choice == "1":
        trades = _DB.get_all_trades()
    elif choice == "2":
        trades = [{"asset_class": "futures", **t} for t in _DB.get_futures_trades()]
    elif choice == "3":
        trades = [{"asset_class": "options", **t} for t in _DB.get_options_trades()]
    elif choice == "4":
        ticker = _p("Ticker").upper()
        ft = [{"asset_class": "futures", **t} for t in _DB.get_futures_trades(ticker=ticker)]
        ot = [{"asset_class": "options", **t} for t in _DB.get_options_trades(ticker=ticker)]
        trades = ft + ot
    elif choice == "5":
        date_from = _p("From date (YYYY-MM-DD)")
        date_to   = _p("To date   (YYYY-MM-DD)")
        trades = _DB.get_all_trades(date_from=date_from, date_to=date_to)
    elif choice == "6":
        state = _p("State (planned/open/closed/cancelled)", "planned").lower()
        trades = [{"asset_class": "futures", **t} for t in _DB.get_futures_trades(state=state)]
    else:
        return

    _clear()
    render_trades_table(trades)
    input("\n  Press Enter to continue...")


# ---------------------------------------------------------------------------
# Edit / Delete
# ---------------------------------------------------------------------------

def edit_trade() -> None:
    _clear()
    print("\n  EDIT TRADE")
    print("  " + "─" * 42)

    ac = ""
    while ac not in ("futures", "options"):
        ac = _p("Asset class (futures/options)", "futures").lower()

    try:
        trade_id = int(_p("Trade ID"))
    except ValueError:
        print("  Invalid ID.")
        input("  Press Enter to continue...")
        return

    trade = _DB.get_trade_by_id(trade_id, ac)
    if trade is None:
        print(f"  Trade #{trade_id} not found in {ac} trades.")
        input("  Press Enter to continue...")
        return

    print(f"\n  Current values for trade #{trade_id}:")
    for k, v in trade.items():
        print(f"    {k}: {v}")

    field = _p("\n  Field to update (e.g. notes, exit_price)").strip()
    if not field or field not in trade:
        print("  Invalid field.")
        input("  Press Enter to continue...")
        return

    new_val_raw = _p(f"  New value for '{field}'")
    # Try to preserve type
    old_val = trade[field]
    try:
        if isinstance(old_val, float):
            new_val: Any = float(new_val_raw)
        elif isinstance(old_val, int) and old_val is not None:
            new_val = int(new_val_raw)
        else:
            new_val = new_val_raw
    except Exception:
        new_val = new_val_raw

    if ac == "futures":
        ok = _DB.update_futures_trade(trade_id, {field: new_val})
    else:
        ok = _DB.update_options_trade(trade_id, {field: new_val})

    print(f"\n  {'Updated.' if ok else 'Update failed.'}")
    input("  Press Enter to continue...")


def delete_trade() -> None:
    _clear()
    print("\n  DELETE TRADE")
    print("  " + "─" * 42)

    ac = ""
    while ac not in ("futures", "options"):
        ac = _p("Asset class (futures/options)", "futures").lower()

    try:
        trade_id = int(_p("Trade ID"))
    except ValueError:
        print("  Invalid ID.")
        input("  Press Enter to continue...")
        return

    trade = _DB.get_trade_by_id(trade_id, ac)
    if trade is None:
        print(f"  Trade #{trade_id} not found.")
        input("  Press Enter to continue...")
        return

    print(f"\n  Trade #{trade_id}: {trade.get('ticker') or trade.get('underlying')} "
          f"{trade.get('direction','').upper()}  "
          f"P&L: {trade.get('pnl_dollars') or trade.get('pnl_points')}")
    confirm = _p("  Delete this trade? (yes/no)", "no").lower()
    if confirm == "yes":
        ok = _DB.delete_trade(trade_id, ac)
        print(f"\n  {'Deleted.' if ok else 'Delete failed.'}")
    else:
        print("\n  Cancelled.")
    input("  Press Enter to continue...")


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

def export_trades() -> None:
    _clear()
    print("\n  EXPORT TRADES TO CSV")
    print("  " + "─" * 42)
    ac = _p("Asset class (futures / options / all)", "all").lower()
    path = _p("Output file path", "trades_export.csv")
    n = _DB.export_csv(path, asset_class=ac)
    print(f"\n  Exported {n} trades to {path}")
    input("  Press Enter to continue...")


def import_trades() -> None:
    _clear()
    print("\n  IMPORT FUTURES TRADES FROM CSV")
    print("  " + "─" * 42)
    path = _p("CSV file path")
    if not path:
        return
    n, errors = _DB.import_futures_csv(path)
    print(f"\n  Imported {n} trades.")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
    input("  Press Enter to continue...")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def run_dashboard() -> None:
    while True:
        _clear()
        print("\n  PERFORMANCE DASHBOARD")
        print("  " + "═" * 42)
        print("    1. Overall dashboard")
        print("    2. Futures trades only")
        print("    3. Options trades only")
        print("    4. By ticker")
        print("    5. By setup type")
        print("    6. By date range")
        print("    7. Daily review")
        print("    0. Back")
        print("  " + "─" * 42)
        choice = input("  Choice: ").strip()

        if choice == "0":
            return

        trades: list[dict] = []

        if choice == "1":
            trades = _DB.get_all_trades()
        elif choice == "2":
            trades = [{"asset_class": "futures", **t} for t in _DB.get_futures_trades()]
        elif choice == "3":
            trades = [{"asset_class": "options", **t} for t in _DB.get_options_trades()]
        elif choice == "4":
            ticker = _p("Ticker").upper()
            ft = [{"asset_class": "futures", **t} for t in _DB.get_futures_trades(ticker=ticker)]
            ot = [{"asset_class": "options", **t} for t in _DB.get_options_trades(ticker=ticker)]
            trades = ft + ot
        elif choice == "5":
            trades = _DB.get_all_trades()
        elif choice == "6":
            date_from = _p("From date (YYYY-MM-DD)")
            date_to   = _p("To date   (YYYY-MM-DD)")
            trades = _DB.get_all_trades(date_from=date_from, date_to=date_to)
        elif choice == "7":
            daily_reviews = _DB.get_daily_reviews()
            _clear()
            if not daily_reviews:
                print("\n  No daily reviews saved yet.")
            else:
                r = daily_reviews[0]
                print(f"\n  DAILY REVIEW — {r['review_date']}")
                print("  " + "─" * 42)
                print(f"  Market notes        : {r.get('market_notes', '') or 'N/A'}")
                print(f"  Psychological notes : {r.get('psychological_notes', '') or 'N/A'}")
                print(f"  Rule violations     : {r.get('rule_violations', '') or 'N/A'}")
                print(f"  Best / Worst setup  : {r.get('best_setup', '') or 'N/A'} / {r.get('worst_setup', '') or 'N/A'}")
                print(f"  Improve tomorrow    : {r.get('what_to_improve', '') or 'N/A'}")
                print(f"  Mistake tags        : {', '.join(r.get('mistake_tags', [])) or 'None'}")
            input("\n  Press Enter to continue...")
            continue
        else:
            continue

        if not trades:
            _clear()
            print("\n  No trades found for that filter.")
            input("  Press Enter to continue...")
            continue

        metrics = compute_metrics(trades)
        _clear()

        if choice == "5":
            render_breakdown(metrics.by_setup, "PERFORMANCE BY SETUP TYPE")
        else:
            render_full_dashboard(metrics)

        input("\n  Press Enter to continue...")


def write_daily_review() -> None:
    _clear()
    print("\n  DAILY REVIEW")
    print("  " + "─" * 42)
    review_date = _p("Review date (YYYY-MM-DD)", datetime.now().strftime("%Y-%m-%d"))
    market_notes = _p("Daily market notes")
    psychological_notes = _p("Psychological notes")
    rule_violations = _p("Rule violations")
    what_to_improve = _p("What to improve tomorrow")
    best_setup = _p("Best setup")
    worst_setup = _p("Worst setup")
    mistake_tags = _pick_mistake_tags()
    screenshots = _p("Screenshot paths (comma-separated)")
    notes = _p("Extra notes")

    review = DailyReview(
        review_date=review_date,
        market_notes=market_notes,
        psychological_notes=psychological_notes,
        rule_violations=rule_violations,
        what_to_improve=what_to_improve,
        best_setup=best_setup,
        worst_setup=worst_setup,
        mistake_tags=mistake_tags,
        screenshot_paths=[s.strip() for s in screenshots.split(",") if s.strip()],
        notes=notes,
    )
    _DB.save_daily_review(review)
    print("\n  Daily review saved.")
    input("\n  Press Enter to continue...")


# ---------------------------------------------------------------------------
# Journal main menu
# ---------------------------------------------------------------------------

def run_journal() -> None:
    while True:
        _clear()
        print("\n  TRADE JOURNAL")
        print("  " + "═" * 42)
        print("    1. Log futures trade")
        print("    2. Plan futures trade")
        print("    3. Log options trade")
        print("    4. View trades")
        print("    5. Edit trade")
        print("    6. Delete trade")
        print("    7. Export trades to CSV")
        print("    8. Import futures trades from CSV")
        print("    9. Write daily review")
        print("    0. Back to main menu")
        print("  " + "─" * 42)
        choice = input("  Choice: ").strip()

        if choice == "1":
            log_futures_trade()
        elif choice == "2":
            plan_futures_trade()
        elif choice == "3":
            log_options_trade()
        elif choice == "4":
            view_trades()
        elif choice == "5":
            edit_trade()
        elif choice == "6":
            delete_trade()
        elif choice == "7":
            export_trades()
        elif choice == "8":
            import_trades()
        elif choice == "9":
            write_daily_review()
        elif choice == "0":
            return
        else:
            print("  Invalid choice.")
            input("  Press Enter to continue...")


def run_journal_menu() -> None:
    run_journal()
