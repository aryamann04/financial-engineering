from __future__ import annotations

import csv
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from journal.models import DailyReview, FuturesTrade, OptionsTrade

_DEFAULT_DB = Path.home() / ".financial-engineering" / "trades.db"

_CREATE_FUTURES = """
CREATE TABLE IF NOT EXISTS futures_trades (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                 TEXT    NOT NULL,
    direction              TEXT    NOT NULL,
    entry_time             TEXT    NOT NULL,
    exit_time              TEXT    NOT NULL,
    entry_price            REAL    NOT NULL,
    exit_price             REAL    NOT NULL,
    stop_price             REAL    NOT NULL,
    setup_type             TEXT    NOT NULL,
    timeframe              TEXT    NOT NULL,
    quantity               REAL    NOT NULL DEFAULT 1,
    planned_target         REAL,
    fees                   REAL    NOT NULL DEFAULT 0,
    notes                  TEXT    NOT NULL DEFAULT '',
    screenshot_path        TEXT,
    mistake_tags           TEXT    NOT NULL DEFAULT '[]',
    did_follow_plan        INTEGER,
    contract_multiplier    REAL,
    atr_5m                 REAL,
    atr_15m                REAL,
    pnl_points             REAL    NOT NULL DEFAULT 0,
    pnl_dollars            REAL,
    r_multiple             REAL,
    holding_period_minutes REAL    NOT NULL DEFAULT 0,
    session_bucket         TEXT    NOT NULL DEFAULT '',
    time_of_day_bucket     TEXT    NOT NULL DEFAULT '',
    reached_1x_atr         INTEGER,
    reached_2x_atr         INTEGER,
    reached_3x_atr         INTEGER,
    state                  TEXT    NOT NULL DEFAULT 'closed',
    analyzer_idea          TEXT    NOT NULL DEFAULT '',
    confluence_score       INTEGER,
    bias_at_entry          TEXT    NOT NULL DEFAULT '',
    confidence_at_entry    TEXT    NOT NULL DEFAULT '',
    atr_at_entry           REAL,
    fvg_involved           INTEGER,
    volume_node_involved   INTEGER,
    session_level_involved INTEGER,
    planned_vs_impulsive   TEXT    NOT NULL DEFAULT '',
    reason_for_entry       TEXT    NOT NULL DEFAULT '',
    created_at             TEXT    NOT NULL
);
"""

_CREATE_OPTIONS = """
CREATE TABLE IF NOT EXISTS options_trades (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    underlying             TEXT    NOT NULL,
    option_type            TEXT    NOT NULL,
    strike                 REAL    NOT NULL,
    expiration             TEXT    NOT NULL,
    entry_premium          REAL    NOT NULL,
    exit_premium           REAL    NOT NULL,
    quantity               INTEGER NOT NULL,
    entry_time             TEXT    NOT NULL,
    exit_time              TEXT    NOT NULL,
    setup_type             TEXT    NOT NULL,
    stop_price             REAL,
    notes                  TEXT    NOT NULL DEFAULT '',
    screenshot_path        TEXT,
    mistake_tags           TEXT    NOT NULL DEFAULT '[]',
    did_follow_plan        INTEGER,
    iv_at_entry            REAL,
    delta_at_entry         REAL,
    gamma_at_entry         REAL,
    theta_at_entry         REAL,
    vega_at_entry          REAL,
    pnl_dollars            REAL    NOT NULL DEFAULT 0,
    return_pct             REAL    NOT NULL DEFAULT 0,
    r_multiple             REAL,
    holding_period_minutes REAL    NOT NULL DEFAULT 0,
    underlying_change_pct  REAL,
    created_at             TEXT    NOT NULL
);
"""

_CREATE_DAILY_REVIEWS = """
CREATE TABLE IF NOT EXISTS daily_reviews (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    review_date          TEXT    NOT NULL UNIQUE,
    market_notes         TEXT    NOT NULL DEFAULT '',
    psychological_notes  TEXT    NOT NULL DEFAULT '',
    rule_violations      TEXT    NOT NULL DEFAULT '',
    what_to_improve      TEXT    NOT NULL DEFAULT '',
    best_setup           TEXT    NOT NULL DEFAULT '',
    worst_setup          TEXT    NOT NULL DEFAULT '',
    mistake_tags         TEXT    NOT NULL DEFAULT '[]',
    screenshot_paths     TEXT    NOT NULL DEFAULT '[]',
    notes                TEXT    NOT NULL DEFAULT '',
    created_at           TEXT    NOT NULL
);
"""


class TradeDatabase:
    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.execute(_CREATE_FUTURES)
            con.execute(_CREATE_OPTIONS)
            con.execute(_CREATE_DAILY_REVIEWS)
            _ensure_column(con, "futures_trades", "analyzer_idea", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(con, "futures_trades", "confluence_score", "INTEGER")
            _ensure_column(con, "futures_trades", "bias_at_entry", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(con, "futures_trades", "confidence_at_entry", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(con, "futures_trades", "atr_at_entry", "REAL")
            _ensure_column(con, "futures_trades", "fvg_involved", "INTEGER")
            _ensure_column(con, "futures_trades", "volume_node_involved", "INTEGER")
            _ensure_column(con, "futures_trades", "session_level_involved", "INTEGER")
            _ensure_column(con, "futures_trades", "planned_vs_impulsive", "TEXT NOT NULL DEFAULT ''")
            _ensure_column(con, "futures_trades", "reason_for_entry", "TEXT NOT NULL DEFAULT ''")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_futures_trade(self, trade: FuturesTrade) -> int:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as con:
            cur = con.execute(
                """INSERT INTO futures_trades
                (ticker, direction, entry_time, exit_time, entry_price, exit_price,
                 stop_price, setup_type, timeframe, quantity, planned_target, fees,
                 notes, screenshot_path, mistake_tags, did_follow_plan,
                 contract_multiplier, atr_5m, atr_15m, pnl_points, pnl_dollars,
                 r_multiple, holding_period_minutes, session_bucket, time_of_day_bucket,
                 reached_1x_atr, reached_2x_atr, reached_3x_atr, state,
                 analyzer_idea, confluence_score, bias_at_entry, confidence_at_entry,
                 atr_at_entry, fvg_involved, volume_node_involved, session_level_involved,
                 planned_vs_impulsive, reason_for_entry, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade.ticker, trade.direction, trade.entry_time, trade.exit_time,
                    trade.entry_price, trade.exit_price, trade.stop_price,
                    trade.setup_type, trade.timeframe, trade.quantity,
                    trade.planned_target, trade.fees, trade.notes,
                    trade.screenshot_path, json.dumps(trade.mistake_tags),
                    _bool_to_int(trade.did_follow_plan),
                    trade.contract_multiplier, trade.atr_5m, trade.atr_15m,
                    trade.pnl_points, trade.pnl_dollars, trade.r_multiple,
                    trade.holding_period_minutes, trade.session_bucket,
                    trade.time_of_day_bucket,
                    _bool_to_int(trade.reached_1x_atr),
                    _bool_to_int(trade.reached_2x_atr),
                    _bool_to_int(trade.reached_3x_atr),
                    getattr(trade, "state", "closed"),
                    getattr(trade, "analyzer_idea", ""),
                    getattr(trade, "confluence_score", None),
                    getattr(trade, "bias_at_entry", ""),
                    getattr(trade, "confidence_at_entry", ""),
                    getattr(trade, "atr_at_entry", None),
                    _bool_to_int(getattr(trade, "fvg_involved", None)),
                    _bool_to_int(getattr(trade, "volume_node_involved", None)),
                    _bool_to_int(getattr(trade, "session_level_involved", None)),
                    getattr(trade, "planned_vs_impulsive", ""),
                    getattr(trade, "reason_for_entry", ""),
                    now,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def save_options_trade(self, trade: OptionsTrade) -> int:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as con:
            cur = con.execute(
                """INSERT INTO options_trades
                (underlying, option_type, strike, expiration, entry_premium, exit_premium,
                 quantity, entry_time, exit_time, setup_type, stop_price, notes,
                 screenshot_path, mistake_tags, did_follow_plan,
                 iv_at_entry, delta_at_entry, gamma_at_entry, theta_at_entry, vega_at_entry,
                 pnl_dollars, return_pct, r_multiple, holding_period_minutes,
                 underlying_change_pct, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade.underlying, trade.option_type, trade.strike,
                    trade.expiration, trade.entry_premium, trade.exit_premium,
                    trade.quantity, trade.entry_time, trade.exit_time,
                    trade.setup_type, trade.stop_price, trade.notes,
                    trade.screenshot_path, json.dumps(trade.mistake_tags),
                    _bool_to_int(trade.did_follow_plan),
                    trade.iv_at_entry, trade.delta_at_entry, trade.gamma_at_entry,
                    trade.theta_at_entry, trade.vega_at_entry,
                    trade.pnl_dollars, trade.return_pct, trade.r_multiple,
                    trade.holding_period_minutes, trade.underlying_change_pct,
                    now,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get_futures_trades(self, **filters: Any) -> list[dict]:
        where, params = _build_where(filters, "futures")
        with self._conn() as con:
            rows = con.execute(
                f"SELECT * FROM futures_trades{where} ORDER BY entry_time DESC", params
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_options_trades(self, **filters: Any) -> list[dict]:
        where, params = _build_where(filters, "options")
        with self._conn() as con:
            rows = con.execute(
                f"SELECT * FROM options_trades{where} ORDER BY entry_time DESC", params
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_all_futures(self, **filters: Any) -> list[dict]:
        """Alias for get_futures_trades for ergonomic use in TUI/dashboard."""
        return self.get_futures_trades(**filters)

    def get_all_trades(self, **filters: Any) -> list[dict]:
        """Return futures and options trades merged, each with an 'asset_class' field."""
        ft = [{"asset_class": "futures", **t} for t in self.get_futures_trades(**filters)]
        ot = [{"asset_class": "options", **t} for t in self.get_options_trades(**filters)]
        combined = ft + ot
        combined.sort(key=lambda r: r.get("entry_time", ""), reverse=True)
        return combined

    def get_trade_by_id(self, trade_id: int, asset_class: str) -> dict | None:
        table = "futures_trades" if asset_class == "futures" else "options_trades"
        with self._conn() as con:
            row = con.execute(f"SELECT * FROM {table} WHERE id=?", (trade_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def get_daily_reviews(self, review_date: str | None = None) -> list[dict]:
        query = "SELECT * FROM daily_reviews"
        params: tuple[object, ...] = ()
        if review_date:
            query += " WHERE review_date=?"
            params = (review_date,)
        query += " ORDER BY review_date DESC"
        with self._conn() as con:
            rows = con.execute(query, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Update / Delete
    # ------------------------------------------------------------------

    def update_futures_trade(self, trade_id: int, updates: dict) -> bool:
        return self._update("futures_trades", trade_id, updates)

    def update_options_trade(self, trade_id: int, updates: dict) -> bool:
        return self._update("options_trades", trade_id, updates)

    def _update(self, table: str, trade_id: int, updates: dict) -> bool:
        if not updates:
            return False
        cols = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [trade_id]
        with self._conn() as con:
            cur = con.execute(f"UPDATE {table} SET {cols} WHERE id=?", vals)
            return cur.rowcount > 0

    def delete_trade(self, trade_id: int, asset_class: str) -> bool:
        table = "futures_trades" if asset_class == "futures" else "options_trades"
        with self._conn() as con:
            cur = con.execute(f"DELETE FROM {table} WHERE id=?", (trade_id,))
            return cur.rowcount > 0

    def save_daily_review(self, review: DailyReview) -> int:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._conn() as con:
            cur = con.execute(
                """INSERT INTO daily_reviews
                (review_date, market_notes, psychological_notes, rule_violations,
                 what_to_improve, best_setup, worst_setup, mistake_tags,
                 screenshot_paths, notes, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(review_date) DO UPDATE SET
                    market_notes=excluded.market_notes,
                    psychological_notes=excluded.psychological_notes,
                    rule_violations=excluded.rule_violations,
                    what_to_improve=excluded.what_to_improve,
                    best_setup=excluded.best_setup,
                    worst_setup=excluded.worst_setup,
                    mistake_tags=excluded.mistake_tags,
                    screenshot_paths=excluded.screenshot_paths,
                    notes=excluded.notes""",
                (
                    review.review_date,
                    review.market_notes,
                    review.psychological_notes,
                    review.rule_violations,
                    review.what_to_improve,
                    review.best_setup,
                    review.worst_setup,
                    json.dumps(review.mistake_tags),
                    json.dumps(review.screenshot_paths),
                    review.notes,
                    now,
                ),
            )
            if cur.lastrowid:
                return cur.lastrowid  # type: ignore[return-value]
        existing = self.get_daily_reviews(review.review_date)
        return int(existing[0]["id"]) if existing else 0

    # ------------------------------------------------------------------
    # CSV export / import
    # ------------------------------------------------------------------

    def export_csv(self, path: str, asset_class: str = "all") -> int:
        trades = self.get_all_trades() if asset_class == "all" else (
            self.get_futures_trades() if asset_class == "futures" else self.get_options_trades()
        )
        if not trades:
            return 0
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        return len(trades)

    def import_futures_csv(self, path: str) -> tuple[int, list[str]]:
        """Import futures trades from CSV. Returns (n_imported, errors)."""
        from journal.models import FuturesTrade, compute_futures_metrics
        imported = 0
        errors: list[str] = []
        with open(path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f), 1):
                try:
                    trade = FuturesTrade(
                        ticker=row["ticker"],
                        direction=row["direction"],
                        entry_time=row["entry_time"],
                        exit_time=row["exit_time"],
                        entry_price=float(row["entry_price"]),
                        exit_price=float(row["exit_price"]),
                        stop_price=float(row["stop_price"]),
                        setup_type=row.get("setup_type", "Manual/Other"),
                        timeframe=row.get("timeframe", ""),
                        quantity=float(row.get("quantity", 1)),
                    )
                    compute_futures_metrics(trade)
                    self.save_futures_trade(trade)
                    imported += 1
                except Exception as exc:
                    errors.append(f"Row {i}: {exc}")
        return imported, errors


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _bool_to_int(val: bool | None) -> int | None:
    if val is None:
        return None
    return 1 if val else 0


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("mistake_tags", "screenshot_paths"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except Exception:
                d[key] = []
    for key in (
        "did_follow_plan", "reached_1x_atr", "reached_2x_atr", "reached_3x_atr",
        "fvg_involved", "volume_node_involved", "session_level_involved",
    ):
        if key in d and d[key] is not None:
            d[key] = bool(d[key])
    return d


def _build_where(filters: dict, asset_class: str) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []

    date_from = filters.get("date_from")
    date_to   = filters.get("date_to")
    ticker    = filters.get("ticker")
    setup     = filters.get("setup_type")
    tf        = filters.get("timeframe")

    if date_from:
        clauses.append("entry_time >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("entry_time <= ?")
        params.append(date_to + " 23:59:59")
    if ticker:
        col = "ticker" if asset_class == "futures" else "underlying"
        clauses.append(f"{col} = ?")
        params.append(ticker.upper())
    if setup:
        clauses.append("setup_type = ?")
        params.append(setup)
    if tf and asset_class == "futures":
        clauses.append("timeframe = ?")
        params.append(tf)

    state = filters.get("state")
    if state and asset_class == "futures":
        clauses.append("state = ?")
        params.append(state)

    planned_vs_impulsive = filters.get("planned_vs_impulsive")
    if planned_vs_impulsive and asset_class == "futures":
        clauses.append("planned_vs_impulsive = ?")
        params.append(planned_vs_impulsive)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _ensure_column(con: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {row["name"] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
