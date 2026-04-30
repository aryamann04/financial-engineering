"""Tests for ATR, session levels, trade metrics, metrics, and storage."""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------

from futures.atr import compute_atr, compute_vwap, get_trend_alignment, resample_to_15m


def _make_df(closes: list[float], highs: list[float] | None = None, lows: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if lows is None:
        lows = [c - 1.0 for c in closes]
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame({"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": [1000] * n}, index=idx)


class TestATR:
    def test_returns_none_when_not_enough_bars(self):
        df = _make_df([100.0] * 5)
        assert compute_atr(df, period=14) is None

    def test_returns_float_with_enough_bars(self):
        df = _make_df([100.0 + i * 0.1 for i in range(20)])
        atr = compute_atr(df, period=14)
        assert atr is not None
        assert atr > 0

    def test_higher_volatility_higher_atr(self):
        low_vol = _make_df(
            [100.0] * 20,
            highs=[100.5] * 20,
            lows=[99.5] * 20,
        )
        high_vol = _make_df(
            [100.0] * 20,
            highs=[105.0] * 20,
            lows=[95.0] * 20,
        )
        atr_low  = compute_atr(low_vol,  period=14)
        atr_high = compute_atr(high_vol, period=14)
        assert atr_low is not None and atr_high is not None
        assert atr_high > atr_low

    def test_none_on_empty_df(self):
        assert compute_atr(pd.DataFrame(), period=14) is None

    def test_vwap_typical_price_weighted(self):
        df = _make_df([100.0] * 10, highs=[110.0] * 10, lows=[90.0] * 10)
        vwap = compute_vwap(df)
        assert vwap is not None
        # Typical price = (110 + 90 + 100) / 3 = 100
        assert abs(vwap - 100.0) < 1e-9

    def test_trend_alignment_bullish(self):
        # Rising price, EMA9 should cross above EMA21
        closes = [100 + i * 2 for i in range(30)]
        df = _make_df(closes)
        df15 = resample_to_15m(df)
        t5, t15, align = get_trend_alignment(df, df15)
        assert t5 == "bullish"

    def test_trend_alignment_bearish(self):
        closes = [200 - i * 2 for i in range(30)]
        df = _make_df(closes)
        df15 = resample_to_15m(df)
        t5, t15, align = get_trend_alignment(df, df15)
        assert t5 == "bearish"

    def test_resample_15m_reduces_rows(self):
        df = _make_df([100.0] * 30)
        df15 = resample_to_15m(df)
        assert len(df15) < len(df)
        assert len(df15) == len(df) // 3


# ---------------------------------------------------------------------------
# Session levels tests
# ---------------------------------------------------------------------------

from config.sessions import SESSIONS, filter_session_bars, ET, get_session_label


class TestSessionLevels:
    def _make_intraday(self) -> pd.DataFrame:
        """24 hours of 5m bars covering Asia, London, NY sessions."""
        start = pd.Timestamp("2024-01-10 18:00", tz="America/New_York")
        idx = pd.date_range(start, periods=288, freq="5min")  # 24 hours
        closes = [100.0 + i * 0.01 for i in range(288)]
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]
        return pd.DataFrame(
            {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": [100] * 288},
            index=idx,
        )

    def test_ny_session_has_bars(self):
        df = self._make_intraday()
        ref = date(2024, 1, 11)
        ny_bars = filter_session_bars(df, SESSIONS["new_york"], ref)
        assert not ny_bars.empty

    def test_asia_session_spans_midnight(self):
        df = self._make_intraday()
        ref = date(2024, 1, 11)
        asia_bars = filter_session_bars(df, SESSIONS["asia"], ref)
        assert not asia_bars.empty
        # Asia bars should include bars from prev day (Jan 10 18:00+) and Jan 11 before 03:00
        dates = set(bar.date() for bar in asia_bars.index)
        assert date(2024, 1, 10) in dates

    def test_london_session_bars_in_range(self):
        df = self._make_intraday()
        ref = date(2024, 1, 11)
        london_bars = filter_session_bars(df, SESSIONS["london"], ref)
        if not london_bars.empty:
            for ts in london_bars.index:
                h = ts.hour
                m = ts.minute
                assert (h * 60 + m) >= (3 * 60)
                assert (h * 60 + m) < (8 * 60 + 30)

    def test_session_label(self):
        assert get_session_label(20, 0) == "Asia"
        assert get_session_label(1,  0) == "Asia"
        assert get_session_label(4,  0) == "London"
        assert get_session_label(10, 0) == "New York"
        assert get_session_label(17, 0) == "Post-Market"

    def test_empty_df_returns_empty(self):
        result = filter_session_bars(pd.DataFrame(), SESSIONS["new_york"], date.today())
        assert result.empty


# ---------------------------------------------------------------------------
# Trade P&L and R multiple tests
# ---------------------------------------------------------------------------

from journal.models import FuturesTrade, OptionsTrade, compute_futures_metrics, compute_options_metrics


class TestFuturesMetrics:
    def _make_trade(self, direction="long", entry=100.0, exit_p=105.0, stop=97.0, qty=1.0, mult=5.0) -> FuturesTrade:
        return FuturesTrade(
            ticker="MES=F", direction=direction,
            entry_time="2024-01-10 09:30", exit_time="2024-01-10 10:15",
            entry_price=entry, exit_price=exit_p, stop_price=stop,
            setup_type="NY Open Range Breakout", timeframe="5m",
            quantity=qty, contract_multiplier=mult,
        )

    def test_long_profitable_pnl(self):
        t = self._make_trade(direction="long", entry=100, exit_p=105)
        compute_futures_metrics(t)
        assert t.pnl_points > 0
        assert t.pnl_dollars is not None and t.pnl_dollars > 0

    def test_short_profitable_pnl(self):
        t = self._make_trade(direction="short", entry=100, exit_p=95)
        compute_futures_metrics(t)
        assert t.pnl_points > 0

    def test_long_losing_pnl(self):
        t = self._make_trade(direction="long", entry=100, exit_p=97)
        compute_futures_metrics(t)
        assert t.pnl_points < 0

    def test_r_multiple_calculation(self):
        # Entry=100, Stop=97 → risk=3. Exit=106 → reward=6 → R=2.0
        t = self._make_trade(direction="long", entry=100, exit_p=106, stop=97)
        compute_futures_metrics(t)
        assert t.r_multiple is not None
        assert abs(t.r_multiple - 2.0) < 1e-9

    def test_r_multiple_loss(self):
        # Entry=100, Stop=97 → risk=3. Exit=98 → reward=-2 → R=-0.667
        t = self._make_trade(direction="long", entry=100, exit_p=98, stop=97)
        compute_futures_metrics(t)
        assert t.r_multiple is not None
        assert t.r_multiple < 0

    def test_holding_period_minutes(self):
        t = self._make_trade()
        compute_futures_metrics(t)
        assert abs(t.holding_period_minutes - 45.0) < 1.0

    def test_pnl_dollars_with_multiplier(self):
        # 5 points × 1 contract × 5.0 multiplier = $25
        t = self._make_trade(direction="long", entry=100, exit_p=105, stop=97, mult=5.0)
        compute_futures_metrics(t)
        assert t.pnl_dollars == pytest.approx(25.0)

    def test_atr_targets(self):
        t = self._make_trade(direction="long", entry=100, exit_p=110, stop=97)
        t.atr_15m = 5.0
        compute_futures_metrics(t)
        assert t.reached_1x_atr is True   # excursion=10 >= 1*5
        assert t.reached_2x_atr is True   # excursion=10 >= 2*5
        assert t.reached_3x_atr is False  # excursion=10 < 3*5=15

    def test_session_bucket_assigned(self):
        t = self._make_trade()
        compute_futures_metrics(t)
        assert t.session_bucket in ("Asia", "London", "New York", "Post-Market", "")


class TestOptionsMetrics:
    def _make_options_trade(self, entry=1.50, exit_p=3.00, qty=2) -> OptionsTrade:
        return OptionsTrade(
            underlying="SPY", option_type="call", strike=450.0,
            expiration="2024-02-16", entry_premium=entry, exit_premium=exit_p,
            quantity=qty, entry_time="2024-01-10 09:35", exit_time="2024-01-10 14:00",
            setup_type="Long Call",
        )

    def test_pnl_long_call_profit(self):
        t = self._make_options_trade(entry=1.50, exit_p=3.00, qty=2)
        compute_options_metrics(t)
        # (3.00 - 1.50) * 2 * 100 = $300
        assert t.pnl_dollars == pytest.approx(300.0)

    def test_pnl_long_call_loss(self):
        t = self._make_options_trade(entry=2.00, exit_p=0.50, qty=1)
        compute_options_metrics(t)
        # (0.50 - 2.00) * 1 * 100 = -$150
        assert t.pnl_dollars == pytest.approx(-150.0)

    def test_return_pct(self):
        t = self._make_options_trade(entry=1.00, exit_p=2.00)
        compute_options_metrics(t)
        assert t.return_pct == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Metrics aggregation tests
# ---------------------------------------------------------------------------

from journal.metrics import compute_metrics, _compute_drawdown


class TestMetrics:
    def _trades(self) -> list[dict]:
        return [
            {"asset_class": "futures", "ticker": "MES=F", "direction": "long",
             "pnl_dollars": 50.0, "r_multiple": 1.5, "setup_type": "NY Open Range Breakout",
             "session_bucket": "New York", "time_of_day_bucket": "Morning",
             "timeframe": "5m", "holding_period_minutes": 30,
             "reached_1x_atr": True, "reached_2x_atr": False, "reached_3x_atr": False},
            {"asset_class": "futures", "ticker": "MES=F", "direction": "short",
             "pnl_dollars": -25.0, "r_multiple": -0.8, "setup_type": "VWAP Reclaim",
             "session_bucket": "New York", "time_of_day_bucket": "Afternoon",
             "timeframe": "15m", "holding_period_minutes": 60,
             "reached_1x_atr": False, "reached_2x_atr": False, "reached_3x_atr": False},
            {"asset_class": "options", "underlying": "SPY", "direction": None,
             "pnl_dollars": 150.0, "r_multiple": 2.0, "setup_type": "Long Call",
             "session_bucket": "", "time_of_day_bucket": "",
             "timeframe": None, "holding_period_minutes": 120,
             "reached_1x_atr": None, "reached_2x_atr": None, "reached_3x_atr": None},
        ]

    def test_total_pnl(self):
        m = compute_metrics(self._trades())
        assert m.total_pnl == pytest.approx(175.0)

    def test_win_rate(self):
        m = compute_metrics(self._trades())
        assert abs(m.win_rate - (2 / 3 * 100)) < 0.01

    def test_profit_factor(self):
        m = compute_metrics(self._trades())
        # Gross profit = 200, gross loss = 25
        assert m.profit_factor == pytest.approx(8.0)

    def test_by_ticker(self):
        m = compute_metrics(self._trades())
        assert "MES=F" in m.by_ticker
        assert m.by_ticker["MES=F"]["n"] == 2

    def test_max_drawdown(self):
        pnls = [100, -50, -30, 200, -10]
        dd = _compute_drawdown(pnls)
        # peak=100, then falls to 20 → dd=80
        assert dd == pytest.approx(80.0)

    def test_empty_trades_returns_zero_metrics(self):
        m = compute_metrics([])
        assert m.total_trades == 0
        assert m.total_pnl == 0.0


# ---------------------------------------------------------------------------
# SQLite storage round-trip tests
# ---------------------------------------------------------------------------

from journal.storage import TradeDatabase


class TestStorage:
    @pytest.fixture
    def db(self, tmp_path):
        return TradeDatabase(db_path=tmp_path / "test_trades.db")

    def test_save_and_retrieve_futures_trade(self, db):
        t = FuturesTrade(
            ticker="MES=F", direction="long",
            entry_time="2024-01-10 09:30", exit_time="2024-01-10 10:00",
            entry_price=4700.0, exit_price=4720.0, stop_price=4685.0,
            setup_type="NY Open Range Breakout", timeframe="5m", quantity=2.0,
            contract_multiplier=5.0,
        )
        compute_futures_metrics(t)
        trade_id = db.save_futures_trade(t)
        assert trade_id > 0

        rows = db.get_futures_trades()
        assert len(rows) == 1
        row = rows[0]
        assert row["ticker"] == "MES=F"
        assert row["direction"] == "long"
        assert row["entry_price"] == pytest.approx(4700.0)
        assert row["pnl_points"] == pytest.approx(40.0)

    def test_save_and_retrieve_options_trade(self, db):
        t = OptionsTrade(
            underlying="SPY", option_type="call", strike=450.0,
            expiration="2024-02-16", entry_premium=2.00, exit_premium=3.50,
            quantity=1, entry_time="2024-01-10 09:40", exit_time="2024-01-10 12:00",
            setup_type="Long Call",
        )
        compute_options_metrics(t)
        trade_id = db.save_options_trade(t)
        assert trade_id > 0

        rows = db.get_options_trades()
        assert len(rows) == 1
        assert rows[0]["underlying"] == "SPY"
        assert rows[0]["pnl_dollars"] == pytest.approx(150.0)

    def test_delete_trade(self, db):
        t = FuturesTrade(
            ticker="MNQ=F", direction="short",
            entry_time="2024-01-10 10:00", exit_time="2024-01-10 10:30",
            entry_price=17000.0, exit_price=16980.0, stop_price=17020.0,
            setup_type="VWAP Reclaim", timeframe="5m", quantity=1.0,
        )
        compute_futures_metrics(t)
        tid = db.save_futures_trade(t)
        ok = db.delete_trade(tid, "futures")
        assert ok
        assert db.get_futures_trades() == []

    def test_update_trade(self, db):
        t = FuturesTrade(
            ticker="MBT=F", direction="long",
            entry_time="2024-01-10 09:30", exit_time="2024-01-10 09:50",
            entry_price=40000.0, exit_price=40200.0, stop_price=39900.0,
            setup_type="Asia High Breakout", timeframe="15m", quantity=1.0,
            notes="",
        )
        compute_futures_metrics(t)
        tid = db.save_futures_trade(t)
        ok = db.update_futures_trade(tid, {"notes": "updated note"})
        assert ok
        row = db.get_trade_by_id(tid, "futures")
        assert row["notes"] == "updated note"

    def test_get_all_trades_combined(self, db):
        ft = FuturesTrade(
            ticker="MES=F", direction="long",
            entry_time="2024-01-10 09:30", exit_time="2024-01-10 10:00",
            entry_price=4700.0, exit_price=4710.0, stop_price=4690.0,
            setup_type="NY Open Range Breakout", timeframe="5m", quantity=1.0,
        )
        compute_futures_metrics(ft)
        db.save_futures_trade(ft)

        ot = OptionsTrade(
            underlying="QQQ", option_type="put", strike=380.0,
            expiration="2024-02-16", entry_premium=1.50, exit_premium=2.50,
            quantity=1, entry_time="2024-01-10 11:00", exit_time="2024-01-10 13:00",
            setup_type="Long Put",
        )
        compute_options_metrics(ot)
        db.save_options_trade(ot)

        all_trades = db.get_all_trades()
        assert len(all_trades) == 2
        asset_classes = {t["asset_class"] for t in all_trades}
        assert asset_classes == {"futures", "options"}

    def test_export_csv(self, db, tmp_path):
        t = FuturesTrade(
            ticker="MES=F", direction="long",
            entry_time="2024-01-10 09:30", exit_time="2024-01-10 09:45",
            entry_price=4700.0, exit_price=4705.0, stop_price=4695.0,
            setup_type="Manual/Other", timeframe="5m", quantity=1.0,
        )
        compute_futures_metrics(t)
        db.save_futures_trade(t)
        out = tmp_path / "out.csv"
        n = db.export_csv(str(out), asset_class="futures")
        assert n == 1
        assert out.exists()
        import csv
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["ticker"] == "MES=F"


# ---------------------------------------------------------------------------
# Trade planner tests
# ---------------------------------------------------------------------------

from futures.planner import assess_trade


class TestPlanner:
    def test_long_basic_r_calculation(self):
        plan = assess_trade(
            symbol="MES=F", direction="long",
            entry=4700.0, stop=4685.0, target=4730.0,
            setup_type="NY Open Range Breakout", timeframe="5m",
            notes="", atr_5m=10.0, atr_15m=15.0,
        )
        assert plan.risk_points == pytest.approx(15.0)
        assert plan.reward_points == pytest.approx(30.0)
        assert plan.r_multiple == pytest.approx(2.0)

    def test_short_r_calculation(self):
        plan = assess_trade(
            symbol="MNQ=F", direction="short",
            entry=18000.0, stop=18030.0, target=17940.0,
            setup_type="VWAP Reclaim", timeframe="15m",
            notes="", atr_5m=20.0, atr_15m=30.0,
        )
        assert plan.risk_points == pytest.approx(30.0)
        assert plan.reward_points == pytest.approx(60.0)
        assert plan.r_multiple == pytest.approx(2.0)

    def test_atr_targets_long(self):
        plan = assess_trade(
            symbol="MES=F", direction="long",
            entry=4700.0, stop=4685.0, target=None,
            setup_type="Manual/Other", timeframe="5m",
            notes="", atr_5m=10.0, atr_15m=20.0,
        )
        # 15m ATR = 20, so 1x = 4720, 2x = 4740, 3x = 4760
        assert plan.target_1x == pytest.approx(4720.0)
        assert plan.target_2x == pytest.approx(4740.0)
        assert plan.target_3x == pytest.approx(4760.0)

    def test_quality_strong(self):
        plan = assess_trade(
            symbol="ES=F", direction="long",
            entry=4700.0, stop=4690.0, target=4730.0,
            setup_type="London High Breakout", timeframe="15m",
            notes="", atr_5m=5.0, atr_15m=10.0,
        )
        # Stop = 10 pts = 1x ATR (well-placed), target = 30 pts = 3x ATR, R = 3.0
        assert plan.quality in ("strong", "decent")

    def test_quality_avoid_wide_stop(self):
        plan = assess_trade(
            symbol="MES=F", direction="long",
            entry=4700.0, stop=4600.0, target=4730.0,
            setup_type="Manual/Other", timeframe="5m",
            notes="", atr_5m=5.0, atr_15m=10.0,
        )
        # Stop = 100 pts = 10x ATR → terrible
        assert plan.quality in ("weak", "avoid")
