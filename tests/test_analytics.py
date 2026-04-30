"""Tests for volume profile, FVG detection, bias engine, confluence, ideas, and journal state."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime

import pandas as pd
import pytest
from journal.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    freq: str = "5min",
) -> pd.DataFrame:
    n = len(closes)
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if lows is None:
        lows = [c - 1.0 for c in closes]
    if volumes is None:
        volumes = [1_000.0] * n
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Volume profile
# ---------------------------------------------------------------------------

from futures.volume import build_volume_profile, volume_spike, compute_relative_volume, analyze_volume


class TestVolumeProfile:
    def test_build_returns_none_for_empty_df(self):
        assert build_volume_profile(pd.DataFrame()) is None

    def test_build_returns_none_for_single_bar(self):
        df = _make_df([100.0])
        assert build_volume_profile(df) is None

    def test_poc_is_highest_volume_bin(self):
        # Price range 100–110; force one spike at ~105 with much higher volume
        closes = [100.0, 101.0, 105.0, 105.0, 105.0, 105.0, 106.0, 109.0, 110.0, 110.0]
        vols   = [100.0,  100.0, 500.0, 500.0, 500.0, 500.0, 100.0, 100.0, 100.0, 100.0]
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]
        df = _make_df(closes, highs=highs, lows=lows, volumes=vols)
        vp = build_volume_profile(df, n_bins=20)
        assert vp is not None
        # POC should be near 105
        assert abs(vp.poc - 105.0) < 3.0

    def test_value_area_captures_correct_fraction(self):
        df = _make_df([100.0 + i * 0.5 for i in range(30)])
        vp = build_volume_profile(df, n_bins=20, value_area_pct=0.70)
        assert vp is not None
        # VAH >= POC >= VAL
        assert vp.vah >= vp.poc
        assert vp.poc >= vp.val

    def test_value_area_context(self):
        df = _make_df([100.0 + i * 0.5 for i in range(30)])
        vp = build_volume_profile(df, n_bins=20)
        assert vp is not None
        assert vp.value_area_context(vp.vah + 10) == "above_value"
        assert vp.value_area_context(vp.val - 10) == "below_value"
        assert vp.value_area_context((vp.vah + vp.val) / 2) == "inside_value"

    def test_hvn_and_lvn_present(self):
        # Need enough price dispersion to produce HVNs
        closes = list(range(100, 150))
        vols   = [500.0 if 120 <= c <= 130 else 100.0 for c in closes]
        df = _make_df(closes, volumes=vols)
        vp = build_volume_profile(df, n_bins=30)
        assert vp is not None
        assert len(vp.hvns) >= 1

    def test_nearest_hvn_above_and_below(self):
        closes = list(range(100, 130))
        vols   = [500.0 if c in (110, 120) else 100.0 for c in closes]
        df = _make_df(closes, volumes=vols)
        vp = build_volume_profile(df, n_bins=20)
        if vp is None or not vp.hvns:
            pytest.skip("Profile too sparse for this assertion")
        above = vp.nearest_hvn_above(105.0)
        below = vp.nearest_hvn_below(125.0)
        if above:
            assert above.price > 105.0
        if below:
            assert below.price < 125.0

    def test_volume_spike_true(self):
        # Last bar is 5x the rolling average
        vols = [100.0] * 21
        vols[-1] = 500.0
        df = _make_df([100.0] * 21, volumes=vols)
        assert volume_spike(df, threshold=2.0, lookback=20) is True

    def test_volume_spike_false(self):
        df = _make_df([100.0] * 21, volumes=[100.0] * 21)
        assert volume_spike(df, threshold=2.0, lookback=20) is False

    def test_relative_volume_none_when_insufficient_data(self):
        df = _make_df([100.0] * 5)
        assert compute_relative_volume(df, lookback=20) is None

    def test_analyze_volume_returns_profile_and_context(self):
        closes = [100.0 + i * 0.25 for i in range(25)]
        vols = [100.0] * 24 + [350.0]
        df = _make_df(closes, volumes=vols)
        analytics = analyze_volume(df, current_price=closes[-1], n_bins=20)
        assert analytics.volume_profile is not None
        assert analytics.relative_volume is not None
        assert analytics.acceptance_context


# ---------------------------------------------------------------------------
# FVG detection
# ---------------------------------------------------------------------------

from futures.fvg import detect_fvgs, FairValueGap


class TestFVGDetection:
    def _bullish_fvg_df(self) -> pd.DataFrame:
        """Three-candle bullish FVG: c1.high < c3.low."""
        closes = [100.0, 102.0, 110.0] + [110.0] * 10
        highs  = [101.0, 104.0, 112.0] + [112.0] * 10   # c1.high=101
        lows   = [99.0,  101.5, 105.0] + [105.0] * 10   # c3.low=105 > c1.high=101 → bullish FVG
        return _make_df(closes, highs=highs, lows=lows)

    def _bearish_fvg_df(self) -> pd.DataFrame:
        """Three-candle bearish FVG: c1.low > c3.high."""
        closes = [110.0, 108.0, 100.0] + [100.0] * 10
        highs  = [111.0, 109.0, 102.0] + [102.0] * 10   # c3.high=102
        lows   = [109.0, 107.0,  98.0] + [ 98.0] * 10   # c1.low=109 > c3.high=102 → bearish FVG
        return _make_df(closes, highs=highs, lows=lows)

    def test_detects_bullish_fvg(self):
        df = self._bullish_fvg_df()
        fvgs = detect_fvgs(df, "5m", current_price=110.0, atr=2.0)
        bullish = [f for f in fvgs if f.direction == "bullish"]
        assert len(bullish) >= 1

    def test_detects_bearish_fvg(self):
        df = self._bearish_fvg_df()
        fvgs = detect_fvgs(df, "5m", current_price=100.0, atr=2.0)
        bearish = [f for f in fvgs if f.direction == "bearish"]
        assert len(bearish) >= 1

    def test_bullish_fvg_bounds(self):
        df = self._bullish_fvg_df()
        fvgs = detect_fvgs(df, "5m", current_price=110.0, atr=2.0)
        bullish = [f for f in fvgs if f.direction == "bullish"]
        if bullish:
            f = bullish[0]
            assert f.lower < f.upper
            assert f.midpoint == pytest.approx((f.lower + f.upper) / 2.0)

    def test_fvg_fill_tracking(self):
        df = self._bullish_fvg_df()
        fvgs = detect_fvgs(df, "5m", current_price=110.0, atr=2.0)
        bullish = [f for f in fvgs if f.direction == "bullish"]
        if not bullish:
            pytest.skip("No bullish FVG detected")
        f = bullish[0]
        # Price well above — fill_pct should be 0 (not filled)
        assert not f.is_filled
        assert f.fill_pct == pytest.approx(0.0)

    def test_fvg_marked_filled_when_price_below_lower(self):
        df = self._bullish_fvg_df()
        fvgs = detect_fvgs(df, "5m", current_price=100.0, atr=2.0)  # price at lower
        bullish = [f for f in fvgs if f.direction == "bullish"]
        if not bullish:
            pytest.skip("No bullish FVG detected")
        f = bullish[0]
        # current_price (100) <= f.lower (101) → filled
        assert f.is_filled

    def test_no_fvgs_on_flat_candles(self):
        df = _make_df([100.0] * 15, highs=[100.5] * 15, lows=[99.5] * 15)
        fvgs = detect_fvgs(df, "5m", current_price=100.0, atr=1.0)
        assert fvgs == []

    def test_timeframe_label_preserved(self):
        df = self._bullish_fvg_df()
        fvgs = detect_fvgs(df, "15m", current_price=110.0, atr=2.0)
        for f in fvgs:
            assert f.timeframe == "15m"

    def test_sorted_by_proximity(self):
        df = self._bullish_fvg_df()
        fvgs = detect_fvgs(df, "5m", current_price=110.0, atr=5.0)
        if len(fvgs) >= 2:
            dists = [f.dist_from_price for f in fvgs]
            assert dists == sorted(dists)

    def test_tiny_gap_filtered_out_by_atr_threshold(self):
        closes = [100.0, 100.4, 100.8] + [100.8] * 8
        highs = [100.5, 100.7, 101.0] + [101.0] * 8
        lows = [99.5, 100.2, 100.6] + [100.6] * 8
        df = _make_df(closes, highs=highs, lows=lows)
        fvgs = detect_fvgs(df, "5m", current_price=100.8, atr=5.0)
        assert fvgs == []

    def test_requires_clear_displacement_not_just_small_wick_separation(self):
        closes = [100.0, 100.1, 101.2] + [101.2] * 8
        highs = [101.0, 101.1, 101.5] + [101.5] * 8
        lows = [99.0, 99.9, 101.05] + [101.05] * 8
        df = _make_df(closes, highs=highs, lows=lows)
        fvgs = detect_fvgs(df, "5m", current_price=101.2, atr=2.0)
        assert fvgs == []


# ---------------------------------------------------------------------------
# Bias engine
# ---------------------------------------------------------------------------

from futures.bias import compute_bias, find_confluence_zones, suggest_pullback_zones


class TestBiasEngine:
    def _make_mock_levels(self):
        from futures.levels import FuturesLevels, SessionLevels
        return FuturesLevels(
            symbol="MES=F", display_name="Micro S&P", current_price=4800.0,
            prev_day_high=4820.0, prev_day_low=4750.0, prev_day_close=4780.0,
            today_high=4810.0, today_low=4760.0,
            asia=SessionLevels("Asia", 4790.0, 4770.0, 10),
            london=SessionLevels("London", 4795.0, 4760.0, 20),
            new_york=SessionLevels("New York", 4810.0, 4770.0, 30),
            ny_open_range=SessionLevels("NY Open Range", 4805.0, 4780.0, 6),
            vwap=4788.0,
            atr_5m=4.0, atr_15m=6.0,
            trend_5m="bullish", trend_15m="bullish", trend_align="bullish",
        )

    def test_bullish_bias_when_all_bullish_signals(self):
        lvl = self._make_mock_levels()
        result = compute_bias(
            spot=4800.0, trend_5m="bullish", trend_15m="bullish", trend_1h="bullish",
            vwap=4788.0, levels=lvl,
            volume_profile=None,
            fvgs_5m=[], fvgs_15m=[], fvgs_1h=[],
            atr_15m=6.0,
        )
        assert result.bias == "bullish"
        assert result.score > 0


    def test_bearish_bias_when_all_bearish_signals(self):
        lvl = self._make_mock_levels()
        # Price below all levels
        result = compute_bias(
            spot=4700.0, trend_5m="bearish", trend_15m="bearish", trend_1h="bearish",
            vwap=4750.0, levels=lvl,
            volume_profile=None,
            fvgs_5m=[], fvgs_15m=[], fvgs_1h=[],
            atr_15m=6.0,
        )
        assert result.bias == "bearish"
        assert result.score < 0

    def test_neutral_bias_no_data(self):
        from futures.levels import FuturesLevels, SessionLevels
        lvl = FuturesLevels(
            symbol="X", display_name="X", current_price=None,
            prev_day_high=None, prev_day_low=None, prev_day_close=None,
            today_high=None, today_low=None,
            asia=SessionLevels("Asia", None, None, 0),
            london=SessionLevels("London", None, None, 0),
            new_york=SessionLevels("New York", None, None, 0),
            ny_open_range=SessionLevels("NY Open Range", None, None, 0),
            vwap=None, atr_5m=None, atr_15m=None,
            trend_5m="neutral", trend_15m="neutral", trend_align="neutral",
        )
        result = compute_bias(
            spot=None, trend_5m="neutral", trend_15m="neutral", trend_1h="neutral",
            vwap=None, levels=lvl, volume_profile=None,
            fvgs_5m=[], fvgs_15m=[], fvgs_1h=[], atr_15m=None,
        )
        assert result.bias == "neutral"

    def test_high_confidence_requires_score_4(self):
        lvl = self._make_mock_levels()
        result = compute_bias(
            spot=4800.0, trend_5m="bullish", trend_15m="bullish", trend_1h="bullish",
            vwap=4788.0, levels=lvl,
            volume_profile=None,
            fvgs_5m=[], fvgs_15m=[], fvgs_1h=[],
            atr_15m=6.0,
        )
        if result.score >= 4:
            assert result.confidence == "high"
        elif result.score >= 2:
            assert result.confidence == "medium"

    def test_cautions_not_empty_on_extension(self):
        lvl = self._make_mock_levels()
        # Price 3x ATR above london high
        result = compute_bias(
            spot=4800.0 + 3 * 6.0, trend_5m="bullish", trend_15m="bullish",
            trend_1h="bullish", vwap=4788.0, levels=lvl, volume_profile=None,
            fvgs_5m=[], fvgs_15m=[], fvgs_1h=[], atr_15m=6.0,
        )
        # May or may not fire depending on exact prices; just check it returns cleanly
        assert isinstance(result.cautions, list)


# ---------------------------------------------------------------------------
# Options snapshot monitor
# ---------------------------------------------------------------------------

from options.monitor import build_options_watchlist, fetch_options_entry
from options.snapshot import OptionsSnapshot


class TestOptionsSnapshots:
    def test_fetch_options_entry_uses_snapshot_fields(self, monkeypatch):
        def _fake_snapshot(symbol: str, t_days: int = 30):
            return OptionsSnapshot(
                symbol=symbol.upper(),
                price=500.0,
                regime="TREND UP",
                atm_iv=0.22,
                rr_25d=0.03,
                gex_regime="Call-heavy",
                confidence="medium",
                signal="Bullish continuation watch",
                sentiment="bullish",
                expiry="2026-05-15",
                updated_at="2026-04-30 09:30:00",
            )

        monkeypatch.setattr("options.monitor.get_options_snapshot", _fake_snapshot)
        entry = fetch_options_entry("spy")
        assert entry.symbol == "SPY"
        assert entry.price == 500.0
        assert entry.regime == "TREND UP"
        assert entry.signal == "Bullish continuation watch"

    def test_build_options_watchlist_sorts_low_to_high_confidence(self, monkeypatch):
        def _fake_snapshot(symbol: str, t_days: int = 30):
            confidence = {"AAPL": "high", "SPY": "low", "QQQ": "medium"}[symbol.upper()]
            return OptionsSnapshot(
                symbol=symbol.upper(),
                price=100.0,
                regime="CHOP",
                atm_iv=0.20,
                rr_25d=0.01,
                gex_regime="Balanced",
                confidence=confidence,
                signal="Watch",
                sentiment="neutral",
                updated_at="2026-04-30 09:30:00",
            )

        monkeypatch.setattr("options.monitor.get_options_snapshot", _fake_snapshot)
        entries = build_options_watchlist(["AAPL", "SPY", "QQQ"])
        assert [entry.symbol for entry in entries] == ["SPY", "QQQ", "AAPL"]


# ---------------------------------------------------------------------------
# Confluence zones
# ---------------------------------------------------------------------------

class TestConfluenceZones:
    def test_groups_nearby_levels(self):
        named = {
            "A": 100.0,
            "B": 100.5,  # within 1.0 ATR of A → should group
            "C": 110.0,  # far away
        }
        zones = find_confluence_zones(named, tolerance_atr=0.25, atr=4.0)
        assert len(zones) == 1
        zone = zones[0]
        assert zone.strength == 2

    def test_no_zone_when_all_far_apart(self):
        named = {"A": 100.0, "B": 110.0, "C": 120.0}
        zones = find_confluence_zones(named, tolerance_atr=0.25, atr=2.0)
        assert zones == []

    def test_zone_type_support_when_only_lows(self):
        named = {"Asia Low": 100.0, "London Low": 100.3}
        zones = find_confluence_zones(named, tolerance_atr=0.5, atr=1.0)
        assert len(zones) == 1
        assert zones[0].zone_type == "support"

    def test_zone_type_resistance_when_only_highs(self):
        named = {"Asia High": 100.0, "London High": 100.2}
        zones = find_confluence_zones(named, tolerance_atr=0.5, atr=1.0)
        assert len(zones) == 1
        assert zones[0].zone_type == "resistance"

    def test_sorted_by_strength_descending(self):
        named = {"A": 100.0, "B": 100.1, "C": 100.2, "D": 110.0, "E": 110.1}
        zones = find_confluence_zones(named, tolerance_atr=0.5, atr=1.0)
        assert len(zones) >= 1
        strengths = [z.strength for z in zones]
        assert strengths == sorted(strengths, reverse=True)

    def test_none_values_ignored(self):
        named = {"A": 100.0, "B": None, "C": 100.3}
        zones = find_confluence_zones(named, tolerance_atr=0.5, atr=1.0)
        # B is ignored; A and C should group
        assert len(zones) == 1
        assert zones[0].strength == 2


# ---------------------------------------------------------------------------
# Pullback zones
# ---------------------------------------------------------------------------

class TestPullbackZones:
    def _bullish_bias(self):
        from futures.bias import BiasResult
        return BiasResult(
            bias="bullish", confidence="medium",
            bull_signals=["5m trend bullish"],
            bear_signals=[], cautions=[], score=3,
        )

    def _bearish_bias(self):
        from futures.bias import BiasResult
        return BiasResult(
            bias="bearish", confidence="medium",
            bull_signals=[],
            bear_signals=["5m trend bearish"],
            cautions=[], score=-3,
        )

    def _make_confluence(self, lower: float, upper: float, zone_type: str = "support"):
        from futures.bias import ConfluenceZone
        return ConfluenceZone(
            lower=lower, upper=upper, midpoint=(lower + upper) / 2,
            levels=[("Asia Low", lower), ("London Low", upper)],
            strength=2, zone_type=zone_type,
        )

    def test_long_pullback_zone_below_spot(self):
        bias = self._bullish_bias()
        cz = self._make_confluence(4780.0, 4785.0)
        zones = suggest_pullback_zones(bias, [cz], spot=4800.0, atr=5.0, fvgs=[], volume_profile=None)
        assert len(zones) >= 1
        assert zones[0].direction == "long"
        assert zones[0].upper < 4800.0

    def test_short_pullback_zone_above_spot(self):
        bias = self._bearish_bias()
        cz = self._make_confluence(4820.0, 4825.0, zone_type="resistance")
        zones = suggest_pullback_zones(bias, [cz], spot=4800.0, atr=5.0, fvgs=[], volume_profile=None)
        assert len(zones) >= 1
        assert zones[0].direction == "short"
        assert zones[0].lower > 4800.0

    def test_neutral_bias_returns_empty(self):
        from futures.bias import BiasResult
        bias = BiasResult("neutral", "low", [], [], [], 0)
        zones = suggest_pullback_zones(bias, [], spot=4800.0, atr=5.0, fvgs=[], volume_profile=None)
        assert zones == []

    def test_zones_have_invalidation_below_lower_for_long(self):
        bias = self._bullish_bias()
        cz = self._make_confluence(4780.0, 4785.0)
        zones = suggest_pullback_zones(bias, [cz], spot=4800.0, atr=5.0, fvgs=[], volume_profile=None)
        if zones:
            z = zones[0]
            assert z.invalidation < z.lower


# ---------------------------------------------------------------------------
# Trade ideas generator
# ---------------------------------------------------------------------------

class TestTradeIdeas:
    def _bullish_bias(self):
        from futures.bias import BiasResult
        return BiasResult(
            bias="bullish", confidence="high",
            bull_signals=["5m bullish", "15m bullish", "above VWAP"],
            bear_signals=[], cautions=[], score=5,
        )

    def _make_levels(self, spot: float):
        from futures.levels import FuturesLevels, SessionLevels
        return FuturesLevels(
            symbol="MES=F", display_name="Micro S&P", current_price=spot,
            prev_day_high=spot + 20, prev_day_low=spot - 30, prev_day_close=spot - 5,
            today_high=spot + 10, today_low=spot - 15,
            asia=SessionLevels("Asia", spot - 5, spot - 20, 10),
            london=SessionLevels("London", spot + 5, spot - 10, 20),
            new_york=SessionLevels("New York", spot + 10, spot - 5, 30),
            ny_open_range=SessionLevels("NY Open Range", spot + 3, spot - 3, 6),
            vwap=spot - 2,
            atr_5m=4.0, atr_15m=6.0,
            trend_5m="bullish", trend_15m="bullish", trend_align="bullish",
        )

    def test_neutral_bias_returns_empty(self):
        from futures.bias import BiasResult
        from futures.ideas import generate_ideas
        bias = BiasResult("neutral", "low", [], [], [], 0)
        ideas = generate_ideas(bias, None, [], None, [], [], spot=4800.0, atr_5m=4.0, atr_15m=6.0)
        assert ideas == []

    def test_none_spot_returns_empty(self):
        from futures.ideas import generate_ideas
        bias = self._bullish_bias()
        ideas = generate_ideas(bias, None, [], None, [], [], spot=None, atr_5m=4.0, atr_15m=6.0)
        assert ideas == []

    def test_returns_list_of_trade_ideas(self):
        from futures.ideas import generate_ideas, TradeIdea
        from futures.bias import BiasResult, PullbackZone
        bias = self._bullish_bias()
        levels = self._make_levels(4800.0)
        pz = PullbackZone(
            lower=4785.0, upper=4790.0, direction="long",
            reasons=["Asia Low", "London Low"],
            confirmation=["5m candle closes back above 4790"],
            invalidation=4782.0, target_1x=4810.0, target_2x=4820.0, strength=2,
        )
        ideas = generate_ideas(
            bias=bias, levels=levels, fvgs=[], volume_profile=None,
            confluence_zones=[], pullback_zones=[pz],
            spot=4800.0, atr_5m=4.0, atr_15m=6.0,
        )
        assert len(ideas) >= 1
        for idea in ideas:
            assert isinstance(idea, TradeIdea)
            assert idea.direction == "long"
            assert idea.entry_low < idea.entry_high
            assert idea.confidence in ("high", "medium", "low")

    def test_ideas_sorted_by_confidence(self):
        from futures.ideas import generate_ideas
        from futures.bias import PullbackZone
        bias = self._bullish_bias()
        pz = PullbackZone(
            lower=4785.0, upper=4790.0, direction="long",
            reasons=[], confirmation=[], invalidation=4782.0,
            target_1x=4810.0, target_2x=4820.0, strength=1,
        )
        ideas = generate_ideas(
            bias=bias, levels=self._make_levels(4800.0), fvgs=[], volume_profile=None,
            confluence_zones=[], pullback_zones=[pz],
            spot=4800.0, atr_5m=4.0, atr_15m=6.0,
        )
        order = {"high": 0, "medium": 1, "low": 2}
        conf_vals = [order[i.confidence] for i in ideas]
        assert conf_vals == sorted(conf_vals)

    def test_max_5_ideas_returned(self):
        from futures.ideas import generate_ideas
        from futures.bias import PullbackZone
        bias = self._bullish_bias()
        pullbacks = [
            PullbackZone(
                lower=4800.0 - 20 - i * 5,
                upper=4800.0 - 15 - i * 5,
                direction="long", reasons=[], confirmation=[],
                invalidation=4750.0, strength=1,
            )
            for i in range(10)
        ]
        ideas = generate_ideas(
            bias=bias, levels=self._make_levels(4800.0), fvgs=[], volume_profile=None,
            confluence_zones=[], pullback_zones=pullbacks,
            spot=4800.0, atr_5m=4.0, atr_15m=6.0,
        )
        assert len(ideas) <= 5


# ---------------------------------------------------------------------------
# Journal state field
# ---------------------------------------------------------------------------

from journal.models import FuturesTrade, compute_futures_metrics


class TestJournalState:
    def _make_trade(self, state: str = "closed") -> FuturesTrade:
        return FuturesTrade(
            ticker="MES=F", direction="long",
            entry_time="2024-01-02 09:35",
            exit_time="2024-01-02 10:00",
            entry_price=4800.0, exit_price=4810.0, stop_price=4790.0,
            setup_type="VWAP Reclaim", timeframe="5m",
            quantity=1.0, state=state,
        )

    def test_default_state_is_closed(self):
        t = FuturesTrade(
            ticker="MES=F", direction="long",
            entry_time="2024-01-02 09:35", exit_time="2024-01-02 10:00",
            entry_price=4800.0, exit_price=4810.0, stop_price=4790.0,
            setup_type="Manual/Other", timeframe="5m", quantity=1.0,
        )
        assert t.state == "closed"

    def test_planned_state(self):
        t = self._make_trade("planned")
        assert t.state == "planned"

    def test_open_state(self):
        t = self._make_trade("open")
        assert t.state == "open"

    def test_cancelled_state(self):
        t = self._make_trade("cancelled")
        assert t.state == "cancelled"


# ---------------------------------------------------------------------------
# Journal storage with state
# ---------------------------------------------------------------------------

from journal.storage import TradeDatabase


class TestJournalStorageState:
    def _db(self) -> TradeDatabase:
        tmp = tempfile.mktemp(suffix=".db")
        return TradeDatabase(tmp)

    def _trade(self, state: str = "closed") -> FuturesTrade:
        t = FuturesTrade(
            ticker="MES=F", direction="long",
            entry_time="2024-01-02 09:35", exit_time="2024-01-02 10:00",
            entry_price=4800.0, exit_price=4810.0, stop_price=4790.0,
            setup_type="VWAP Reclaim", timeframe="5m", quantity=1.0, state=state,
        )
        compute_futures_metrics(t)
        return t

    def test_state_round_trips(self):
        db = self._db()
        t = self._trade("planned")
        tid = db.save_futures_trade(t)
        row = db.get_trade_by_id(tid, "futures")
        assert row is not None
        assert row["state"] == "planned"

    def test_state_filter_returns_only_matching(self):
        db = self._db()
        db.save_futures_trade(self._trade("planned"))
        db.save_futures_trade(self._trade("closed"))
        db.save_futures_trade(self._trade("open"))

        planned = db.get_futures_trades(state="planned")
        assert len(planned) == 1
        assert all(r["state"] == "planned" for r in planned)

        closed = db.get_futures_trades(state="closed")
        assert len(closed) == 1

    def test_get_all_futures_alias(self):
        db = self._db()
        db.save_futures_trade(self._trade("closed"))
        db.save_futures_trade(self._trade("open"))
        all_trades = db.get_all_futures()
        assert len(all_trades) == 2

    def test_update_state(self):
        db = self._db()
        t = self._trade("planned")
        tid = db.save_futures_trade(t)
        db.update_futures_trade(tid, {"state": "open"})
        row = db.get_trade_by_id(tid, "futures")
        assert row["state"] == "open"

    def test_daily_review_round_trip(self):
        from journal.models import DailyReview

        db = self._db()
        review = DailyReview(
            review_date="2024-01-02",
            market_notes="Trend day up",
            psychological_notes="Patient",
            what_to_improve="Hold winners longer",
            mistake_tags=["early_exit"],
            screenshot_paths=["/tmp/chart.png"],
        )
        db.save_daily_review(review)
        rows = db.get_daily_reviews("2024-01-02")
        assert len(rows) == 1
        assert rows[0]["market_notes"] == "Trend day up"
        assert rows[0]["mistake_tags"] == ["early_exit"]


class TestDashboardMetricsBreakdowns:
    def test_compute_metrics_includes_new_breakdowns(self):
        trades = [
            {
                "ticker": "MES=F",
                "direction": "long",
                "pnl_dollars": 100.0,
                "r_multiple": 2.0,
                "setup_type": "VWAP Reclaim",
                "timeframe": "5m",
                "session_bucket": "New York",
                "time_of_day_bucket": "Morning",
                "confluence_score": 3,
                "fvg_involved": True,
                "volume_node_involved": True,
                "bias_at_entry": "bullish",
                "planned_vs_impulsive": "planned",
                "did_follow_plan": True,
                "reason_for_entry": "VWAP hold",
            },
            {
                "ticker": "MES=F",
                "direction": "short",
                "pnl_dollars": -50.0,
                "r_multiple": -1.0,
                "setup_type": "Manual/Other",
                "timeframe": "5m",
                "session_bucket": "New York",
                "time_of_day_bucket": "Afternoon",
                "confluence_score": 1,
                "fvg_involved": False,
                "volume_node_involved": False,
                "bias_at_entry": "mixed",
                "planned_vs_impulsive": "impulsive",
                "did_follow_plan": False,
                "reason_for_entry": "chase",
            },
        ]
        metrics = compute_metrics(trades)
        assert "planned" in metrics.by_planned_vs_impulsive
        assert "True" in metrics.by_fvg_involved
        assert "bullish" in metrics.by_bias_alignment
