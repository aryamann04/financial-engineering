from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from types import SimpleNamespace

from config.sessions import ET
from config.settings import Settings, load_settings
from core.assistant import AssistantResponse, answer_query
from core.backtest import BacktestSummary, run_bias_backtest
from core.data import MarketDataBundle, fetch_market_data
from core.fvg import FVGZone, detect_fvg_zones
from core.indicators import atr_value, rolling_range, session_range, trend_snapshot, vwap_value
from core.liquidity import KeyLevel, LiquidityEvent, detect_key_levels, detect_liquidity_sweeps
from core.macro import macro_context_dict
from core.risk import RiskPlan, build_risk_plan
from core.signals import TimeframeSnapshot, alignment_status, summarize_timeframe
from core.structure import StructureState, detect_structure
from futures.bias import compute_bias, find_confluence_zones, suggest_pullback_zones
from futures.ideas import generate_ideas
from futures.volume import analyze_volume


@dataclass
class TradeAnalysis:
    symbol: str
    display_name: str
    chart_timeframe: str
    timestamp: str
    bias: str
    confidence: int
    regime: str
    alignment: str
    structure_summary: str
    key_levels: list[dict]
    nearby_fvgs: list[dict]
    liquidity_events: list[dict]
    macro_context: dict[str, dict]
    setup: dict
    warnings: list[str]
    snapshots: dict[str, dict]
    risk_plan: dict | None
    backtest_summary: dict | None
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TradeEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._analysis_cache: dict[tuple[str, str], TradeAnalysis] = {}
        self._previous_cache: dict[tuple[str, str], TradeAnalysis] = {}

    def previous_analysis(self, symbol: str, timeframe: str) -> TradeAnalysis | None:
        return self._previous_cache.get((symbol.upper(), timeframe))

    def analyze(self, symbol: str, timeframe: str = "5m") -> TradeAnalysis:
        data = fetch_market_data(symbol, self.settings)
        analysis = self._build_analysis(data, timeframe=timeframe)
        key = (analysis.symbol.upper(), timeframe)
        previous = self._analysis_cache.get(key)
        if previous is not None:
            self._previous_cache[key] = previous
        self._analysis_cache[key] = analysis
        return analysis

    def query(self, symbol: str, query: str, timeframe: str = "5m") -> AssistantResponse:
        analysis = self.analyze(symbol, timeframe=timeframe)
        previous = self.previous_analysis(analysis.symbol, timeframe)
        return answer_query(query, analysis, previous=previous)

    def _build_analysis(self, data: MarketDataBundle, timeframe: str) -> TradeAnalysis:
        current_price = data.spot
        settings = self.settings
        frames = data.frames
        selected = frames.get(timeframe)
        if selected is None or selected.empty:
            selected = frames["5m"]

        volume = analyze_volume(frames["5m"], current_price, n_bins=settings.volume_bins, spike_threshold=settings.vol_spike_threshold) if not frames["5m"].empty and current_price is not None else None
        trend_1h = trend_snapshot(frames["1h"]).trend if not frames["1h"].empty else "neutral"
        key_levels = detect_key_levels(data.daily, frames["5m"], atr=atr_value(frames["15m"]) or atr_value(frames["5m"]))

        snapshots: dict[str, TimeframeSnapshot] = {}
        structure_by_tf: dict[str, StructureState] = {}
        fvg_by_tf: dict[str, list[FVGZone]] = {}

        for tf, frame in frames.items():
            if frame is None or frame.empty:
                continue
            atr = atr_value(frame, period=settings.atr_period)
            structure = detect_structure(frame.tail(160))
            structure_by_tf[tf] = structure
            trend = trend_snapshot(frame)
            fvgs = detect_fvg_zones(frame.tail(180), tf, current_price, atr, settings.fvg_min_size_atr) if current_price is not None else []
            fvg_by_tf[tf] = fvgs
            sess_high, sess_low = session_range(frame.tail(60))
            snapshots[tf] = summarize_timeframe(
                timeframe=tf,
                trend=trend,
                structure=structure,
                current_price=current_price,
                atr=atr,
                recent_range=rolling_range(frame, lookback=min(20, len(frame))),
                session_high=sess_high,
                session_low=sess_low,
                vwap=vwap_value(frame.tail(120)),
                nearby_fvgs=fvgs,
            )

        alignment = alignment_status(snapshots)
        chart_structure = structure_by_tf.get(timeframe) or structure_by_tf.get("5m") or StructureState("range", "neutral", summary="No structure state.")
        chart_snapshot = snapshots.get(timeframe) or snapshots.get("5m")
        chart_atr = chart_snapshot.atr if chart_snapshot else None

        fvgs_5m = fvg_by_tf.get("5m", [])
        fvgs_15m = fvg_by_tf.get("15m", [])
        fvgs_1h = fvg_by_tf.get("1h", [])

        legacy_5m = [
            SimpleNamespace(direction=zone.direction, upper=zone.upper, lower=zone.lower, timeframe=zone.timeframe, is_filled=zone.status == "fully mitigated")
            for zone in fvgs_5m
        ]
        legacy_15m = [
            SimpleNamespace(direction=zone.direction, upper=zone.upper, lower=zone.lower, timeframe=zone.timeframe, is_filled=zone.status == "fully mitigated")
            for zone in fvgs_15m
        ]
        legacy_1h = [
            SimpleNamespace(direction=zone.direction, upper=zone.upper, lower=zone.lower, timeframe=zone.timeframe, is_filled=zone.status == "fully mitigated")
            for zone in fvgs_1h
        ]

        snapshot_5m = snapshots.get("5m") or chart_snapshot
        snapshot_15m = snapshots.get("15m") or chart_snapshot

        bias_result = compute_bias(
            spot=current_price,
            trend_5m=snapshot_5m.trend if snapshot_5m else "neutral",
            trend_15m=snapshot_15m.trend if snapshot_15m else "neutral",
            trend_1h=trend_1h,
            vwap=snapshot_5m.vwap if snapshot_5m else None,
            levels=None,
            volume_profile=volume.volume_profile if volume else None,
            fvgs_5m=legacy_5m,
            fvgs_15m=legacy_15m,
            fvgs_1h=legacy_1h,
            atr_15m=snapshot_15m.atr if snapshot_15m else None,
        )

        confluence = find_confluence_zones(
            {level.name: level.price for level in key_levels},
            tolerance_atr=settings.confluence_tolerance_atr,
            atr=snapshot_15m.atr if snapshot_15m else None,
        )
        all_legacy_fvgs = [
            SimpleNamespace(
                direction=zone.direction,
                upper=zone.upper,
                lower=zone.lower,
                timeframe=zone.timeframe,
                is_filled=zone.status == "fully mitigated",
                midpoint=zone.midpoint,
                atr_mult=None,
                age_bars=0,
                size=zone.upper - zone.lower,
            )
            for zone in (fvgs_5m + fvgs_15m + fvgs_1h)
        ]
        pullbacks = suggest_pullback_zones(
            bias_result,
            confluence,
            current_price or 0.0,
            snapshot_15m.atr if snapshot_15m else None,
            all_legacy_fvgs,
            volume.volume_profile if volume else None,
        )
        ideas = generate_ideas(
            bias_result,
            levels=None,
            fvgs=all_legacy_fvgs,
            volume_profile=volume.volume_profile if volume else None,
            confluence_zones=confluence,
            pullback_zones=pullbacks,
            spot=current_price,
            atr_5m=snapshot_5m.atr if snapshot_5m else None,
            atr_15m=snapshot_15m.atr if snapshot_15m else None,
        )

        liquidity_events = detect_liquidity_sweeps(selected, key_levels, atr=chart_atr)
        nearby_fvgs = sorted((fvg_by_tf.get(timeframe, []) + fvg_by_tf.get("15m", []) + fvg_by_tf.get("1h", [])), key=lambda zone: zone.distance)[:6]

        bias = bias_result.bias if bias_result.bias in {"bullish", "bearish", "neutral"} else "neutral"
        if chart_structure.regime == "range" and alignment == "conflicted":
            bias = "neutral"

        confidence = 40
        if alignment in {"bullish", "bearish"}:
            confidence += 25
        if bias_result.confidence == "high":
            confidence += 20
        elif bias_result.confidence == "medium":
            confidence += 10
        if liquidity_events:
            confidence += 5
        confidence = max(0, min(confidence, 100))

        risk_plan = None
        if ideas:
            top_idea = ideas[0]
            built = build_risk_plan(
                current_price=current_price,
                direction=top_idea.direction,
                atr=chart_atr,
                key_levels=key_levels,
                nearby_fvgs=nearby_fvgs,
                entry_low=top_idea.entry_low,
                entry_high=top_idea.entry_high,
            )
            risk_plan = asdict(built) if built else None
            setup = {
                "type": top_idea.setup_type,
                "entry_zone": f"{top_idea.entry_low:.5g} - {top_idea.entry_high:.5g}",
                "invalidation": f"{top_idea.invalidation:.5g}",
                "targets": [f"{target:.5g}" for target in [top_idea.target_1x, top_idea.target_2x, top_idea.target_3x] if target is not None],
                "reasoning": top_idea.reasons[:4],
            }
        else:
            fallback = build_risk_plan(current_price, "long" if bias == "bullish" else "short", chart_atr, key_levels, nearby_fvgs) if bias in {"bullish", "bearish"} else None
            risk_plan = asdict(fallback) if fallback else None
            setup = {
                "type": "No clean setup" if bias == "neutral" else f"{bias.title()} bias, but wait for pullback",
                "entry_zone": risk_plan["entry_zone"] if risk_plan else "N/A",
                "invalidation": risk_plan["invalidation"] if risk_plan else "N/A",
                "targets": risk_plan["targets"] if risk_plan else [],
                "reasoning": [
                    "Price is mid-range." if bias == "neutral" else f"{bias.title()} bias exists, but current location looks extended.",
                ],
            }

        backtest = run_bias_backtest(selected, timeframe=timeframe)
        macro = macro_context_dict(settings)
        warnings = list(data.errors)
        warnings.extend(bias_result.cautions[:3])
        if not settings.fred_api_key:
            warnings.append("FRED_API_KEY not configured; macro panel is running in optional mode.")

        if chart_structure.regime == "range" and bias != "neutral":
            warnings.append("Bias leans directional, but price structure is still range-like.")

        return TradeAnalysis(
            symbol=data.symbol,
            display_name=data.display_name,
            chart_timeframe=timeframe,
            timestamp=datetime.now(tz=ET).isoformat(),
            bias=bias,
            confidence=confidence,
            regime=chart_structure.regime,
            alignment=alignment,
            structure_summary=chart_structure.summary,
            key_levels=[
                {"label": level.name, "price": level.price, "kind": level.kind, "timeframe": level.timeframe, "source": level.source}
                for level in sorted(key_levels, key=lambda level: abs((current_price or level.price) - level.price))[:10]
            ],
            nearby_fvgs=[
                {
                    "timeframe": zone.timeframe,
                    "direction": zone.direction,
                    "range": f"{zone.lower:.5g} - {zone.upper:.5g}",
                    "status": zone.status,
                    "distance": round(zone.distance, 6),
                    "source": zone.source,
                }
                for zone in nearby_fvgs
            ],
            liquidity_events=[
                {
                    "type": event.event_type,
                    "direction": event.direction,
                    "level": event.level_name,
                    "price": event.price,
                    "description": event.description,
                    "timestamp": str(event.timestamp),
                    "source": event.source,
                }
                for event in liquidity_events
            ],
            macro_context=macro,
            setup=setup,
            warnings=warnings,
            snapshots={tf: asdict(snapshot) for tf, snapshot in snapshots.items()},
            risk_plan=risk_plan,
            backtest_summary=asdict(backtest) if backtest else None,
            sources=[
                f"based on current {tf} bars" for tf in snapshots
            ] + ["based on cached FRED context" if settings.fred_api_key else "FRED unavailable"],
        )
