from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.analysis import TradeAnalysis


@dataclass(frozen=True)
class AssistantResponse:
    intent: str
    answer: str


def parse_intent(query: str) -> str:
    text = query.lower()
    if "what changed" in text or "changed since" in text:
        return "changes"
    if "macro" in text or "rates" in text or "fred" in text:
        return "macro"
    if "risk" in text or "reward" in text or "r multiple" in text:
        return "risk"
    if "fvg" in text or "imbalance" in text:
        return "fvg"
    if "liquidity" in text or "sweep" in text or "levels" in text:
        return "liquidity"
    if "structure" in text or "regime" in text:
        return "structure"
    if "bias" in text or "bullish" in text or "bearish" in text or "neutral" in text:
        return "bias"
    if "setup" in text:
        return "setup"
    return "summary"


def answer_query(query: str, analysis: TradeAnalysis, previous: TradeAnalysis | None = None) -> AssistantResponse:
    intent = parse_intent(query)

    if intent == "bias":
        answer = (
            f"{analysis.bias.title()} bias with {analysis.confidence}% confidence; regime is {analysis.regime}. "
            f"Based on current {analysis.chart_timeframe} bars and multi-timeframe alignment {analysis.alignment}."
        )
    elif intent == "liquidity":
        levels = ", ".join(level["label"] for level in analysis.key_levels[:4]) or "no nearby levels"
        events = analysis.liquidity_events[:2]
        event_text = events[0]["description"] if events else "No recent sweep/reclaim event."
        answer = f"Nearest liquidity references: {levels}. {event_text} Source: based on current {analysis.chart_timeframe} bars."
    elif intent == "fvg":
        zones = analysis.nearby_fvgs[:2]
        if zones:
            parts = [f"{zone['direction']} {zone['timeframe']} FVG {zone['range']} ({zone['status']})" for zone in zones]
            answer = f"Nearby imbalances: {'; '.join(parts)}. Source: based on current timeframe FVG scan."
        else:
            answer = "No nearby active FVGs in the tracked windows. Source: based on current timeframe FVG scan."
    elif intent == "structure":
        answer = f"Structure reads as {analysis.regime}. {analysis.structure_summary} Source: based on current {analysis.chart_timeframe} swing structure."
    elif intent == "macro":
        if not analysis.macro_context:
            answer = "Macro context is unavailable because no FRED snapshot is cached. Source: FRED configuration check."
        else:
            points = []
            for key in ("DGS10", "DGS2", "FEDFUNDS", "VIXCLS"):
                item = analysis.macro_context.get(key)
                if item and item.get("value") is not None:
                    points.append(f"{item['label']} {item['value']}")
            answer = f"Relevant macro reads: {', '.join(points) if points else 'no FRED values available'}. Source: cached FRED context."
    elif intent == "risk":
        if analysis.risk_plan:
            answer = (
                f"Analytical setup framing: entry {analysis.risk_plan['entry_zone']}, invalidation {analysis.risk_plan['invalidation']}, "
                f"targets {', '.join(analysis.risk_plan['targets'])}. Approx R: {analysis.risk_plan['r_multiple']}. "
                f"Source: {analysis.risk_plan['source']}."
            )
        else:
            answer = "No risk framing available because the engine did not find a clean entry zone. Source: current setup engine output."
    elif intent == "setup":
        setup = analysis.setup
        answer = (
            f"{setup['type']}. Entry zone {setup['entry_zone']}. Invalidation {setup['invalidation']}. "
            f"{setup['reasoning'][0]} Source: current shared setup engine."
        )
    elif intent == "changes":
        if previous is None:
            answer = "No prior refresh is available yet, so there is no delta to compare. Source: in-memory refresh history."
        else:
            changes: list[str] = []
            if previous.bias != analysis.bias:
                changes.append(f"bias changed from {previous.bias} to {analysis.bias}")
            if previous.regime != analysis.regime:
                changes.append(f"regime changed from {previous.regime} to {analysis.regime}")
            if previous.setup["type"] != analysis.setup["type"]:
                changes.append(f"setup shifted from {previous.setup['type']} to {analysis.setup['type']}")
            answer = (
                f"{'; '.join(changes) if changes else 'No major state change since the last refresh.'} "
                "Source: current analysis compared with the previous cached snapshot."
            )
    else:
        answer = (
            f"{analysis.bias.title()} bias, {analysis.regime}, alignment {analysis.alignment}. "
            f"Current framing: {analysis.setup['type']}. Source: based on shared analysis from current bars."
        )

    return AssistantResponse(intent=intent, answer=answer)
