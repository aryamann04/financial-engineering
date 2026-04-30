from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from futures.bias import BiasResult, ConfluenceZone, PullbackZone
    from futures.fvg import FairValueGap
    from futures.levels import FuturesLevels
    from futures.volume import VolumeProfile


@dataclass
class TradeIdea:
    """
    A structured, context-driven trade idea.
    Uses cautious language: 'possible setup', 'watch zone', not a signal.
    """
    direction: str           # 'long' | 'short'
    setup_type: str
    entry_low: float
    entry_high: float
    confirmation: list[str]
    invalidation: float
    target_1x: float | None
    target_2x: float | None
    target_3x: float | None
    r_estimate: float | None  # estimated R if triggered from midpoint to 2x ATR
    confidence: str           # 'high' | 'medium' | 'low'
    reasons: list[str]
    warnings: list[str]

    @property
    def entry_mid(self) -> float:
        return (self.entry_low + self.entry_high) / 2.0


def generate_ideas(
    bias: "BiasResult",
    levels: "FuturesLevels | None",
    fvgs: list["FairValueGap"],
    volume_profile: "VolumeProfile | None",
    confluence_zones: list["ConfluenceZone"],
    pullback_zones: list["PullbackZone"],
    spot: float | None,
    atr_5m: float | None,
    atr_15m: float | None,
) -> list[TradeIdea]:
    """
    Generate structured trade ideas from current context.
    Produces at most 5 ideas, sorted by confidence then relevance.
    """
    if spot is None or bias.bias == "neutral":
        return []

    ideas: list[TradeIdea] = []
    ref = atr_15m or atr_5m or 0.0
    is_long = bias.bias == "bullish"
    sign = 1 if is_long else -1
    direction = "long" if is_long else "short"

    # --- 1. Pullback continuation (from pullback zones) ---
    for pz in pullback_zones:
        if not ref:
            continue
        risk = abs(pz.entry_mid - pz.invalidation) if hasattr(pz, "entry_mid") else ref
        entry_mid = (pz.lower + pz.upper) / 2.0
        risk = abs(entry_mid - pz.invalidation)
        if risk < 1e-9:
            risk = ref * 0.5
        t2 = spot + sign * 2 * ref
        r_est = abs(t2 - entry_mid) / risk if risk > 0 else None

        ideas.append(TradeIdea(
            direction=direction,
            setup_type="Pullback Continuation",
            entry_low=pz.lower,  entry_high=pz.upper,
            confirmation=pz.confirmation,
            invalidation=pz.invalidation,
            target_1x=spot + sign * ref,
            target_2x=t2,
            target_3x=spot + sign * 3 * ref,
            r_estimate=r_est,
            confidence="medium" if bias.confidence in ("high", "medium") else "low",
            reasons=pz.reasons,
            warnings=bias.cautions[:2],
        ))

    # --- 2. FVG retest ---
    fvg_dir = "bullish" if is_long else "bearish"
    target_fvgs = [
        f for f in fvgs
        if f.direction == fvg_dir and not f.is_filled
        and ((is_long and f.upper < spot) or (not is_long and f.lower > spot))
    ]
    if target_fvgs and ref:
        nearest = min(target_fvgs, key=lambda f: abs(f.midpoint - spot))
        risk = abs(nearest.midpoint - (nearest.lower - ref * 0.5 if is_long else nearest.upper + ref * 0.5))
        risk = max(risk, ref * 0.3)
        t2 = spot + sign * 2 * ref
        r_est = abs(t2 - nearest.midpoint) / risk if risk > 0 else None

        size_tag = f"{nearest.atr_mult:.1f}x ATR" if nearest.atr_mult else f"{nearest.size:.4g} pts"
        ideas.append(TradeIdea(
            direction=direction,
            setup_type=f"FVG Retest ({nearest.timeframe})",
            entry_low=nearest.lower,  entry_high=nearest.upper,
            confirmation=[
                f"Price retraces into {nearest.lower:.5g}–{nearest.upper:.5g}",
                f"{'Bullish' if is_long else 'Bearish'} candle closes {('above' if is_long else 'below')} FVG",
                "Pullback volume lighter than the impulsive leg",
            ],
            invalidation=(nearest.lower - ref * 0.5) if is_long else (nearest.upper + ref * 0.5),
            target_1x=spot + sign * ref,
            target_2x=t2,
            target_3x=spot + sign * 3 * ref,
            r_estimate=r_est,
            confidence="medium" if nearest.atr_mult and nearest.atr_mult >= 0.25 else "low",
            reasons=[
                f"{nearest.timeframe} {fvg_dir} FVG at {nearest.lower:.5g}–{nearest.upper:.5g} ({size_tag})",
                f"Gap formed {nearest.age_bars} bars ago — still active",
                f"Overall bias: {bias.bias}",
            ],
            warnings=[
                "FVG may fully fill before reversing",
                "Monitor candle structure at FVG entry",
            ] + bias.cautions[:1],
        ))

    # --- 3. VWAP reclaim / hold ---
    if levels and levels.vwap and ref:
        vwap = levels.vwap
        dist = abs(spot - vwap)
        if dist < ref * 1.5:
            holding_above = spot > vwap and is_long
            about_to_reclaim = spot < vwap and is_long and dist < ref * 0.5

            if holding_above or about_to_reclaim:
                tag = "VWAP Hold" if holding_above else "VWAP Reclaim"
                risk = ref * 0.6
                ideas.append(TradeIdea(
                    direction=direction,
                    setup_type=tag,
                    entry_low=vwap - ref * 0.2,
                    entry_high=vwap + ref * 0.2,
                    confirmation=[
                        f"Price holds {'above' if is_long else 'below'} VWAP ({vwap:.5g})",
                        "Volume normal or declining on test",
                        f"{'Higher low' if is_long else 'Lower high'} prints near VWAP",
                    ],
                    invalidation=vwap - ref * 0.5 if is_long else vwap + ref * 0.5,
                    target_1x=spot + sign * ref,
                    target_2x=spot + sign * 2 * ref,
                    target_3x=spot + sign * 3 * ref,
                    r_estimate=(2 * ref) / risk,
                    confidence="medium",
                    reasons=[
                        f"Price is {'above' if is_long else 'below'} VWAP — intraday bias aligned",
                        f"Bias: {bias.bias} ({bias.confidence} confidence)",
                    ],
                    warnings=bias.cautions[:2],
                ))

    # --- 4. Value area breakout / rejection ---
    if volume_profile and ref:
        vp = volume_profile
        # Breakout above VAH
        if is_long and abs(spot - vp.vah) < ref * 0.5 and spot > vp.vah:
            ideas.append(TradeIdea(
                direction="long",
                setup_type="Value Area Breakout (Above VAH)",
                entry_low=vp.vah, entry_high=vp.vah + ref * 0.3,
                confirmation=[
                    f"Price holds above VAH ({vp.vah:.5g}) on retest",
                    "Volume expands on the breakout bar",
                    "No strong rejection back inside value",
                ],
                invalidation=vp.poc,
                target_1x=vp.vah + ref, target_2x=vp.vah + 2 * ref, target_3x=vp.vah + 3 * ref,
                r_estimate=(2 * ref) / max(spot - vp.poc, ref * 0.5),
                confidence="medium",
                reasons=[
                    f"Price broke above value area high ({vp.vah:.5g})",
                    f"POC ({vp.poc:.5g}) as natural stop reference",
                ],
                warnings=["Value area breakouts often retest VAH before continuing"] + bias.cautions[:1],
            ))
        # Breakout below VAL
        elif not is_long and abs(spot - vp.val) < ref * 0.5 and spot < vp.val:
            ideas.append(TradeIdea(
                direction="short",
                setup_type="Value Area Breakdown (Below VAL)",
                entry_low=vp.val - ref * 0.3, entry_high=vp.val,
                confirmation=[
                    f"Price holds below VAL ({vp.val:.5g}) on retest",
                    "Volume expands on the breakdown bar",
                    "No strong recovery back inside value",
                ],
                invalidation=vp.poc,
                target_1x=vp.val - ref, target_2x=vp.val - 2 * ref, target_3x=vp.val - 3 * ref,
                r_estimate=(2 * ref) / max(vp.poc - spot, ref * 0.5),
                confidence="medium",
                reasons=[
                    f"Price broke below value area low ({vp.val:.5g})",
                    f"POC ({vp.poc:.5g}) as natural stop reference",
                ],
                warnings=["Value area breakdowns often retest VAL before continuing"] + bias.cautions[:1],
            ))

    # --- Sort by confidence then R estimate ---
    order = {"high": 0, "medium": 1, "low": 2}
    ideas.sort(key=lambda x: (order.get(x.confidence, 3), -(x.r_estimate or 0)))
    return ideas[:5]


def format_idea_rich(idea: TradeIdea, idx: int, ref_atr: float | None) -> str:
    """Render a trade idea as a Rich markup string."""
    conf_color = {"high": "green", "medium": "yellow", "low": "dim"}.get(idea.confidence, "white")
    dir_color  = "green" if idea.direction == "long" else "red"
    dir_label  = "▲ LONG" if idea.direction == "long" else "▼ SHORT"

    lines = [
        f"[bold]Idea {idx}: {idea.setup_type}[/bold]",
        f"  Direction    : [{dir_color}]{dir_label}[/{dir_color}]",
        f"  Confidence   : [{conf_color}]{idea.confidence.upper()}[/{conf_color}]",
        f"  Entry zone   : [cyan]{idea.entry_low:.5g} – {idea.entry_high:.5g}[/cyan]",
        f"  Invalidation : [red]{idea.invalidation:.5g}[/red]",
    ]
    for mult, t in [(1, idea.target_1x), (2, idea.target_2x), (3, idea.target_3x)]:
        if t is not None:
            lines.append(f"  {mult}x ATR tgt  : [green]{t:.5g}[/green]")
    if idea.r_estimate is not None:
        lines.append(f"  R estimate   : [yellow]{idea.r_estimate:.1f}R[/yellow] (approx)")

    lines.append("  [bold]Why:[/bold]")
    for r in idea.reasons:
        lines.append(f"    [dim]•[/dim] {r}")

    lines.append("  [bold]Confirmation:[/bold]")
    for c in idea.confirmation:
        lines.append(f"    [dim]•[/dim] {c}")

    if idea.warnings:
        lines.append("  [yellow]Caution:[/yellow]")
        for w in idea.warnings:
            lines.append(f"    [yellow]![/yellow] {w}")

    return "\n".join(lines)
