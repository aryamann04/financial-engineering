from __future__ import annotations

from dataclasses import dataclass, field

SETUP_TYPES: list[str] = [
    "Asia High Breakout",
    "Asia Low Breakdown",
    "London High Breakout",
    "London Low Breakdown",
    "NY Open Range Breakout",
    "NY Open Range Breakdown",
    "VWAP Reclaim",
    "Liquidity Sweep & Reclaim",
    "Shallow Pullback Continuation",
    "Prior Day High Breakout",
    "Prior Day Low Breakdown",
    "Manual/Other",
]

TIMEFRAMES: list[str] = ["1m", "5m", "15m", "30m", "1h"]


@dataclass
class TradePlan:
    symbol: str
    direction: str              # 'long' | 'short'
    entry: float
    stop: float
    target: float | None
    setup_type: str
    timeframe: str
    notes: str
    # Calculated
    risk_points: float
    reward_points: float | None
    r_multiple: float | None
    atr_5m: float | None
    atr_15m: float | None
    stop_atr_5m_mult: float | None
    stop_atr_15m_mult: float | None
    target_atr_15m_mult: float | None
    target_1x: float | None
    target_2x: float | None
    target_3x: float | None
    r_at_1x: float | None
    r_at_2x: float | None
    r_at_3x: float | None
    has_room: bool
    quality: str                # 'strong' | 'decent' | 'weak' | 'avoid' | 'unknown'
    quality_reason: str


def assess_trade(
    symbol: str,
    direction: str,
    entry: float,
    stop: float,
    target: float | None,
    setup_type: str,
    timeframe: str,
    notes: str,
    atr_5m: float | None,
    atr_15m: float | None,
) -> TradePlan:
    sign = 1 if direction == "long" else -1
    risk_points = abs(entry - stop)
    if risk_points < 1e-10:
        risk_points = 1e-10

    reward_points: float | None = None
    r_multiple: float | None = None
    if target is not None:
        reward_points = abs(target - entry)
        r_multiple = reward_points / risk_points

    ref_atr = atr_15m or atr_5m

    stop_atr_5m_mult   = (risk_points / atr_5m)  if atr_5m  and atr_5m  > 0 else None
    stop_atr_15m_mult  = (risk_points / atr_15m) if atr_15m and atr_15m > 0 else None
    target_atr_15m_mult = (reward_points / atr_15m) if (reward_points and atr_15m and atr_15m > 0) else None

    target_1x = target_2x = target_3x = None
    r_at_1x = r_at_2x = r_at_3x = None

    if ref_atr and ref_atr > 0:
        target_1x = entry + sign * 1 * ref_atr
        target_2x = entry + sign * 2 * ref_atr
        target_3x = entry + sign * 3 * ref_atr
        r_at_1x = abs(target_1x - entry) / risk_points
        r_at_2x = abs(target_2x - entry) / risk_points
        r_at_3x = abs(target_3x - entry) / risk_points

    has_room = bool(r_at_2x and r_at_2x >= 1.5)

    quality, reason = _assess_quality(
        risk_points=risk_points,
        r_multiple=r_multiple,
        stop_atr_mult=stop_atr_15m_mult or stop_atr_5m_mult,
        r_at_2x=r_at_2x,
        has_room=has_room,
        atr=ref_atr,
    )

    return TradePlan(
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        setup_type=setup_type,
        timeframe=timeframe,
        notes=notes,
        risk_points=risk_points,
        reward_points=reward_points,
        r_multiple=r_multiple,
        atr_5m=atr_5m,
        atr_15m=atr_15m,
        stop_atr_5m_mult=stop_atr_5m_mult,
        stop_atr_15m_mult=stop_atr_15m_mult,
        target_atr_15m_mult=target_atr_15m_mult,
        target_1x=target_1x,
        target_2x=target_2x,
        target_3x=target_3x,
        r_at_1x=r_at_1x,
        r_at_2x=r_at_2x,
        r_at_3x=r_at_3x,
        has_room=has_room,
        quality=quality,
        quality_reason=reason,
    )


def _assess_quality(
    risk_points: float,
    r_multiple: float | None,
    stop_atr_mult: float | None,
    r_at_2x: float | None,
    has_room: bool,
    atr: float | None,
) -> tuple[str, str]:
    if atr is None:
        return "unknown", "ATR unavailable — cannot assess trade quality."

    reasons: list[str] = []
    score = 0

    if stop_atr_mult is not None:
        if stop_atr_mult < 0.3:
            reasons.append(f"stop is very tight ({stop_atr_mult:.2f}x ATR) — susceptible to noise")
            score -= 1
        elif stop_atr_mult <= 1.2:
            reasons.append(f"stop is well-placed ({stop_atr_mult:.2f}x ATR)")
            score += 2
        elif stop_atr_mult <= 2.0:
            reasons.append(f"stop is moderately wide ({stop_atr_mult:.1f}x ATR)")
            score += 1
        else:
            reasons.append(f"stop is very wide ({stop_atr_mult:.1f}x ATR) — R/R compressed")
            score -= 1

    if r_at_2x is not None:
        if r_at_2x >= 2.5:
            reasons.append(f"{r_at_2x:.1f}R at 2x ATR target — excellent room")
            score += 2
        elif r_at_2x >= 1.5:
            reasons.append(f"{r_at_2x:.1f}R at 2x ATR target — adequate room")
            score += 1
        else:
            reasons.append(f"only {r_at_2x:.1f}R at 2x ATR — not enough room for preferred target")
            score -= 1

    if r_multiple is not None:
        if r_multiple >= 3.0:
            reasons.append(f"planned R ({r_multiple:.1f}R) is strong")
            score += 2
        elif r_multiple >= 2.0:
            reasons.append(f"planned R ({r_multiple:.1f}R) is good")
            score += 1
        elif r_multiple < 1.5:
            reasons.append(f"planned R ({r_multiple:.1f}R) is too low")
            score -= 1

    if not has_room:
        score -= 1

    if score >= 4:
        quality = "strong"
    elif score >= 2:
        quality = "decent"
    elif score >= 0:
        quality = "weak"
    else:
        quality = "avoid"

    reason = "; ".join(reasons) if reasons else "Setup meets minimum criteria."
    return quality, reason


def format_plan_display(plan: TradePlan) -> str:
    """Render a trade plan analysis as a clean terminal string."""
    sep = "─" * 60
    lines = [sep, f"  TRADE PLAN — {plan.symbol}", sep]

    def fv(val: float | None, dec: int = 5) -> str:
        return f"{val:.{dec}g}" if val is not None else "N/A"

    lines += [
        f"  Direction     : {plan.direction.upper()}",
        f"  Setup Type    : {plan.setup_type}",
        f"  Timeframe     : {plan.timeframe}",
        f"  Entry         : {fv(plan.entry)}",
        f"  Stop          : {fv(plan.stop)}",
        f"  Target        : {fv(plan.target)}",
        "",
        "  RISK / REWARD",
        "  " + "─" * 42,
        f"  Risk (points) : {fv(plan.risk_points)}",
    ]
    if plan.reward_points is not None:
        lines.append(f"  Reward(points): {fv(plan.reward_points)}")
    if plan.r_multiple is not None:
        lines.append(f"  R multiple    : {plan.r_multiple:.2f}R")

    lines += [
        "",
        "  ATR CONTEXT",
        "  " + "─" * 42,
        f"  5m  ATR       : {fv(plan.atr_5m)}",
        f"  15m ATR       : {fv(plan.atr_15m)}",
    ]
    if plan.stop_atr_15m_mult is not None:
        lines.append(f"  Stop = {plan.stop_atr_15m_mult:.2f}x 15m ATR")
    elif plan.stop_atr_5m_mult is not None:
        lines.append(f"  Stop = {plan.stop_atr_5m_mult:.2f}x 5m ATR")

    lines += [
        "",
        f"  ATR Targets (ref ATR = {fv(plan.atr_15m or plan.atr_5m)}):",
        f"  1x ATR : {fv(plan.target_1x)}"
        + (f"  → {plan.r_at_1x:.2f}R" if plan.r_at_1x else ""),
        f"  2x ATR : {fv(plan.target_2x)}"
        + (f"  → {plan.r_at_2x:.2f}R" if plan.r_at_2x else ""),
        f"  3x ATR : {fv(plan.target_3x)}"
        + (f"  → {plan.r_at_3x:.2f}R" if plan.r_at_3x else ""),
        "",
        "  ASSESSMENT",
        "  " + "─" * 42,
        f"  Quality       : {plan.quality.upper()}",
        f"  Reason        : {plan.quality_reason}",
        f"  Has room (2x) : {'Yes' if plan.has_room else 'No'}",
    ]

    if plan.notes:
        lines += ["", f"  Notes         : {plan.notes}"]

    lines.append(sep)
    return "\n".join(lines)
