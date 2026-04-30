from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from futures.levels import FuturesLevels
    from futures.volume import VolumeProfile
    from futures.fvg import FairValueGap


@dataclass
class BiasResult:
    bias: str           # 'bullish' | 'bearish' | 'mixed' | 'neutral'
    confidence: str     # 'high' | 'medium' | 'low'
    bull_signals: list[str]
    bear_signals: list[str]
    cautions: list[str]
    score: int          # positive = bullish lean, negative = bearish lean


@dataclass
class ConfluenceZone:
    lower: float
    upper: float
    midpoint: float
    levels: list[tuple[str, float]]   # (level_name, price) pairs
    strength: int                     # number of confluent levels
    zone_type: str                    # 'support' | 'resistance' | 'mixed'


@dataclass
class PullbackZone:
    lower: float
    upper: float
    direction: str        # 'long' | 'short'
    reasons: list[str]
    confirmation: list[str]
    invalidation: float
    target_1x: float | None = None
    target_2x: float | None = None
    target_3x: float | None = None
    strength: int = 0


# ---------------------------------------------------------------------------
# Bias engine
# ---------------------------------------------------------------------------

def compute_bias(
    spot: float | None,
    trend_5m: str,
    trend_15m: str,
    trend_1h: str,
    vwap: float | None,
    levels: "FuturesLevels | None",
    volume_profile: "VolumeProfile | None",
    fvgs_5m: list["FairValueGap"],
    fvgs_15m: list["FairValueGap"],
    fvgs_1h: list["FairValueGap"],
    atr_15m: float | None,
) -> BiasResult:
    """
    Rule-based bias assessment. Does NOT give buy/sell signals.
    Outputs context for discretionary decision-making.
    """
    bull: list[str] = []
    bear: list[str] = []
    cautions: list[str] = []

    if spot is None:
        return BiasResult("neutral", "low", [], [], ["No price data available"], 0)

    # --- Trend ---
    trend_map = {"bullish": 1, "bearish": -1, "neutral": 0}
    if trend_5m == "bullish":
        bull.append("5m trend bullish (price > EMA9 > EMA21)")
    elif trend_5m == "bearish":
        bear.append("5m trend bearish (price < EMA9 < EMA21)")

    if trend_15m == "bullish":
        bull.append("15m trend bullish — intermediate timeframe aligned")
    elif trend_15m == "bearish":
        bear.append("15m trend bearish — intermediate timeframe aligned")

    if trend_1h == "bullish":
        bull.append("1h trend bullish — higher timeframe context is long")
    elif trend_1h == "bearish":
        bear.append("1h trend bearish — higher timeframe context is short")

    # --- VWAP ---
    if vwap:
        if spot > vwap:
            bull.append(f"Price above VWAP ({vwap:.5g}) — intraday long bias")
        else:
            bear.append(f"Price below VWAP ({vwap:.5g}) — intraday short bias")

    # --- Session levels ---
    if levels:
        _check_level(spot, levels.prev_day_high, "prev day high", bull, bear, atr_15m)
        _check_level(spot, levels.prev_day_low,  "prev day low",  bull, bear, atr_15m)
        _check_level(spot, levels.asia.high,     "Asia high",     bull, bear, atr_15m)
        _check_level(spot, levels.asia.low,      "Asia low",      bull, bear, atr_15m)
        _check_level(spot, levels.london.high,   "London high",   bull, bear, atr_15m)
        _check_level(spot, levels.london.low,    "London low",    bull, bear, atr_15m)

    # --- Volume profile ---
    if volume_profile:
        vp = volume_profile
        ctx = vp.value_area_context(spot)
        if ctx == "above_value":
            bull.append(f"Price above value area (VAH {vp.vah:.5g}) — bullish context")
        elif ctx == "below_value":
            bear.append(f"Price below value area (VAL {vp.val:.5g}) — bearish context")
        else:
            cautions.append(
                f"Price inside value area ({vp.val:.5g}–{vp.vah:.5g}) — chop/mean-reversion context"
            )

        hvn_above = vp.nearest_hvn_above(spot)
        hvn_below = vp.nearest_hvn_below(spot)
        if hvn_above and atr_15m and (hvn_above.price - spot) < 2 * atr_15m:
            cautions.append(
                f"HVN resistance ~{hvn_above.price:.5g} within {(hvn_above.price - spot):.4g} pts"
            )
        if hvn_below and atr_15m and (spot - hvn_below.price) < 1.5 * atr_15m:
            bull.append(
                f"HVN support ~{hvn_below.price:.5g} holding within {(spot - hvn_below.price):.4g} pts"
            )

    # --- FVGs ---
    all_fvgs = fvgs_5m + fvgs_15m + fvgs_1h
    bull_fvgs_below = [f for f in all_fvgs if f.direction == "bullish" and f.upper < spot and not f.is_filled]
    bear_fvgs_above = [f for f in all_fvgs if f.direction == "bearish" and f.lower > spot and not f.is_filled]

    if bull_fvgs_below:
        nearest = min(bull_fvgs_below, key=lambda f: spot - f.upper)
        bull.append(
            f"Bullish {nearest.timeframe} FVG support at {nearest.lower:.5g}–{nearest.upper:.5g}"
        )
    if bear_fvgs_above:
        nearest = min(bear_fvgs_above, key=lambda f: f.lower - spot)
        bear.append(
            f"Bearish {nearest.timeframe} FVG resistance at {nearest.lower:.5g}–{nearest.upper:.5g}"
        )

    # 1h FVG carries extra weight (counted double in score)
    bull_fvgs_below_1h = [f for f in fvgs_1h if f.direction == "bullish" and f.upper < spot and not f.is_filled]
    bear_fvgs_above_1h = [f for f in fvgs_1h if f.direction == "bearish" and f.lower > spot and not f.is_filled]
    extra_bull = len(bull_fvgs_below_1h)
    extra_bear = len(bear_fvgs_above_1h)

    # --- Extension caution ---
    if atr_15m and atr_15m > 0 and levels:
        for name, lvl in [
            ("London high", levels.london.high),
            ("Asia high",   levels.asia.high),
            ("NY OR high",  levels.ny_open_range.high if levels.ny_open_range else None),
            ("London low",  levels.london.low),
            ("Asia low",    levels.asia.low),
        ]:
            if lvl is None:
                continue
            ext = (spot - lvl) / atr_15m
            if abs(ext) > 2.0:
                direction = "above" if ext > 0 else "below"
                cautions.append(
                    f"Price is {abs(ext):.1f}x 15m ATR {direction} {name} — possible extension, avoid chase"
                )
                break

    # --- Score and classify ---
    score = len(bull) - len(bear) + extra_bull - extra_bear

    if score >= 4:
        bias, conf = "bullish", "high"
    elif score >= 2:
        bias, conf = "bullish", "medium"
    elif score <= -4:
        bias, conf = "bearish", "high"
    elif score <= -2:
        bias, conf = "bearish", "medium"
    elif -1 <= score <= 1:
        bias, conf = "neutral", "low"
    else:
        bias, conf = "mixed", "low"

    return BiasResult(
        bias=bias, confidence=conf,
        bull_signals=bull, bear_signals=bear,
        cautions=cautions, score=score,
    )


def _check_level(
    spot: float,
    level: float | None,
    name: str,
    bull: list[str],
    bear: list[str],
    atr: float | None,
) -> None:
    if level is None:
        return
    tol = atr * 0.15 if atr else 0  # within 0.15 ATR = "at the level"
    if spot > level + tol:
        bull.append(f"Price above {name} ({level:.5g})")
    elif spot < level - tol:
        bear.append(f"Price below {name} ({level:.5g})")


# ---------------------------------------------------------------------------
# Confluence engine
# ---------------------------------------------------------------------------

def find_confluence_zones(
    named_levels: dict[str, float | None],
    tolerance_atr: float = 0.25,
    atr: float | None = None,
) -> list[ConfluenceZone]:
    """
    Group levels within tolerance_atr × ATR of each other into confluence zones.
    Returns zones with 2+ members, sorted by strength descending.
    """
    tol = (atr * tolerance_atr) if (atr and atr > 0) else tolerance_atr

    valid = sorted(
        [(name, price) for name, price in named_levels.items() if price is not None],
        key=lambda x: x[1],
    )
    if len(valid) < 2:
        return []

    groups: list[list[tuple[str, float]]] = [[valid[0]]]
    for name, price in valid[1:]:
        if price - groups[-1][-1][1] <= tol:
            groups[-1].append((name, price))
        else:
            groups.append([(name, price)])

    zones = [_to_zone(g) for g in groups if len(g) >= 2]
    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones


def _to_zone(group: list[tuple[str, float]]) -> ConfluenceZone:
    prices = [p for _, p in group]
    lo, hi = min(prices), max(prices)
    mid = (lo + hi) / 2.0

    name_blob = " ".join(n.lower() for n, _ in group)
    support_hits = sum(1 for kw in ("low", "support", "hvn", "bullish", "val") if kw in name_blob)
    resist_hits  = sum(1 for kw in ("high", "resistance", "bearish", "vah") if kw in name_blob)
    zone_type = "support" if support_hits > resist_hits else ("resistance" if resist_hits > support_hits else "mixed")

    return ConfluenceZone(
        lower=lo, upper=hi, midpoint=mid,
        levels=group, strength=len(group), zone_type=zone_type,
    )


# ---------------------------------------------------------------------------
# Pullback suggestion engine
# ---------------------------------------------------------------------------

def suggest_pullback_zones(
    bias: BiasResult,
    confluence_zones: list[ConfluenceZone],
    spot: float,
    atr: float | None,
    fvgs: list,
    volume_profile,
) -> list[PullbackZone]:
    """
    Suggest possible pullback zones.
    Language is intentionally cautious — these are watch zones, not signals.
    """
    if bias.bias not in ("bullish", "bearish"):
        return []

    is_long = bias.bias == "bullish"
    sign = 1 if is_long else -1
    ref_atr = atr or 0.0

    zones: list[PullbackZone] = []

    for cz in confluence_zones[:6]:
        # Longs want support below; shorts want resistance above
        if is_long and cz.upper >= spot:
            continue
        if not is_long and cz.lower <= spot:
            continue

        reasons = [f"{name} @ {price:.5g}" for name, price in cz.levels]

        # FVG reinforcement?
        fvg_dir = "bullish" if is_long else "bearish"
        nearby = next(
            (f for f in fvgs if f.direction == fvg_dir and not f.is_filled
             and abs(f.midpoint - cz.midpoint) <= ref_atr * 0.5),
            None,
        )
        if nearby:
            reasons.append(
                f"Reinforced by {nearby.timeframe} {fvg_dir} FVG "
                f"({nearby.lower:.5g}–{nearby.upper:.5g})"
            )

        # HVN reinforcement?
        if volume_profile:
            hvn = (
                volume_profile.nearest_hvn_below(spot) if is_long
                else volume_profile.nearest_hvn_above(spot)
            )
            if hvn and abs(hvn.price - cz.midpoint) <= ref_atr * 0.5:
                reasons.append(f"HVN support ~{hvn.price:.5g} nearby")

        inv_offset = ref_atr * 0.5 if ref_atr else cz.lower * 0.001
        invalidation = cz.lower - inv_offset if is_long else cz.upper + inv_offset

        confirmation = (
            [
                f"5m candle closes back above {cz.upper:.5g}",
                "Pullback volume below average (no heavy selling)",
                "Higher low forms on 1m/5m chart",
            ]
            if is_long else
            [
                f"5m candle closes back below {cz.lower:.5g}",
                "Rally volume below average (no heavy buying)",
                "Lower high forms on 1m/5m chart",
            ]
        )

        t1 = spot + sign * ref_atr     if ref_atr else None
        t2 = spot + sign * 2 * ref_atr if ref_atr else None
        t3 = spot + sign * 3 * ref_atr if ref_atr else None

        zones.append(PullbackZone(
            lower=cz.lower, upper=cz.upper,
            direction="long" if is_long else "short",
            reasons=reasons, confirmation=confirmation,
            invalidation=invalidation,
            target_1x=t1, target_2x=t2, target_3x=t3,
            strength=cz.strength + (1 if nearby else 0),
        ))

    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones[:3]
