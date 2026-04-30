from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from futures.atr import compute_vwap


@dataclass
class VolumeNode:
    price: float
    volume: float
    node_type: str   # 'poc' | 'hvn' | 'lvn' | 'vah' | 'val'


@dataclass
class VolumeProfile:
    """
    Approximate volume profile built from OHLCV candle data.
    Volume is distributed proportionally across each candle's high-low range.
    Label all output as APPROXIMATE — yfinance volume is imperfect.
    """
    prices: list[float]       # bin centre prices, low → high
    volumes: list[float]
    bin_size: float
    poc: float                # Point of Control — highest-volume bin
    poc_volume: float
    vah: float                # Value Area High (default: 70% of volume)
    val: float                # Value Area Low
    hvns: list[VolumeNode]    # High-volume nodes
    lvns: list[VolumeNode]    # Low-volume nodes
    total_volume: float
    errors: list[str] = field(default_factory=list)

    # --- proximity helpers ---

    def _nearest(self, nodes: list[VolumeNode], price: float, above: bool) -> VolumeNode | None:
        candidates = [n for n in nodes if (n.price > price) == above]
        if not candidates:
            return None
        return min(candidates, key=lambda n: abs(n.price - price))

    def nearest_hvn_above(self, price: float) -> VolumeNode | None:
        return self._nearest(self.hvns, price, above=True)

    def nearest_hvn_below(self, price: float) -> VolumeNode | None:
        return self._nearest(self.hvns, price, above=False)

    def nearest_lvn_above(self, price: float) -> VolumeNode | None:
        return self._nearest(self.lvns, price, above=True)

    def nearest_lvn_below(self, price: float) -> VolumeNode | None:
        return self._nearest(self.lvns, price, above=False)

    def is_above_value(self, price: float) -> bool:
        return price > self.vah

    def is_below_value(self, price: float) -> bool:
        return price < self.val

    def is_inside_value(self, price: float) -> bool:
        return self.val <= price <= self.vah

    def value_area_context(self, price: float) -> str:
        if self.is_above_value(price):
            return "above_value"
        if self.is_below_value(price):
            return "below_value"
        return "inside_value"


@dataclass
class VolumeAnalytics:
    approximate: bool
    volume_ma: float | None
    relative_volume: float | None
    vwap: float | None
    highest_volume_prices: list[float]
    highest_volume_candles: list[dict]
    spike_count: int
    last_spike_direction: str
    volume_profile: VolumeProfile | None
    acceptance_context: str
    rejection_context: str


def build_volume_profile(
    df: pd.DataFrame,
    n_bins: int = 50,
    value_area_pct: float = 0.70,
) -> VolumeProfile | None:
    """
    Build an approximate volume profile from OHLCV intraday bars.
    Returns None if data is insufficient.
    """
    if df is None or df.empty or len(df) < 2:
        return None
    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return None

    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())
    if price_max <= price_min:
        return None

    bin_size = (price_max - price_min) / n_bins
    if bin_size <= 0:
        return None

    vol_bins: dict[int, float] = defaultdict(float)

    for _, row in df.iterrows():
        vol = float(row["Volume"])
        if vol <= 0:
            continue
        lo, hi = float(row["Low"]), float(row["High"])
        bar_range = hi - lo

        if bar_range < bin_size * 0.1:
            idx = min(int((float(row["Close"]) - price_min) / bin_size), n_bins - 1)
            vol_bins[idx] += vol
        else:
            lo_bin = max(0, int((lo - price_min) / bin_size))
            hi_bin = min(n_bins - 1, int((hi - price_min) / bin_size))
            n_covered = max(hi_bin - lo_bin + 1, 1)
            per_bin = vol / n_covered
            for b in range(lo_bin, hi_bin + 1):
                vol_bins[b] += per_bin

    if not vol_bins:
        return None

    prices  = [price_min + (i + 0.5) * bin_size for i in range(n_bins)]
    volumes = [vol_bins.get(i, 0.0) for i in range(n_bins)]

    total = sum(volumes)
    if total <= 0:
        return None

    # --- POC ---
    poc_idx = int(np.argmax(volumes))
    poc = prices[poc_idx]
    poc_vol = volumes[poc_idx]

    # --- Value area (expand from POC until value_area_pct of volume captured) ---
    va_vol = volumes[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx

    while va_vol < total * value_area_pct and (lo_idx > 0 or hi_idx < n_bins - 1):
        vol_up   = volumes[hi_idx + 1] if hi_idx < n_bins - 1 else 0.0
        vol_down = volumes[lo_idx - 1] if lo_idx > 0          else 0.0
        if vol_up >= vol_down:
            hi_idx += 1
            va_vol += volumes[hi_idx]
        else:
            lo_idx -= 1
            va_vol += volumes[lo_idx]

    vah = prices[hi_idx]
    val = prices[lo_idx]

    # --- HVN / LVN via statistical thresholds ---
    vol_arr = np.array(volumes)
    nonzero = vol_arr[vol_arr > 0]
    if len(nonzero) >= 4:
        mean, std = float(nonzero.mean()), float(nonzero.std())
        hvn_thr = mean + 0.5 * std
        lvn_thr = max(mean - 0.5 * std, 0.01 * mean)
    else:
        hvn_thr, lvn_thr = float("inf"), 0.0

    hvns = [VolumeNode(prices[i], volumes[i], "hvn") for i in range(n_bins) if volumes[i] >= hvn_thr]
    lvns = [VolumeNode(prices[i], volumes[i], "lvn") for i in range(n_bins) if 0 < volumes[i] <= lvn_thr]

    return VolumeProfile(
        prices=prices, volumes=volumes, bin_size=bin_size,
        poc=poc, poc_volume=poc_vol, vah=vah, val=val,
        hvns=hvns, lvns=lvns, total_volume=total,
    )


def compute_relative_volume(df: pd.DataFrame, lookback: int = 20) -> float | None:
    """Last bar's volume relative to the rolling average of the prior `lookback` bars."""
    if df is None or len(df) < lookback + 1:
        return None
    avg = float(df["Volume"].iloc[-(lookback + 1):-1].mean())
    if avg <= 0:
        return None
    return float(df["Volume"].iloc[-1]) / avg


def compute_volume_ma(df: pd.DataFrame, lookback: int = 20) -> float | None:
    if df is None or len(df) < lookback:
        return None
    ma = df["Volume"].tail(lookback).mean()
    return float(ma) if pd.notna(ma) else None


def volume_spike(df: pd.DataFrame, threshold: float = 1.8, lookback: int = 20) -> bool:
    rv = compute_relative_volume(df, lookback)
    return rv is not None and rv >= threshold


def analyze_volume(
    df: pd.DataFrame,
    current_price: float | None,
    n_bins: int = 50,
    lookback: int = 20,
    spike_threshold: float = 1.8,
) -> VolumeAnalytics:
    profile = build_volume_profile(df, n_bins=n_bins) if df is not None and not df.empty else None
    volume_ma = compute_volume_ma(df, lookback=lookback) if df is not None else None
    relative_volume = compute_relative_volume(df, lookback=lookback) if df is not None else None
    vwap = compute_vwap(df) if df is not None and not df.empty else None

    candles: list[dict] = []
    spike_count = 0
    last_spike_direction = "none"
    highest_volume_prices: list[float] = []
    acceptance_context = "Volume read unavailable."
    rejection_context = ""

    if df is not None and not df.empty:
        ranked = df.sort_values("Volume", ascending=False).head(5)
        for ts, row in ranked.iterrows():
            candles.append(
                {
                    "time": ts,
                    "price": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "direction": "up" if float(row["Close"]) >= float(row["Open"]) else "down",
                }
            )
        highest_volume_prices = [c["price"] for c in candles]

        if len(df) >= lookback + 1:
            trailing_avg = df["Volume"].rolling(lookback).mean()
            spikes = df[trailing_avg.notna() & (df["Volume"] >= trailing_avg * spike_threshold)]
            spike_count = len(spikes)
            if not spikes.empty:
                last = spikes.iloc[-1]
                last_spike_direction = "bullish" if float(last["Close"]) >= float(last["Open"]) else "bearish"

    if profile and current_price is not None:
        ctx = profile.value_area_context(current_price)
        if ctx == "above_value":
            acceptance_context = "Above value and holding; bullish acceptance if pullbacks stay above VAH/near HVN."
            rejection_context = "Watch for rejection only if price loses VAH and accepts back inside value."
        elif ctx == "below_value":
            acceptance_context = "Below value and rejecting value; bearish acceptance if rallies fail near VAL/near HVN."
            rejection_context = "Watch for squeeze risk if price reclaims VAL and holds inside value."
        else:
            acceptance_context = "Inside value; more likely rotational or mean-reverting until value breaks."
            rejection_context = "LVNs can still act as fast-move zones if price escapes value with volume."

    return VolumeAnalytics(
        approximate=True,
        volume_ma=volume_ma,
        relative_volume=relative_volume,
        vwap=vwap,
        highest_volume_prices=highest_volume_prices,
        highest_volume_candles=candles,
        spike_count=spike_count,
        last_spike_direction=last_spike_direction,
        volume_profile=profile,
        acceptance_context=acceptance_context,
        rejection_context=rejection_context,
    )


def render_volume_profile(
    vp: VolumeProfile,
    current_price: float,
    bar_width: int = 22,
    max_rows: int = 40,
) -> str:
    """
    Render volume profile as a coloured Rich markup string (high → low).
    Uses █ and ░ block characters. Colours: yellow=POC, blue=HVN, dim=LVN,
    cyan=VAH/VAL, green=current price row.
    """
    if not vp.prices:
        return "  [dim]No volume profile data.[/dim]"

    max_vol = max(vp.volumes) if vp.volumes else 1
    half_bin = vp.bin_size / 2.0

    def _is(price: float, nodes: list[VolumeNode]) -> bool:
        return any(abs(price - n.price) <= half_bin for n in nodes)

    rows: list[tuple[float, float]] = sorted(
        zip(vp.prices, vp.volumes), key=lambda x: x[0], reverse=True
    )
    # Trim to max_rows rows nearest to current price
    if len(rows) > max_rows:
        rows_with_idx = [(i, p, v) for i, (p, v) in enumerate(rows)]
        rows_with_idx.sort(key=lambda x: abs(x[1] - current_price))
        keep = set(x[0] for x in rows_with_idx[:max_rows])
        rows = [(p, v) for i, (p, v) in enumerate(rows) if i in keep]
        rows.sort(key=lambda x: x[0], reverse=True)

    lines: list[str] = []
    for price, vol in rows:
        if vol <= 0:
            continue
        filled = int(vol / max_vol * bar_width)
        empty  = bar_width - filled
        bar    = "█" * filled + "░" * empty

        is_current = abs(price - current_price) <= half_bin
        is_poc     = abs(price - vp.poc) <= half_bin
        is_hvn     = _is(price, vp.hvns)
        is_lvn     = _is(price, vp.lvns)
        is_vah     = abs(price - vp.vah) <= half_bin
        is_val     = abs(price - vp.val) <= half_bin

        tags: list[str] = []
        if is_poc: tags.append("[yellow bold]POC[/yellow bold]")
        if is_vah: tags.append("[cyan]VAH[/cyan]")
        if is_val: tags.append("[cyan]VAL[/cyan]")
        if is_hvn and not is_poc: tags.append("[blue]HVN[/blue]")
        if is_lvn: tags.append("[dim]LVN[/dim]")
        tag_str = "  " + " ".join(tags) if tags else ""

        marker = "[green]►[/green]" if is_current else " "

        if is_poc:
            colour = "yellow"
        elif is_hvn:
            colour = "blue"
        elif is_lvn:
            colour = "dim"
        elif is_vah or is_val:
            colour = "cyan"
        elif is_current:
            colour = "green"
        else:
            colour = "white"

        lines.append(
            f"  {marker} [{colour}]{price:>10.4f}  {bar}[/{colour}]  "
            f"[dim]{vol:>9,.0f}[/dim]{tag_str}"
        )

    return "\n".join(lines)


def render_volume_analytics(analytics: VolumeAnalytics, current_price: float | None) -> str:
    lines = ["[bold]VOLUME PROFILE / VOLUME NODES[/bold]", ""]
    lines.append("[dim]Approximate only — built from yfinance OHLCV bars.[/dim]")
    lines.append("")
    lines.append(f"  [cyan]Volume MA[/cyan]       : {analytics.volume_ma:,.0f}" if analytics.volume_ma is not None else "  [cyan]Volume MA[/cyan]       : N/A")
    lines.append(
        f"  [cyan]Relative Volume[/cyan]: {analytics.relative_volume:.2f}x"
        if analytics.relative_volume is not None else
        "  [cyan]Relative Volume[/cyan]: N/A"
    )
    lines.append(f"  [cyan]VWAP[/cyan]            : {analytics.vwap:.5g}" if analytics.vwap is not None else "  [cyan]VWAP[/cyan]            : N/A")
    lines.append(f"  [cyan]Volume Spikes[/cyan]   : {analytics.spike_count}  last={analytics.last_spike_direction}")
    lines.append("")

    vp = analytics.volume_profile
    if vp:
        lines.append(f"  [yellow]POC[/yellow]             : {vp.poc:.5g}")
        lines.append(f"  [cyan]VAH / VAL[/cyan]       : {vp.vah:.5g} / {vp.val:.5g}")
        if current_price is not None:
            lines.append(f"  [cyan]Context[/cyan]         : {vp.value_area_context(current_price).replace('_', ' ')}")
        hvn_above = vp.nearest_hvn_above(current_price) if current_price is not None else None
        hvn_below = vp.nearest_hvn_below(current_price) if current_price is not None else None
        lvn_above = vp.nearest_lvn_above(current_price) if current_price is not None else None
        lvn_below = vp.nearest_lvn_below(current_price) if current_price is not None else None
        lines.append(f"  [blue]Nearest HVN above[/blue] : {hvn_above.price:.5g}" if hvn_above else "  [blue]Nearest HVN above[/blue] : N/A")
        lines.append(f"  [blue]Nearest HVN below[/blue] : {hvn_below.price:.5g}" if hvn_below else "  [blue]Nearest HVN below[/blue] : N/A")
        lines.append(f"  [magenta]Nearest LVN above[/magenta] : {lvn_above.price:.5g}" if lvn_above else "  [magenta]Nearest LVN above[/magenta] : N/A")
        lines.append(f"  [magenta]Nearest LVN below[/magenta] : {lvn_below.price:.5g}" if lvn_below else "  [magenta]Nearest LVN below[/magenta] : N/A")
        lines.append("")
        lines.append(f"  [green]Acceptance[/green]    : {analytics.acceptance_context}")
        lines.append(f"  [yellow]Caution[/yellow]      : {analytics.rejection_context}")
        lines.append("")
        lines.append(render_volume_profile(vp, current_price or vp.poc))
    else:
        lines.append("[dim]No volume profile data available.[/dim]")

    if analytics.highest_volume_candles:
        lines.append("")
        lines.append("  [bold]Highest-Volume Candles[/bold]")
        for candle in analytics.highest_volume_candles[:3]:
            direction_color = "green" if candle["direction"] == "up" else "red"
            lines.append(
                f"  [{direction_color}]•[/{direction_color}] "
                f"{candle['time']} close={candle['price']:.5g} vol={candle['volume']:,.0f}"
            )

    return "\n".join(lines)
