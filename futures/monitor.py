from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from config.tickers import resolve_ticker


@dataclass
class MonitorEntry:
    symbol: str
    display_name: str
    price: float | None
    daily_chg_pct: float | None
    trend_5m: str
    trend_15m: str
    trend_1h: str
    bias: str
    confidence: str
    above_vwap: bool | None
    session: str
    atr_15m: float | None
    atr_regime: str
    vol_spike: bool
    nearest_level: str
    nearest_fvg_dir: str    # '' | 'bullish' | 'bearish'
    nearest_fvg_tf: str
    near_key_level: bool
    best_idea: str
    errors: list[str] = field(default_factory=list)


def fetch_monitor_entry(user_input: str) -> MonitorEntry:
    """Fetch a compact snapshot for the multi-ticker monitor table."""
    from futures.data import fetch_futures_data
    from futures.levels import compute_levels
    from futures.volume import analyze_volume, volume_spike
    from futures.fvg import detect_fvgs
    from futures.bias import compute_bias, find_confluence_zones, suggest_pullback_zones
    from futures.ideas import generate_ideas
    from futures.atr import compute_ema, resample_to_15m
    from config.sessions import get_session_label
    from config.settings import load_settings
    from datetime import datetime, timezone

    yf_symbol, spec = resolve_ticker(user_input)
    display = spec["name"] if spec else yf_symbol
    errors: list[str] = []

    try:
        data = fetch_futures_data(user_input)
        settings = load_settings()
    except Exception as exc:
        return MonitorEntry(
            symbol=yf_symbol, display_name=display,
            price=None, daily_chg_pct=None,
            trend_5m="?", trend_15m="?", trend_1h="?",
            bias="?", confidence="?",
            above_vwap=None, session="?", atr_15m=None,
            atr_regime="?", vol_spike=False, nearest_level="", nearest_fvg_dir="", nearest_fvg_tf="", near_key_level=False,
            best_idea="", errors=[str(exc)],
        )

    try:
        levels = compute_levels(data)
    except Exception as exc:
        errors.append(f"Levels: {exc}")
        levels = None

    spot = data.spot
    atr_15m = levels.atr_15m if levels else None
    atr_5m  = levels.atr_5m  if levels else None
    trend_5m  = levels.trend_5m  if levels else "neutral"
    trend_15m = levels.trend_15m if levels else "neutral"
    vwap = levels.vwap if levels else None

    # 1h trend
    trend_1h = "neutral"
    if not data.intraday_1h.empty:
        try:
            close = data.intraday_1h["Close"]
            ema9  = compute_ema(close, 9).iloc[-1]
            ema21 = compute_ema(close, 21).iloc[-1]
            price_1h = close.iloc[-1]
            if price_1h > ema9 > ema21:
                trend_1h = "bullish"
            elif price_1h < ema9 < ema21:
                trend_1h = "bearish"
        except Exception:
            pass

    # Volume analytics
    vspike = False
    volume_analytics = None
    if not data.intraday_5m.empty:
        try:
            vspike = volume_spike(data.intraday_5m)
            volume_analytics = analyze_volume(data.intraday_5m, spot)
        except Exception:
            pass

    # FVGs
    fvgs_5m = fvgs_15m = fvgs_1h = []
    if spot is not None:
        try:
            if not data.intraday_5m.empty:
                fvgs_5m  = detect_fvgs(
                    data.intraday_5m, "5m", spot, atr_5m,
                    min_size_atr=settings.fvg_min_size_atr,
                )
            if not data.intraday_5m.empty:
                df15 = resample_to_15m(data.intraday_5m)
                fvgs_15m = detect_fvgs(
                    df15, "15m", spot, atr_15m,
                    min_size_atr=settings.fvg_min_size_atr,
                )
            if not data.intraday_1h.empty:
                fvgs_1h  = detect_fvgs(
                    data.intraday_1h, "1h", spot, atr_15m,
                    min_size_atr=settings.fvg_min_size_atr,
                )
        except Exception:
            pass

    all_fvgs = fvgs_5m + fvgs_15m + fvgs_1h

    # Bias
    bias_result = None
    try:
        bias_result = compute_bias(
            spot=spot, trend_5m=trend_5m, trend_15m=trend_15m, trend_1h=trend_1h,
            vwap=vwap, levels=levels,
            volume_profile=volume_analytics.volume_profile if volume_analytics else None,
            fvgs_5m=fvgs_5m, fvgs_15m=fvgs_15m, fvgs_1h=fvgs_1h, atr_15m=atr_15m,
        )
        bias = bias_result.bias
        conf = bias_result.confidence
    except Exception:
        bias, conf = "?", "?"

    # VWAP relationship
    above_vwap: bool | None = None
    if spot is not None and vwap is not None:
        above_vwap = spot > vwap

    # Session label
    session = ""
    try:
        now_et = datetime.now().astimezone()
        session = get_session_label(now_et.hour, now_et.minute)
    except Exception:
        pass

    # Daily change
    daily_chg: float | None = None
    try:
        if not data.daily_df.empty and len(data.daily_df) >= 2:
            prev = float(data.daily_df["Close"].iloc[-2])
            curr = float(data.daily_df["Close"].iloc[-1])
            daily_chg = (curr - prev) / prev * 100 if prev else None
    except Exception:
        pass

    # Nearest key level
    nearest_level = ""
    near_key_level = False
    if levels and spot is not None and atr_5m:
        named = {
            "PD High": levels.prev_day_high, "PD Low": levels.prev_day_low,
            "Asia High": levels.asia.high, "Asia Low": levels.asia.low,
            "London High": levels.london.high, "London Low": levels.london.low,
            "VWAP": levels.vwap,
        }
        closest = None
        closest_dist = float("inf")
        for name, lvl in named.items():
            if lvl is None:
                continue
            d = abs(spot - lvl)
            if d < closest_dist:
                closest_dist = d
                closest = f"{name} ({d:.4g} pts)"
        nearest_level = closest or ""
        near_key_level = closest_dist <= max(atr_5m * 0.5, 1e-9)

    # Nearest FVG
    nearest_fvg_dir = nearest_fvg_tf = ""
    if all_fvgs and spot is not None:
        nearest = min(all_fvgs, key=lambda f: f.dist_from_price)
        nearest_fvg_dir = nearest.direction
        nearest_fvg_tf  = nearest.timeframe

    best_idea = ""
    if bias_result and levels and spot is not None:
        try:
            confluence = find_confluence_zones(
                {
                    "Prev Day High": levels.prev_day_high,
                    "Prev Day Low": levels.prev_day_low,
                    "London High": levels.london.high,
                    "London Low": levels.london.low,
                    "VWAP": levels.vwap,
                    "POC": volume_analytics.volume_profile.poc if volume_analytics and volume_analytics.volume_profile else None,
                    "VAH": volume_analytics.volume_profile.vah if volume_analytics and volume_analytics.volume_profile else None,
                    "VAL": volume_analytics.volume_profile.val if volume_analytics and volume_analytics.volume_profile else None,
                },
                atr=atr_15m,
            )
            pullbacks = suggest_pullback_zones(
                bias_result, confluence, spot, atr_15m, all_fvgs,
                volume_analytics.volume_profile if volume_analytics else None,
            )
            ideas = generate_ideas(
                bias_result, levels, all_fvgs,
                volume_analytics.volume_profile if volume_analytics else None,
                confluence, pullbacks, spot, atr_5m, atr_15m,
            )
            if ideas:
                best_idea = ideas[0].setup_type
        except Exception:
            pass

    atr_regime = "unknown"
    if atr_15m and spot:
        pct = atr_15m / max(abs(spot), 1e-9)
        if pct >= 0.01:
            atr_regime = "expanding"
        elif pct >= 0.004:
            atr_regime = "normal"
        else:
            atr_regime = "compressed"

    return MonitorEntry(
        symbol=yf_symbol, display_name=display,
        price=spot, daily_chg_pct=daily_chg,
        trend_5m=trend_5m, trend_15m=trend_15m, trend_1h=trend_1h,
        bias=bias, confidence=conf,
        above_vwap=above_vwap, session=session, atr_15m=atr_15m,
        atr_regime=atr_regime, vol_spike=vspike, nearest_level=nearest_level,
        nearest_fvg_dir=nearest_fvg_dir, nearest_fvg_tf=nearest_fvg_tf,
        near_key_level=near_key_level, best_idea=best_idea, errors=errors,
    )


def build_watchlist(symbols: list[str], max_workers: int = 4) -> list[MonitorEntry]:
    """Fetch monitor data for all symbols in parallel."""
    entries: list[MonitorEntry] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_monitor_entry, s): s for s in symbols}
        for fut in as_completed(futures):
            try:
                entries.append(fut.result())
            except Exception as exc:
                sym = futures[fut]
                yf_sym, spec = resolve_ticker(sym)
                entries.append(MonitorEntry(
                    symbol=yf_sym, display_name=spec["name"] if spec else yf_sym,
                    price=None, daily_chg_pct=None,
                    trend_5m="?", trend_15m="?", trend_1h="?",
                    bias="err", confidence="",
                    above_vwap=None, session="", atr_15m=None,
                    atr_regime="?", vol_spike=False, nearest_level="", near_key_level=False,
                    nearest_fvg_dir="", nearest_fvg_tf="",
                    best_idea="", errors=[str(exc)],
                ))
    rank = {"low": 0, "medium": 1, "high": 2, "?": 3, "": 3}
    entries.sort(key=lambda e: (rank.get(e.confidence, 3), e.symbol))
    return entries
