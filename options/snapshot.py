from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from analyzer.data import cache_dir, get_fast_price, get_history, get_option_chain
from analyzer.news import aggregate_sentiment, fetch_news
from options.analytics.regime import detect_regime


@dataclass
class OptionsSnapshot:
    symbol: str
    price: float | None
    regime: str
    atm_iv: float | None
    rr_25d: float | None
    gex_regime: str
    confidence: str
    signal: str
    sentiment: str
    expiry: str = ""
    updated_at: str = ""
    errors: list[str] = field(default_factory=list)


def _snapshot_path() -> Path:
    return cache_dir() / "options_snapshots.json"


def _load_cache() -> dict[str, dict]:
    path = _snapshot_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict[str, dict]) -> None:
    path = _snapshot_path()
    path.write_text(json.dumps(cache, indent=2))


def _nearest_iv(df: pd.DataFrame, strike: float) -> float | None:
    if df.empty or "strike" not in df or "impliedVolatility" not in df:
        return None
    valid = df[df["impliedVolatility"] > 0]
    if valid.empty:
        return None
    idx = (valid["strike"] - strike).abs().idxmin()
    try:
        return float(valid.loc[idx, "impliedVolatility"])
    except Exception:
        return None


def _gex_regime(calls: pd.DataFrame, puts: pd.DataFrame, spot: float | None) -> tuple[str, str, str]:
    if spot is None or calls.empty or puts.empty:
        return "low", "No setup", "Balanced"

    band = max(abs(spot) * 0.03, 1.0)
    call_near = calls[(calls["strike"] >= spot - band) & (calls["strike"] <= spot + band)].copy()
    put_near = puts[(puts["strike"] >= spot - band) & (puts["strike"] <= spot + band)].copy()
    call_oi = float(call_near.get("openInterest", pd.Series(dtype=float)).fillna(0).sum()) if not call_near.empty else 0.0
    put_oi = float(put_near.get("openInterest", pd.Series(dtype=float)).fillna(0).sum()) if not put_near.empty else 0.0
    ratio = (call_oi + 1.0) / (put_oi + 1.0)

    if ratio >= 1.25:
        return "medium", "Call-side interest near spot", "Call-heavy"
    if ratio <= 0.80:
        return "medium", "Put-side interest near spot", "Put-heavy"
    return "low", "Balanced dealer positioning", "Balanced"


def _regime_for_symbol(symbol: str) -> str:
    try:
        intraday = get_history(symbol, period="1d", interval="5m", auto_adjust=True, ttl_seconds=90)
        if intraday.empty:
            return "N/A"
        return detect_regime(intraday[["Open", "High", "Low", "Close", "Volume"]]).get("regime", "N/A")
    except Exception:
        return "N/A"


def _build_snapshot(symbol: str, t_days: int) -> OptionsSnapshot:
    errors: list[str] = []
    price = get_fast_price(symbol)
    sentiment, _ = aggregate_sentiment(fetch_news(symbol, max_items=4))
    regime = _regime_for_symbol(symbol)

    try:
        chain = get_option_chain(symbol, t_days=t_days, ttl_seconds=180)
        calls = chain.calls
        puts = chain.puts
    except Exception as exc:
        return OptionsSnapshot(
            symbol=symbol,
            price=price,
            regime=regime,
            atm_iv=None,
            rr_25d=None,
            gex_regime="Unavailable",
            confidence="low",
            signal="Snapshot unavailable",
            sentiment=sentiment,
            updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            errors=[str(exc)],
        )

    atm_iv = None
    rr_25d = None
    if price is not None:
        call_atm = _nearest_iv(calls, price)
        put_atm = _nearest_iv(puts, price)
        if call_atm is not None and put_atm is not None:
            atm_iv = (call_atm + put_atm) / 2.0
        elif call_atm is not None:
            atm_iv = call_atm
        elif put_atm is not None:
            atm_iv = put_atm

        call_otm = _nearest_iv(calls, price * 1.05)
        put_otm = _nearest_iv(puts, price * 0.95)
        if call_otm is not None and put_otm is not None:
            rr_25d = call_otm - put_otm

    confidence, signal, gex_regime = _gex_regime(calls, puts, price)
    if regime in {"TREND UP", "EXTENDED UP"} and sentiment == "bullish":
        confidence = "high" if confidence == "medium" else "medium"
        signal = "Bullish continuation watch"
    elif regime in {"TREND DOWN", "EXTENDED DOWN"} and sentiment == "bearish":
        confidence = "high" if confidence == "medium" else "medium"
        signal = "Bearish continuation watch"
    elif regime == "CHOP":
        signal = "Range / mean-reversion watch"
    elif regime == "COMPRESSION":
        signal = "Compression, wait for expansion"

    if atm_iv is None:
        errors.append("ATM IV unavailable from current snapshot.")

    return OptionsSnapshot(
        symbol=symbol,
        price=price,
        regime=regime,
        atm_iv=atm_iv,
        rr_25d=rr_25d,
        gex_regime=gex_regime,
        confidence=confidence,
        signal=signal,
        sentiment=sentiment,
        expiry=chain.expiry,
        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        errors=errors,
    )


def get_options_snapshot(symbol: str, t_days: int = 30, ttl_seconds: int = 180, force_refresh: bool = False) -> OptionsSnapshot:
    symbol = symbol.upper().strip()
    cache = _load_cache()
    cached = cache.get(symbol)
    if not force_refresh and cached:
        try:
            updated_at = datetime.strptime(cached["updated_at"], "%Y-%m-%d %H:%M:%S")
            if datetime.now() - updated_at <= timedelta(seconds=ttl_seconds):
                return OptionsSnapshot(**cached)
        except Exception:
            pass

    snapshot = _build_snapshot(symbol, t_days)
    cache[symbol] = asdict(snapshot)
    _save_cache(cache)
    return snapshot
