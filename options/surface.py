from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import json

import numpy as np
import pandas as pd

from analyzer.data import cache_dir, get_fast_price, get_history, get_ticker
from analyzer.formatting import format_percent
from config.settings import load_settings
from fixed_income.core.bootstrap import zc_yield
from options.analytics.regime import detect_regime
from options.volatility.iv import bs_iv
from options.volatility.marketvols import _mid_price
from options.volatility.svi import SVI


def _bucket(ttl_seconds: int) -> int:
    return int(datetime.now().timestamp()) // max(int(ttl_seconds), 1)


@lru_cache(maxsize=128)
def _expiry_chain_cached(symbol: str, expiry: str, bucket: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker = get_ticker(symbol)
    chain = ticker.option_chain(expiry)
    return chain.calls.copy(), chain.puts.copy()


@dataclass
class ChainSnapshot:
    symbol: str
    expiry: str
    maturity: float
    spot: float
    rate: float
    dividend_yield: float
    calls: pd.DataFrame
    puts: pd.DataFrame
    svi_call: SVI | None
    svi_put: SVI | None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VolRegime:
    label: str
    iv_rank: float | None
    rv_iv_ratio: float | None
    term_structure: str
    skew: str
    rationale: list[str]


def available_expiries(symbol: str) -> list[str]:
    ticker = get_ticker(symbol)
    return list(ticker.options or [])


def _time_to_expiry(expiry: str) -> float:
    return max((datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days / 365.0, 1 / 365.0)


def _normalize_chain(df: pd.DataFrame, spot: float, maturity: float, rate: float, dividend_yield: float, option_type: str) -> pd.DataFrame:
    working = df.copy()
    for column in ("strike", "bid", "ask", "lastPrice", "openInterest", "volume", "impliedVolatility"):
        if column in working:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working["mid"] = working.apply(_mid_price, axis=1)
    working["spread"] = (working["ask"].fillna(0.0) - working["bid"].fillna(0.0)).clip(lower=0.0)
    working["spread_pct"] = np.where(working["mid"] > 0, working["spread"] / working["mid"], np.nan)
    working["bs_iv"] = working.apply(
        lambda row: bs_iv(float(row["mid"]), spot, float(row["strike"]), maturity, rate, dividend_yield, option_type) if pd.notna(row.get("mid")) and float(row["mid"]) > 0 else np.nan,
        axis=1,
    )
    working["liquidity_score"] = (
        np.log1p(working.get("openInterest", pd.Series(dtype=float)).fillna(0.0))
        + np.log1p(working.get("volume", pd.Series(dtype=float)).fillna(0.0))
        - (working["spread_pct"].fillna(1.0) * 4.0)
    )
    working["option_type"] = option_type
    working = working.replace([np.inf, -np.inf], np.nan)
    return working


def load_chain_snapshot(symbol: str, expiry: str, ttl_seconds: int = 180) -> ChainSnapshot:
    settings = load_settings()
    spot = get_fast_price(symbol) or 0.0
    maturity = _time_to_expiry(expiry)
    rate = zc_yield(maturity)
    ticker = get_ticker(symbol)
    dividend_yield = 0.0
    try:
        dividend_yield = float(ticker.info.get("dividendYield", 0.0) or 0.0)
        if dividend_yield > 0.20:
            dividend_yield /= 100.0
    except Exception:
        dividend_yield = 0.0

    warnings: list[str] = []
    calls_raw, puts_raw = _expiry_chain_cached(symbol.upper(), expiry, _bucket(ttl_seconds))
    calls = _normalize_chain(calls_raw, spot, maturity, rate, dividend_yield, "call")
    puts = _normalize_chain(puts_raw, spot, maturity, rate, dividend_yield, "put")

    def _svi(df: pd.DataFrame, option_type: str) -> SVI | None:
        valid = df[["strike", "bs_iv"]].dropna()
        if len(valid) < 5:
            warnings.append(f"{option_type} surface has too few valid IV points for SVI.")
            return None
        model = SVI(spot, maturity, rate, dividend_yield, option_type, strikes=valid["strike"].to_numpy(dtype=float), implied_vols=valid["bs_iv"].to_numpy(dtype=float))
        warnings.extend(model.warnings)
        return model

    return ChainSnapshot(
        symbol=symbol.upper(),
        expiry=expiry,
        maturity=maturity,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        calls=calls,
        puts=puts,
        svi_call=_svi(calls, "call"),
        svi_put=_svi(puts, "put"),
        warnings=warnings,
    )


def nearest_expiries(symbol: str, count: int = 2) -> list[str]:
    expiries = available_expiries(symbol)
    today = datetime.today()
    return sorted(expiries, key=lambda value: abs(datetime.strptime(value, "%Y-%m-%d") - today))[:count]


def _iv_history_path():
    return cache_dir() / "options_iv_history.json"


def record_atm_iv(symbol: str, atm_iv: float | None) -> None:
    if atm_iv is None:
        return
    path = _iv_history_path()
    try:
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            payload = {}
    except Exception:
        payload = {}
    history = payload.get(symbol.upper(), [])
    history.append({"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "atm_iv": float(atm_iv)})
    payload[symbol.upper()] = history[-500:]
    path.write_text(json.dumps(payload, indent=2))


def iv_rank(symbol: str, current_iv: float | None) -> float | None:
    if current_iv is None:
        return None
    path = _iv_history_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        history = payload.get(symbol.upper(), [])
        values = [float(item["atm_iv"]) for item in history if "atm_iv" in item]
    except Exception:
        return None
    if len(values) < 5:
        return None
    low, high = min(values), max(values)
    if high - low <= 1e-9:
        return 50.0
    return max(0.0, min(100.0, ((current_iv - low) / (high - low)) * 100.0))


def atm_iv(snapshot: ChainSnapshot) -> float | None:
    if snapshot.spot <= 0:
        return None
    call_row = snapshot.calls.iloc[(snapshot.calls["strike"] - snapshot.spot).abs().argsort().iloc[0]] if not snapshot.calls.empty else None
    put_row = snapshot.puts.iloc[(snapshot.puts["strike"] - snapshot.spot).abs().argsort().iloc[0]] if not snapshot.puts.empty else None
    ivs = []
    for row in (call_row, put_row):
        if row is not None and pd.notna(row.get("bs_iv")) and float(row["bs_iv"]) > 0:
            ivs.append(float(row["bs_iv"]))
    return float(np.mean(ivs)) if ivs else None


def realized_vs_implied(symbol: str, current_iv: float | None) -> float | None:
    if current_iv is None or current_iv <= 0:
        return None
    try:
        intraday = get_history(symbol, period="1d", interval="5m", auto_adjust=True, ttl_seconds=120)
        if intraday.empty:
            return None
        regime = detect_regime(intraday[["Open", "High", "Low", "Close", "Volume"]], atm_iv=current_iv)
        return regime.get("rv_iv_ratio")
    except Exception:
        return None


def term_structure_label(symbol: str, expiries: list[str]) -> tuple[str, list[str]]:
    if len(expiries) < 2:
        return "insufficient", ["Only one expiry was available."]
    notes: list[str] = []
    ivs: list[float] = []
    for expiry in expiries[:2]:
        snapshot = load_chain_snapshot(symbol, expiry)
        value = atm_iv(snapshot)
        if value is not None:
            ivs.append(value)
            notes.append(f"{expiry}: ATM IV {format_percent(value * 100, decimals=1)}")
    if len(ivs) < 2:
        return "insufficient", notes or ["Could not read ATM IV from nearby expiries."]
    if ivs[0] > ivs[1] * 1.05:
        return "backwardation", notes
    if ivs[1] > ivs[0] * 1.05:
        return "contango", notes
    return "flat", notes


def classify_vol_regime(symbol: str, snapshot: ChainSnapshot) -> VolRegime:
    current_iv = atm_iv(snapshot)
    record_atm_iv(symbol, current_iv)
    rank = iv_rank(symbol, current_iv)
    rv_iv = realized_vs_implied(symbol, current_iv)
    expiries = nearest_expiries(symbol, count=2)
    term_structure, term_notes = term_structure_label(symbol, expiries)

    skew_value = None
    skew_note = "skew unavailable"
    try:
        otm_put = snapshot.puts.iloc[(snapshot.puts["strike"] - snapshot.spot * 0.95).abs().argsort().iloc[0]]
        otm_call = snapshot.calls.iloc[(snapshot.calls["strike"] - snapshot.spot * 1.05).abs().argsort().iloc[0]]
        if pd.notna(otm_put.get("bs_iv")) and pd.notna(otm_call.get("bs_iv")):
            skew_value = float(otm_put["bs_iv"]) - float(otm_call["bs_iv"])
            skew_note = "put skew rich" if skew_value > 0.02 else "call skew rich" if skew_value < -0.02 else "balanced skew"
    except Exception:
        pass

    rationale = [*term_notes]
    if rank is not None:
        rationale.append(f"IV rank {rank:.0f}")
    if rv_iv is not None:
        rationale.append(f"RV/IV {rv_iv:.2f}")
    rationale.append(skew_note)

    if rank is not None and rank >= 70:
        label = "high_iv"
    elif rank is not None and rank <= 30:
        label = "low_iv"
    elif rv_iv is not None and rv_iv > 1.1:
        label = "realized_breakout"
    else:
        label = "balanced"
    return VolRegime(label=label, iv_rank=rank, rv_iv_ratio=rv_iv, term_structure=term_structure, skew=skew_note, rationale=rationale)
