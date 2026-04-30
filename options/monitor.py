from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from analyzer.data import cache_dir, get_fast_price, get_history, get_option_chain
from analyzer.formatting import format_percent, format_price
from analyzer.macro import fetch_macro_context
from analyzer.news import NewsItem, aggregate_sentiment, fetch_news
from config.settings import load_settings
from futures.fvg import detect_fvgs
from journal.storage import TradeDatabase
from options.snapshot import get_options_snapshot


_BASE_OPTIONS_WATCHLIST = ["SPY", "QQQ"]
_DYNAMIC_CANDIDATES = ["AAPL", "NVDA", "TSLA", "AMD", "META", "AMZN", "MSFT", "GOOGL", "NFLX", "JPM"]
_SECTOR_MAP = {
    "AAPL": "XLK",
    "NVDA": "SMH",
    "TSLA": "XLY",
    "AMD": "SMH",
    "META": "XLC",
    "AMZN": "XLY",
    "MSFT": "XLK",
    "GOOGL": "XLC",
    "NFLX": "XLC",
    "JPM": "XLF",
    "SPY": "QQQ",
    "QQQ": "SPY",
    "SPX": "SPY",
}
_MARKET_PROXY = {"SPX": "^GSPC"}


@dataclass
class SignalBucket:
    bias: str
    summary: str


@dataclass
class OptionsMonitorEntry:
    symbol: str
    price: float | None
    regime: str
    atm_iv: float | None
    rr_25d: float | None
    gex_regime: str
    confidence: str
    signal: str
    sentiment: str
    price_action: SignalBucket
    macro: SignalBucket
    correlated_assets: SignalBucket
    indicators: SignalBucket
    volume_flow: SignalBucket
    news_snippet: str
    news_sentiment_score: float | None
    unusual_options_activity: float | None
    iv_rank: float | None
    alignment_score: float | None
    gamma_flip: float | None = None
    max_gamma_strike: float | None = None
    max_pain: float | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class RecommendedSymbol:
    symbol: str
    score: float
    reason: str


def _market_symbol(symbol: str) -> str:
    return _MARKET_PROXY.get(symbol.upper(), symbol.upper())


def _iv_history_path() -> Path:
    return cache_dir() / "options_iv_history.json"


def _recommendation_cache_path() -> Path:
    return cache_dir() / "options_watchlist_recommendations.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _append_iv_history(symbol: str, atm_iv: float | None) -> None:
    if atm_iv is None:
        return
    path = _iv_history_path()
    payload = _load_json(path)
    history = payload.get(symbol, [])
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append({"ts": now, "atm_iv": float(atm_iv)})
    cutoff = datetime.now() - timedelta(days=90)
    trimmed = []
    for item in history[-500:]:
        try:
            if datetime.strptime(item["ts"], "%Y-%m-%d %H:%M:%S") >= cutoff:
                trimmed.append(item)
        except Exception:
            continue
    payload[symbol] = trimmed
    _save_json(path, payload)


def _iv_rank(symbol: str, atm_iv: float | None) -> float | None:
    if atm_iv is None:
        return None
    payload = _load_json(_iv_history_path())
    history = payload.get(symbol, [])
    values = [float(item["atm_iv"]) for item in history if item.get("atm_iv") is not None]
    if len(values) < 5:
        return None
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-9:
        return 50.0
    return max(0.0, min(100.0, (atm_iv - lo) / (hi - lo) * 100.0))


def _rsi(series: pd.Series, window: int = 14) -> float | None:
    if series is None or len(series) < window + 1:
        return None
    delta = series.diff()
    gains = delta.clip(lower=0).rolling(window).mean()
    losses = (-delta.clip(upper=0)).rolling(window).mean()
    if losses.iloc[-1] == 0:
        return 100.0
    rs = gains.iloc[-1] / losses.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def _macd_state(series: pd.Series) -> str:
    if len(series) < 35:
        return "insufficient data"
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-1] > signal.iloc[-1] > 0:
        return "bullish MACD confirmation"
    if macd.iloc[-1] < signal.iloc[-1] < 0:
        return "bearish MACD confirmation"
    return "MACD mixed"


def _volume_ratio(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or len(df) < 20:
        return None
    avg = float(df["Volume"].iloc[-20:-1].mean()) if len(df) >= 21 else float(df["Volume"].mean())
    if avg <= 0:
        return None
    return float(df["Volume"].iloc[-1]) / avg


def _compute_max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    strikes = sorted({float(v) for v in pd.concat([calls.get("strike", pd.Series(dtype=float)), puts.get("strike", pd.Series(dtype=float))]).dropna().tolist()})
    if not strikes:
        return None
    losses: list[tuple[float, float]] = []
    for spot in strikes:
        call_loss = 0.0
        put_loss = 0.0
        for _, row in calls.iterrows():
            strike = float(row.get("strike", 0) or 0)
            oi = float(row.get("openInterest", 0) or 0)
            call_loss += max(spot - strike, 0.0) * oi * 100.0
        for _, row in puts.iterrows():
            strike = float(row.get("strike", 0) or 0)
            oi = float(row.get("openInterest", 0) or 0)
            put_loss += max(strike - spot, 0.0) * oi * 100.0
        losses.append((spot, call_loss + put_loss))
    return min(losses, key=lambda item: item[1])[0] if losses else None


def _unusual_options_activity(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    combined = pd.concat([calls, puts], ignore_index=True)
    if combined.empty:
        return None
    volume = pd.to_numeric(combined.get("volume"), errors="coerce").fillna(0.0)
    oi = pd.to_numeric(combined.get("openInterest"), errors="coerce").fillna(0.0)
    total_oi = float(oi.sum())
    total_volume = float(volume.sum())
    if total_oi <= 0 and total_volume <= 0:
        return None
    ratio = total_volume / max(total_oi, 1.0)
    hot_contracts = float((volume > (oi * 1.2 + 50)).sum())
    return ratio + hot_contracts * 0.15


def _cointegration_context(symbol: str, benchmark: str) -> tuple[float | None, str]:
    try:
        left = get_history(_market_symbol(symbol), period="6mo", interval="1d", auto_adjust=True, ttl_seconds=900)
        right = get_history(_market_symbol(benchmark), period="6mo", interval="1d", auto_adjust=True, ttl_seconds=900)
        if left.empty or right.empty:
            return None, "correlation context unavailable"
        df = pd.DataFrame({
            "left": left["Close"],
            "right": right["Close"],
        }).dropna()
        if len(df) < 40:
            return None, "correlation context unavailable"
        corr = float(df["left"].pct_change().corr(df["right"].pct_change()))
        x = df["right"].to_numpy()
        y = df["left"].to_numpy()
        beta, alpha = np.polyfit(x, y, 1)
        residual = y - (beta * x + alpha)
        lagged = residual[:-1]
        delta = np.diff(residual)
        denom = np.dot(lagged, lagged)
        if len(delta) < 10 or denom <= 1e-9:
            return corr, f"{benchmark} corr {corr:+.2f}; spread test unavailable"
        phi = float(np.dot(lagged, delta) / denom)
        stable = phi < -0.03
        tag = "mean-reverting spread" if stable else "no clear spread stationarity"
        return corr, f"{benchmark} corr {corr:+.2f}; EG-style residual check: {tag}"
    except Exception:
        return None, "correlation context unavailable"


def _price_action_bucket(symbol: str, price: float | None, regime: str) -> SignalBucket:
    market_symbol = _market_symbol(symbol)
    try:
        intraday = get_history(market_symbol, period="1d", interval="5m", auto_adjust=True, ttl_seconds=120)
        if intraday.empty or price is None:
            return SignalBucket("neutral", "Intraday structure unavailable")
        high = float(intraday["High"].max())
        low = float(intraday["Low"].min())
        pos = ((price - low) / max(high - low, 1e-9)) * 100.0
        momentum = float(intraday["Close"].pct_change(6).iloc[-1] * 100) if len(intraday) >= 7 else 0.0
        atr = float((intraday["High"] - intraday["Low"]).rolling(14).mean().iloc[-1]) if len(intraday) >= 14 else None
        fvgs = detect_fvgs(intraday[["Open", "High", "Low", "Close", "Volume"]], "5m", price, atr=atr)
        if regime in {"TREND UP", "EXTENDED UP"} and pos > 65 and momentum > 0:
            bias = "bullish"
        elif regime in {"TREND DOWN", "EXTENDED DOWN"} and pos < 35 and momentum < 0:
            bias = "bearish"
        else:
            bias = "neutral"
        fvg_note = "active FVG nearby" if fvgs else "no active FVG nearby"
        return SignalBucket(
            bias,
            f"{regime}; {pos:.0f}% of session range; {momentum:+.2f}% short-term momentum; {fvg_note}",
        )
    except Exception:
        return SignalBucket("neutral", "Price-action read unavailable")


def _macro_bucket(news_items: list[NewsItem]) -> SignalBucket:
    macro_rows = fetch_macro_context()
    vix_row = next((row for row in macro_rows if row.label == "VIX"), None)
    yield_row = next((row for row in macro_rows if row.label == "10Y Yield"), None)
    sentiment, score = aggregate_sentiment(news_items)
    if vix_row and "risk-on" in vix_row.interpretation and sentiment != "bearish":
        bias = "bullish"
    elif vix_row and "risk-off" in vix_row.interpretation:
        bias = "bearish"
    else:
        bias = "neutral"
    parts = []
    if vix_row:
        parts.append(f"{vix_row.label} {vix_row.value} ({vix_row.interpretation})")
    if yield_row:
        parts.append(f"{yield_row.label} {yield_row.value} ({yield_row.interpretation})")
    parts.append(f"AI headline sentiment {sentiment} ({score:+.1f})")
    return SignalBucket(bias, "; ".join(parts))


def _correlated_bucket(symbol: str, price_action_bias: str) -> SignalBucket:
    benchmark = _SECTOR_MAP.get(symbol, "SPY")
    corr, summary = _cointegration_context(symbol, benchmark)
    if corr is None:
        return SignalBucket("neutral", summary)
    if corr > 0.7 and price_action_bias == "bullish":
        return SignalBucket("bullish", summary)
    if corr > 0.7 and price_action_bias == "bearish":
        return SignalBucket("bearish", summary)
    return SignalBucket("neutral", summary)


def _indicators_bucket(symbol: str, price: float | None) -> SignalBucket:
    try:
        hist = get_history(_market_symbol(symbol), period="6mo", interval="1d", auto_adjust=True, ttl_seconds=900)
        if hist.empty or price is None:
            return SignalBucket("neutral", "Daily indicator set unavailable")
        close = hist["Close"]
        sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        rsi = _rsi(close)
        macd_state = _macd_state(close)
        if sma20 and sma50 and price > sma20 > sma50 and (rsi is None or rsi < 72):
            bias = "bullish"
        elif sma20 and sma50 and price < sma20 < sma50 and (rsi is None or rsi > 28):
            bias = "bearish"
        else:
            bias = "neutral"
        pieces = [
            f"20d MA {format_price(sma20)}",
            f"50d MA {format_price(sma50)}",
            f"RSI {rsi:.1f}" if rsi is not None else "RSI unavailable",
            macd_state,
        ]
        return SignalBucket(bias, "; ".join(pieces))
    except Exception:
        return SignalBucket("neutral", "Daily indicator set unavailable")


def _flow_bucket(
    calls: pd.DataFrame | None,
    puts: pd.DataFrame | None,
    symbol: str,
    price: float | None,
    unusual_score: float | None,
    gamma_flip: float | None,
    max_gamma_strike: float | None,
    max_pain: float | None,
) -> SignalBucket:
    if calls is None or puts is None or calls.empty or puts.empty:
        return SignalBucket("neutral", "Options chain unavailable from current data provider")
    try:
        intraday = get_history(_market_symbol(symbol), period="5d", interval="1d", auto_adjust=True, ttl_seconds=900)
        vol_ratio = _volume_ratio(intraday)
    except Exception:
        vol_ratio = None
    pieces = []
    if vol_ratio is not None:
        pieces.append(f"vol {vol_ratio:.2f}x 20d avg")
    if unusual_score is not None:
        pieces.append(f"unusual options score {unusual_score:.2f}")
    if gamma_flip is not None and price is not None:
        direction = "above" if gamma_flip > price else "below"
        pieces.append(f"gamma flip {format_price(gamma_flip)} ({direction} spot)")
    if max_gamma_strike is not None:
        pieces.append(f"max gamma {format_price(max_gamma_strike)}")
    if max_pain is not None:
        pieces.append(f"max pain {format_price(max_pain)}")
    bias = "neutral"
    if gamma_flip is not None and price is not None:
        if price > gamma_flip:
            bias = "bullish"
        elif price < gamma_flip:
            bias = "bearish"
    return SignalBucket(bias, "; ".join(pieces) if pieces else "Flow context unavailable")


def _recent_trade_alignment(symbol: str) -> float | None:
    try:
        db = TradeDatabase(load_settings().db_path)
        trades = db.get_options_trades(underlying=symbol)
        if not trades:
            return 0.0
        recent = trades[:5]
        score = 0.0
        for trade in recent:
            score += 1.0
            if trade.get("setup_type") in {"Gamma Scalp", "Vol Trade", "Earnings Trade"}:
                score += 0.5
        return score
    except Exception:
        return None


def fetch_options_entry(symbol: str, t_days: int = 30) -> OptionsMonitorEntry:
    symbol = symbol.upper()
    errors: list[str] = []
    market_symbol = _market_symbol(symbol)
    snapshot = get_options_snapshot(market_symbol if symbol == "SPX" else symbol, t_days=t_days)
    price = snapshot.price or get_fast_price(market_symbol)
    news_items = fetch_news(market_symbol, max_items=4)
    news_sentiment, news_score = aggregate_sentiment(news_items)
    news_snippet = news_items[0].title[:140] if news_items else "No recent headline available"
    _append_iv_history(symbol, snapshot.atm_iv)
    iv_rank = _iv_rank(symbol, snapshot.atm_iv)

    calls = puts = None
    unusual_score = None
    gamma_flip = None
    max_gamma_strike = None
    max_pain = None
    if symbol != "SPX":
        try:
            chain = get_option_chain(symbol, t_days=t_days, ttl_seconds=180)
            calls = chain.calls
            puts = chain.puts
            unusual_score = _unusual_options_activity(calls, puts)
            max_pain = _compute_max_pain(calls, puts)
            combined = pd.concat([
                calls[["strike", "openInterest", "impliedVolatility"]].assign(side="call"),
                puts[["strike", "openInterest", "impliedVolatility"]].assign(side="put"),
            ], ignore_index=True)
            if price is not None and not combined.empty:
                combined = combined.copy()
                combined["distance"] = (pd.to_numeric(combined["strike"], errors="coerce") - price).abs()
                combined = combined.dropna(subset=["strike", "openInterest", "impliedVolatility", "distance"])
                if not combined.empty:
                    weighted = combined["openInterest"] * combined["impliedVolatility"]
                    idx = weighted.idxmax()
                    max_gamma_strike = float(combined.loc[idx, "strike"])
                    gamma_flip = float(combined.loc[combined["distance"].idxmin(), "strike"])
        except Exception as exc:
            errors.append(f"Options chain: {exc}")
    else:
        errors.append("SPX chain analytics unavailable from the current retail data feed; price context is shown via ^GSPC.")

    price_action = _price_action_bucket(symbol, price, snapshot.regime)
    macro = _macro_bucket(news_items)
    correlated = _correlated_bucket(symbol, price_action.bias)
    indicators = _indicators_bucket(symbol, price)
    volume_flow = _flow_bucket(calls, puts, symbol, price, unusual_score, gamma_flip, max_gamma_strike, max_pain)
    alignment_score = _recent_trade_alignment(symbol)

    confidence_rank = {"high": 2, "medium": 1, "low": 0}
    confidence = snapshot.confidence
    if alignment_score and alignment_score >= 2 and confidence_rank.get(confidence, 0) < 2:
        confidence = "high"

    return OptionsMonitorEntry(
        symbol=symbol,
        price=price,
        regime=snapshot.regime,
        atm_iv=snapshot.atm_iv,
        rr_25d=snapshot.rr_25d,
        gex_regime=snapshot.gex_regime,
        confidence=confidence,
        signal=snapshot.signal,
        sentiment=news_sentiment,
        price_action=price_action,
        macro=macro,
        correlated_assets=correlated,
        indicators=indicators,
        volume_flow=volume_flow,
        news_snippet=news_snippet,
        news_sentiment_score=news_score,
        unusual_options_activity=unusual_score,
        iv_rank=iv_rank,
        alignment_score=alignment_score,
        gamma_flip=gamma_flip,
        max_gamma_strike=max_gamma_strike,
        max_pain=max_pain,
        errors=list(snapshot.errors) + errors,
    )


def _recommendation_reason(entry: OptionsMonitorEntry) -> str:
    parts = []
    if entry.iv_rank is not None:
        parts.append(f"IV rank {entry.iv_rank:.0f}")
    if entry.unusual_options_activity is not None:
        parts.append(f"flow {entry.unusual_options_activity:.2f}")
    if entry.alignment_score:
        parts.append(f"idea alignment {entry.alignment_score:.1f}")
    return ", ".join(parts) if parts else entry.signal


def recommend_options_watchlist(t_days: int = 30, ttl_seconds: int = 900) -> list[str]:
    cache_path = _recommendation_cache_path()
    payload = _load_json(cache_path)
    now = datetime.now()
    cached_ts = payload.get("updated_at")
    if cached_ts:
        try:
            updated_at = datetime.strptime(cached_ts, "%Y-%m-%d %H:%M:%S")
            if now - updated_at <= timedelta(seconds=ttl_seconds):
                symbols = payload.get("symbols") or []
                if symbols:
                    return symbols
        except Exception:
            pass

    entries = build_options_watchlist(_DYNAMIC_CANDIDATES, t_days=t_days, max_workers=4)
    scored: list[RecommendedSymbol] = []
    for entry in entries:
        score = 0.0
        if entry.iv_rank is not None:
            score += entry.iv_rank / 25.0
        elif entry.atm_iv is not None:
            score += entry.atm_iv * 10.0
        if entry.unusual_options_activity is not None:
            score += min(entry.unusual_options_activity * 3.0, 5.0)
        if entry.alignment_score is not None:
            score += min(entry.alignment_score, 3.0)
        if entry.confidence == "high":
            score += 2.0
        elif entry.confidence == "medium":
            score += 1.0
        if entry.sentiment == "bullish" and entry.price_action.bias == "bullish":
            score += 0.5
        if entry.sentiment == "bearish" and entry.price_action.bias == "bearish":
            score += 0.5
        scored.append(RecommendedSymbol(entry.symbol, score, _recommendation_reason(entry)))

    scored.sort(key=lambda item: (-item.score, item.symbol))
    selected = _BASE_OPTIONS_WATCHLIST + [item.symbol for item in scored[:2] if item.symbol not in _BASE_OPTIONS_WATCHLIST]
    payload = {
        "updated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "symbols": selected,
        "reasons": {item.symbol: item.reason for item in scored[:5]},
    }
    _save_json(cache_path, payload)
    return selected


def build_options_watchlist(symbols: list[str], t_days: int = 30, max_workers: int = 4) -> list[OptionsMonitorEntry]:
    entries: list[OptionsMonitorEntry] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_options_entry, s, t_days): s for s in symbols}
        for fut in as_completed(futures):
            try:
                entries.append(fut.result())
            except Exception as exc:
                symbol = futures[fut]
                entries.append(
                    OptionsMonitorEntry(
                        symbol=symbol.upper(),
                        price=None,
                        regime="ERR",
                        atm_iv=None,
                        rr_25d=None,
                        gex_regime="ERR",
                        confidence="low",
                        signal="Unavailable",
                        sentiment="neutral",
                        price_action=SignalBucket("neutral", "Unavailable"),
                        macro=SignalBucket("neutral", "Unavailable"),
                        correlated_assets=SignalBucket("neutral", "Unavailable"),
                        indicators=SignalBucket("neutral", "Unavailable"),
                        volume_flow=SignalBucket("neutral", "Unavailable"),
                        news_snippet="No recent headline available",
                        news_sentiment_score=None,
                        unusual_options_activity=None,
                        iv_rank=None,
                        alignment_score=None,
                        errors=[str(exc)],
                    )
                )
    rank = {"low": 0, "medium": 1, "high": 2}
    entries.sort(key=lambda e: (rank.get(e.confidence, 0), e.symbol))
    return entries
