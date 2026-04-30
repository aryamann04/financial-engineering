from __future__ import annotations

from dataclasses import dataclass

from analyzer.data import get_history


@dataclass
class MacroIndicator:
    label: str
    value: str
    change: str
    interpretation: str


def _snapshot(symbol: str) -> tuple[float | None, float | None]:
    try:
        hist = get_history(symbol, period="5d", interval="1d", auto_adjust=True, ttl_seconds=300)
        if hist.empty:
            return None, None
        close = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else close
        chg = ((close - prev) / prev * 100) if prev else 0.0
        return close, chg
    except Exception:
        return None, None


def fetch_macro_context() -> list[MacroIndicator]:
    symbols = [
        ("^VIX", "VIX", "Fear gauge"),
        ("DX-Y.NYB", "DXY", "Dollar trend"),
        ("^TNX", "10Y Yield", "Rates pressure"),
        ("XLK", "Tech", "Growth leadership"),
        ("XLF", "Financials", "Credit / banks"),
    ]
    out: list[MacroIndicator] = []
    for symbol, label, interp in symbols:
        value, chg = _snapshot(symbol)
        if value is None:
            out.append(MacroIndicator(label, "N/A", "N/A", interp))
            continue
        if label == "VIX":
            read = "risk-on" if value < 18 else ("caution" if value < 24 else "risk-off")
            value_str = f"{value:.2f}"
        elif label == "10Y Yield":
            read = "rates easing" if (chg or 0) < -0.5 else ("rates pressing higher" if (chg or 0) > 0.5 else "stable")
            value_str = f"{value/10:.2f}%"
        else:
            read = "up" if (chg or 0) > 0.3 else ("down" if (chg or 0) < -0.3 else "flat")
            value_str = f"{value:.2f}"
        out.append(MacroIndicator(label, value_str, f"{chg:+.2f}%" if chg is not None else "N/A", f"{interp}: {read}"))
    return out
