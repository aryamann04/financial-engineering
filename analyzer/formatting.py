from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


def infer_tick_size(value: float | None) -> float:
    if value is None:
        return 0.01
    abs_value = abs(float(value))
    if abs_value >= 1000:
        return 0.25
    if abs_value >= 200:
        return 0.1
    if abs_value >= 20:
        return 0.01
    if abs_value >= 1:
        return 0.01
    if abs_value >= 0.1:
        return 0.001
    if abs_value >= 0.01:
        return 0.0001
    return 0.000001


def _decimal_places(tick_size: float) -> int:
    text = f"{tick_size:.10f}".rstrip("0")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def format_price(
    value: float | int | None,
    *,
    tick_size: float | None = None,
    none_text: str = "N/A",
    prefix: str = "",
) -> str:
    if value is None:
        return none_text
    tick = tick_size if tick_size and tick_size > 0 else infer_tick_size(float(value))
    places = _decimal_places(tick)
    try:
        quant = Decimal(str(tick))
        rounded = Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        rounded = Decimal("0")
    text = f"{rounded:,.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{prefix}{text}"


def format_percent(
    value: float | None,
    *,
    decimals: int = 1,
    none_text: str = "N/A",
    suffix: str = "%",
    signed: bool = False,
) -> str:
    if value is None:
        return none_text
    sign = "+" if signed else ""
    return f"{value:{sign},.{decimals}f}{suffix}"
