from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import requests

from config.settings import Settings

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

DEFAULT_SERIES = {
    "DGS10": "10Y Treasury",
    "DGS2": "2Y Treasury",
    "FEDFUNDS": "Fed Funds",
    "CPIAUCSL": "CPI",
    "UNRATE": "Unemployment",
    "VIXCLS": "VIX",
}


@dataclass(frozen=True)
class MacroPoint:
    series_id: str
    label: str
    value: float | None
    date: str | None
    source: str
    error: str | None = None


def _bucket(ttl_seconds: int) -> int:
    return int(time.time()) // max(int(ttl_seconds), 1)


@lru_cache(maxsize=128)
def _fetch_latest(series_id: str, api_key: str, bucket: int) -> MacroPoint:
    try:
        response = requests.get(
            FRED_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 12,
            },
            timeout=10,
        )
        response.raise_for_status()
        observations = response.json().get("observations", [])
        for obs in observations:
            value = obs.get("value")
            if value in {None, ".", ""}:
                continue
            return MacroPoint(
                series_id=series_id,
                label=DEFAULT_SERIES.get(series_id, series_id),
                value=float(value),
                date=obs.get("date"),
                source=f"based on cached FRED {series_id}",
            )
        return MacroPoint(series_id, DEFAULT_SERIES.get(series_id, series_id), None, None, f"based on cached FRED {series_id}", "No observations returned.")
    except Exception as exc:
        return MacroPoint(series_id, DEFAULT_SERIES.get(series_id, series_id), None, None, f"based on cached FRED {series_id}", str(exc))


def fetch_macro_points(settings: Settings, series_ids: list[str] | None = None, ttl_seconds: int = 1800) -> list[MacroPoint]:
    if not settings.fred_api_key:
        return [
            MacroPoint(series_id, DEFAULT_SERIES.get(series_id, series_id), None, None, "FRED unavailable", "Missing FRED_API_KEY.")
            for series_id in (series_ids or list(DEFAULT_SERIES))
        ]

    bucket = _bucket(ttl_seconds)
    points = [_fetch_latest(series_id, settings.fred_api_key, bucket) for series_id in (series_ids or list(DEFAULT_SERIES))]
    point_map = {point.series_id: point for point in points}
    if point_map.get("DGS10") and point_map.get("DGS2"):
        dgs10 = point_map["DGS10"].value
        dgs2 = point_map["DGS2"].value
        if dgs10 is not None and dgs2 is not None:
            points.append(
                MacroPoint(
                    "YC_SPREAD",
                    "10Y-2Y Spread",
                    dgs10 - dgs2,
                    point_map["DGS10"].date,
                    "derived from cached FRED DGS10 and DGS2",
                )
            )
    return points


def macro_context_dict(settings: Settings) -> dict[str, dict[str, str | float | None]]:
    context: dict[str, dict[str, str | float | None]] = {}
    for point in fetch_macro_points(settings):
        context[point.series_id] = {
            "label": point.label,
            "value": point.value,
            "date": point.date,
            "source": point.source,
            "error": point.error,
        }
    return context
