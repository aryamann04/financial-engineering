from __future__ import annotations

import functools
import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import requests
from matplotlib.ticker import FuncFormatter


_FIELD_MAP: dict[str, float] = {
    "BC_1MONTH": 1 / 12,
    "BC_2MONTH": 2 / 12,
    "BC_3MONTH": 3 / 12,
    "BC_4MONTH": 4 / 12,
    "BC_6MONTH": 6 / 12,
    "BC_1YEAR": 1.0,
    "BC_2YEAR": 2.0,
    "BC_3YEAR": 3.0,
    "BC_5YEAR": 5.0,
    "BC_7YEAR": 7.0,
    "BC_10YEAR": 10.0,
    "BC_20YEAR": 20.0,
    "BC_30YEAR": 30.0,
}

_FALLBACK_YIELDS: dict[float, float] = {
    1 / 12: 0.043,
    2 / 12: 0.043,
    3 / 12: 0.043,
    4 / 12: 0.043,
    6 / 12: 0.042,
    1.0: 0.041,
    2.0: 0.039,
    3.0: 0.038,
    5.0: 0.039,
    7.0: 0.040,
    10.0: 0.043,
    20.0: 0.047,
    30.0: 0.046,
}

_FRED_TENORS: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 3 / 12,
    "DGS6MO": 6 / 12,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".treasury_cache.json")
_LIVE_TTL = 600
_MAX_CACHE_AGE_SECONDS = 60 * 60 * 24 * 7
_DATA_NS = "http://schemas.microsoft.com/ado/2007/08/dataservices"
_TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml?data=daily_treasury_yield_curve"
    "&field_tdr_date_value={yyyymm}"
)
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/xml,text/xml,application/json,*/*",
}


@dataclass(frozen=True)
class TreasuryCurveSnapshot:
    curve: dict[float, float]
    source: str
    fetched_at: str
    cache_age_seconds: int = 0
    is_stale: bool = False
    error_message: str | None = None
    details: list[str] = field(default_factory=list)


_LAST_STATUS: TreasuryCurveSnapshot | None = None


def _build_snapshot(
    curve: dict[float, float],
    *,
    source: str,
    fetched_at: str | None = None,
    cache_age_seconds: int = 0,
    is_stale: bool = False,
    error_message: str | None = None,
    details: list[str] | None = None,
) -> TreasuryCurveSnapshot:
    return TreasuryCurveSnapshot(
        curve=dict(curve),
        source=source,
        fetched_at=fetched_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        cache_age_seconds=cache_age_seconds,
        is_stale=is_stale,
        error_message=error_message,
        details=list(details or []),
    )


def _parse_treasury_xml(xml_text: str) -> dict[float, float] | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    entries = list(root.iter("{http://www.w3.org/2005/Atom}entry")) or list(root.iter("entry"))
    entries.reverse()

    for entry in entries:
        curve: dict[float, float] = {}
        for field_name, maturity in _FIELD_MAP.items():
            el = entry.find(f".//{{{_DATA_NS}}}{field_name}") or entry.find(f".//{field_name}")
            if el is None or not el.text:
                continue
            try:
                val = float(el.text.strip())
            except (TypeError, ValueError):
                continue
            if 0 < val < 30:
                curve[maturity] = val / 100.0
        if len(curve) >= 5:
            return curve
    return None


def _business_days_back(n: int):
    today = datetime.today()
    count = 0
    delta = 0
    while count < n:
        candidate = today - timedelta(days=delta)
        delta += 1
        if candidate.weekday() < 5:
            yield candidate
            count += 1


def _series_latest_from_fred(series_id: str, api_key: str, timeout: int = 4) -> float | None:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 10,
    }
    resp = requests.get(_FRED_URL, params=params, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    observations = payload.get("observations", [])
    for obs in observations:
        raw = obs.get("value")
        if raw in (None, "."):
            continue
        val = float(raw)
        if 0 < val < 30:
            return val / 100.0
    return None


def _fetch_fred_snapshot() -> TreasuryCurveSnapshot | None:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None

    curve: dict[float, float] = {}
    errors: list[str] = []
    for series_id, maturity in _FRED_TENORS.items():
        try:
            value = _series_latest_from_fred(series_id, api_key)
        except requests.Timeout:
            errors.append(f"FRED timed out while loading {series_id}.")
            continue
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            if status in {401, 403}:
                errors.append("FRED rejected the API key while loading Treasury yields.")
                break
            errors.append(f"FRED returned HTTP {status} for {series_id}.")
            continue
        except requests.RequestException:
            errors.append(f"FRED network error while loading {series_id}.")
            continue
        except (TypeError, ValueError):
            errors.append(f"FRED returned unreadable data for {series_id}.")
            continue
        if value is not None:
            curve[maturity] = value
        else:
            errors.append(f"FRED returned no usable observation for {series_id}.")

    if len(curve) >= 5:
        return _build_snapshot(
            curve,
            source="FRED",
            details=errors,
            error_message=None if len(curve) == len(_FRED_TENORS) else "FRED loaded a partial Treasury curve; missing tenors were omitted.",
        )
    if errors:
        return _build_snapshot({}, source="FRED", error_message="FRED Treasury yield fetch failed.", details=errors)
    return None


def _fetch_treasury_snapshot() -> TreasuryCurveSnapshot | None:
    errors: list[str] = []
    seen_months: set[str] = set()
    for date in _business_days_back(5):
        yyyymm = date.strftime("%Y%m")
        if yyyymm in seen_months:
            continue
        seen_months.add(yyyymm)
        url = _TREASURY_URL.format(yyyymm=yyyymm)
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=4)
            resp.raise_for_status()
            curve = _parse_treasury_xml(resp.text)
        except requests.Timeout:
            errors.append(f"Treasury XML feed timed out for {yyyymm}.")
            continue
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            errors.append(f"Treasury XML feed returned HTTP {status} for {yyyymm}.")
            continue
        except requests.RequestException:
            errors.append(f"Treasury XML feed network error for {yyyymm}.")
            continue
        if curve:
            return _build_snapshot(curve, source="Treasury.gov", details=errors)
        errors.append(f"Treasury XML feed returned no usable curve for {yyyymm}.")

    if errors:
        return _build_snapshot({}, source="Treasury.gov", error_message="Treasury.gov yield fetch failed.", details=errors)
    return None


def _load_cache_snapshot() -> TreasuryCurveSnapshot | None:
    if not os.path.exists(_CACHE_PATH):
        return None
    try:
        with open(_CACHE_PATH) as fh:
            data = json.load(fh)
        curve = {float(k): float(v) for k, v in data.get("yields", {}).items()}
        fetched_at = data.get("fetched_at") or data.get("date") or ""
        source = data.get("source", "cache")
        parsed = datetime.strptime(fetched_at[:19], "%Y-%m-%d %H:%M:%S") if fetched_at else None
        age = int((datetime.now() - parsed).total_seconds()) if parsed else 0
        is_stale = age > _LIVE_TTL
        return _build_snapshot(
            curve,
            source=source,
            fetched_at=fetched_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cache_age_seconds=max(age, 0),
            is_stale=is_stale,
            error_message="Using cached Treasury curve because live sources failed.",
        )
    except Exception:
        return None


def _save_cache(snapshot: TreasuryCurveSnapshot) -> None:
    try:
        payload = {
            "fetched_at": snapshot.fetched_at,
            "source": snapshot.source,
            "yields": {str(k): v for k, v in snapshot.curve.items()},
        }
        with open(_CACHE_PATH, "w") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass


@functools.lru_cache(maxsize=8)
def _fetch_treasury_curve_cached(bucket: int) -> TreasuryCurveSnapshot:
    del bucket
    statuses: list[str] = []

    fred_snapshot = _fetch_fred_snapshot()
    if fred_snapshot and fred_snapshot.curve:
        _save_cache(fred_snapshot)
        return fred_snapshot
    if fred_snapshot and fred_snapshot.error_message:
        statuses.extend([fred_snapshot.error_message, *fred_snapshot.details])

    treasury_snapshot = _fetch_treasury_snapshot()
    if treasury_snapshot and treasury_snapshot.curve:
        _save_cache(treasury_snapshot)
        return treasury_snapshot
    if treasury_snapshot and treasury_snapshot.error_message:
        statuses.extend([treasury_snapshot.error_message, *treasury_snapshot.details])

    cached = _load_cache_snapshot()
    if cached and cached.curve:
        age = cached.cache_age_seconds
        if age <= _MAX_CACHE_AGE_SECONDS:
            return _build_snapshot(
                cached.curve,
                source=f"{cached.source} cache",
                fetched_at=cached.fetched_at,
                cache_age_seconds=age,
                is_stale=True,
                error_message=(
                    f"Live Treasury data failed, so the dashboard is using cached {cached.source} yields "
                    f"from {cached.fetched_at} ({age // 60} minutes old)."
                ),
                details=statuses,
            )

    return _build_snapshot(
        _FALLBACK_YIELDS,
        source="hardcoded fallback",
        is_stale=True,
        error_message=(
            "Live Treasury data failed and no recent cache was available. "
            "The dashboard is showing hardcoded fallback yields for continuity only."
        ),
        details=statuses or ["No usable live or cached Treasury curve was available."],
    )


def get_treasury_curve_snapshot(force_refresh: bool = False) -> TreasuryCurveSnapshot:
    global _LAST_STATUS
    if force_refresh:
        _fetch_treasury_curve_cached.cache_clear()
    bucket = int(time.time()) // _LIVE_TTL
    snapshot = _fetch_treasury_curve_cached(bucket)
    _LAST_STATUS = snapshot
    return snapshot


def fetch_treasury_curve() -> dict[float, float]:
    return dict(get_treasury_curve_snapshot().curve)


def last_treasury_curve_status() -> TreasuryCurveSnapshot:
    return _LAST_STATUS or get_treasury_curve_snapshot()


def treasury_curve_status_message() -> str | None:
    snapshot = last_treasury_curve_status()
    return snapshot.error_message


def treasury_yield(t: float) -> float:
    curve = fetch_treasury_curve()
    maturities = sorted(curve.keys())
    if not maturities:
        return _FALLBACK_YIELDS.get(3 / 12, 0.043)
    if t <= maturities[0]:
        return curve[maturities[0]]
    if t >= maturities[-1]:
        return curve[maturities[-1]]
    t1 = max(m for m in maturities if m <= t)
    t2 = min(m for m in maturities if m > t)
    return curve[t1] + (t - t1) / (t2 - t1) * (curve[t2] - curve[t1])


def plot_yield_curve() -> None:
    snapshot = get_treasury_curve_snapshot()
    curve = snapshot.curve
    maturities = sorted(curve.keys())
    yields_pct = [curve[m] * 100 for m in maturities]
    labels = ["1M", "2M", "3M", "4M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]

    today = datetime.today().strftime("%Y-%m-%d")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(maturities)), yields_pct, marker="o")
    plt.xticks(range(len(maturities)), labels[: len(maturities)], rotation=45)
    title = f"Treasury Par Yield Curve  —  {today} ({snapshot.source})"
    plt.title(title)
    plt.xlabel("Maturity")
    plt.ylabel("Yield")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}%"))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
