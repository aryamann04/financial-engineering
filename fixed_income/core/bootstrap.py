"""
bootstrap.py — Treasury curve bootstrapping to zero-coupon yields.

Pipeline:
  1. fetch_treasury_curve()  → par yields (single network call, TTL-cached)
  2. build_instruments()     → par instruments built directly from that curve
  3. bootstrap()             → discount factors via iterative stripping
  4. full_disc_factors()     → interpolated monthly grid
  5. zero_coupon_yields()    → semi-annual ZC yields

Caching:
  get_zc_yields() / zc_yield() use a 10-minute TTL lru_cache so the full
  bootstrap pipeline runs at most once per TTL window.

  fetch_treasury_curve() itself is also TTL-cached in marketyields.py, so
  even if build_instruments() were to call treasury_yield() multiple times,
  each call would return instantly from the in-memory cache after the first.
"""

import time
import functools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import fsolve

from fixed_income.core.marketyields import (
    fetch_treasury_curve,
    treasury_yield,
    _FALLBACK_YIELDS,
)


# ---------------------------------------------------------------------------
# Curve interpolation helper (operates on a plain dict — no network calls)
# ---------------------------------------------------------------------------

def _interp_curve(curve: dict[float, float], t: float) -> float:
    """Linear interpolation of a {maturity: yield} dict.  No network calls."""
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


# ---------------------------------------------------------------------------
# Instrument construction
# ---------------------------------------------------------------------------

def par_price(y: float, c: float, T: float) -> float:
    """Theoretical par price of a coupon bond (zero-coupon convention for T < 1)."""
    if T < 1:
        return 100 / (1 + y * T)
    periods = int(T * 2)
    c_rate = c / 2
    cash_flows = np.full(periods, c_rate * 100)
    cash_flows[-1] += 100
    discounted = np.array([(1 + y / 2) ** (-n) for n in range(1, periods + 1)])
    return float((cash_flows * discounted).sum())


def build_instruments(curve: dict | None = None) -> list[dict]:
    """
    Build par instruments from Treasury yields.

    Accepts a pre-fetched `curve` dict to avoid any redundant network calls.
    If None, fetches once via fetch_treasury_curve() (which is TTL-cached).
    All interpolation is done directly on the dict — no per-tenor treasury_yield()
    calls, so no extra network round-trips regardless of how many tenors we need.
    """
    if curve is None:
        curve = fetch_treasury_curve()

    T_grid = [1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]
    return [
        {
            "maturity":     t,
            "coupon":       _interp_curve(curve, t),
            "market price": par_price(_interp_curve(curve, t),
                                      _interp_curve(curve, t), t),
        }
        for t in T_grid
    ]


def print_instruments(instruments: list[dict]) -> None:
    for i in instruments:
        print(
            f"Maturity: {i['maturity']:.2f}y  "
            f"Coupon: {i['coupon']*100:.3f}%  "
            f"Market Price: {i['market price']:.3f}"
        )


# ---------------------------------------------------------------------------
# Discount factor initialisation (short end — money-market ACT/360)
# ---------------------------------------------------------------------------

def init_disc_factors(instruments: list[dict],
                      curve: dict | None = None) -> dict[float, float]:
    """
    Bootstrap short-end discount factors using ACT/360 convention.
    Uses the passed `curve` (or fetches once) for yield lookup — no per-tenor
    calls to treasury_yield().
    """
    if curve is None:
        curve = fetch_treasury_curve()

    short_t = [1/12, 2/12, 3/12, 4/12, 6/12]
    D: dict[float, float] = {0.0: 1.0}
    for t in short_t:
        y = _interp_curve(curve, t)
        D[t] = 1.0 / (1 + y * (t * 365 / 360))
    return D


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

def interpolate_d(t: float, d_factors: dict[float, float],
                  method: str = "log-linear") -> float:
    t = float(t)
    if t <= 0:
        return 1.0
    if t in d_factors:
        return d_factors[t]
    known = sorted(d_factors.keys())
    t1 = max((k for k in known if k < t), default=None)
    t2 = min((k for k in known if k > t), default=None)
    if t1 is None or t2 is None:
        raise ValueError(f"Cannot interpolate for t={t}: insufficient discount factors.")
    D1, D2 = d_factors[t1], d_factors[t2]
    if method == "linear":
        return D1 + (t - t1) / (t2 - t1) * (D2 - D1)
    return D1 * (D2 / D1) ** ((t - t1) / (t2 - t1))


def bootstrap(instruments: list[dict],
              interpolation: str = "linear") -> dict[float, float]:
    # Re-use the curve already embedded in build_instruments output
    # init_disc_factors also uses the cached curve (no extra fetch)
    curve = fetch_treasury_curve()   # instant: TTL-cached
    d_factors = init_disc_factors(instruments, curve=curve)
    known_times = sorted(d_factors.keys())

    for inst in instruments:
        maturity = float(inst["maturity"])
        if maturity in d_factors:
            continue

        c_rate = inst["coupon"]
        price  = inst["market price"]
        periods = int(maturity * 2)

        cash_flows = np.array([c_rate / 2 * 100] * periods)
        cash_flows[-1] += 100
        times = [float(t) for t in np.arange(0.5, maturity + 0.1, 0.5)]

        def equation(d_tn, _mat=maturity, _times=times, _cf=cash_flows):
            d_temp = d_factors.copy()
            d_temp[_mat] = d_tn[0]
            pv = sum(
                _cf[i] * interpolate_d(_times[i], d_temp, interpolation)
                for i in range(len(_times))
            )
            return pv - price

        sol = fsolve(equation, [0.9])
        d_factors[maturity] = float(sol[0])
        known_times.append(maturity)
        known_times.sort()

    return d_factors


def full_disc_factors(d_factors: dict[float, float],
                      method: str = "log-linear") -> dict[float, float]:
    times = np.arange(1/12, 30 + 1/12, 1/12)
    return {
        t: d_factors[t] if t in d_factors else interpolate_d(t, d_factors, method)
        for t in times
    }


def zero_coupon_yields(d_factors: dict[float, float]) -> dict[float, float]:
    return {
        t: 2 * (d_factors[t] ** (-1 / (2 * t)) - 1)
        for t in d_factors
        if t > 0
    }


# ---------------------------------------------------------------------------
# TTL-cached public interface
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _get_zc_yields_cached(bucket: int) -> dict[float, float]:
    """Compute yield curve once per TTL bucket.  fetch_treasury_curve() is
    also TTL-cached, so the whole pipeline makes exactly one network call."""
    curve       = fetch_treasury_curve()          # instant on cache hit
    instruments = build_instruments(curve)         # no extra fetches
    return zero_coupon_yields(full_disc_factors(bootstrap(instruments)))


def get_zc_yields(ttl_seconds: int = 600) -> dict[float, float]:
    """ZC yields, recomputed at most once per ttl_seconds window."""
    bucket = int(time.time()) // ttl_seconds
    return _get_zc_yields_cached(bucket)


def zc_yield(t: float, ttl_seconds: int = 600) -> float:
    """
    Zero-coupon yield (decimal) closest to maturity t (years).
    Guaranteed non-crashing: falls back to hardcoded rate if yields unavailable.
    """
    try:
        yields = get_zc_yields(ttl_seconds)
        if yields:
            return yields[min(yields, key=lambda tau: abs(tau - t))]
    except Exception:
        pass
    # Last-resort fallback: interpolate hardcoded par yields
    return _interp_curve(_FALLBACK_YIELDS, t)


def get_disc_factors() -> dict[float, float]:
    curve = fetch_treasury_curve()
    return full_disc_factors(bootstrap(build_instruments(curve)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_zc_yields() -> None:
    yields = get_zc_yields()
    curve  = fetch_treasury_curve()

    fig, ax = plt.subplots(figsize=(10, 6))
    today = datetime.today()

    maturities = sorted(yields.keys())
    zc_dates = [today + timedelta(days=int(t * 365)) for t in maturities]
    zc_vals  = [yields[t] * 100 for t in maturities]

    T_grid  = sorted(curve.keys())
    t_dates = [today + timedelta(days=int(t * 365)) for t in T_grid]
    t_vals  = [curve[t] * 100 for t in T_grid]

    ax.plot(zc_dates, zc_vals, "-o", color="orange", label="bootstrapped ZC yields")
    ax.plot(t_dates,  t_vals,  "-o", color="blue",   label="Treasury par yields")

    ax.xaxis.set_major_locator(
        mdates.YearLocator(base=1, month=today.month, day=today.day)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Maturity date")
    ax.set_ylabel("Yield (%)")
    ax.set_title(
        f"Bootstrapped ZC vs Treasury Par Yields  —  {today.strftime('%Y-%m-%d')}"
    )
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
