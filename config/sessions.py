from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    crosses_midnight: bool = False


SESSIONS: dict[str, SessionWindow] = {
    "asia":          SessionWindow("Asia",             18,  0,  3,  0, crosses_midnight=True),
    "london":        SessionWindow("London",            3,  0,  8, 30),
    "new_york":      SessionWindow("New York",          8, 30, 16,  0),
    "ny_open_range": SessionWindow("NY Open Range",     9, 30, 10,  0),
    "cme_open_range":SessionWindow("CME Open Range",    8, 30,  9,  0),
    "ny_morning":    SessionWindow("NY Morning",        9, 30, 12,  0),
    "ny_afternoon":  SessionWindow("NY Afternoon",     12,  0, 16,  0),
}


def filter_session_bars(
    df: pd.DataFrame,
    session: SessionWindow,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """
    Return rows of df that fall within the given session window.
    df.index must be timezone-aware (or will be treated as UTC).
    reference_date is the calendar date used as "today" for non-midnight-crossing sessions.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert(ET)

    today = reference_date if reference_date is not None else idx_et[-1].date()
    yesterday = today - timedelta(days=1)

    dates_et = pd.Series(idx_et.date, index=df.index)
    bar_mins  = pd.Series(idx_et.hour * 60 + idx_et.minute, index=df.index)

    s_min = session.start_hour * 60 + session.start_minute
    e_min = session.end_hour   * 60 + session.end_minute

    if session.crosses_midnight:
        mask = (
            ((dates_et == yesterday) & (bar_mins >= s_min)) |
            ((dates_et == today)     & (bar_mins <  e_min))
        )
    else:
        mask = (
            (dates_et == today) &
            (bar_mins >= s_min) &
            (bar_mins <  e_min)
        )

    result = df[mask.values].copy()
    result.index = idx_et[mask.values]
    return result


def get_session_label(hour: int, minute: int) -> str:
    """Return which named session a given ET hour:minute falls in."""
    t = hour * 60 + minute
    if t >= 18 * 60 or t < 3 * 60:
        return "Asia"
    if t < 8 * 60 + 30:
        return "London"
    if t < 16 * 60:
        return "New York"
    return "Post-Market"


def get_time_of_day_label(hour: int, minute: int) -> str:
    t = hour * 60 + minute
    if t < 9 * 60 + 30:
        return "Pre-Market"
    if t < 12 * 60:
        return "Morning"
    if t < 16 * 60:
        return "Afternoon"
    return "After-Hours"
