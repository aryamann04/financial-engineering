from __future__ import annotations

from dataclasses import dataclass

from config.settings import load_settings, save_settings
from options.monitor import recommend_options_watchlist


@dataclass
class Watchlists:
    options: list[str]
    futures: list[str]


def load_watchlists() -> Watchlists:
    settings = load_settings()
    futures = settings.futures_watchlist or settings.watchlist or ["MES=F"]
    saved_options = [s.upper() for s in (settings.options_watchlist or []) if s.strip() and s.upper() != "SPX"]
    legacy_defaults = {"SPY", "QQQ", "IWM", "AAPL"}
    if not saved_options or set(saved_options) == legacy_defaults:
        options = recommend_options_watchlist(t_days=settings.options_default_t_days)
    else:
        options = saved_options
    return Watchlists(
        options=[s.upper() for s in options if s.strip()],
        futures=[s.upper() for s in futures if s.strip()],
    )


def save_watchlists(options: list[str], futures: list[str]) -> None:
    settings = load_settings()
    settings.options_watchlist = [s.upper() for s in options if s.strip() and s.upper() != "SPX"]
    settings.futures_watchlist = [s.upper() for s in futures if s.strip()]
    settings.watchlist = list(settings.futures_watchlist)
    if settings.options_watchlist:
        settings.default_options_symbol = settings.options_watchlist[0]
    if settings.futures_watchlist:
        settings.default_symbol = settings.futures_watchlist[0]
    save_settings(settings)


def manage_watchlists() -> None:
    wl = load_watchlists()
    print("\nWatchlists")
    print(f"Options: {', '.join(wl.options) if wl.options else 'None'}")
    print(f"Futures: {', '.join(wl.futures) if wl.futures else 'None'}")
    options_raw = input("Options watchlist (comma-separated, blank keeps current): ").strip()
    futures_raw = input("Futures watchlist (comma-separated, blank keeps current): ").strip()
    if options_raw:
        wl.options = [s.strip().upper() for s in options_raw.split(",") if s.strip()]
    if futures_raw:
        wl.futures = [s.strip().upper() for s in futures_raw.split(",") if s.strip()]
    save_watchlists(wl.options, wl.futures)
