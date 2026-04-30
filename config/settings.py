from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

_CONFIG_PATH = Path.home() / ".financial-engineering" / "config.json"


@dataclass
class Settings:
    # Watchlist for multi-ticker monitor
    watchlist: list[str] = field(
        default_factory=lambda: ["MES=F", "MNQ=F", "MBT=F", "M6E=F"]
    )
    futures_watchlist: list[str] = field(
        default_factory=lambda: ["MES=F", "MNQ=F", "MBT=F", "M6E=F"]
    )
    options_watchlist: list[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "AAPL"]
    )
    default_symbol: str = "MES=F"
    default_options_symbol: str = "SPY"
    yfinance_symbols: dict[str, str] = field(default_factory=dict)
    contract_multipliers: dict[str, float] = field(default_factory=dict)
    tick_values: dict[str, float] = field(default_factory=dict)

    # Session & data
    open_range_minutes: int = 30      # NY open range window in minutes
    atr_period: int = 14
    timezone: str = "America/New_York"
    session_windows: dict[str, list[int]] = field(default_factory=dict)
    default_chart_intervals: list[str] = field(default_factory=lambda: ["5m", "15m", "1h"])
    options_default_t_days: int = 30
    options_snapshot_ttl_seconds: int = 180

    # Volume profile
    volume_bins: int = 50
    value_area_pct: float = 0.70      # fraction of volume for value area (0.70 = 70%)
    vol_spike_threshold: float = 1.8  # relative volume multiple to flag as spike

    # Confluence & FVG
    confluence_tolerance_atr: float = 0.25   # levels within this × ATR form a zone
    fvg_min_size_atr: float = 0.12           # FVGs smaller than this × ATR are ignored

    # TUI
    use_tui: bool = True      # set False to always use simple menu
    color: bool = True
    tui_enabled: bool = True

    # Journal
    db_path: str = str(Path.home() / ".financial-engineering" / "trades.db")


def load_settings() -> Settings:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                data = json.load(f)
            s = Settings()
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return s
        except Exception:
            pass
    return Settings()


def save_settings(settings: Settings) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(asdict(settings), f, indent=2)


def settings_menu() -> None:
    """Interactive settings editor."""
    import os
    settings = load_settings()

    def _clear() -> None:
        os.system("clear" if os.name != "nt" else "cls")

    while True:
        _clear()
        print("\n  SETTINGS")
        print("  " + "─" * 42)
        d = asdict(settings)
        for i, (k, v) in enumerate(d.items(), 1):
            print(f"  {i:>2}. {k:<35}: {v}")
        print("\n   0. Save and exit")
        print("   x. Exit without saving")
        print("  " + "─" * 42)
        choice = input("  Edit field # (or 0/x): ").strip()

        if choice == "0":
            save_settings(settings)
            print("  Settings saved.")
            input("  Press Enter to continue...")
            return
        if choice.lower() == "x":
            return

        try:
            idx = int(choice) - 1
            keys = list(d.keys())
            if 0 <= idx < len(keys):
                key = keys[idx]
                cur = getattr(settings, key)
                new_raw = input(f"  New value for {key} [{cur}]: ").strip()
                if not new_raw:
                    continue
                if isinstance(cur, bool):
                    setattr(settings, key, new_raw.lower() in ("true", "1", "yes"))
                elif isinstance(cur, int):
                    setattr(settings, key, int(new_raw))
                elif isinstance(cur, float):
                    setattr(settings, key, float(new_raw))
                elif isinstance(cur, list):
                    setattr(settings, key, [s.strip() for s in new_raw.split(",")])
                else:
                    setattr(settings, key, new_raw)
        except (ValueError, IndexError):
            print("  Invalid choice.")
            input("  Press Enter to continue...")
