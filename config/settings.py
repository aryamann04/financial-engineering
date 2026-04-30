from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency in some environments
    load_dotenv = None

_CONFIG_PATH = Path.home() / ".financial-engineering" / "config.json"


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


@dataclass
class Settings:
    # Watchlists
    watchlist: list[str] = field(default_factory=lambda: ["MES=F", "MNQ=F", "MBT=F", "M6E=F"])
    futures_watchlist: list[str] = field(default_factory=lambda: ["MES=F", "MNQ=F", "MBT=F", "M6E=F"])
    options_watchlist: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM", "AAPL"])
    default_symbol: str = "MES=F"
    default_options_symbol: str = "SPY"
    yfinance_symbols: dict[str, str] = field(default_factory=dict)
    contract_multipliers: dict[str, float] = field(default_factory=dict)
    tick_values: dict[str, float] = field(default_factory=dict)

    # Session & data
    open_range_minutes: int = 30
    atr_period: int = 14
    timezone: str = "America/New_York"
    session_windows: dict[str, list[int]] = field(default_factory=dict)
    default_chart_intervals: list[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    options_default_t_days: int = 30
    options_snapshot_ttl_seconds: int = 180
    default_refresh_seconds: int = 15
    recent_intraday_days: int = 3
    intraday_lookback_bars: int = 240
    data_dir: str = str(Path.home() / ".financial-engineering" / "data")

    # Secrets / integrations
    fred_api_key: str = ""
    openai_api_key: str = ""

    # Volume profile
    volume_bins: int = 50
    value_area_pct: float = 0.70
    vol_spike_threshold: float = 1.8

    # Confluence & FVG
    confluence_tolerance_atr: float = 0.25
    fvg_min_size_atr: float = 0.12

    # UI
    use_tui: bool = True
    color: bool = True
    tui_enabled: bool = True

    # Journal
    db_path: str = str(Path.home() / ".financial-engineering" / "trades.db")

    @property
    def data_path(self) -> Path:
        path = Path(self.data_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path


def _apply_json_overrides(settings: Settings) -> Settings:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _CONFIG_PATH.exists():
        return settings

    try:
        with open(_CONFIG_PATH) as f:
            data = json.load(f)
    except Exception:
        return settings

    for key, value in data.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    return settings


def _apply_env_overrides(settings: Settings) -> Settings:
    settings.default_symbol = os.environ.get("DEFAULT_SYMBOL", settings.default_symbol).strip() or settings.default_symbol
    settings.default_refresh_seconds = _env_int("DEFAULT_REFRESH_SECONDS", settings.default_refresh_seconds)
    settings.data_dir = os.environ.get("DATA_DIR", settings.data_dir).strip() or settings.data_dir
    settings.fred_api_key = os.environ.get("FRED_API_KEY", settings.fred_api_key).strip()
    settings.openai_api_key = os.environ.get("OPENAI_API_KEY", settings.openai_api_key).strip()
    settings.futures_watchlist = _env_list("FUTURES_WATCHLIST", settings.futures_watchlist)
    settings.watchlist = list(settings.futures_watchlist)
    settings.options_watchlist = _env_list("OPTIONS_WATCHLIST", settings.options_watchlist)
    settings.use_tui = _env_bool("USE_TUI", settings.use_tui)
    settings.tui_enabled = _env_bool("TUI_ENABLED", settings.tui_enabled)
    settings.recent_intraday_days = _env_int("RECENT_INTRADAY_DAYS", settings.recent_intraday_days)
    settings.intraday_lookback_bars = _env_int("INTRADAY_LOOKBACK_BARS", settings.intraday_lookback_bars)
    return settings


def load_settings() -> Settings:
    _load_dotenv()
    settings = Settings()
    settings = _apply_json_overrides(settings)
    settings = _apply_env_overrides(settings)
    return settings


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
