# Trade Terminal

A unified terminal-based system for futures and options analysis, built around an integrated analytics pipeline and a persistent trade journal.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)

---

## Overview

Trade Terminal launches a single entry point that routes into two fully-featured TUI workspaces — one for futures, one for options — plus a trade journal and performance dashboard. The system is built around a `UnifiedAnalyzerApp` (Textual) that serves as the hub, with each workspace running its own analytics pipeline on demand.

The options workspace runs a multi-stage pipeline: option chain fetch → BS implied vol computation → SVI surface calibration → dealer GEX model → intraday regime detection → FRED macro screen. The futures workspace runs a parallel pipeline: OHLCV fetch → ATR/EMA/VWAP → session levels (Asia/London/NY) → fair value gap detection → volume profile → bias engine → trade idea generation. All sessions share a single SQLite journal with per-trade R-multiple tracking, ATR hit rates, and performance breakdowns.

---

## Features

### Unified Hub
- 7-tab Textual TUI: Dashboard, Options, Futures, Watchlists, Journal, Performance, Help
- Dashboard aggregates macro context (VIX, DXY, 10Y yield, sector ETFs), live news sentiment, and watchlist snapshots for both asset classes
- Watchlists for both options and futures are persisted and editable in-app
- Keyboard-driven navigation with hotkeys for direct workspace access (`o`, `f`, `j`, `d`)

### Options Workspace (13 tabs)
- **Option chain**: calls, puts, and straddle tables with market IV, SVI model IV, bid/ask, and BUY/SELL/REVIEW action signals based on model–market edge
- **SVI surface**: Stochastic Volatility Inspired calibration for calls and puts; 25Δ risk reversal, butterfly, put/call skew slopes, IV term structure across nearest expirations
- **Dealer GEX model**: net dealer gamma exposure in $B, gamma flip point, call/put walls, charm (Δ/day), vanna (Δ/vol-pt), max pain, max gamma strike
- **Regime detection**: intraday classification — TREND UP, TREND DOWN, CHOP, COMPRESSION, EXTENDED UP/DOWN — using EMA9/21/50 stack and VWAP deviation; RV/IV ratio
- **Volume analytics**: put/call volume ratio, unusual volume flags (vol/OI ≥ 1.25×), top strikes by volume and OI
- **FRED macro screen**: yield curve (10Y–2Y), IG/HY credit spreads, CPI, UNRATE, NFCI, SOFR/EFFR (requires free FRED API key)
- **Tactical interpretation**: mechanical plain-English summary derived from regime, GEX regime, RV/IV ratio, 25Δ RR, butterfly, and GEX level proximity
- **Monitor tab**: watchlist snapshot table (symbol, price, regime, ATM IV, RR, GEX, confidence, sentiment, signal)
- Symbol cycling within watchlist with `[` / `]`; manual refresh with `r`

### Futures Workspace (8 tabs)
- **Session levels**: previous day H/L/C, today H/L, Asia/London/NY session H/L, NY open range H/L, VWAP; ATR targets at 1×/2×/3× for both directions
- **ATR computation**: EMA-based 5m and 15m ATR; trend classification per timeframe (bullish/bearish/neutral/mixed)
- **Fair value gaps (FVGs)**: detected across 5m, 15m, and 1h timeframes; sized relative to ATR, fill percentage, age in bars, proximity to spot
- **Volume profile**: POC, VAH, VAL (70% value area), relative volume; high/low volume node identification
- **Bias engine**: multi-factor scoring across trend alignment, VWAP position, session levels, volume nodes, and FVGs; outputs `BiasResult` (bullish/bearish/mixed/neutral) with confidence (high/medium/low), bull/bear signal lists, and cautions
- **Confluence zones**: clusters of named levels within a configurable ATR tolerance; typed as support/resistance
- **Trade ideas**: auto-generated from bias + levels + FVGs + volume profile + pullback zones; includes entry zone, stop (invalidation), 1×/2×/3× ATR targets, setup type, and entry reasons
- **Exit guidance**: when an open trade is detected in the journal, computes unrealized R, distance to stop and target, next barrier, and proximity to ATR multiples
- **Multi-ticker monitor**: watchlist table with price, daily Δ%, 5m/15m/1h trend, bias, confidence, VWAP position, volume spike flag, ATR regime, best idea

### Trade Journal
- SQLite database at `~/.financial-engineering/trades.db`
- Separate schemas for `FuturesTrade` and `OptionsTrade`; trade lifecycle states: `planned`, `open`, `closed`, `cancelled`
- Computed fields: P&L (points and dollars), R-multiple, holding period, session bucket, time-of-day bucket, ATR hit rates (1×/2×/3×)
- Structured metadata: setup type, timeframe, mistake tags, planned vs impulsive, bias/confidence at entry, FVG/volume node/session level involvement, confluence score
- Daily review entries with psychological notes, best/worst setup, mistake tags
- Performance metrics: win rate, profit factor, expectancy, avg/median R, max drawdown; breakdowns by ticker, setup, timeframe, session, time-of-day, confluence score, bias alignment, planned vs impulsive, reason for entry

### Fixed Income (standalone)
- Bond pricing with bootstrapped discount factors from live Treasury yield curve data
- Zero coupon bonds and options on ZCBs (binomial tree)
- Caplets, floorlets, swaps, swaptions
- Treasury yield curve and bootstrapped zero coupon yields via Treasury.gov

---

## Architecture

```
python main.py
    └── analyzer.unified.run_unified_analyzer()
            └── UnifiedAnalyzerApp (Textual, 7 tabs)
                    ├── Dashboard tab
                    │     ├── analyzer.macro   — VIX, DXY, 10Y, XLK, XLF snapshots
                    │     ├── analyzer.news    — news + sentiment aggregation
                    │     ├── options.monitor  — options watchlist snapshots
                    │     └── futures.monitor  — futures watchlist snapshots
                    │
                    ├── Options tab → OptionsTUI (Textual, 13 tabs)
                    │     └── options.core.analyzer.Analyzer
                    │           ├── options.volatility.marketvols  — chain fetch, BS IVs
                    │           ├── options.volatility.svi         — SVI calibration
                    │           ├── options.analytics.regime       — intraday regime
                    │           ├── options.analytics.gamma_model  — dealer GEX
                    │           ├── options.analytics.macro_screen — FRED indicators
                    │           └── fixed_income.core.bootstrap    — risk-free rate
                    │
                    ├── Futures tab → FuturesTUI (Textual, 8 tabs)
                    │     ├── futures.data     — OHLCV fetch + session parsing
                    │     ├── futures.levels   — ATR, EMA, VWAP, session levels
                    │     ├── futures.volume   — volume profile (POC, VAH, VAL)
                    │     ├── futures.fvg      — FVG detection across timeframes
                    │     ├── futures.bias     — bias engine + confluence zones
                    │     ├── futures.ideas    — trade idea generation
                    │     └── futures.signals  — alerts + exit guidance
                    │
                    ├── Journal tab  ─┐
                    └── Performance  ─┴─ journal.storage / journal.metrics / journal.dashboard
```

The `Analyzer` class loads all data on construction via a 4-way `ThreadPoolExecutor` (chain, intraday, dividend yield, macro screen in parallel) and exposes a thread-safe `refresh()` / `start_refresh()` interface for background updates while the TUI remains interactive.

---

## Project Structure

```
financial-engineering/
├── main.py                      # entry point — bootstraps venv, calls run_unified_analyzer()
├── requirements.txt
│
├── analyzer/                    # unified hub + shared utilities
│   ├── unified.py               # UnifiedAnalyzerApp + run_unified_analyzer()
│   ├── macro.py                 # VIX/DXY/yield/sector snapshots
│   ├── news.py                  # news fetch + sentiment
│   ├── watchlists.py            # options/futures watchlist persistence (JSON)
│   ├── data.py                  # TTL-cached yfinance history wrapper
│   └── formatting.py            # price/percent formatting
│
├── options/
│   ├── tui.py                   # OptionsTUI — 13-tab Textual app
│   ├── monitor.py               # options watchlist snapshot builder
│   ├── snapshot.py              # lightweight cached options snapshot
│   ├── core/
│   │   ├── analyzer.py          # Analyzer class — full options pipeline
│   │   ├── option.py            # Option pricing primitives
│   │   ├── strategy.py          # multi-leg strategy builder
│   │   └── pricing/
│   │       ├── pricing.py       # Black-Scholes + binomial pricing
│   │       └── montecarlo.py    # Monte Carlo simulation
│   ├── volatility/
│   │   ├── iv.py                # BS IV (Newton + Brent fallback)
│   │   ├── svi.py               # SVI surface calibration
│   │   └── marketvols.py        # chain fetch, BS IV batch computation, vol surface
│   ├── analytics/
│   │   ├── regime.py            # intraday regime classification
│   │   ├── gamma_model.py       # dealer GEX model (flip, walls, charm, vanna)
│   │   ├── gamma_zones.py       # OI ladder + pin zone heuristic
│   │   ├── macro_screen.py      # FRED API macro indicators
│   │   └── visualizer.py        # matplotlib charts (dashboard, OI ladder)
│   ├── exotics/
│   │   ├── asian.py             # Asian options (MC)
│   │   ├── digital.py           # digital call/put
│   │   └── range.py             # range accruals
│   └── utilities/
│       ├── marketdata.py        # MarketDataFetcher (spot, div yield, chain)
│       └── printer.py           # tabular output helpers
│
├── futures/
│   ├── tui.py                   # FuturesTUI — 8-tab Textual app
│   ├── cli.py                   # futures workspace CLI + trade planner
│   ├── monitor.py               # multi-ticker watchlist monitor
│   ├── data.py                  # FuturesData fetch (spot, 5m/1h OHLCV, sessions)
│   ├── levels.py                # session levels, VWAP, ATR, trend
│   ├── atr.py                   # ATR computation + EMA + 15m resample
│   ├── volume.py                # volume profile (POC, VAH, VAL, relative vol)
│   ├── fvg.py                   # FairValueGap detection
│   ├── bias.py                  # BiasResult, ConfluenceZone, PullbackZone
│   ├── ideas.py                 # TradeIdea generation
│   ├── signals.py               # AlertMessage, ExitGuidance
│   └── planner.py               # R-multiple assessment, trade plan formatting
│
├── journal/
│   ├── models.py                # FuturesTrade, OptionsTrade, DailyReview dataclasses
│   ├── storage.py               # TradeDatabase — SQLite CRUD
│   ├── metrics.py               # PerformanceMetrics computation
│   ├── dashboard.py             # Rich text renderers for journal/performance views
│   └── cli.py                   # interactive journal menus
│
├── config/
│   ├── settings.py              # Settings dataclass + JSON persistence
│   ├── tickers.py               # FUTURES_TICKER_MAP + CONTRACT_SPECS
│   └── sessions.py              # session/time-of-day label helpers
│
├── fixed_income/
│   ├── core/
│   │   ├── bond.py              # Bond pricing (bootstrapped discount factors)
│   │   ├── bootstrap.py         # ZC yield curve bootstrapping from Treasury.gov
│   │   ├── marketyields.py      # live Treasury yield curve fetch + cache
│   │   ├── fred_client.py       # FRED API client
│   │   └── zcb.py               # ZeroCouponBond (binomial tree)
│   └── derivatives/
│       ├── caplet.py            # Caplet pricing
│       ├── floorlet.py          # Floorlet pricing
│       ├── swap.py              # Interest rate swap
│       ├── swaption.py          # Swaption
│       └── zcb_option.py        # Option on ZCB
│
└── tests/
    ├── test_iv.py               # IV inversion correctness
    ├── test_futures.py          # futures levels, ATR, FVG, bias, ideas
    └── test_analytics.py        # regime, GEX model, macro screen
```

---

## Tech Stack

| Layer | Libraries |
|---|---|
| TUI framework | `textual >= 0.52.0`, `rich` |
| Market data | `yfinance`, `curl_cffi` |
| Numerics | `numpy`, `pandas`, `scipy` |
| Charting | `matplotlib` |
| HTTP / scraping | `requests`, `beautifulsoup4` |
| Table formatting | `tabulate` |
| Storage | `sqlite3` (stdlib) |
| Rates data | Treasury.gov (scrape), FRED API (optional) |

---

## Setup

**Prerequisites**: Python 3.11+

```bash
git clone https://github.com/aryamann04/financial-engineering.git
cd financial-engineering

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Optional — FRED macro indicators** (free API key):

```bash
export FRED_API_KEY=your_key_here
```

Get a key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). Without it, the Macro tab displays a setup prompt and the rest of the app functions normally.

---

## Usage

```bash
python main.py
```

This launches the unified Textual TUI. The app auto-bootstraps the `.venv` if present and `FE_SKIP_VENV_BOOTSTRAP` is not set.

Alternatively, run the analyzer package directly:

```bash
python -m analyzer
```

### Keyboard Reference

**Unified Hub**

| Key | Action |
|---|---|
| `←` / `→` | Switch tabs |
| `o` | Open options workspace (first watchlist symbol) |
| `f` | Open futures workspace (first watchlist symbol) |
| `j` | Open journal menu |
| `d` | Open performance dashboard |
| `r` | Refresh all data |
| `s` | Save watchlist edits (from Watchlists tab) |
| `q` | Quit |

**Options Workspace**

| Key | Action |
|---|---|
| `←` / `→` | Switch tabs |
| `[` / `]` | Previous / next watchlist symbol |
| `r` | Force full refresh |
| `l` | Log an options trade |
| `p` | Plan a trade |
| `j` | Open journal |
| `q` | Quit to hub |

**Futures Workspace**

| Key | Action |
|---|---|
| `←` / `→` | Switch tabs |
| `[` / `]` | Previous / next watchlist symbol |
| `↑` / `↓` | Navigate trade ideas |
| `Enter` | Plan from selected idea |
| `r` | Refresh |
| `l` | Log a futures trade |
| `p` | Plan a trade |
| `j` | Open journal |
| `q` | Quit to hub |

---

## Example Workflow

**1. Morning review in the hub**

Open the app with `python main.py`. The Dashboard tab loads macro context (VIX, DXY, 10Y yield, XLK, XLF) and news sentiment for SPY alongside your watchlist snapshots. Press `r` to refresh.

**2. Futures pre-market setup (MES)**

Press `f` to enter the futures workspace on MES (or the first symbol in your futures watchlist). Check the Levels tab for prev day H/L, Asia/London ranges, and the NY open range boundaries. Review the Bias tab for the multi-factor bias score, confluence zones, and pullback watch zones. The Ideas tab surfaces specific setups (e.g. "Asia High Breakout") with entry zone, invalidation level, and ATR targets.

**3. Options analysis (SPY)**

Press `q` to return to the hub, then `o` for the options workspace. Start on the Overview tab (ATM IV, regime, GEX regime, signal, confidence). Navigate to Summary for the full GEX model readout and tactical interpretation. Check the Gamma tab for the gamma flip point, call/put walls, charm, and vanna. Use the Surface tab for the IV term structure and 25Δ risk reversal.

**4. Log a trade**

From either workspace, press `l` to open the trade log prompt. The futures planner pre-fills ATR values and prompts for direction, entry, stop, target, setup type, and timeframe. The journal computes R-multiple, session bucket, and ATR hit rates automatically.

**5. Review performance**

In the hub, press `d` for the performance dashboard. Breakdowns by setup type, session, timeframe, and planned vs impulsive are shown alongside overall metrics (win rate, profit factor, expectancy, max drawdown).

---

## Configuration

Settings are stored at `~/.financial-engineering/config.json` and can be edited in-app from the settings menu or directly as JSON.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `default_symbol` | `MES=F` | Default futures symbol |
| `default_options_symbol` | `SPY` | Default options symbol |
| `options_default_t_days` | `30` | Target DTE for option chain resolution |
| `options_snapshot_ttl_seconds` | `180` | Options snapshot cache TTL |
| `confluence_tolerance_atr` | `0.25` | ATR multiple for level clustering |
| `fvg_min_size_atr` | `0.12` | Minimum FVG size relative to ATR |
| `vol_spike_threshold` | `1.8` | Relative volume multiple for spike flag |
| `value_area_pct` | `0.70` | Volume profile value area fraction |
| `db_path` | `~/.financial-engineering/trades.db` | Journal database path |

Watchlists can be edited directly from the Watchlists tab in the hub and saved with `s`.

---

## Limitations

- **Data latency**: all market data comes from yfinance, which has inherent delays and occasional gaps. There is no real-time streaming; data is fetched on demand and on manual refresh.
- **Options chain quality**: yfinance option chains can have stale quotes, wide spreads during off-hours, or missing strikes. The pipeline applies a soft liquidity filter but degenerate surfaces still occur.
- **SVI calibration**: SVI fitting can fail for illiquid surfaces or near-expiry chains. The pipeline falls back to raw BS IVs when calibration fails, and calibration status is shown in the Diagnostics tab.
- **Regime detection**: based on intraday 5m bars from yfinance, period="1d". Intraday data is unavailable outside market hours; regime defaults to UNAVAILABLE.
- **FRED macro**: requires a free API key. Without it, the Macro tab in both the options workspace and the unified dashboard is unavailable.
- **Fixed income module**: standalone and not integrated into either TUI workspace. Access the pricing classes directly via Python.
- **No execution**: Trade Terminal is a read-only analysis tool. Nothing connects to a brokerage or places orders.

---

## Future Improvements

- Real-time data via WebSocket feeds (e.g. Polygon.io, Interactive Brokers) to replace yfinance polling
- Streaming GEX updates as option chain ticks arrive
- Options strategy builder integrated directly into the options workspace, connected to the journal planner
- Deeper fixed income integration: yield curve overlaid in macro tab, treasury curve as rate input selector
- Backtesting layer for setup-type performance validation against historical data
- Export to CSV / markdown for journal entries and performance reports
