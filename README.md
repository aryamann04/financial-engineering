# Trade Terminal

Trade Terminal is a speed-first discretionary trading support system for futures and options. It combines a shared market-analysis core, a fast terminal workflow, a Streamlit cockpit, a local query assistant, and an options strategy recommendation engine that ranks multi-leg trades from live option-chain context.

It is decision-support and backtesting software, not financial advice and not an execution engine.

## Overview

The repo now has one shared analysis direction:

- `core/` for futures-oriented market state, liquidity, structure, FVGs, risk framing, macro context, and the local assistant
- `options/` for option-chain analytics, volatility surfaces, pricing models, and strategy recommendations
- `interfaces/tui.py` for the fast CLI/TUI workflow
- `interfaces/streamlit_app.py` for the visual cockpit
- legacy Textual options/futures workspaces are still available and preserved

## Key Features

### Futures / Trading Terminal

- multi-timeframe snapshot across `1m`, `5m`, `15m`, `1h`
- structure, liquidity, sweep detection, FVGs, ATR, VWAP, regime, risk framing
- local assistant with grounded answers based on the current analysis object
- lightweight heuristic backtest summary

### Options Strategy Recommendation Engine

- evaluates live option chains and recommends:
  - bull call spreads
  - bear call spreads
  - bull put spreads
  - bear put spreads
  - calendars
  - diagonals
  - long straddles
  - long strangles
  - call butterflies
  - iron butterflies
  - call condors
  - iron condors
  - covered calls
  - cash-secured puts
  - ratio spreads with explicit warnings
- strategy ranking uses:
  - valuation edge
  - model agreement
  - volatility regime fit
  - liquidity quality
  - complexity penalty
  - risk definition
  - directional fit

### Pricing Models

- Black-Scholes baseline
- Black-76 baseline
- binomial pricing for American-style checks
- SVI smile support when enough points exist
- local-volatility approximation from smile interpolation
- SABR smile fit with graceful fallback
- Heston scaffold is present as a documented non-live placeholder
- ensemble fair value from available models with confidence based on model dispersion

### Volatility Regime Logic

- ATM IV tracking and IV rank
- realized vol vs implied vol
- term structure shape from nearby expiries
- skew read from OTM put vs OTM call IV
- regime labels such as `high_iv`, `low_iv`, `realized_breakout`, `balanced`

### Streamlit Dashboard

- futures cockpit tab
- options strategies tab
- strategy ranking table with filters
- detailed strategy card
- payoff curve at expiry
- P/L heatmap across price and IV shifts
- model comparison table
- risk / reward summary
- warnings and regime rationale

### Terminal / CLI

- futures analysis commands
- scan mode
- export mode
- assistant query mode
- options strategy recommendations with filters

## Architecture

```text
main.py
├── CLI mode -> interfaces/tui.py
└── legacy mode -> analyzer/unified.py

core/
├── data.py
├── resampling.py
├── indicators.py
├── structure.py
├── fvg.py
├── liquidity.py
├── signals.py
├── risk.py
├── analysis.py
├── backtest.py
├── macro.py
└── assistant.py

options/
├── models.py
├── surface.py
├── strategies.py
├── risk.py
├── recommender.py
├── core/
├── analytics/
└── volatility/

interfaces/
├── tui.py
└── streamlit_app.py
```

The important design rule is that trading and strategy logic stays in shared modules, not in UI code.

## Local Setup

### 1. Clone and install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

`.env.example` contains:

```env
FRED_API_KEY=
DATA_DIR=
DEFAULT_SYMBOL=
DEFAULT_REFRESH_SECONDS=
OPENAI_API_KEY=
```

Useful optional environment variables:

- `FUTURES_WATCHLIST`
- `OPTIONS_WATCHLIST`
- `RECENT_INTRADAY_DAYS`
- `INTRADAY_LOOKBACK_BARS`

### 3. FRED setup

Set `FRED_API_KEY` to enable cached macro context for:

- `DGS10`
- `DGS2`
- `FEDFUNDS`
- `CPIAUCSL`
- `UNRATE`
- `VIXCLS`

The app still runs if no key is configured.

## Running The Project

### Terminal / CLI mode

```bash
python3 main.py analyze M6E --timeframe 5m
python3 main.py scan MES M6E MNQ
python3 main.py levels MES
python3 main.py setup M6E
python3 main.py ask M6E "Why is the terminal bearish?"
python3 main.py export M6E --output exports/m6e_analysis.json
python3 main.py backtest MES
```

### Options strategy CLI

```bash
python3 main.py options SPY
python3 main.py options SPY --view bullish
python3 main.py options SPY --view neutral --max-risk 500 --min-score 20
python3 main.py options SPY --strategy-type calendar
```

### Streamlit dashboard

```bash
streamlit run interfaces/streamlit_app.py
```

### Legacy Textual workspaces

```bash
python3 main.py legacy
```

The existing options Textual dashboard now includes a strategy recommendations panel.

## Docker Setup

### Build

```bash
docker build -t trade-terminal .
```

### Run Streamlit in Docker

```bash
docker run --rm -p 8501:8501 --env-file .env trade-terminal
```

### Run CLI in Docker

```bash
docker run --rm -it --env-file .env trade-terminal python main.py analyze MES
docker run --rm -it --env-file .env trade-terminal python main.py options SPY --view neutral
```

### Docker Compose

```bash
docker compose up --build
```

## Strategy Recommendation Output

Each recommendation includes:

- strategy name
- underlying
- expiry
- legs
- net debit or credit
- max profit / max loss
- breakevens
- aggregate Greeks
- ensemble fair value
- dollar and percentage edge
- volatility regime rationale
- liquidity warnings
- final score
- why the strategy fits
- what would invalidate it
- payoff curve
- P/L heatmap points

## Testing

Run the full test suite:

```bash
pytest -q
```

Current tests cover:

- IV round-trips and surface behavior
- futures analytics
- shared analysis schema
- strategy payoff and breakeven logic
- aggregated Greeks
- ensemble fair value logic
- recommendation generation
- liquidity warnings
- invalid chain handling

## CI

GitHub Actions is configured in `.github/workflows/tests.yml`.

It runs:

- dependency install
- CLI smoke checks
- import smoke checks
- full `pytest -q`

## Speed Notes

- intraday work uses recent rolling windows by default
- option-chain math is reused from cached chain snapshots where possible
- SVI and smile-dependent models only run when enough valid data exists
- recommendation ranking penalizes unreliable or illiquid structures instead of crashing
- expensive model paths fall back gracefully when calibration quality is weak

## Assumptions And Limitations

- market data comes from `yfinance`, which is suitable for monitoring and research but not exchange-grade execution
- strategy pricing and Greeks are analytical approximations
- SABR is implemented as a lightweight smile fit, not a full institutional calibration stack
- Heston is scaffolded as unavailable rather than forced into an unreliable calibration path
- probability of profit is a simple payoff-grid estimate, not a full distributional model
- quote staleness is only as good as the upstream chain metadata
- no trade execution is performed

## Disclaimer

Trade Terminal is for research, discretionary review, journaling, and strategy framing. It does not guarantee correctness, suitability, or profitability, and it should not be treated as investment advice.
