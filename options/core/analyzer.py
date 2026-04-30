"""
analyzer.py — Institutional-style intraday options surface analyzer.

Pipeline (single chain fetch):
  1. Resolve expiry → fetch option chain ONCE
  2. Filter liquid strikes (soft filter)
  3. Compute BS implied vols from mid prices (vectorized)
  4. Calibrate SVI for calls and puts (graceful fallback)
  5. Build analysis tables (model price, edge, action)
  6. Compute surface diagnostics (ATM IV, skew, RR, butterfly)
  7. Fetch intraday OHLCV → detect regime (TREND UP/DOWN/CHOP/COMPRESSION/…)
  8. Compute dealer GEX model (gamma flip, call/put walls, charm, vanna)
  9. Fetch macro screen from FRED (yield curve, credit, employment, inflation)
 10. Build IV term structure (up to 2 nearest expirations)
 11. Launch curses TUI

TUI tabs:
  SUMMARY | CALLS | PUTS | STRADDLE | TOP OPPS | GAMMA | REGIME | MACRO | SURFACE | DIAGNOSTICS

All data fetching happens in __init__; the TUI is read-only after that.
"""

import concurrent.futures
import curses
import threading
import time
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate

from options.volatility.marketvols import fetch_chain, compute_bs_ivs, get_bid_asks, _mid_price
from options.volatility.iv import bs_iv
from options.core.pricing.pricing import bs_price
from options.utilities.marketdata import MarketDataFetcher
from options.volatility.svi import SVI
from fixed_income.core.bootstrap import zc_yield
from fixed_income.core.marketyields import last_treasury_curve_status
from options.analytics.regime import detect_regime
from options.analytics.gamma_zones import compute_gamma_zones, oi_ladder_summary
from options.analytics.gamma_model import compute_dealer_gex
from options.analytics.macro_screen import MacroScreen
from options.analytics import visualizer


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class Analyzer:
    def __init__(self, ticker: str, T: float, market_match: bool = True, fetcher=None):
        self.ticker  = ticker.upper()
        self.T_input = T

        # ── persistent objects (survive across refreshes) ─────────────────
        if fetcher is None:
            self.fetcher = MarketDataFetcher(ticker, T, creation_date=None)
        else:
            self.fetcher = fetcher

        # ── refresh state ─────────────────────────────────────────────────
        self._refreshing:    bool           = False
        self._refresh_error: str | None     = None
        self._last_refresh:  datetime | None = None
        self._refresh_lock   = threading.Lock()

        # ── load all data (blocking; raises on chain failure) ─────────────
        self._do_load()

    # ------------------------------------------------------------------
    # Core data-loading pipeline — called on init and every refresh
    # ------------------------------------------------------------------

    def _do_load(self) -> None:
        """
        Fetch all market data and compute all derived quantities.
        Designed to be re-called on refresh without re-creating the object.
        Raises RuntimeError if the option chain cannot be fetched (hard failure).
        All other failures are soft — appended to self.warnings.
        """
        warnings: list[str] = []

        # ── rates (TTL-cached; fast after first call) ──────────────────
        r = zc_yield(self.T_input)
        treasury_status = last_treasury_curve_status()
        if treasury_status.error_message:
            warnings.append(treasury_status.error_message)

        # ── update spot price (fast_info; no history download) ─────────
        self.fetcher._spot_price = self.fetcher._fetch_spot()
        S_0 = self.fetcher.current_price()

        # ── parallel I/O: chain + intraday + dividend yield ────────────
        T = self.T_input

        def _fetch_chain():
            return fetch_chain(self.fetcher.yfTicker, T)

        def _fetch_intraday():
            df = self.fetcher.yfTicker.history(interval="5m", period="1d")
            return df[["Open", "High", "Low", "Close", "Volume"]] if not df.empty else None

        def _fetch_div():
            return self.fetcher.dividend_yield()

        def _fetch_macro():
            return MacroScreen()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            f_chain    = pool.submit(_fetch_chain)
            f_intraday = pool.submit(_fetch_intraday)
            f_div      = pool.submit(_fetch_div)
            f_macro    = pool.submit(_fetch_macro)

            try:
                calls_df, puts_df, expiry_str, chain_warnings = f_chain.result()
                warnings.extend(chain_warnings)
            except Exception as e:
                raise RuntimeError(f"[Analyzer] Cannot fetch option chain: {e}") from e

            intraday_raw = None
            try:
                intraday_raw = f_intraday.result()
                if intraday_raw is not None:
                    intraday_raw.index = pd.to_datetime(intraday_raw.index)
            except Exception as e:
                warnings.append(f"Intraday data fetch failed: {e}")

            try:
                q = f_div.result()
            except Exception:
                q = 0.0

            macro_screen: MacroScreen | None = None
            try:
                macro_screen = f_macro.result()
                if macro_screen and macro_screen.warnings:
                    # Only surface non-key warnings (missing API key is expected)
                    for w in macro_screen.warnings:
                        if "FRED_API_KEY" not in w:
                            warnings.append(f"MacroScreen: {w}")
            except Exception as e:
                warnings.append(f"MacroScreen fetch failed: {e}")

        # ── resolve T from expiry ─────────────────────────────────────
        try:
            T_resolved = (
                datetime.strptime(expiry_str, "%Y-%m-%d") - datetime.today()
            ).days / 365.0
        except Exception:
            T_resolved = T

        if T_resolved <= 0:
            raise ValueError(f"[Analyzer] Resolved expiry {expiry_str} is in the past.")

        # ── BS IVs ────────────────────────────────────────────────────
        call_strikes, call_bs_ivs = compute_bs_ivs(calls_df, S_0, T_resolved, r, q, "call")
        put_strikes,  put_bs_ivs  = compute_bs_ivs(puts_df,  S_0, T_resolved, r, q, "put")

        if len(call_strikes) < 2:
            warnings.append("Very few call strikes with valid BS IV; surface may be unreliable.")
        if len(put_strikes) < 2:
            warnings.append("Very few put strikes with valid BS IV; surface may be unreliable.")

        # ── SVI calibration ───────────────────────────────────────────
        svi_c = SVI(S_0, T_resolved, r, q, "call", strikes=call_strikes, implied_vols=call_bs_ivs)
        svi_p = SVI(S_0, T_resolved, r, q, "put",  strikes=put_strikes,  implied_vols=put_bs_ivs)
        warnings.extend(svi_c.warnings)
        warnings.extend(svi_p.warnings)

        # ── All CPU-bound computations ────────────────────────────────
        # Temporarily assign needed attrs so _build_tables etc. can use self.*
        self.r   = r
        self.S_0 = S_0
        self.q   = q
        self.T   = T_resolved
        self.expiry_str = expiry_str
        self.svi_c = svi_c
        self.svi_p = svi_p
        self._raw_calls_df = calls_df
        self._raw_puts_df  = puts_df

        calls_out, puts_out, straddle_out, top5c, top5p, top5s = self._build_tables(
            calls_df, puts_df, call_strikes, call_bs_ivs, put_strikes, put_bs_ivs
        )
        surface = self._compute_surface_diagnostics(call_strikes, call_bs_ivs, put_strikes, put_bs_ivs)
        self.surface = surface   # needed by _compute_term_structure
        flags   = self._compute_opportunity_flags_from(calls_out, puts_out, svi_c, svi_p)
        ts      = self._compute_term_structure(max_expirations=2)

        intraday_df = (
            intraday_raw
            if (intraday_raw is not None and not intraday_raw.empty)
            else None
        )
        regime: dict = {"regime": "UNAVAILABLE"}
        if intraday_df is not None:
            try:
                regime = detect_regime(intraday_df, atm_iv=surface.get("atm_iv_call"))
            except Exception as e:
                warnings.append(f"Regime detection failed: {e}")
        else:
            warnings.append("Intraday data unavailable; regime classification skipped.")

        # ── Dealer GEX model ──────────────────────────────────────────────
        gamma_profile: dict = {
            "gex_by_strike": pd.DataFrame(), "gex_profile": pd.DataFrame(),
            "flip_point": None, "call_walls": [], "put_walls": [],
            "charm_exposure": 0.0, "vanna_exposure": 0.0,
            "current_gex": 0.0, "regime": "N/A", "warnings": [],
        }
        try:
            gamma_profile = compute_dealer_gex(
                calls_df, puts_df, S_0, T_resolved, r, q
            )
            warnings.extend(gamma_profile.get("warnings", []))
        except Exception as e:
            warnings.append(f"Dealer GEX model failed: {e}")

        # ── OI heuristic (kept for OI ladder plot) ────────────────────
        gamma_zones: dict = {
            "ladder_df": pd.DataFrame(), "pin_zones": [], "call_wall": None,
            "put_wall": None, "call_dom": [], "put_dom": [], "warnings": [],
        }
        try:
            gamma_zones = compute_gamma_zones(calls_df, puts_df, S_0)
        except Exception:
            pass

        # ── Atomic swap — update all public attributes at once ────────
        # Hold the lock only for the attribute assignments (microseconds).
        with self._refresh_lock:
            self.warnings       = warnings
            self.calls_df       = calls_out
            self.puts_df        = puts_out
            self.straddle_df    = straddle_out
            self.top5_calls     = top5c
            self.top5_puts      = top5p
            self.top5_straddle  = top5s
            self.surface        = surface
            self.flags          = flags
            self.term_structure = ts
            self.intraday_df    = intraday_df
            self.regime         = regime
            self.gamma_profile  = gamma_profile
            self.gamma_zones    = gamma_zones
            self.macro_screen   = macro_screen
            self._last_refresh  = datetime.now()

    # ------------------------------------------------------------------
    # Public refresh interface
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """
        Re-fetch all market data and recompute all derived quantities.

        Safe to call from any thread.  If a refresh is already running,
        this call returns immediately without starting a second one.
        Errors during soft-path (intraday, regime, gamma) are absorbed into
        self.warnings.  Hard failures (chain fetch) set self._refresh_error
        and leave the previous data intact.
        """
        if self._refreshing:
            return
        self._refreshing    = True
        self._refresh_error = None
        try:
            self._do_load()
        except Exception as e:
            self._refresh_error = str(e)
        finally:
            self._refreshing = False

    def start_refresh(self) -> threading.Thread:
        """
        Start a background refresh and return the thread.
        The TUI calls this so the UI stays responsive during the fetch.
        """
        t = threading.Thread(target=self.refresh, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    # Term structure
    # ------------------------------------------------------------------

    def _compute_term_structure(self, max_expirations: int = 2) -> dict[str, float]:
        """
        Fetch ATM call IV for up to `max_expirations` nearest expirations.

        The primary expiry (self.expiry_str) is handled without a network call
        since we already have its ATM IV from the main chain fetch.  Additional
        expirations each cost one option_chain() call — capped at max_expirations
        to keep startup latency bounded.

        Returns {expiry_str: atm_iv_decimal}.
        """
        ts: dict[str, float] = {}

        # Seed with already-computed ATM IV for the primary expiry
        atm_c = self.surface.get("atm_iv_call")
        if atm_c is not None and self.expiry_str:
            ts[self.expiry_str] = atm_c

        try:
            exp_dates = self.fetcher.yfTicker.options
            if not exp_dates:
                return ts
            today = datetime.today()
            sorted_exps = sorted(
                (e for e in exp_dates if e != self.expiry_str),
                key=lambda x: abs(datetime.strptime(x, "%Y-%m-%d") - today),
            )[:max_expirations]

            for expiry in sorted_exps:
                try:
                    T_exp = (
                        datetime.strptime(expiry, "%Y-%m-%d") - today
                    ).days / 365.0
                    if T_exp <= 0:
                        continue
                    chain_calls = self.fetcher.yfTicker.option_chain(expiry).calls
                    nearest = chain_calls.iloc[
                        (chain_calls["strike"] - self.S_0).abs().argsort().iloc[:3]
                    ]
                    atm_ivs = []
                    for _, row in nearest.iterrows():
                        mid = _mid_price(row)
                        if mid <= 0:
                            continue
                        iv = bs_iv(mid, self.S_0, float(row["strike"]),
                                   T_exp, self.r, self.q, "call")
                        if iv and np.isfinite(iv) and 0.01 < iv < 3.0:
                            atm_ivs.append(iv)
                    if atm_ivs:
                        ts[expiry] = float(np.mean(atm_ivs))
                except Exception:
                    continue
        except Exception:
            pass
        return ts

    # ------------------------------------------------------------------
    # Table building
    # ------------------------------------------------------------------

    def _build_tables(self, calls_df, puts_df,
                      call_strikes, call_bs_ivs,
                      put_strikes, put_bs_ivs):
        """Build CALLS, PUTS, STRADDLE, and TOP-5 DataFrames."""

        def _compute_leg(df_chain, strikes, bs_ivs, option_type, svi):
            indexed = df_chain.set_index("strike")
            rows = []

            for K, bs_iv_val in zip(strikes, bs_ivs):
                model_vol = svi.svi_vol(K)
                if model_vol is None or model_vol <= 0:
                    model_vol = bs_iv_val

                model_price = bs_price(
                    self.S_0, K, self.T, self.r, model_vol, self.q, option_type
                )

                # Bid/ask with lastPrice fallback (after-hours safe)
                bid, ask = None, None
                if K in indexed.index:
                    row  = indexed.loc[K]
                    bid  = float(row.get("bid", 0) or 0) or None
                    ask  = float(row.get("ask", 0) or 0) or None
                    if bid is None and ask is None:
                        last = float(row.get("lastPrice", 0) or 0)
                        if last > 0:
                            bid = ask = last

                if bid is not None and ask is not None and model_price is not None:
                    if model_price < bid:
                        edge, action = bid - model_price, "SELL"
                    elif model_price > ask:
                        edge, action = ask - model_price, "BUY"
                    else:
                        edge, action = 0.0, "-"
                else:
                    edge, action = None, "N/A"

                rows.append({
                    "Strike":        K,
                    "Market IV":     f"{bs_iv_val * 100:.2f}%" if bs_iv_val else "N/A",
                    "Model IV":      f"{model_vol * 100:.2f}%",
                    "Bid":           round(bid,   2) if bid   is not None else "N/A",
                    "Ask":           round(ask,   2) if ask   is not None else "N/A",
                    "Model Price":   round(model_price, 2) if model_price is not None else "N/A",
                    "Est. Edge ($)": round(abs(edge), 2) if edge is not None else "N/A",
                    "Action":        action,
                    "Type":          option_type.upper(),
                })

            df = pd.DataFrame(rows)
            if not df.empty:
                df["Est. Edge ($)"] = pd.to_numeric(df["Est. Edge ($)"], errors="coerce")
            return df

        calls_out = _compute_leg(calls_df, call_strikes, call_bs_ivs, "call", self.svi_c)
        puts_out  = _compute_leg(puts_df,  put_strikes,  put_bs_ivs,  "put",  self.svi_p)

        def _top5(df):
            if df.empty:
                return df
            return df.reindex(
                df["Est. Edge ($)"].abs().nlargest(5).index
            ).reset_index(drop=True)

        top5_calls  = _top5(calls_out)
        top5_puts   = _top5(puts_out)

        if not calls_out.empty and not puts_out.empty:
            straddle_df = pd.merge(
                calls_out, puts_out, on="Strike", suffixes=("_Call", "_Put")
            )[[
                "Bid_Call", "Ask_Call", "Model Price_Call", "Est. Edge ($)_Call", "Action_Call",
                "Strike",
                "Bid_Put",  "Ask_Put",  "Model Price_Put",  "Est. Edge ($)_Put",  "Action_Put",
            ]]
            all_rows = pd.concat([calls_out, puts_out], ignore_index=True)
            top5_straddle = _top5(all_rows)[[
                "Type", "Strike", "Market IV", "Model IV",
                "Bid", "Ask", "Model Price", "Est. Edge ($)", "Action"
            ]]
        else:
            straddle_df   = pd.DataFrame()
            top5_straddle = pd.DataFrame()

        return calls_out, puts_out, straddle_df, top5_calls, top5_puts, top5_straddle

    # ------------------------------------------------------------------
    # Surface diagnostics
    # ------------------------------------------------------------------

    def _compute_surface_diagnostics(self, call_strikes, call_ivs,
                                     put_strikes, put_ivs) -> dict:
        diag = {
            "atm_iv_call":         None,
            "atm_iv_put":          None,
            "call_skew":           None,
            "put_skew":            None,
            "risk_reversal_25d":   None,
            "butterfly_25d":       None,
            "iv_range_call":       None,
            "iv_range_put":        None,
            "n_call_strikes":      int(len(call_strikes)),
            "n_put_strikes":       int(len(put_strikes)),
            "svi_calibrated_call": self.svi_c.calibrated,
            "svi_calibrated_put":  self.svi_p.calibrated,
            "svi_rmse_call":       self.svi_c.rmse,
            "svi_rmse_put":        self.svi_p.rmse,
        }

        def _atm_iv(strikes, ivs):
            if len(strikes) == 0:
                return None
            idx = np.argmin(np.abs(strikes - self.S_0))
            return float(ivs[idx])

        def _approx_25d_strike(strikes, ivs, option_type):
            if len(strikes) < 3:
                return None, None
            mask = strikes > self.S_0 if option_type == "call" else strikes < self.S_0
            otm_s, otm_v = strikes[mask], ivs[mask]
            if len(otm_s) == 0:
                return None, None
            target = self.S_0 * (1.10 if option_type == "call" else 0.90)
            idx = np.argmin(np.abs(otm_s - target))
            return float(otm_s[idx]), float(otm_v[idx])

        diag["atm_iv_call"] = _atm_iv(call_strikes, call_ivs)
        diag["atm_iv_put"]  = _atm_iv(put_strikes,  put_ivs)

        if len(call_ivs) >= 2:
            diag["iv_range_call"] = float(call_ivs.max() - call_ivs.min())
        if len(put_ivs) >= 2:
            diag["iv_range_put"]  = float(put_ivs.max()  - put_ivs.min())

        if len(put_strikes) >= 3:
            log_m = np.log(put_strikes / self.S_0)
            diag["put_skew"] = float(np.polyfit(log_m, put_ivs, 1)[0])

        if len(call_strikes) >= 3:
            log_m = np.log(call_strikes / self.S_0)
            diag["call_skew"] = float(np.polyfit(log_m, call_ivs, 1)[0])

        _, iv_otm_call = _approx_25d_strike(call_strikes, call_ivs, "call")
        _, iv_otm_put  = _approx_25d_strike(put_strikes,  put_ivs,  "put")
        atm_avg = None
        if diag["atm_iv_call"] and diag["atm_iv_put"]:
            atm_avg = (diag["atm_iv_call"] + diag["atm_iv_put"]) / 2

        if iv_otm_call is not None and iv_otm_put is not None:
            diag["risk_reversal_25d"] = iv_otm_call - iv_otm_put
            if atm_avg is not None:
                diag["butterfly_25d"] = (iv_otm_call + iv_otm_put) / 2 - atm_avg

        return diag

    # ------------------------------------------------------------------
    # Opportunity flags
    # ------------------------------------------------------------------

    def _compute_opportunity_flags(self) -> list:
        """Convenience wrapper — uses current self.calls_df / puts_df."""
        return self._compute_opportunity_flags_from(
            self.calls_df, self.puts_df, self.svi_c, self.svi_p
        )

    def _compute_opportunity_flags_from(self, calls_df, puts_df, svi_c, svi_p) -> list:
        flags = []

        def _flag_edges(df, label):
            if df.empty:
                return
            edges = pd.to_numeric(df["Est. Edge ($)"], errors="coerce").dropna()
            if len(edges) < 3:
                return
            threshold = edges.mean() + edges.std()
            for _, row in df.iterrows():
                edge_val = pd.to_numeric(row["Est. Edge ($)"], errors="coerce")
                if pd.notna(edge_val) and edge_val > threshold:
                    flags.append({
                        "Strike":  row["Strike"],
                        "Type":    label,
                        "Action":  row["Action"],
                        "Message": f"Edge ${edge_val:.2f} > {threshold:.2f} (1σ threshold)",
                    })

        def _flag_iv_jumps(strikes, ivs, label):
            if len(strikes) < 3:
                return
            diffs = np.abs(np.diff(ivs))
            mean_diff, std_diff = diffs.mean(), diffs.std()
            for i, d in enumerate(diffs):
                if d > mean_diff + 2 * std_diff:
                    flags.append({
                        "Strike":  float(strikes[i]),
                        "Type":    label,
                        "Action":  "REVIEW",
                        "Message": f"IV jump {d*100:.1f}pp between "
                                   f"{strikes[i]:.0f}→{strikes[i+1]:.0f} (>2σ)",
                    })

        _flag_edges(calls_df, "CALL")
        _flag_edges(puts_df,  "PUT")
        _flag_iv_jumps(svi_c.strikes, svi_c.implied_vols, "CALL IV")
        _flag_iv_jumps(svi_p.strikes, svi_p.implied_vols, "PUT IV")

        return flags

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def plot_vols(self) -> None:
        self.svi_c.plot_svi()
        self.svi_p.plot_svi()

    def plot_dashboard(self) -> None:
        """Launch the 3-panel matplotlib intraday dashboard."""
        if self.intraday_df is None or self.intraday_df.empty:
            print("[Analyzer] No intraday data; cannot plot dashboard.")
            return
        visualizer.plot_intraday_dashboard(
            ticker=self.ticker,
            intraday_df=self.intraday_df,
            gamma_zones=self.gamma_zones,
            surface=self.surface,
            regime=self.regime,
            term_structure=self.term_structure,
        )

    def plot_oi_ladder(self) -> None:
        """Launch the OI ladder matplotlib chart."""
        ladder = oi_ladder_summary(self.gamma_zones, self.S_0)
        visualizer.plot_oi_ladder(ladder, self.S_0,
                                  title=f"{self.ticker} OI Ladder — {self.expiry_str}")

    # ------------------------------------------------------------------
    # Curses TUI
    # ------------------------------------------------------------------

    def _build_views(self) -> dict:
        """Snapshot of current DataFrames for the TUI.  Called after each refresh."""
        with self._refresh_lock:
            return {
                "SUMMARY":     None,
                "CALLS":       self.calls_df,
                "PUTS":        self.puts_df,
                "STRADDLE":    self.straddle_df,
                "TOP OPPS":    self.top5_straddle,
                "GAMMA":       self._gamma_gex_df(),
                "REGIME":      self._regime_df(),
                "MACRO":       None,           # rendered as text via _macro_lines()
                "SURFACE":     self._surface_df(),
                "DIAGNOSTICS": self._diagnostics_df(),
            }

    def launch_analyzer(self, auto_refresh_seconds: int | None = None) -> None:
        """
        Launch interactive curses terminal.

        Parameters
        ----------
        auto_refresh_seconds : int or None
            If set, automatically re-fetch all data every N seconds.
            None (default) disables auto-refresh; use 'r' for manual refresh.

        Tabs:  SUMMARY | CALLS | PUTS | STRADDLE | TOP OPPS | GAMMA | REGIME | SURFACE | DIAGNOSTICS
        Keys:  ← / → (or h/l) — switch tabs
               ↑ / ↓ (or k/j) — scroll
               q               — quit
               r               — force refresh now
               p               — launch matplotlib dashboard
               o               — launch OI ladder chart
        """
        views        = self._build_views()
        view_keys    = list(views.keys())
        current_idx  = 0
        scroll_off   = 0
        _TEXT_TABS   = {"SUMMARY", "MACRO"}

        # Auto-refresh state
        _last_auto_refresh = time.monotonic()

        def _get_table_lines(key):
            df = views[key]
            if df is None or (hasattr(df, "empty") and df.empty):
                return ["  (no data for this view)"]
            return tabulate(df, headers="keys", tablefmt="grid",
                            showindex=False).splitlines()

        def draw(screen):
            nonlocal current_idx, scroll_off

            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_BLACK,  curses.COLOR_WHITE)   # strike col
            curses.init_pair(2, curses.COLOR_GREEN,  -1)                   # BUY / bullish
            curses.init_pair(3, curses.COLOR_RED,    -1)                   # SELL / bearish
            curses.init_pair(4, curses.COLOR_YELLOW, -1)                   # REVIEW / warn
            curses.init_pair(5, curses.COLOR_CYAN,   -1)                   # header info
            curses.init_pair(6, curses.COLOR_WHITE,  curses.COLOR_BLUE)    # tab selected
            curses.curs_set(0)
            screen.nodelay(1)
            screen.timeout(100)

            while True:
                now_mono = time.monotonic()

                # ── Auto-refresh timer ───────────────────────────────────
                if (
                    auto_refresh_seconds is not None
                    and not self._refreshing
                    and now_mono - _last_auto_refresh >= auto_refresh_seconds
                ):
                    self.start_refresh()
                    _last_auto_refresh = now_mono

                # ── Swap in updated views after a completed refresh ───────
                # _refreshing just flipped False → views are stale → rebuild.
                # We detect this by checking _last_refresh against what we
                # built views from; simplest: rebuild whenever not refreshing
                # and last_refresh is newer than the views snapshot.
                if not self._refreshing and self._last_refresh is not None:
                    # _views_as_of tracks the timestamp of the last build
                    if not hasattr(draw, "_views_as_of") or \
                            draw._views_as_of != self._last_refresh:
                        views = self._build_views()
                        draw._views_as_of = self._last_refresh
                        # Keep scroll position, but don't jump to top
                        if self._refresh_error:
                            pass   # leave views unchanged on hard failure

                screen.clear()
                h, w = screen.getmaxyx()

                # ── Header bar ──────────────────────────────────────────
                reg_label = self.regime.get("regime", "N/A")
                atm_c = self.surface.get("atm_iv_call")
                atm_p = self.surface.get("atm_iv_put")
                rr    = self.surface.get("risk_reversal_25d")
                fly   = self.surface.get("butterfly_25d")
                rv    = self.regime.get("realized_vol")

                hdr_parts = [
                    f" {self.ticker}",
                    f"S=${self.S_0:.2f}",
                    f"r={self.r*100:.2f}%",
                    f"q={self.q*100:.2f}%",
                    f"T={self.T:.3f}y",
                    f"exp:{self.expiry_str}",
                ]
                if atm_c is not None:
                    hdr_parts.append(f"ATM-C:{atm_c*100:.1f}%")
                if atm_p is not None:
                    hdr_parts.append(f"ATM-P:{atm_p*100:.1f}%")
                if rr is not None:
                    hdr_parts.append(f"RR:{rr*100:+.1f}pp")
                if fly is not None:
                    hdr_parts.append(f"FLY:{fly*100:+.1f}pp")
                if rv is not None:
                    hdr_parts.append(f"RV:{rv*100:.1f}%")
                hdr_parts.append(f"[{reg_label}]")

                # ── Refresh status tag ───────────────────────────────────
                if self._refreshing:
                    refresh_tag = " ↻ REFRESHING..."
                    tag_attr    = curses.color_pair(4) | curses.A_BOLD
                elif self._refresh_error:
                    refresh_tag = f" ✗ ERR:{self._refresh_error[:30]}"
                    tag_attr    = curses.color_pair(3) | curses.A_BOLD
                elif self._last_refresh is not None:
                    age_s  = int((datetime.now() - self._last_refresh).total_seconds())
                    if auto_refresh_seconds:
                        nxt = auto_refresh_seconds - age_s
                        refresh_tag = f" ✔ {age_s}s ago | next:{max(nxt,0)}s"
                    else:
                        refresh_tag = f" ✔ {age_s}s ago"
                    tag_attr = curses.color_pair(5)
                else:
                    refresh_tag = ""
                    tag_attr    = curses.A_NORMAL

                header = " | ".join(hdr_parts) + " "
                try:
                    screen.addstr(0, 0, header[:w-1],
                                  curses.color_pair(5) | curses.A_BOLD)
                    # Append refresh tag right-aligned
                    tag_x = max(0, w - len(refresh_tag) - 2)
                    screen.addstr(0, tag_x, refresh_tag[:w-tag_x-1], tag_attr)
                except curses.error:
                    pass

                # ── Tab bar ──────────────────────────────────────────────
                x = 0
                for i, k in enumerate(view_keys):
                    label = f"[{k}]" if i == current_idx else f" {k} "
                    attr  = (curses.color_pair(6) | curses.A_BOLD
                             if i == current_idx else curses.A_BOLD)
                    try:
                        screen.addstr(1, x, label, attr)
                    except curses.error:
                        pass
                    x += len(label) + 1

                # ── Content ───────────────────────────────────────────────
                view_name = view_keys[current_idx]

                if view_name in _TEXT_TABS:
                    if view_name == "MACRO":
                        lines = self._macro_lines()
                    else:
                        lines = self._summary_lines()
                    for row_i, line in enumerate(
                        lines[scroll_off: scroll_off + h - 4], start=3
                    ):
                        if row_i >= h - 1:
                            break
                        attr = curses.A_NORMAL
                        if any(r in line for r in (
                            "TREND UP", "EXTENDED UP", "TREND DOWN",
                            "EXTENDED DOWN", "CHOP", "COMPRESSION", "UNAVAILABLE"
                        )):
                            up_words   = ("TREND UP", "EXTENDED UP")
                            down_words = ("TREND DOWN", "EXTENDED DOWN")
                            if any(w in line for w in up_words):
                                attr = curses.color_pair(2) | curses.A_BOLD
                            elif any(w in line for w in down_words):
                                attr = curses.color_pair(3) | curses.A_BOLD
                            else:
                                attr = curses.color_pair(4) | curses.A_BOLD
                        try:
                            screen.addstr(row_i, 0, line[:w-1], attr)
                        except curses.error:
                            pass
                else:
                    lines = _get_table_lines(view_name)
                    for row_i, line in enumerate(
                        lines[scroll_off: scroll_off + h - 4], start=3
                    ):
                        if row_i >= h - 1:
                            break
                        if line and line[0] in "+┌└├─":
                            try:
                                screen.addstr(row_i, 0, line[:w-1])
                            except curses.error:
                                pass
                            continue
                        cells = line.split("│")
                        cx = 0
                        for col_i, cell in enumerate(cells):
                            val  = cell.strip()
                            text = cell + "│"
                            if col_i == 1:
                                attr = curses.color_pair(1)
                            elif val == "BUY":
                                attr = curses.color_pair(2) | curses.A_BOLD
                            elif val == "SELL":
                                attr = curses.color_pair(3) | curses.A_BOLD
                            elif val in ("REVIEW", "FLAG", "CALL WALL", "PUT WALL"):
                                attr = curses.color_pair(4) | curses.A_BOLD
                            elif "PIN" in val:
                                attr = curses.color_pair(5) | curses.A_BOLD
                            else:
                                attr = curses.A_NORMAL
                            try:
                                screen.addstr(row_i, cx, text[:w - cx - 1], attr)
                            except curses.error:
                                pass
                            cx += len(text)

                # ── Key legend ───────────────────────────────────────────
                if auto_refresh_seconds:
                    legend = (f" q:quit  ←/→:tab  ↑/↓:scroll"
                              f"  r:refresh(auto:{auto_refresh_seconds}s)"
                              f"  p:chart  o:OI ")
                else:
                    legend = " q:quit  ←/→:tab  ↑/↓:scroll  r:refresh  p:chart  o:OI "
                try:
                    screen.addstr(h - 1, 0, legend[:w-1], curses.A_DIM)
                except curses.error:
                    pass

                screen.refresh()

                key = screen.getch()
                if key == ord("q"):
                    break
                elif key in (curses.KEY_RIGHT, ord("l")):
                    current_idx = (current_idx + 1) % len(view_keys)
                    scroll_off  = 0
                elif key in (curses.KEY_LEFT, ord("h")):
                    current_idx = (current_idx - 1) % len(view_keys)
                    scroll_off  = 0
                elif key in (curses.KEY_DOWN, ord("j")):
                    if view_name in _TEXT_TABS:
                        if view_name == "MACRO":
                            n_lines = len(self._macro_lines())
                        else:
                            n_lines = len(self._summary_lines())
                    else:
                        n_lines = len(_get_table_lines(view_name))
                    if scroll_off + h - 4 < n_lines:
                        scroll_off += 1
                elif key in (curses.KEY_UP, ord("k")):
                    if scroll_off > 0:
                        scroll_off -= 1
                elif key == ord("r") and not self._refreshing:
                    # Manual refresh — non-blocking background thread
                    self.start_refresh()
                    _last_auto_refresh = time.monotonic()
                elif key == ord("p"):
                    curses.endwin()
                    self.plot_dashboard()
                    screen.refresh()
                elif key == ord("o"):
                    curses.endwin()
                    self.plot_oi_ladder()
                    screen.refresh()

        curses.wrapper(draw)

    # ------------------------------------------------------------------
    # SUMMARY tab — plain text lines
    # ------------------------------------------------------------------

    def _summary_lines(self) -> list[str]:
        """
        Generate the formatted text lines for the SUMMARY tab.
        Returns a list of strings to be rendered row-by-row.
        """
        s   = self.surface
        reg = self.regime
        ts  = self.term_structure

        W = 72   # target box width

        def _box(title: str, rows: list[str]) -> list[str]:
            bar   = "─" * (W - 2)
            lines = [f"┌{bar}┐"]
            lines.append(f"│  {title.upper():<{W-4}}│")
            lines.append(f"├{bar}┤")
            for r in rows:
                lines.append(f"│  {r:<{W-4}}│")
            lines.append(f"└{bar}┘")
            return lines

        fmt_iv  = lambda v: f"{v*100:.1f}%" if v is not None else "N/A"
        fmt_pp  = lambda v: f"{v*100:+.1f}pp" if v is not None else "N/A"

        out: list[str] = [""]

        # ── REGIME ───────────────────────────────────────────────────────
        regime_label = reg.get("regime", "UNAVAILABLE")
        price        = reg.get("price")
        vwap         = reg.get("vwap")
        vwap_diff    = reg.get("vwap_pct_diff")
        e9, e21, e50 = reg.get("ema9"), reg.get("ema21"), reg.get("ema50")
        rv           = reg.get("realized_vol")
        rv_iv        = reg.get("rv_iv_ratio")
        atm_c        = s.get("atm_iv_call")

        ema_str = "N/A"
        if e9 and e21 and e50:
            aligned = reg.get("ema_aligned")
            dir_str = "↑ aligned UP" if aligned is True else ("↓ aligned DOWN" if aligned is False else "— mixed")
            ema_str = f"EMA9={e9:.2f}  EMA21={e21:.2f}  EMA50={e50:.2f}  {dir_str}"

        rv_str = "N/A"
        if rv is not None and atm_c is not None:
            rv_str = f"RV={rv*100:.1f}%  ATM-IV={atm_c*100:.1f}%  RV/IV={rv_iv:.2f}" if rv_iv else f"RV={rv*100:.1f}%"
        elif rv is not None:
            rv_str = f"RV={rv*100:.1f}%  (ATM IV unavailable)"

        reg_rows = [
            f"State: {regime_label}",
            f"Price: {price:.4f}    VWAP: {vwap:.4f}    ({vwap_diff:+.3f}% vs VWAP)"
            if price and vwap and vwap_diff is not None
            else "Price/VWAP: N/A",
            ema_str,
            rv_str,
        ]
        out.extend(_box("Regime & Market State", reg_rows))
        out.append("")

        # ── DEALER GEX MODEL ──────────────────────────────────────────────
        gp        = self.gamma_profile
        net_gex   = gp.get("current_gex", 0.0)
        gex_reg   = gp.get("regime", "N/A")
        flip      = gp.get("flip_point")
        c_walls   = gp.get("call_walls", [])
        p_walls   = gp.get("put_walls",  [])
        max_gamma = gp.get("max_gamma_strike")
        max_pain  = gp.get("max_pain")
        charm     = gp.get("charm_exposure")
        vanna     = gp.get("vanna_exposure")
        dealer_note = gp.get("dealer_note")

        gex_b     = net_gex / 1e9 if net_gex else None
        if flip is not None:
            flip_rel = "ABOVE" if flip > self.S_0 else "BELOW"
            flip_str = f"{flip:.2f}  ({flip_rel} current spot by {abs(flip-self.S_0):.2f})"
        else:
            flip_str = "N/A (no zero-crossing in ±15% range)"

        gex_rows = [
            f"Convention: dealer LONG calls / SHORT puts (SpotGamma style)",
            f"Net dealer GEX: ${gex_b:+.2f}B  [{gex_reg}]"
            if gex_b is not None else "Net dealer GEX: N/A",
            f"Gamma flip point: {flip_str}",
            "Call walls (top OTM call GEX strikes): "
            + "  ".join(f"{w:.0f}" for w in c_walls[:3]) if c_walls else "Call walls: N/A",
            "Put walls  (top OTM put GEX strikes):  "
            + "  ".join(f"{w:.0f}" for w in p_walls[:3]) if p_walls else "Put walls:  N/A",
            f"Max gamma strike: {max_gamma:.2f}" if max_gamma is not None else "Max gamma strike: N/A",
            f"Max pain: {max_pain:.2f}" if max_pain is not None else "Max pain: N/A",
            f"Charm exposure: {charm:+.2f} Δ/day  (daily dealer delta rebalancing need)"
            if charm is not None else "Charm: N/A",
            f"Vanna exposure: {vanna:+.2f} Δ/vol-pt  (delta sensitivity to vol moves)"
            if vanna is not None else "Vanna: N/A",
            dealer_note or "Dealer flow note: N/A",
        ]
        out.extend(_box("Dealer Gamma Exposure (GEX Model)", gex_rows))
        out.append("")

        # ── VOLATILITY ────────────────────────────────────────────────────
        atm_p    = s.get("atm_iv_put")
        rr       = s.get("risk_reversal_25d")
        fly      = s.get("butterfly_25d")
        ps       = s.get("put_skew")
        cs       = s.get("call_skew")
        svi_cal  = s.get("svi_calibrated_call")
        rmse_c   = s.get("svi_rmse_call")

        atm_avg  = None
        if atm_c and atm_p:
            atm_avg = (atm_c + atm_p) / 2

        iv_rv_str = "N/A"
        if atm_avg and rv is not None and rv > 0:
            iv_rv_str = f"{atm_avg / rv:.2f}×"

        svi_str = ("CALIBRATED" if svi_cal else "FAILED")
        if rmse_c is not None:
            svi_str += f"  (RMSE: {rmse_c*100:.3f}%)"

        ts_str = "N/A"
        if ts:
            ts_str = "  ".join(
                f"{exp[-5:]}:{iv*100:.1f}%"
                for exp, iv in sorted(ts.items())[:5]
            )

        vol_rows = [
            f"ATM IV:  call={fmt_iv(atm_c)}  put={fmt_iv(atm_p)}"
            + (f"  avg={fmt_iv(atm_avg)}" if atm_avg else ""),
            f"25Δ RR: {fmt_pp(rr)}   25Δ Fly: {fmt_pp(fly)}",
            f"Skew:   put={ps:.3f}" + (f"  call={cs:.3f}" if cs else "")
            if ps else "Skew: N/A",
            f"IV/RV:  {iv_rv_str}",
            f"Surface: {svi_str}",
            f"Term:   {ts_str}",
        ]
        out.extend(_box("Volatility Surface", vol_rows))
        out.append("")

        # ── MACRO SNAPSHOT ────────────────────────────────────────────────
        ms = self.macro_screen
        if ms is not None and ms._available:
            d = ms.data
            t10y2y   = d.get("t10y2y")
            hy_oas   = d.get("hy_oas")
            ig_oas   = d.get("ig_oas")
            cpi_yoy  = d.get("cpi_yoy")
            unrate   = d.get("unrate")
            nfci     = d.get("nfci")
            sofr_val = d.get("sofr")
            effr_val = d.get("effr")
            macro_rows = [
                f"Yield curve: 10Y–2Y={t10y2y:.0f}bps"
                + ("  INVERTED" if t10y2y is not None and t10y2y < 0 else "")
                if t10y2y is not None else "Yield curve: N/A",
                f"Credit:  IG={ig_oas:.0f}bps  HY={hy_oas:.0f}bps"
                if ig_oas is not None and hy_oas is not None else "Credit: N/A",
                f"Inflation: CPI YoY={cpi_yoy:.1f}%"
                if cpi_yoy is not None else "Inflation: N/A",
                f"Labour:  UNRATE={unrate:.1f}%"
                if unrate is not None else "Labour: N/A",
                f"Conditions (NFCI): {nfci:+.3f}  "
                + ("LOOSE" if nfci is not None and nfci < -0.3
                   else "TIGHT" if nfci is not None and nfci > 0.3
                   else "NEUTRAL")
                if nfci is not None else "NFCI: N/A",
                f"Rates:  EFFR={effr_val:.2f}%  SOFR={sofr_val:.2f}%"
                if effr_val is not None and sofr_val is not None else "Rates: N/A",
                "→ See MACRO tab for full analysis",
            ]
            out.extend(_box("Macro Snapshot (FRED)", macro_rows))
            out.append("")

        # ── INTERPRETATION ────────────────────────────────────────────────
        interp = _generate_interpretation(
            regime_label, atm_c, rv, rv_iv, rr, fly,
            c_walls[0] if c_walls else None,
            p_walls[0] if p_walls else None,
            self.S_0, net_gex, gex_reg, flip
        )
        out.extend(_box("Tactical Interpretation", interp))
        out.append("")

        return out

    # ------------------------------------------------------------------
    # Helper DataFrames for GAMMA, REGIME, MACRO, SURFACE, DIAGNOSTICS tabs
    # ------------------------------------------------------------------

    def _gamma_gex_df(self) -> pd.DataFrame:
        """
        Return the dealer GEX table for the GAMMA tab.
        Shows per-strike breakdown sorted nearest-to-spot first,
        with GEX in $M, plus summary header rows.
        """
        gp = self.gamma_profile
        gex_df = gp.get("gex_by_strike", pd.DataFrame())

        # Build a summary header block
        net_gex  = gp.get("current_gex", 0.0)
        regime   = gp.get("regime", "N/A")
        flip     = gp.get("flip_point")
        c_walls  = gp.get("call_walls", [])
        p_walls  = gp.get("put_walls", [])
        charm    = gp.get("charm_exposure", 0.0)
        vanna    = gp.get("vanna_exposure", 0.0)
        max_gamma = gp.get("max_gamma_strike")
        max_pain = gp.get("max_pain")

        header_rows = [
            {"Strike": "── Net GEX ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": f"${net_gex/1e9:+.2f}B",
             "Put OI": "", "Put GEX($M)": "", "Net GEX($M)": "",
             "Zone": regime},
            {"Strike": "── Flip Point ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": "",
             "Put OI": "", "Put GEX($M)": "",
             "Net GEX($M)": f"{flip:.2f}" if flip else "N/A",
             "Zone": ""},
            {"Strike": "── Call Walls ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": "  ".join(f"{w:.0f}" for w in c_walls),
             "Put OI": "", "Put GEX($M)": "", "Net GEX($M)": "",
             "Zone": ""},
            {"Strike": "── Put Walls ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": "",
             "Put OI": "", "Put GEX($M)": "  ".join(f"{w:.0f}" for w in p_walls),
             "Net GEX($M)": "", "Zone": ""},
            {"Strike": "── Charm ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": f"{charm:+.2f} Δ/day",
             "Put OI": "", "Put GEX($M)": "", "Net GEX($M)": "",
             "Zone": ""},
            {"Strike": "── Vanna ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": f"{vanna:+.2f} Δ/vol-pt",
             "Put OI": "", "Put GEX($M)": "", "Net GEX($M)": "",
             "Zone": ""},
            {"Strike": "── Max Gamma ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": f"{max_gamma:.2f}" if max_gamma is not None else "N/A",
             "Put OI": "", "Put GEX($M)": "", "Net GEX($M)": "",
             "Zone": ""},
            {"Strike": "── Max Pain ──", "Δ%": "",
             "Call OI": "", "Call GEX($M)": f"{max_pain:.2f}" if max_pain is not None else "N/A",
             "Put OI": "", "Put GEX($M)": "", "Net GEX($M)": "",
             "Zone": ""},
        ]
        header_df = pd.DataFrame(header_rows)

        if gex_df.empty:
            return header_df

        # Ensure all columns exist in both frames before concat
        all_cols = list(gex_df.columns)
        for col in all_cols:
            if col not in header_df.columns:
                header_df[col] = ""
        for col in header_df.columns:
            if col not in gex_df.columns:
                gex_df = gex_df.copy()
                gex_df[col] = ""

        combined = pd.concat(
            [header_df[all_cols], gex_df[all_cols]], ignore_index=True
        )
        return combined

    def _regime_df(self) -> pd.DataFrame:
        reg = self.regime
        rows = [
            ("Regime",         reg.get("regime", "N/A")),
            ("Price",          f"{reg.get('price', 'N/A')}"),
            ("VWAP",           f"{reg.get('vwap', 'N/A')}"),
            ("VWAP Δ%",        f"{reg.get('vwap_pct_diff', 'N/A'):+.3f}%"
                               if reg.get("vwap_pct_diff") is not None else "N/A"),
            ("EMA 9",          f"{reg.get('ema9', 'N/A')}"),
            ("EMA 21",         f"{reg.get('ema21', 'N/A')}"),
            ("EMA 50",         f"{reg.get('ema50', 'N/A')}"),
            ("EMA Aligned",    str(reg.get("ema_aligned", "N/A"))),
            ("Intraday RV",    f"{reg.get('realized_vol', 0)*100:.2f}%"
                               if reg.get("realized_vol") else "N/A"),
            ("RV/IV Ratio",    f"{reg.get('rv_iv_ratio', 'N/A')}"),
            ("Bars Used",      str(reg.get("n_bars", "N/A"))),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def _surface_df(self) -> pd.DataFrame:
        s = self.surface
        fmt_iv   = lambda v: f"{v*100:.2f}%"   if v is not None else "N/A"
        fmt_pp   = lambda v: f"{v*100:+.2f}pp" if v is not None else "N/A"
        fmt_rmse = lambda v: f"{v*100:.3f}%"   if v is not None else "N/A"

        rows = [
            ("ATM IV (calls)",        fmt_iv(s.get("atm_iv_call"))),
            ("ATM IV (puts)",         fmt_iv(s.get("atm_iv_put"))),
            ("Put skew (slope)",      f"{s['put_skew']:.4f}"  if s.get("put_skew")  else "N/A"),
            ("Call skew (slope)",     f"{s['call_skew']:.4f}" if s.get("call_skew") else "N/A"),
            ("25Δ Risk Reversal",     fmt_pp(s.get("risk_reversal_25d"))),
            ("25Δ Butterfly",         fmt_pp(s.get("butterfly_25d"))),
            ("IV range (calls)",      fmt_pp(s.get("iv_range_call"))),
            ("IV range (puts)",       fmt_pp(s.get("iv_range_put"))),
            ("Call strikes",          str(s.get("n_call_strikes", "N/A"))),
            ("Put strikes",           str(s.get("n_put_strikes",  "N/A"))),
            ("SVI calibrated (call)", str(s.get("svi_calibrated_call"))),
            ("SVI calibrated (put)",  str(s.get("svi_calibrated_put"))),
            ("SVI RMSE (call)",       fmt_rmse(s.get("svi_rmse_call"))),
            ("SVI RMSE (put)",        fmt_rmse(s.get("svi_rmse_put"))),
        ]
        # Add term structure
        if self.term_structure:
            rows.append(("── Term Structure ──", ""))
            for exp, iv in sorted(self.term_structure.items()):
                rows.append((f"  {exp}", fmt_iv(iv)))

        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def _diagnostics_df(self) -> pd.DataFrame:
        rows = []
        for w in self.warnings:
            rows.append({"Category": "WARNING", "Detail": w})
        for f in self.flags:
            rows.append({
                "Category": f"FLAG:{f['Type']}",
                "Detail":   f"K={f['Strike']:.0f}  {f['Action']}  {f['Message']}",
            })
        if not rows:
            rows.append({"Category": "OK", "Detail": "No warnings or flags"})
        return pd.DataFrame(rows)

    def _macro_lines(self) -> list[str]:
        """
        Generate the formatted text lines for the MACRO tab.
        Shows indicator table followed by quantitative interpretation.
        """
        ms = self.macro_screen
        W  = 72

        def _box(title: str, rows: list[str]) -> list[str]:
            bar   = "─" * (W - 2)
            lines = [f"┌{bar}┐"]
            lines.append(f"│  {title.upper():<{W-4}}│")
            lines.append(f"├{bar}┤")
            for r in rows:
                lines.append(f"│  {r:<{W-4}}│")
            lines.append(f"└{bar}┘")
            return lines

        out: list[str] = [""]

        if ms is None or not ms._available:
            unavail_rows = [
                "FRED API key not configured.",
                "Set the FRED_API_KEY environment variable to enable:",
                "  export FRED_API_KEY=<your_key>",
                "",
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html",
            ]
            out.extend(_box("Macro Screen — FRED Data", unavail_rows))
            return out

        # ── Indicator table ──────────────────────────────────────────────
        df = ms.as_df()
        indicator_rows = []
        for _, row in df.iterrows():
            ind_str = f"{row['Indicator']:<28}  {row['Value']:<14}  {row['Signal']}"
            indicator_rows.append(ind_str)
        out.extend(_box("Macro Indicators (FRED)", indicator_rows))
        out.append("")

        # ── Interpretation ───────────────────────────────────────────────
        interp_lines = ms.interp_lines()
        wrapped: list[str] = []
        for line in interp_lines:
            # Word-wrap at W-4 chars so it fits in the box
            while len(line) > W - 4:
                cut = line[:W-4].rfind(" ")
                if cut < 10:
                    cut = W - 4
                wrapped.append(line[:cut])
                line = "  " + line[cut:].lstrip()
            wrapped.append(line)
        out.extend(_box("Quantitative Macro Interpretation", wrapped))
        out.append("")

        return out


# ---------------------------------------------------------------------------
# Tactical interpretation generator
# ---------------------------------------------------------------------------

def _generate_interpretation(
    regime: str,
    atm_iv: float | None,
    rv: float | None,
    rv_iv_ratio: float | None,
    rr: float | None,
    fly: float | None,
    call_wall: float | None,
    put_wall: float | None,
    spot: float,
    net_gex: float = 0.0,
    gex_regime: str = "N/A",
    flip_point: float | None = None,
) -> list[str]:
    """
    Generate 4–8 plain-English interpretation lines from the surface/regime state.
    Thresholds are quantitative and grounded in market microstructure research.
    This is NOT investment advice — it is a mechanical summary of observed metrics.
    """
    lines: list[str] = []

    # Regime line
    regime_commentary = {
        "TREND UP":       "Price trending up — VWAP and EMA stack aligned bullish. "
                          "Momentum strategies favoured; avoid fading the trend.",
        "TREND DOWN":     "Price trending down — VWAP and EMA stack aligned bearish. "
                          "Momentum strategies favoured; avoid fading the trend.",
        "CHOP":           "Range-bound — EMA stack mixed, no clear directional bias. "
                          "Mean-reversion setups preferred; reduce breakout risk.",
        "COMPRESSION":    "Volatility compression — RV well below IV. "
                          "Historically precedes an expansion; watch for directional catalyst.",
        "EXTENDED UP":    "Trend up but price >2σ extended above VWAP. "
                          "Pullback or mean-reversion risk elevated near-term.",
        "EXTENDED DOWN":  "Trend down but price >2σ extended below VWAP. "
                          "Short-cover bounce risk elevated; avoid chasing.",
        "UNAVAILABLE":    "Intraday data unavailable. Regime classification skipped.",
        "UNKNOWN":        "Insufficient intraday bars for regime classification.",
    }
    lines.append(regime_commentary.get(regime, f"Regime: {regime}."))

    # Dealer gamma regime
    if gex_regime not in ("N/A", ""):
        gex_b = net_gex / 1e9 if net_gex else 0
        if gex_regime == "LONG GAMMA":
            lines.append(
                f"Dealer GEX: {gex_b:+.2f}B [LONG GAMMA] — dealers are net long gamma. "
                "Expect mean-reverting price action; large moves tend to be faded by dealer hedging."
            )
        else:
            lines.append(
                f"Dealer GEX: {gex_b:+.2f}B [SHORT GAMMA] — dealers are net short gamma. "
                "Expect trending/amplified price action; dealer hedging adds momentum to moves."
            )
        if flip_point is not None:
            rel  = "above" if flip_point > spot else "below"
            dist = abs(flip_point - spot)
            lines.append(
                f"Gamma flip at {flip_point:.2f} ({rel} spot by {dist:.2f}). "
                "Crossing the flip inverts dealer hedging behaviour."
            )

    # IV vs RV
    if rv_iv_ratio is not None:
        if rv_iv_ratio > 1.15:
            lines.append(
                f"RV/IV={rv_iv_ratio:.2f}: realized vol > implied — "
                "options appear cheap; vol buyers have statistical edge over this window."
            )
        elif rv_iv_ratio < 0.70:
            lines.append(
                f"RV/IV={rv_iv_ratio:.2f}: IV elevated vs RV — "
                "premium-selling environment; theta decay working in sellers' favour."
            )
        else:
            lines.append(f"RV/IV={rv_iv_ratio:.2f}: IV and RV broadly in line.")
    elif atm_iv is not None:
        lines.append(f"ATM IV = {atm_iv*100:.1f}% (intraday RV not available).")

    # Skew / RR
    if rr is not None:
        rr_pct = rr * 100
        if rr_pct < -1.5:
            lines.append(
                f"Put skew elevated (25Δ RR {rr_pct:+.1f}pp): "
                "options market pricing significant downside tail. "
                "Protective puts expensive; consider spreads to reduce premium cost."
            )
        elif rr_pct > 1.5:
            lines.append(
                f"Call skew elevated (25Δ RR {rr_pct:+.1f}pp): "
                "upside demand dominant. "
                "Calls carry premium; credit call spreads offer better risk/reward than naked calls."
            )
        else:
            lines.append(f"Skew neutral (25Δ RR {rr_pct:+.1f}pp).")

    # Butterfly
    if fly is not None and abs(fly * 100) > 0.3:
        fly_pct = fly * 100
        if fly_pct > 0:
            lines.append(
                f"Butterfly {fly_pct:+.1f}pp — wings expensive relative to body. "
                "Market pricing fat tails; straddles/strangles relatively cheap vs. risk reversals."
            )
        else:
            lines.append(
                f"Butterfly {fly_pct:+.1f}pp — wings cheap relative to body. "
                "Condors and iron flies offer good risk/reward given wing compression."
            )

    # Key GEX levels
    level_parts = []
    if call_wall:
        level_parts.append(f"call wall {call_wall:.0f} (dealer long-gamma OTM call concentration)")
    if put_wall:
        level_parts.append(f"put wall {put_wall:.0f} (dealer short-gamma OTM put concentration)")
    if level_parts:
        lines.append("GEX levels: " + ", ".join(level_parts) + ".")

    lines.append("─" * 60)
    lines.append("NOT investment advice. Mechanical summary of observed metrics only.")
    return lines
