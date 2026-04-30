"""
visualizer.py — Matplotlib dashboards for the intraday options analyzer.

Charts
------
  plot_intraday_dashboard(ticker, intraday_df, gamma_zones, surface, regime)
    3-panel figure:
      Panel 1 — price + VWAP + EMA9/21 + gamma zone overlays
      Panel 2 — intraday volume bars
      Panel 3 — IV term structure (ATM IV vs expiry)

  plot_oi_ladder(oi_ladder_df, S_0)
    Horizontal bar chart of call/put OI by strike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Intraday dashboard
# ---------------------------------------------------------------------------

def plot_intraday_dashboard(
    ticker: str,
    intraday_df: pd.DataFrame,
    gamma_zones: dict,
    surface: dict,
    regime: dict,
    term_structure: dict[str, float] | None = None,
) -> None:
    """
    3-panel intraday dashboard.

    Parameters
    ----------
    ticker        : symbol string
    intraday_df   : OHLCV DataFrame from regime.fetch_intraday()
    gamma_zones   : dict from gamma_zones.compute_gamma_zones()
    surface       : surface diagnostics dict from Analyzer._compute_surface_diagnostics()
    regime        : regime dict from regime.detect_regime()
    term_structure: {expiry_str: atm_iv_decimal} or None
    """
    if intraday_df is None or intraday_df.empty:
        print("[visualizer] No intraday data; skipping dashboard.")
        return

    from options.analytics.regime import compute_vwap, compute_ema

    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[3, 1, 1.5], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])   # price + overlays
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # volume
    ax3 = fig.add_subplot(gs[2])   # term structure

    close  = intraday_df["Close"]
    times  = intraday_df.index
    vwap   = compute_vwap(intraday_df)
    ema9   = compute_ema(close, 9)
    ema21  = compute_ema(close, 21)

    # ── Panel 1: Price ─────────────────────────────────────────────────
    ax1.plot(times, close,  color="#1f77b4", linewidth=1.2, label="Price")
    ax1.plot(times, vwap,   color="orange",  linewidth=1.0, linestyle="--", label="VWAP")
    ax1.plot(times, ema9,   color="lime",    linewidth=0.8, linestyle=":",  label="EMA9")
    ax1.plot(times, ema21,  color="magenta", linewidth=0.8, linestyle=":",  label="EMA21")

    # Gamma zone overlays
    call_wall = gamma_zones.get("call_wall")
    put_wall  = gamma_zones.get("put_wall")
    if call_wall:
        ax1.axhline(call_wall, color="red",   linestyle="-.", linewidth=0.9,
                    label=f"Call Wall {call_wall:.0f}")
    if put_wall:
        ax1.axhline(put_wall,  color="green", linestyle="-.", linewidth=0.9,
                    label=f"Put Wall {put_wall:.0f}")

    # Pin zones (shade ±0.5% band around top pin)
    pin_zones = gamma_zones.get("pin_zones", [])
    if pin_zones:
        top_pin = pin_zones[0]["strike"]
        ax1.axhspan(top_pin * 0.995, top_pin * 1.005, alpha=0.10,
                    color="yellow", label=f"Pin ~{top_pin:.0f}")

    # Regime label in corner
    reg_label  = regime.get("regime", "UNKNOWN")
    reg_colors = {
        "TREND UP":      "limegreen",
        "TREND DOWN":    "tomato",
        "CHOP":          "gold",
        "COMPRESSION":   "cyan",
        "EXTENDED UP":   "lime",
        "EXTENDED DOWN": "red",
        "UNKNOWN":       "gray",
    }
    ax1.text(
        0.01, 0.97, f"Regime: {reg_label}",
        transform=ax1.transAxes, fontsize=10, fontweight="bold",
        color=reg_colors.get(reg_label, "white"),
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.75),
    )

    atm_c = surface.get("atm_iv_call")
    if atm_c is not None:
        ax1.text(
            0.99, 0.97, f"ATM IV: {atm_c*100:.1f}%",
            transform=ax1.transAxes, fontsize=9,
            color="white", horizontalalignment="right", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", alpha=0.75),
        )

    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker} — Intraday Dashboard", fontsize=13)
    ax1.legend(loc="upper left", fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Volume ───────────────────────────────────────────────
    ax2.bar(times, intraday_df["Volume"], color="#4a90d9", alpha=0.6, width=0.003)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="x", labelbottom=False)

    # ── Panel 3: IV Term Structure ────────────────────────────────────
    if term_structure and len(term_structure) >= 2:
        # Sort by maturity
        sorted_ts = sorted(term_structure.items(),
                           key=lambda kv: kv[0])   # expiry str sorts chrono
        labels = [kv[0] for kv in sorted_ts]
        ivs    = [kv[1] * 100 for kv in sorted_ts]

        ax3.plot(range(len(labels)), ivs, marker="o", color="#f0a500",
                 linewidth=1.5, markersize=6, label="ATM IV")
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax3.set_ylabel("ATM IV (%)")
        ax3.set_title("IV Term Structure", fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Mark the primary expiry in the analyzer
        # (first in sorted_ts; mark with a dashed vertical line)
        ax3.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    else:
        ax3.text(0.5, 0.5, "Term structure unavailable",
                 transform=ax3.transAxes, ha="center", va="center", color="gray")
        ax3.set_title("IV Term Structure", fontsize=10)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# OI ladder chart
# ---------------------------------------------------------------------------

def plot_oi_ladder(
    oi_ladder_df: pd.DataFrame,
    S_0: float,
    title: str = "Open Interest by Strike",
) -> None:
    """
    Horizontal bar chart showing call and put OI by strike.

    Call OI extends to the right (positive); put OI extends to the left
    (negative), creating a classic back-to-back ladder.
    """
    if oi_ladder_df is None or oi_ladder_df.empty:
        print("[visualizer] No OI ladder data; skipping chart.")
        return

    df = oi_ladder_df.copy()

    # Standardise column names (accept either raw or summary format)
    rename = {}
    for raw, nice in [("call_oi", "Call OI"), ("put_oi", "Put OI"),
                      ("strike", "Strike")]:
        if raw in df.columns:
            rename[raw] = nice
    df = df.rename(columns=rename)

    if "Strike" not in df.columns:
        print("[visualizer] OI ladder missing 'Strike' column.")
        return

    strikes   = df["Strike"].values
    call_oi   = df.get("Call OI", pd.Series(np.zeros(len(df)))).values
    put_oi    = df.get("Put OI",  pd.Series(np.zeros(len(df)))).values

    fig, ax = plt.subplots(figsize=(9, max(6, len(strikes) * 0.28)))

    y_pos = np.arange(len(strikes))
    ax.barh(y_pos,  call_oi, color="#1f77b4", alpha=0.8, label="Call OI")
    ax.barh(y_pos, -put_oi,  color="#d62728", alpha=0.8, label="Put OI")

    # Spot price marker
    atm_idx = np.argmin(np.abs(strikes - S_0))
    ax.axhline(y_pos[atm_idx], color="gold", linestyle="--",
               linewidth=1.0, label=f"Spot ≈ {S_0:.1f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{s:.0f}" for s in strikes], fontsize=8)
    ax.set_xlabel("Open Interest  (calls →, puts ←)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.3)

    # Zero line
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.show()
