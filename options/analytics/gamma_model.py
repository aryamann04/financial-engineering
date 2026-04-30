"""
gamma_model.py — Production dealer gamma exposure (GEX) model.

Convention (SpotGamma / SqueezeMetrics style)
---------------------------------------------
  Calls: dealers are net LONG (covered-call writers sell calls to dealers).
         → dealer holds positive gamma on calls → stabilising.
  Puts : dealers are net SHORT (hedgers buy puts from dealers).
         → dealer holds negative gamma on puts → destabilising.

  Net GEX = Σ_calls(γ·OI·100·S) − Σ_puts(γ·OI·100·S)

  Positive Net GEX → dealers are net long gamma → mean-reverting environment.
  Negative Net GEX → dealers are net short gamma → trending / amplifying environment.

Greeks
------
  gamma  = N'(d1) / (S·σ·√T)            [Δ per $1 spot move]
  vanna  = −N'(d1)·d2 / σ               [Δ per vol-point move]
  charm  = dΔ/dt  (annualised)          [Δ per year of time decay]

GEX profile (sticky-strike model)
----------------------------------
  Compute Net GEX at 200 spot levels in [0.85·S, 1.15·S].
  Each option's implied vol is held constant at its current market IV
  (sticky-strike assumption).  The zero-crossing of this profile is the
  "gamma flip point" — the level where dealer hedging behaviour reverses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Greek helpers (vectorised-friendly)
# ---------------------------------------------------------------------------

def _d1d2(S: np.ndarray | float,
          K: np.ndarray | float,
          T: float,
          r: float,
          sigma: np.ndarray | float,
          q: float = 0.0):
    """Compute d1 and d2. Mask invalid σ/T with NaN."""
    sqrtT = np.sqrt(max(T, 1e-8))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
    return d1, d2


def _bs_gamma_vec(S: np.ndarray,
                  K: np.ndarray,
                  T: float,
                  r: float,
                  sigma: np.ndarray,
                  q: float = 0.0) -> np.ndarray:
    """
    Vectorised BS gamma.  S may be a 2-D grid (n_spots × n_strikes),
    K and sigma must be 1-D arrays of shape (n_strikes,).
    Returns array of same shape as S.
    """
    sqrtT = np.sqrt(max(T, 1e-8))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        gamma = norm.pdf(d1) / (S * sigma * sqrtT)
    gamma = np.where(np.isfinite(gamma), gamma, 0.0)
    return gamma


def _bs_vanna_scalar(S: float, K: float, T: float, r: float,
                     sigma: float, q: float = 0.0) -> float:
    """vanna = −N'(d1)·d2 / σ"""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    return float(-norm.pdf(d1) * d2 / sigma)


def _bs_charm_scalar(S: float, K: float, T: float, r: float,
                     sigma: float, q: float = 0.0) -> float:
    """
    charm = dDelta/dt (annualised).
    Approximation for q ≈ 0:
      charm ≈ −N'(d1) · [2(r−q)T − d2·σ·√T] / (2T·σ·√T)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    num = 2.0 * (r - q) * T - d2 * sigma * sqrtT
    denom = 2.0 * T * sigma * sqrtT
    if denom == 0:
        return 0.0
    return float(-norm.pdf(d1) * num / denom)


# ---------------------------------------------------------------------------
# Per-strike GEX at current spot
# ---------------------------------------------------------------------------

def _strike_gex(df: pd.DataFrame,
                S_0: float, T: float, r: float, q: float,
                sign: float) -> pd.DataFrame:
    """
    Compute per-strike GEX, vanna, and charm for one leg (calls or puts).

    Parameters
    ----------
    df    : chain DataFrame with columns 'strike', 'openInterest', 'impliedVolatility'
    sign  : +1.0 for calls (dealer long), -1.0 for puts (dealer short)

    Returns
    -------
    DataFrame with Strike, OI, Sigma, Gamma, GEX_dollar, Vanna_exp, Charm_daily
    """
    rows = []
    for _, row in df.iterrows():
        K     = float(row.get("strike", 0) or 0)
        oi    = float(row.get("openInterest", 0) or 0)
        sigma = float(row.get("impliedVolatility", 0) or 0)
        if K <= 0 or oi <= 0 or sigma <= 0 or sigma > 5.0:
            continue

        gamma  = float(_bs_gamma_vec(S_0, K, T, r, sigma, q))
        gex_d  = sign * gamma * oi * 100.0 * (S_0 ** 2) * 0.01
        vanna  = sign * _bs_vanna_scalar(S_0, K, T, r, sigma, q) * oi * 100.0
        charm  = sign * _bs_charm_scalar(S_0, K, T, r, sigma, q) * oi * 100.0 / 252.0

        rows.append({
            "Strike":       K,
            "OI":           int(oi),
            "Sigma":        sigma,
            "Gamma":        gamma,
            "GEX_dollar":   gex_d,
            "Vanna_exp":    vanna,
            "Charm_daily":  charm,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GEX profile (sticky-strike) across spot range
# ---------------------------------------------------------------------------

def _gex_profile(call_df_raw: pd.DataFrame,
                 put_df_raw:  pd.DataFrame,
                 S_0: float, T: float, r: float, q: float,
                 n_points: int = 200) -> pd.DataFrame:
    """
    Evaluate net dealer GEX at n_points evenly spaced spot levels
    between 0.85·S and 1.15·S using the sticky-strike assumption.

    Returns DataFrame with columns ['Spot', 'Net_GEX'].
    """

    def _extract(df: pd.DataFrame):
        strikes, ois, sigmas = [], [], []
        for _, row in df.iterrows():
            K     = float(row.get("strike", 0) or 0)
            oi    = float(row.get("openInterest", 0) or 0)
            sigma = float(row.get("impliedVolatility", 0) or 0)
            if K > 0 and oi > 0 and 0 < sigma < 5.0:
                strikes.append(K)
                ois.append(oi)
                sigmas.append(sigma)
        return np.array(strikes), np.array(ois), np.array(sigmas)

    c_K, c_oi, c_sig = _extract(call_df_raw)
    p_K, p_oi, p_sig = _extract(put_df_raw)

    spots = np.linspace(0.85 * S_0, 1.15 * S_0, n_points)

    net_gex = np.zeros(n_points)

    # Calls: dealers long → positive GEX
    if len(c_K) > 0:
        # spots: (n_points,1)  ×  K: (n_calls,)  → broadcast
        S_grid  = spots[:, None]       # (n, 1)
        K_grid  = c_K[None, :]        # (1, m)
        sig_grid = c_sig[None, :]      # (1, m)
        g = _bs_gamma_vec(S_grid, K_grid, T, r, sig_grid, q)   # (n, m)
        # GEX at each spot, for each call: gamma × oi × 100 × spot
        gex_c = (g * c_oi[None, :] * 100.0 * (S_grid ** 2) * 0.01).sum(axis=1)  # (n,)
        net_gex += gex_c

    # Puts: dealers short → negative GEX
    if len(p_K) > 0:
        S_grid  = spots[:, None]
        K_grid  = p_K[None, :]
        sig_grid = p_sig[None, :]
        g = _bs_gamma_vec(S_grid, K_grid, T, r, sig_grid, q)
        gex_p = (g * p_oi[None, :] * 100.0 * (S_grid ** 2) * 0.01).sum(axis=1)
        net_gex -= gex_p

    return pd.DataFrame({"Spot": spots, "Net_GEX": net_gex})


# ---------------------------------------------------------------------------
# Flip-point finder
# ---------------------------------------------------------------------------

def _find_flip(profile_df: pd.DataFrame) -> float | None:
    """
    Find the spot price where Net_GEX crosses zero (sign change).
    Uses linear interpolation for sub-tick precision.
    Returns None if no crossing is found in the profile range.
    """
    spots = profile_df["Spot"].values
    gex   = profile_df["Net_GEX"].values
    signs = np.sign(gex)
    # Find first sign change (excluding zero)
    for i in range(len(signs) - 1):
        if signs[i] != 0 and signs[i + 1] != 0 and signs[i] != signs[i + 1]:
            x1, x2 = spots[i], spots[i + 1]
            y1, y2 = gex[i],   gex[i + 1]
            if y2 - y1 == 0:
                return float((x1 + x2) / 2)
            return float(x1 - y1 * (x2 - x1) / (y2 - y1))
    return None


def _max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float | None:
    strikes = sorted(
        {
            float(v)
            for v in pd.concat(
                [calls_df.get("strike", pd.Series(dtype=float)), puts_df.get("strike", pd.Series(dtype=float))]
            ).dropna().tolist()
        }
    )
    if not strikes:
        return None
    best_strike = None
    best_loss = None
    for spot in strikes:
        total_loss = 0.0
        for _, row in calls_df.iterrows():
            strike = float(row.get("strike", 0) or 0)
            oi = float(row.get("openInterest", 0) or 0)
            total_loss += max(spot - strike, 0.0) * oi * 100.0
        for _, row in puts_df.iterrows():
            strike = float(row.get("strike", 0) or 0)
            oi = float(row.get("openInterest", 0) or 0)
            total_loss += max(strike - spot, 0.0) * oi * 100.0
        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_strike = spot
    return best_strike


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def compute_dealer_gex(
    calls_df: pd.DataFrame,
    puts_df:  pd.DataFrame,
    S_0: float,
    T: float,
    r: float,
    q: float = 0.0,
    top_walls: int = 3,
    display_window_pct: float = 0.12,
) -> dict:
    """
    Compute the full dealer gamma exposure (GEX) model.

    Parameters
    ----------
    calls_df, puts_df : yfinance chain DataFrames
        Must have columns: strike, openInterest, impliedVolatility
    S_0         : current spot price
    T           : time to expiry in years
    r           : risk-free rate (decimal)
    q           : continuous dividend yield (decimal)
    top_walls   : number of top GEX strikes to report as call/put walls

    Returns
    -------
    dict with:
      gex_by_strike  : DataFrame — per-strike GEX breakdown
      gex_profile    : DataFrame — Net_GEX vs Spot across ±15% range
      flip_point     : float | None — spot where net GEX = 0
      call_walls     : list[float] — top call-GEX OTM strikes
      put_walls      : list[float] — top put-GEX OTM strikes (by absolute GEX)
      charm_exposure : float — total daily delta change (Δ/day)
      vanna_exposure : float — total vanna (Δ per vol-point)
      current_gex    : float — net GEX in $ at current spot
      regime         : str — "LONG GAMMA" | "SHORT GAMMA"
      warnings       : list[str]
    """
    warnings_out: list[str] = []

    if T <= 0:
        warnings_out.append("GEX: T ≤ 0 — cannot compute gamma model.")
        return _empty_result(warnings_out)

    # ── Per-strike GEX tables ────────────────────────────────────────────
    call_gex_df = _strike_gex(calls_df, S_0, T, r, q, sign= 1.0)
    put_gex_df  = _strike_gex(puts_df,  S_0, T, r, q, sign=-1.0)

    if call_gex_df.empty and put_gex_df.empty:
        warnings_out.append("GEX: No valid strikes with IV and OI — model unavailable.")
        return _empty_result(warnings_out)

    # ── Aggregate ────────────────────────────────────────────────────────
    net_gex      = (call_gex_df["GEX_dollar"].sum() if not call_gex_df.empty else 0.0) \
                 + (put_gex_df["GEX_dollar"].sum()  if not put_gex_df.empty  else 0.0)
    charm_total  = (call_gex_df["Charm_daily"].sum() if not call_gex_df.empty else 0.0) \
                 + (put_gex_df["Charm_daily"].sum()  if not put_gex_df.empty  else 0.0)
    vanna_total  = (call_gex_df["Vanna_exp"].sum()  if not call_gex_df.empty else 0.0) \
                 + (put_gex_df["Vanna_exp"].sum()   if not put_gex_df.empty  else 0.0)

    lower_bound = S_0 * (1.0 - display_window_pct)
    upper_bound = S_0 * (1.0 + display_window_pct)
    call_display_df = call_gex_df[
        (call_gex_df["Strike"] >= lower_bound) & (call_gex_df["Strike"] <= upper_bound)
    ].copy() if not call_gex_df.empty else call_gex_df
    put_display_df = put_gex_df[
        (put_gex_df["Strike"] >= lower_bound) & (put_gex_df["Strike"] <= upper_bound)
    ].copy() if not put_gex_df.empty else put_gex_df
    display_call_pool = call_display_df if not call_display_df.empty else call_gex_df
    display_put_pool = put_display_df if not put_display_df.empty else put_gex_df

    # ── Walls — OTM strikes ranked by |GEX| near current spot ───────────
    if not display_call_pool.empty:
        otm_calls = display_call_pool[display_call_pool["Strike"] > S_0].copy()
        call_walls = (
            otm_calls.nlargest(top_walls, "GEX_dollar")["Strike"].tolist()
            if not otm_calls.empty else []
        )
    else:
        call_walls = []

    if not display_put_pool.empty:
        otm_puts = display_put_pool[display_put_pool["Strike"] < S_0].copy()
        put_walls = (
            otm_puts.nsmallest(top_walls, "GEX_dollar")["Strike"].tolist()
            if not otm_puts.empty else []
        )
    else:
        put_walls = []

    # ── Combined per-strike table (for display) ──────────────────────────
    # Merge calls and puts on strike; fill missing with 0
    def _label_side(df, col_suffix):
        if df.empty:
            return pd.DataFrame(columns=["Strike", f"GEX_{col_suffix}",
                                         f"OI_{col_suffix}"])
        out = df[["Strike", "GEX_dollar", "OI"]].copy()
        out.columns = ["Strike", f"GEX_{col_suffix}", f"OI_{col_suffix}"]
        return out

    c_tbl = _label_side(display_call_pool, "Call")
    p_tbl = _label_side(display_put_pool, "Put")
    merged = pd.merge(c_tbl, p_tbl, on="Strike", how="outer").fillna(0)
    merged["Net_GEX"]  = merged.get("GEX_Call", 0) + merged.get("GEX_Put", 0)
    merged["GEX_M_Call"] = (merged.get("GEX_Call", 0) / 1e6).round(1)
    merged["GEX_M_Put"]  = (merged.get("GEX_Put",  0) / 1e6).round(1)
    merged["GEX_M_Net"]  = (merged["Net_GEX"] / 1e6).round(1)
    merged["Δ%"]         = ((merged["Strike"] - S_0) / S_0 * 100).round(2)

    # Zone labels
    call_wall_set = set(call_walls)
    put_wall_set  = set(put_walls)
    def _zone(row):
        z = []
        if abs(row["Strike"] - S_0) / S_0 < 0.001:
            z.append("~ATM")
        if row["Strike"] in call_wall_set:
            z.append("CALL WALL")
        if row["Strike"] in put_wall_set:
            z.append("PUT WALL")
        return ", ".join(z) if z else ""
    merged["Zone"] = merged.apply(_zone, axis=1)
    merged = merged.sort_values("Strike", ascending=False).reset_index(drop=True)

    display_cols = ["Strike", "Δ%", "OI_Call", "GEX_M_Call",
                    "OI_Put", "GEX_M_Put", "GEX_M_Net", "Zone"]
    # Keep only columns that exist
    display_cols = [c for c in display_cols if c in merged.columns]
    gex_by_strike = merged[display_cols].rename(columns={
        "OI_Call":    "Call OI",
        "GEX_M_Call": "Call GEX($M)",
        "OI_Put":     "Put OI",
        "GEX_M_Put":  "Put GEX($M)",
        "GEX_M_Net":  "Net GEX($M)",
    })

    # ── GEX profile across spot range ────────────────────────────────────
    try:
        profile = _gex_profile(calls_df, puts_df, S_0, T, r, q)
        flip    = _find_flip(profile)
    except Exception as e:
        warnings_out.append(f"GEX profile computation failed: {e}")
        profile = pd.DataFrame(columns=["Spot", "Net_GEX"])
        flip    = None

    calls_relevant = calls_df[
        (pd.to_numeric(calls_df.get("strike"), errors="coerce") >= lower_bound)
        & (pd.to_numeric(calls_df.get("strike"), errors="coerce") <= upper_bound)
    ].copy()
    puts_relevant = puts_df[
        (pd.to_numeric(puts_df.get("strike"), errors="coerce") >= lower_bound)
        & (pd.to_numeric(puts_df.get("strike"), errors="coerce") <= upper_bound)
    ].copy()
    max_pain = _max_pain(calls_relevant if not calls_relevant.empty else calls_df, puts_relevant if not puts_relevant.empty else puts_df)
    max_gamma_strike = None
    if not merged.empty and "Net_GEX" in merged.columns:
        strongest = merged.iloc[merged["Net_GEX"].abs().argmax()]
        max_gamma_strike = float(strongest["Strike"])

    # ── Regime ───────────────────────────────────────────────────────────
    regime = "LONG GAMMA" if net_gex >= 0 else "SHORT GAMMA"
    dealer_note = (
        "Dealers likely dampen moves around large gamma strikes and hedge with the tape."
        if regime == "LONG GAMMA"
        else "Dealers likely hedge against the tape, which can amplify directional moves."
    )

    if len(call_gex_df) < 3:
        warnings_out.append("GEX: Fewer than 3 call strikes with valid IV — results may be imprecise.")
    if len(put_gex_df) < 3:
        warnings_out.append("GEX: Fewer than 3 put strikes with valid IV — results may be imprecise.")

    return {
        "gex_by_strike":  gex_by_strike,
        "gex_profile":    profile,
        "flip_point":     flip,
        "call_walls":     call_walls,
        "put_walls":      put_walls,
        "max_gamma_strike": max_gamma_strike,
        "max_pain":       max_pain,
        "charm_exposure": float(charm_total),
        "vanna_exposure": float(vanna_total),
        "current_gex":    float(net_gex),
        "regime":         regime,
        "dealer_note":    dealer_note,
        "warnings":       warnings_out,
    }


def _empty_result(warnings: list[str]) -> dict:
    return {
        "gex_by_strike":  pd.DataFrame(),
        "gex_profile":    pd.DataFrame(columns=["Spot", "Net_GEX"]),
        "flip_point":     None,
        "call_walls":     [],
        "put_walls":      [],
        "max_gamma_strike": None,
        "max_pain":       None,
        "charm_exposure": 0.0,
        "vanna_exposure": 0.0,
        "current_gex":    0.0,
        "regime":         "N/A",
        "dealer_note":    "",
        "warnings":       warnings,
    }
