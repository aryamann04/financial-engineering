"""
gamma_zones.py — OI/volume concentration heuristics for equity options.

DISCLAIMER: This is NOT a formal dealer gamma / charm / vanna model.
These are open interest and volume concentration heuristics that identify
strikes where large OI accumulates.  They may create pin risk and loose
support/resistance references, but MUST NOT be used as precise risk levels.

Zone types
----------
  pin_zone   — top-N strikes by total OI (potential magnetic / pin levels)
  call_wall  — highest-OI OTM call strike (informal resistance proxy)
  put_wall   — highest-OI OTM put strike (informal support proxy)
  call_dom   — OTM call strikes where call OI >> put OI
  put_dom    — OTM put strikes where put OI >> call OI
"""

import numpy as np
import pandas as pd


def compute_gamma_zones(
    calls_df: pd.DataFrame,
    puts_df: pd.DataFrame,
    S_0: float,
    top_n: int = 5,
) -> dict:
    """
    Build OI/volume concentration heuristics from the option chain.

    Parameters
    ----------
    calls_df  : yfinance calls DataFrame (must have strike, openInterest, volume)
    puts_df   : yfinance puts DataFrame
    S_0       : current spot price
    top_n     : number of top strikes per category

    Returns
    -------
    dict with keys:
      ladder_df     — full strike-level OI/volume ladder DataFrame
      pin_zones     — list of dicts {strike, total_oi, call_oi, put_oi}
      call_wall     — float or None
      put_wall      — float or None
      call_dom      — list of dicts {strike, call_oi, put_oi}
      put_dom       — list of dicts {strike, call_oi, put_oi}
      warnings      — list of data quality notes
    """
    warnings_out: list[str] = []

    def _safe(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)

    # ── Build unified strike ladder ──────────────────────────────────────
    c = calls_df[["strike", "openInterest", "volume"]].copy()
    p = puts_df[["strike",  "openInterest", "volume"]].copy()
    c.columns = ["strike", "call_oi", "call_vol"]
    p.columns = ["strike", "put_oi",  "put_vol"]
    for col in ["strike", "call_oi", "call_vol"]:
        c[col] = _safe(c[col])
    for col in ["strike", "put_oi", "put_vol"]:
        p[col] = _safe(p[col])

    ladder = (
        pd.merge(c, p, on="strike", how="outer")
        .fillna(0)
        .assign(
            total_oi  = lambda df: df["call_oi"]  + df["put_oi"],
            total_vol = lambda df: df["call_vol"] + df["put_vol"],
        )
        .sort_values("strike")
        .reset_index(drop=True)
    )

    if ladder.empty or ladder["total_oi"].sum() == 0:
        warnings_out.append("OI data is empty or all-zero; gamma zones unavailable.")
        return {
            "ladder_df": ladder,
            "pin_zones": [],
            "call_wall": None,
            "put_wall":  None,
            "call_dom":  [],
            "put_dom":   [],
            "warnings":  warnings_out,
        }

    # ── Pin zones: highest total OI ──────────────────────────────────────
    pin_zones = (
        ladder.nlargest(top_n, "total_oi")
        [["strike", "total_oi", "call_oi", "put_oi"]]
        .to_dict("records")
    )

    # ── Call wall: highest OI OTM call ───────────────────────────────────
    otm_calls = ladder[ladder["strike"] > S_0]
    if not otm_calls.empty and otm_calls["call_oi"].max() > 0:
        call_wall = float(otm_calls.nlargest(1, "call_oi")["strike"].iloc[0])
    else:
        call_wall = None

    # ── Put wall: highest OI OTM put ─────────────────────────────────────
    otm_puts = ladder[ladder["strike"] < S_0]
    if not otm_puts.empty and otm_puts["put_oi"].max() > 0:
        put_wall = float(otm_puts.nlargest(1, "put_oi")["strike"].iloc[0])
    else:
        put_wall = None

    # ── Dominated strikes ────────────────────────────────────────────────
    call_dom = (
        ladder[
            (ladder["strike"] > S_0)
            & (ladder["call_oi"] > 2 * ladder["put_oi"])
            & (ladder["call_oi"] > 0)
        ]
        .nlargest(top_n, "call_oi")
        [["strike", "call_oi", "put_oi"]]
        .to_dict("records")
    )

    put_dom = (
        ladder[
            (ladder["strike"] < S_0)
            & (ladder["put_oi"] > 2 * ladder["call_oi"])
            & (ladder["put_oi"] > 0)
        ]
        .nlargest(top_n, "put_oi")
        [["strike", "call_oi", "put_oi"]]
        .to_dict("records")
    )

    return {
        "ladder_df": ladder,
        "pin_zones": pin_zones,
        "call_wall": call_wall,
        "put_wall":  put_wall,
        "call_dom":  call_dom,
        "put_dom":   put_dom,
        "warnings":  warnings_out,
    }


def oi_ladder_summary(
    zones: dict,
    S_0: float,
    n_show: int = 12,
) -> pd.DataFrame:
    """
    Return a tidy strike ladder centered on the current spot, with zone labels.

    Shows n_show strikes above and below S_0, sorted descending (nearest
    above first, nearest below second).
    """
    df = zones.get("ladder_df", pd.DataFrame())
    if df.empty:
        return df

    above = df[df["strike"] >= S_0].head(n_show)
    below = df[df["strike"] <  S_0].tail(n_show)
    ladder = (
        pd.concat([below, above])
        .sort_values("strike", ascending=False)
        .reset_index(drop=True)
    )

    pin_strikes = {z["strike"] for z in zones.get("pin_zones", [])}
    call_wall   = zones.get("call_wall")
    put_wall    = zones.get("put_wall")

    def _label(row: pd.Series) -> str:
        k = row["strike"]
        labels: list[str] = []
        if k in pin_strikes:
            labels.append("PIN")
        if call_wall is not None and k == call_wall:
            labels.append("CALL WALL")
        if put_wall is not None and k == put_wall:
            labels.append("PUT WALL")
        if abs(k - S_0) / S_0 < 0.001:
            labels.append("~ATM")
        return ", ".join(labels) if labels else ""

    ladder = ladder.copy()
    ladder["Zone"] = ladder.apply(_label, axis=1)
    ladder["Δ%"]   = ((ladder["strike"] - S_0) / S_0 * 100).round(2)

    return (
        ladder[["strike", "Δ%", "call_oi", "put_oi", "total_oi", "Zone"]]
        .rename(columns={
            "strike":   "Strike",
            "call_oi":  "Call OI",
            "put_oi":   "Put OI",
            "total_oi": "Total OI",
        })
    )
