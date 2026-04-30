"""
macro_screen.py — Real-time macro indicator dashboard via FRED.

Tracks
------
  Yield curve   : T10Y2Y (10Y–2Y slope), T10Y3M (10Y–3M), DGS2, DGS10
  Policy rates  : EFFR (effective Fed funds), SOFR (overnight)
  Credit        : BAMLC0A0CM (IG OAS bps), BAMLH0A0HYM2 (HY OAS bps)
  Inflation     : CPIAUCSL (CPI YoY%), CPILFESL (Core CPI YoY%), T10YIE (10Y breakeven)
  Employment    : UNRATE (%), PAYEMS (MoM change, thousands), ICSA (initial claims, thousands)
  Conditions    : NFCI (Chicago Fed — 0 = neutral, + = tight, − = loose)

All data sourced from FRED (fredapi).  Requires FRED_API_KEY env var.
Returns None for any series if unavailable.

Interpretation thresholds are grounded in standard macro-finance practice
(e.g. Blinder & Watson on inversions, Fed Z1, BofA spread research).
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

try:
    from fixed_income.core.fred_client import (
        get_fred, fred_latest, fred_yoy, fred_mom
    )
    _FRED_AVAILABLE = True
except Exception:
    _FRED_AVAILABLE = False


# ---------------------------------------------------------------------------
# Series registry — (series_id, display_name, units, fetch_mode)
# fetch_mode: "latest" | "yoy" | "mom"
# ---------------------------------------------------------------------------

_SERIES: list[tuple[str, str, str, str, str]] = [
    # key,          series_id,         name,               units,   mode
    ("t10y2y",      "T10Y2Y",          "10Y–2Y Slope",     "bps",   "latest"),
    ("t10y3m",      "T10Y3M",          "10Y–3M Slope",     "bps",   "latest"),
    ("dgs2",        "DGS2",            "2Y Treasury",      "%",     "latest"),
    ("dgs10",       "DGS10",           "10Y Treasury",     "%",     "latest"),
    ("effr",        "EFFR",            "Fed Funds (EFFR)", "%",     "latest"),
    ("sofr",        "SOFR",            "SOFR",             "%",     "latest"),
    ("ig_oas",      "BAMLC0A0CM",      "IG OAS",           "bps",   "latest"),
    ("hy_oas",      "BAMLH0A0HYM2",    "HY OAS",           "bps",   "latest"),
    ("cpi_yoy",     "CPIAUCSL",        "CPI (YoY)",        "%",     "yoy"),
    ("core_cpi_yoy","CPILFESL",        "Core CPI (YoY)",   "%",     "yoy"),
    ("breakeven",   "T10YIE",          "10Y Breakeven",    "%",     "latest"),
    ("unrate",      "UNRATE",          "Unemployment",     "%",     "latest"),
    ("payems_mom",  "PAYEMS",          "Nonfarm Payrolls", "k MoM", "mom"),
    ("icsa",        "ICSA",            "Initial Claims",   "k",     "latest"),
    ("nfci",        "NFCI",            "Chicago NFCI",     "σ",     "latest"),
]

# T10Y2Y and T10Y3M from FRED are in percentage points (e.g. 0.5 = 50 bps).
# Multiply by 100 to convert to basis points for display.
_CONVERT_TO_BPS = {"t10y2y", "t10y3m"}


# ---------------------------------------------------------------------------
# MacroScreen
# ---------------------------------------------------------------------------

class MacroScreen:
    """
    Fetch and interpret key macro indicators from FRED.

    Usage
    -----
        ms = MacroScreen()          # fetches on init (TTL-cached)
        df = ms.as_df()             # formatted DataFrame for table display
        lines = ms.interp_lines()   # interpretation strings for TUI
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.data:   dict[str, float | None] = {}
        self.warnings: list[str] = []
        self._available = False
        self._fetch_all()

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_all(self) -> None:
        if not _FRED_AVAILABLE:
            self.warnings.append("FRED client not importable.")
            return

        if get_fred() is None:
            self.warnings.append(
                "FRED_API_KEY not set — macro screen unavailable. "
                "Export FRED_API_KEY=<your_key> to enable."
            )
            return

        self._available = True
        ttl = self.ttl_seconds

        for key, series_id, _name, _units, mode in _SERIES:
            try:
                if mode == "yoy":
                    val = fred_yoy(series_id, ttl_seconds=ttl)
                elif mode == "mom":
                    val = fred_mom(series_id, ttl_seconds=ttl)
                else:
                    val = fred_latest(series_id, ttl_seconds=ttl)

                # Unit conversion: T10Y2Y / T10Y3M → bps
                if val is not None and key in _CONVERT_TO_BPS:
                    val = val * 100.0

                self.data[key] = val
            except Exception as e:
                self.data[key] = None
                self.warnings.append(f"FRED fetch failed for {series_id}: {e}")

    # ------------------------------------------------------------------
    # Table view
    # ------------------------------------------------------------------

    def as_df(self) -> pd.DataFrame:
        """Return a formatted DataFrame suitable for the TUI table renderer."""
        if not self._available:
            return pd.DataFrame([{"Indicator": "FRED unavailable",
                                   "Value": "Set FRED_API_KEY env var",
                                   "Signal": ""}])

        rows = []
        for key, series_id, name, units, _mode in _SERIES:
            val = self.data.get(key)
            if val is None:
                val_str = "N/A"
            elif units == "%":
                val_str = f"{val:.2f}%"
            elif units == "bps":
                val_str = f"{val:.1f} bps"
            elif units == "k MoM":
                val_str = f"{val:+.1f}k"
            elif units == "k":
                val_str = f"{val:.1f}k"
            elif units == "σ":
                val_str = f"{val:+.3f}"
            else:
                val_str = f"{val:.3f}"

            signal = _signal(key, val)
            rows.append({
                "Indicator": name,
                "Value":     val_str,
                "Signal":    signal,
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------

    def interp_lines(self) -> list[str]:
        """
        Generate 6–10 quantitative interpretation lines for the TUI MACRO tab.
        Each line is grounded in a specific macro-finance threshold or rule.
        """
        if not self._available:
            return [
                "FRED macro screen disabled.",
                "Set the FRED_API_KEY environment variable to enable.",
                "  export FRED_API_KEY=<your_key>",
            ]

        d = self.data
        lines: list[str] = []

        # ── Yield curve ──────────────────────────────────────────────────
        t10y2y = d.get("t10y2y")
        t10y3m = d.get("t10y3m")
        dgs2   = d.get("dgs2")
        dgs10  = d.get("dgs10")

        if t10y2y is not None:
            bps = t10y2y
            if bps < -75:
                lines.append(
                    f"YIELD CURVE: 10Y-2Y = {bps:.0f}bps — deep inversion. "
                    "Historical recession predictor (Rudebusch/Bauer, 2022); "
                    "credit and equity stress often lag by 12–18 months."
                )
            elif bps < -25:
                lines.append(
                    f"YIELD CURVE: 10Y-2Y = {bps:.0f}bps — inverted. "
                    "Late-cycle signal; monitor credit spreads for follow-through."
                )
            elif bps < 50:
                lines.append(
                    f"YIELD CURVE: 10Y-2Y = {bps:.0f}bps — flat to slightly steep. "
                    "Neutral; growth/inflation balance broadly priced in."
                )
            else:
                lines.append(
                    f"YIELD CURVE: 10Y-2Y = {bps:.0f}bps — steepening. "
                    "Bull steepener (Fed easing) or bear steepener (inflation risk)."
                    f" 2Y={dgs2:.2f}% / 10Y={dgs10:.2f}%." if dgs2 and dgs10 else ""
                )

        if t10y3m is not None:
            if t10y3m < -50:
                lines.append(
                    f"10Y–3M = {t10y3m:.0f}bps — negative. "
                    "Estrella/Mishkin model: historically highest-predictive inversion."
                )

        # ── Policy / rates ───────────────────────────────────────────────
        effr = d.get("effr")
        sofr = d.get("sofr")
        if effr is not None and sofr is not None:
            spread_bps = (sofr - effr) * 100
            lines.append(
                f"POLICY: EFFR={effr:.2f}%  SOFR={sofr:.2f}%  "
                f"(SOFR-EFFR spread: {spread_bps:+.1f}bps). "
                + ("Near zero spread — repo market functioning normally." if abs(spread_bps) < 5
                   else "Elevated spread — watch for overnight funding stress.")
            )

        # ── Credit ───────────────────────────────────────────────────────
        ig  = d.get("ig_oas")
        hy  = d.get("hy_oas")
        if ig is not None and hy is not None:
            hy_ig = hy - ig
            ig_label = ("TIGHT (<80bps, risk-on)" if ig < 80
                        else "NORMAL (80–130bps)" if ig < 130
                        else "WIDE (>130bps, risk-off)")
            hy_label = ("TIGHT (<300bps, risk-on)" if hy < 300
                        else "NORMAL (300–500bps)" if hy < 500
                        else "STRESSED (>500bps, risk-off)" if hy < 800
                        else "CRISIS (>800bps)")
            lines.append(
                f"CREDIT: IG OAS={ig:.0f}bps [{ig_label}]  "
                f"HY OAS={hy:.0f}bps [{hy_label}]  "
                f"HY–IG={hy_ig:.0f}bps."
            )

        # ── Inflation ────────────────────────────────────────────────────
        cpi       = d.get("cpi_yoy")
        core_cpi  = d.get("core_cpi_yoy")
        breakeven = d.get("breakeven")

        if cpi is not None and core_cpi is not None:
            infl_label = ("ELEVATED >3% — hawkish pressure on Fed" if cpi > 3.0
                          else "ABOVE TARGET 2–3% — moderating, still above 2%" if cpi > 2.0
                          else "AT/BELOW TARGET <2% — disinflationary space")
            lines.append(
                f"INFLATION: CPI YoY={cpi:.1f}%  Core CPI={core_cpi:.1f}%  "
                f"[{infl_label}]."
                + (f"  10Y breakeven={breakeven:.2f}% "
                   f"({'expectations elevated' if breakeven > 2.5 else 'anchored'})."
                   if breakeven is not None else "")
            )

        # ── Employment ───────────────────────────────────────────────────
        unrate     = d.get("unrate")
        payems_mom = d.get("payems_mom")
        icsa       = d.get("icsa")

        if unrate is not None:
            ur_label = ("TIGHT (<4%)" if unrate < 4.0
                        else "NORMAL (4–5%)" if unrate < 5.0
                        else "LOOSE (>5%)")
            payems_str = (f"  NFP MoM={payems_mom:+.0f}k "
                          f"({'strong' if payems_mom > 200 else 'moderate' if payems_mom > 100 else 'weak'})"
                          if payems_mom is not None else "")
            icsa_str = (f"  Claims={icsa:.0f}k "
                        f"({'low' if icsa < 200 else 'normal' if icsa < 275 else 'rising'})"
                        if icsa is not None else "")
            lines.append(
                f"EMPLOYMENT: UNRATE={unrate:.1f}% [{ur_label}]"
                f"{payems_str}{icsa_str}."
            )

        # ── Financial conditions ─────────────────────────────────────────
        nfci = d.get("nfci")
        if nfci is not None:
            nfci_label = ("VERY LOOSE (<−0.5)" if nfci < -0.5
                          else "LOOSE (−0.5–0)" if nfci < 0
                          else "NEUTRAL (≈0)" if abs(nfci) < 0.1
                          else "TIGHT (>0)" if nfci < 0.5
                          else "VERY TIGHT (>0.5)")
            lines.append(
                f"FINANCIAL CONDITIONS (NFCI): {nfci:+.3f} [{nfci_label}]. "
                "Index = 0 is historical average; + = tighter than average."
            )

        # ── Cross-asset read ─────────────────────────────────────────────
        lines.extend(_cross_asset_read(d))

        if self.warnings:
            lines.append("─" * 70)
            for w in self.warnings[:3]:
                lines.append(f"  ⚠  {w}")

        lines.append("─" * 70)
        lines.append("Source: FRED (St. Louis Fed).  NOT investment advice.")
        return lines


# ---------------------------------------------------------------------------
# Signal labels
# ---------------------------------------------------------------------------

def _signal(key: str, val: float | None) -> str:
    """Return a short signal label (BULLISH / BEARISH / NEUTRAL / N/A)."""
    if val is None:
        return "N/A"
    if key == "t10y2y":
        return "BEARISH" if val < -50 else "CAUTION" if val < 0 else "NEUTRAL"
    if key == "t10y3m":
        return "BEARISH" if val < -50 else "NEUTRAL"
    if key == "ig_oas":
        return "BULLISH" if val < 80 else "NEUTRAL" if val < 130 else "BEARISH"
    if key == "hy_oas":
        return "BULLISH" if val < 300 else "NEUTRAL" if val < 500 else "BEARISH"
    if key == "cpi_yoy":
        return "BEARISH" if val > 3.5 else "CAUTION" if val > 2.5 else "NEUTRAL"
    if key == "core_cpi_yoy":
        return "BEARISH" if val > 3.5 else "CAUTION" if val > 2.5 else "NEUTRAL"
    if key == "unrate":
        return "BULLISH" if val < 4.0 else "NEUTRAL" if val < 5.0 else "BEARISH"
    if key == "nfci":
        return "BULLISH" if val < -0.3 else "NEUTRAL" if val < 0.2 else "BEARISH"
    if key == "payems_mom":
        return "BULLISH" if val > 200 else "NEUTRAL" if val > 100 else "BEARISH"
    if key == "icsa":
        return "BULLISH" if val < 200 else "NEUTRAL" if val < 275 else "BEARISH"
    if key == "breakeven":
        return "NEUTRAL" if 2.0 <= val <= 2.5 else "CAUTION"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Cross-asset synthesis
# ---------------------------------------------------------------------------

def _cross_asset_read(d: dict) -> list[str]:
    """
    Generate 1–2 cross-asset synthesis lines from the combined macro picture.
    Based on standard macro-finance risk-regime classification.
    """
    lines: list[str] = []
    t10y2y = d.get("t10y2y")
    hy     = d.get("hy_oas")
    cpi    = d.get("cpi_yoy")
    nfci   = d.get("nfci")
    unrate = d.get("unrate")

    risk_score = 0   # + = risk-on, − = risk-off
    n = 0

    if t10y2y is not None:
        risk_score += (1 if t10y2y > 50 else -1 if t10y2y < -50 else 0)
        n += 1
    if hy is not None:
        risk_score += (1 if hy < 300 else -1 if hy > 500 else 0)
        n += 1
    if nfci is not None:
        risk_score += (1 if nfci < -0.3 else -1 if nfci > 0.3 else 0)
        n += 1
    if unrate is not None:
        risk_score += (1 if unrate < 4.0 else -1 if unrate > 5.5 else 0)
        n += 1

    if n >= 2:
        avg = risk_score / n
        if avg >= 0.5:
            lines.append(
                "CROSS-ASSET READ: Risk-on environment — credit tight, conditions "
                "loose, labour market resilient."
            )
        elif avg <= -0.5:
            lines.append(
                "CROSS-ASSET READ: Risk-off environment — credit spreads widening / "
                "curve inverted / conditions tightening."
            )
        else:
            lines.append(
                "CROSS-ASSET READ: Mixed signals — no clear macro regime; "
                "monitor for divergence across credit, rates and labour."
            )

    # Stagflation flag
    if cpi is not None and unrate is not None:
        if cpi > 3.5 and unrate > 4.5:
            lines.append(
                "STAGFLATION WATCH: Elevated inflation concurrent with rising "
                "unemployment — historically challenging for both bonds and equities."
            )

    return lines
