"""
svi.py — SVI (Stochastic Volatility Inspired) volatility surface calibration.

Changes vs. prior version:
  - Accepts pre-computed (strikes, implied_vols) arrays directly; no fetcher dependency
  - Guards against empty / too-few-point inputs before calling curve_fit
  - Calibration failures set self.calibrated=False instead of crashing
  - svi_vol() returns None when not calibrated
  - Calibration quality stored in self.rmse
  - svi_vols property is vectorized (no Python loop)
  - plot_svi y-axis now correctly shows percentages (values × 100)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Minimum number of data points required to fit the 5-parameter SVI model
_MIN_POINTS = 5


class SVICalibrationError(Exception):
    """Raised when SVI calibration cannot proceed due to insufficient or invalid data."""
    pass


class SVI:
    def __init__(self, S_0: float, T: float, r: float, q: float,
                 option_type: str, fetcher=None,
                 strikes: np.ndarray = None, implied_vols: np.ndarray = None):
        """
        Parameters
        ----------
        S_0, T, r, q, option_type : standard option parameters
        fetcher : optional MarketDataFetcher — used if strikes/implied_vols not provided
        strikes : pre-computed strike array (decimal moneyness not required)
        implied_vols : pre-computed IV array in DECIMAL (0.25 = 25%)

        If both fetcher and strikes/implied_vols are supplied, the explicit arrays
        take priority and no network call is made.
        """
        self.S_0        = S_0
        self.T          = T
        self.r          = r
        self.q          = q
        self.option_type = option_type
        self.ticker     = fetcher.ticker if fetcher is not None else "unknown"

        self.calibrated = False
        self.rmse       = None
        self.warnings: list[str] = []

        # -- Obtain strikes and vols (prefer explicit arrays over fetcher) --
        if strikes is not None and implied_vols is not None:
            self.strikes      = np.asarray(strikes, dtype=float)
            self.implied_vols = np.asarray(implied_vols, dtype=float)
        elif fetcher is not None:
            raw_strikes, raw_vols_pct, _ = fetcher.plot_implied_vols(
                r=r, option_type=option_type, plot=False
            )
            # fetcher returns vols in percent (e.g. 25.0); convert to decimal
            self.strikes      = np.array(raw_strikes, dtype=float)
            self.implied_vols = np.array(raw_vols_pct, dtype=float) / 100.0
        else:
            raise ValueError("SVI requires either a fetcher or explicit (strikes, implied_vols)")

        # Remove NaNs / infs / non-positive vols
        valid = (
            np.isfinite(self.strikes)
            & np.isfinite(self.implied_vols)
            & (self.implied_vols > 0)
            & (self.strikes > 0)
        )
        self.strikes      = self.strikes[valid]
        self.implied_vols = self.implied_vols[valid]

        # SVI model parameters (set by calibrate())
        self.a     = None
        self.b     = None
        self.rho   = None
        self.m     = None
        self.sigma = None

        self._try_calibrate()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _try_calibrate(self):
        """Attempt calibration; set calibrated=False with a warning on failure."""
        n = len(self.strikes)
        if n < _MIN_POINTS:
            msg = (
                f"SVI calibration skipped: need ≥{_MIN_POINTS} points, got {n}. "
                f"Using fallback (model vol will be None)."
            )
            self.warnings.append(msg)
            return

        try:
            self.calibrate()
        except (RuntimeError, ValueError) as e:
            msg = f"SVI calibration failed: {e}. Using fallback."
            self.warnings.append(msg)

    def calibrate(self):
        k_i = self.log_moneyness()
        w_i = self.implied_vols ** 2 * self.T   # total variance

        if len(k_i) == 0:
            raise ValueError("empty log-moneyness array")

        p0 = [
            float(np.min(w_i)),
            float((np.max(w_i) - np.min(w_i)) / (np.max(k_i) - np.min(k_i) + 1e-6)),
            0.0,
            float(np.mean(k_i)),
            float(np.std(k_i)) if np.std(k_i) > 0 else 0.1,
        ]

        lower = [-np.inf,  0.0, -0.999, float(np.min(k_i)),  1e-6]
        upper = [ np.inf, np.inf,  0.999, float(np.max(k_i)), np.inf]

        params, _ = curve_fit(
            self.svi_variance, k_i, w_i,
            p0=p0, bounds=(lower, upper), maxfev=20000
        )
        self.a, self.b, self.rho, self.m, self.sigma = params
        self.calibrated = True

        # Calibration quality (RMSE in vol space)
        w_fit = self.svi_variance(k_i, *params)
        vol_fit  = np.sqrt(np.maximum(w_fit, 0) / self.T)
        self.rmse = float(np.sqrt(np.mean((vol_fit - self.implied_vols) ** 2)))

        # Warn if the fit produces negative total variance for any strike
        if np.any(w_fit < 0):
            self.warnings.append(
                "SVI fit produced negative total variance for some strikes — "
                "calibration may be unreliable."
            )

    # ------------------------------------------------------------------
    # Variance / vol evaluation
    # ------------------------------------------------------------------

    def svi_variance(self, k, a, b, rho, m, sigma):
        """SVI parameterisation: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    def svi_vol(self, K: float):
        """Return the SVI-implied vol for strike K, or None if not calibrated."""
        if not self.calibrated:
            return None
        k = np.log(float(K) / self.forward_price(self.S_0, self.T, self.r, self.q))
        w = self.svi_variance(k, self.a, self.b, self.rho, self.m, self.sigma)
        return float(np.sqrt(max(w, 0.0) / self.T))

    @property
    def svi_vols(self) -> np.ndarray:
        """Vectorized SVI vols for all calibration strikes. Returns array of None if uncalibrated."""
        if not self.calibrated:
            return np.array([None] * len(self.strikes))
        k_arr = np.log(self.strikes / self.forward_price(self.S_0, self.T, self.r, self.q))
        w_arr = self.svi_variance(k_arr, self.a, self.b, self.rho, self.m, self.sigma)
        return np.sqrt(np.maximum(w_arr, 0.0) / self.T)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def forward_price(self, S_0, T, r, q) -> float:
        return S_0 * np.exp((r - q) * T)

    def log_moneyness(self) -> np.ndarray:
        F = self.forward_price(self.S_0, self.T, self.r, self.q)
        return np.log(self.strikes / F)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot_svi(self):
        if not self.calibrated:
            print(f"[SVI] Cannot plot — calibration failed. Warnings: {self.warnings}")
            return

        svi_vols_pct    = self.svi_vols * 100          # decimal → percent for display
        market_vols_pct = self.implied_vols * 100

        _, ax = plt.subplots()
        ax.plot(self.strikes, market_vols_pct,
                label="market implied vol", marker="o", linestyle="-", color="blue")
        ax.plot(self.strikes, svi_vols_pct,
                label="SVI-calibrated vol", marker="o", linestyle="--", color="orange")
        ax.axvline(x=self.S_0, color="black", linestyle="--",
                   label=f"spot: ${self.S_0:.2f}")

        rmse_label = f"RMSE: {self.rmse * 100:.2f}%" if self.rmse is not None else ""
        ax.set_xlabel("strike")
        ax.set_ylabel("implied vol (%)")
        ax.set_title(
            f"{self.ticker} SVI Vol Surface  "
            f"(T={self.T:.2f}y, {self.option_type}s)  {rmse_label}"
        )
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
