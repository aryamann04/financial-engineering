"""
tests/test_iv.py — Validation tests for the IV pipeline.

Run with:  pytest tests/test_iv.py -v
No network access required — all tests use synthetic data.
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from options.core.pricing.pricing import bs_price
from options.volatility.iv import bs_iv
from options.volatility.svi import SVI, SVICalibrationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_trip(S, K, T, r, q, sigma, option_type, tol=1e-4):
    """Compute BS price then recover sigma via bs_iv; return absolute error."""
    price = bs_price(S, K, T, r, sigma, q, option_type)
    assert price is not None and np.isfinite(price), \
        f"bs_price returned invalid value: {price}"
    iv = bs_iv(price, S, K, T, r, q, option_type)
    assert iv is not None and np.isfinite(iv), \
        f"bs_iv returned {iv} for {option_type} S={S} K={K} T={T} σ={sigma}"
    err = abs(iv - sigma)
    assert err < tol, f"Round-trip error {err:.2e} > {tol} for {option_type} S={S} K={K} σ={sigma}"
    return err


# ---------------------------------------------------------------------------
# bs_price: sigma=0 guard
# ---------------------------------------------------------------------------

class TestBsPrice:
    def test_call_sigma_zero_returns_intrinsic(self):
        S, K, r, T = 100, 90, 0.05, 1.0
        result = bs_price(S, K, T, r, sigma=0.0, q=0, option_type="call")
        expected = max(S * np.exp(-0 * T) - K * np.exp(-r * T), 0)
        assert abs(result - expected) < 1e-10

    def test_put_sigma_zero_returns_intrinsic(self):
        S, K, r, T = 80, 100, 0.05, 1.0
        result = bs_price(S, K, T, r, sigma=0.0, q=0, option_type="put")
        expected = max(K * np.exp(-r * T) - S * np.exp(-0 * T), 0)
        assert abs(result - expected) < 1e-10

    def test_call_sigma_zero_atm_returns_zero(self):
        # ATM at sigma=0: intrinsic of a forward contract at-the-money = 0
        result = bs_price(100, 100, 1.0, 0.0, sigma=0.0, q=0, option_type="call")
        assert result == 0.0

    def test_result_finite_for_reasonable_inputs(self):
        price = bs_price(100, 100, 1.0, 0.05, sigma=0.25, q=0.02, option_type="call")
        assert np.isfinite(price) and price > 0


# ---------------------------------------------------------------------------
# bs_iv: round-trip accuracy across moneyness / expiry grid
# ---------------------------------------------------------------------------

class TestBsIvRoundTrip:
    """
    Bug 1 fix verification: put intrinsic is now option-type-aware.
    Bug 2 fix verification: ATM options recover correctly (initial guess was 0).
    Bug 3 fix verification: brentq fallback catches cases NR misses.
    """

    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("S,K,T,sigma", [
        # ATM — Bug 2 regression test
        (100, 100,  1.0, 0.20),
        (100, 100,  0.1, 0.20),
        (100, 100,  0.1, 0.05),
        # OTM calls
        (100, 110,  1.0, 0.25),
        (100, 120,  0.5, 0.30),
        # ITM calls / deep ITM puts — Bug 1 regression test
        (100,  80,  1.0, 0.30),
        ( 80, 150,  1.0, 0.30),   # deep ITM put (K >> S): previously returned None
        # Short-dated
        (100, 100, 0.04, 0.25),   # ~2 weeks
        # High vol
        (100, 100,  1.0, 1.50),
    ])
    def test_round_trip(self, S, K, T, sigma, option_type):
        r, q = 0.05, 0.01
        _round_trip(S, K, T, r, q, sigma, option_type, tol=1e-4)

    def test_deep_itm_put_was_broken(self):
        """Explicit regression: deep ITM put used to return None (Bug 1)."""
        S, K, T, r, q, sigma = 80, 150, 1.0, 0.05, 0.0, 0.30
        price = bs_price(S, K, T, r, sigma, q, "put")
        iv = bs_iv(price, S, K, T, r, q, "put")
        assert iv is not None, "Deep ITM put returned None — Bug 1 regression"
        assert abs(iv - sigma) < 1e-3


# ---------------------------------------------------------------------------
# bs_iv: invalid inputs return None (not crash)
# ---------------------------------------------------------------------------

class TestBsIvEdgeCases:
    def test_t_zero_returns_none(self):
        assert bs_iv(5.0, 100, 100, T=0, r=0.05, q=0) is None

    def test_negative_t_returns_none(self):
        assert bs_iv(5.0, 100, 100, T=-1, r=0.05, q=0) is None

    def test_none_market_returns_none(self):
        assert bs_iv(None, 100, 100, T=1.0, r=0.05, q=0) is None

    def test_nan_market_returns_none(self):
        assert bs_iv(float("nan"), 100, 100, T=1.0, r=0.05, q=0) is None

    def test_market_below_intrinsic_call(self):
        # Market price below call intrinsic: no valid IV — expect None or at intrinsic nudge
        S, K, r, T = 100, 80, 0.05, 1.0
        intrinsic = max(S * np.exp(-0 * T) - K * np.exp(-r * T), 0)
        result = bs_iv(intrinsic - 1.0, S, K, T, r, 0, "call")
        # Should not crash; either None or nudged to a valid IV
        assert result is None or (np.isfinite(result) and result > 0)

    def test_put_call_parity_consistency(self):
        """
        For European options: put price = call price - S + K*exp(-rT).
        Both should give the same implied vol.
        """
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.20
        call_price = bs_price(S, K, T, r, sigma, q, "call")
        put_price  = bs_price(S, K, T, r, sigma, q, "put")
        iv_call = bs_iv(call_price, S, K, T, r, q, "call")
        iv_put  = bs_iv(put_price,  S, K, T, r, q, "put")
        assert iv_call is not None and iv_put is not None
        assert abs(iv_call - iv_put) < 1e-4, \
            f"Call IV {iv_call:.4f} vs put IV {iv_put:.4f} differ (put-call parity violated)"


# ---------------------------------------------------------------------------
# SVI: empty / insufficient data handled gracefully
# ---------------------------------------------------------------------------

class TestSVI:
    _BASE = dict(S_0=100, T=1.0, r=0.05, q=0.01, option_type="call")

    def test_empty_strikes_no_crash(self):
        svi = SVI(**self._BASE, strikes=np.array([]), implied_vols=np.array([]))
        assert not svi.calibrated
        assert svi.svi_vol(100) is None

    def test_too_few_points_no_crash(self):
        # 3 points < _MIN_POINTS=5
        strikes = np.array([90., 100., 110.])
        ivs     = np.array([0.22, 0.20, 0.21])
        svi = SVI(**self._BASE, strikes=strikes, implied_vols=ivs)
        assert not svi.calibrated
        assert svi.svi_vol(100) is None
        assert len(svi.warnings) > 0

    def test_nan_in_vols_filtered(self):
        strikes = np.array([85., 90., 95., 100., 105., 110., 115.])
        ivs     = np.array([0.30, float("nan"), 0.25, 0.20, 0.22, 0.25, 0.28])
        svi = SVI(**self._BASE, strikes=strikes, implied_vols=ivs)
        # NaN should be stripped; 6 valid points should allow calibration
        assert svi.calibrated

    def test_successful_calibration_round_trip(self):
        """SVI calibrated on BS IV grid should recover vols within ~1%."""
        sigma_true = 0.25
        strikes    = np.linspace(80, 120, 10)
        ivs        = np.array([
            bs_iv(bs_price(100, K, 1.0, 0.05, sigma_true, 0.0, "call"),
                  100, K, 1.0, 0.05, 0.0, "call")
            for K in strikes
        ])
        # Remove None entries
        valid = np.array([v is not None for v in ivs])
        svi = SVI(**self._BASE, strikes=strikes[valid],
                  implied_vols=np.array(ivs)[valid].astype(float))
        assert svi.calibrated
        # SVI vol at ATM should be close to true vol
        atm_vol = svi.svi_vol(100)
        assert atm_vol is not None
        assert abs(atm_vol - sigma_true) < 0.02, \
            f"SVI ATM vol {atm_vol:.4f} far from true {sigma_true}"

    def test_svi_vols_vectorized(self):
        """svi_vols property should return an array, not a list."""
        strikes = np.linspace(85, 115, 8)
        ivs     = np.full(8, 0.22)
        svi = SVI(**self._BASE, strikes=strikes, implied_vols=ivs)
        if svi.calibrated:
            vols = svi.svi_vols
            assert isinstance(vols, np.ndarray)
            assert len(vols) == len(strikes)
            assert all(np.isfinite(v) and v > 0 for v in vols)


# ---------------------------------------------------------------------------
# Monte Carlo: European matches BS to within statistical noise
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_european_call_matches_bs(self):
        from options.core.pricing.montecarlo import monte_carlo_european
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.20
        np.random.seed(0)
        mc  = monte_carlo_european(S, K, T, r, q, sigma, "call", simulations=200_000)
        bs  = bs_price(S, K, T, r, sigma, q, "call")
        assert abs(mc - bs) < 0.05, f"MC={mc:.4f} vs BS={bs:.4f}"

    def test_european_put_matches_bs(self):
        from options.core.pricing.montecarlo import monte_carlo_european
        S, K, T, r, q, sigma = 100, 110, 1.0, 0.05, 0.02, 0.25
        np.random.seed(1)
        # Deep ITM put has large payoff variance → needs more sims for tight tolerance
        mc  = monte_carlo_european(S, K, T, r, q, sigma, "put", simulations=500_000)
        bs  = bs_price(S, K, T, r, sigma, q, "put")
        assert abs(mc - bs) < 0.10, f"MC={mc:.4f} vs BS={bs:.4f}"


# ---------------------------------------------------------------------------
# marketvols: compute_bs_ivs handles NaN / zero prices
# ---------------------------------------------------------------------------

class TestComputeBsIVs:
    def test_skips_zero_mid_price(self):
        from options.volatility.marketvols import compute_bs_ivs
        df = __import__("pandas").DataFrame({
            "strike": [90, 100, 110],
            "bid":    [0.0, 4.5, 0.0],
            "ask":    [0.0, 5.0, 0.0],
        })
        strikes, ivs = compute_bs_ivs(df, 100, 1.0, 0.05, 0.0, "call")
        # Only strike 100 has a non-zero mid price
        assert len(strikes) == 1
        assert strikes[0] == 100

    def test_returns_valid_ivs(self):
        from options.volatility.marketvols import compute_bs_ivs
        sigma = 0.25
        # Construct synthetic chain prices from known vol
        ks = [85, 90, 95, 100, 105, 110, 115]
        prices = [bs_price(100, K, 1.0, 0.05, sigma, 0.0, "call") for K in ks]
        df = __import__("pandas").DataFrame({
            "strike": ks,
            "bid":    [p * 0.98 for p in prices],
            "ask":    [p * 1.02 for p in prices],
        })
        strikes, ivs = compute_bs_ivs(df, 100, 1.0, 0.05, 0.0, "call")
        assert len(strikes) == 7
        for iv in ivs:
            assert abs(iv - sigma) < 0.01, f"IV {iv:.4f} far from true {sigma}"
