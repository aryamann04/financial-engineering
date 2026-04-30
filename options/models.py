from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, exp, log, sqrt
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from options.core.pricing.pricing import binom_price, bs_price


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + erf(np.asarray(x) / sqrt(2.0)))


def _norm_pdf(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x)
    return (1.0 / sqrt(2.0 * np.pi)) * np.exp(-0.5 * arr * arr)


@dataclass(frozen=True)
class PricingInputs:
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    implied_vol: float
    option_type: str
    american: bool = False


@dataclass
class ModelOutput:
    name: str
    price: float | None
    volatility: float | None
    weight: float
    warnings: list[str] = field(default_factory=list)


def bs_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
    option_type: str,
) -> dict[str, float]:
    if maturity <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = (log(spot / strike) + (rate - dividend_yield + 0.5 * sigma * sigma) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    pdf = float(_norm_pdf(d1))
    if option_type == "call":
        delta = exp(-dividend_yield * maturity) * float(_norm_cdf(d1))
        rho = (strike * maturity * exp(-rate * maturity) * float(_norm_cdf(d2))) / 100.0
        theta = (
            -(spot * sigma * exp(-dividend_yield * maturity) * pdf) / (2 * sqrt(maturity))
            - dividend_yield * spot * exp(-dividend_yield * maturity) * float(_norm_cdf(d1))
            + rate * strike * exp(-rate * maturity) * float(_norm_cdf(d2))
        ) / 365.0
    else:
        delta = -exp(-dividend_yield * maturity) * float(_norm_cdf(-d1))
        rho = (-strike * maturity * exp(-rate * maturity) * float(_norm_cdf(-d2))) / 100.0
        theta = (
            -(spot * sigma * exp(-dividend_yield * maturity) * pdf) / (2 * sqrt(maturity))
            + dividend_yield * spot * exp(-dividend_yield * maturity) * float(_norm_cdf(-d1))
            - rate * strike * exp(-rate * maturity) * float(_norm_cdf(-d2))
        ) / 365.0
    gamma = (exp(-dividend_yield * maturity) * pdf) / (spot * sigma * sqrt(maturity))
    vega = (spot * exp(-dividend_yield * maturity) * pdf * sqrt(maturity)) / 100.0
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def bs_price_vectorized(
    spot: float,
    strikes: np.ndarray,
    maturity: float,
    rate: float,
    sigma: np.ndarray,
    dividend_yield: float,
    option_type: str,
) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = max(float(maturity), 1e-9)
    sigma_safe = np.clip(sigma, 1e-8, None)
    d1 = (np.log(np.maximum(spot, 1e-12) / np.maximum(strikes, 1e-12)) + (rate - dividend_yield + 0.5 * sigma_safe**2) * t) / (sigma_safe * np.sqrt(t))
    d2 = d1 - sigma_safe * np.sqrt(t)
    if option_type == "call":
        price = spot * np.exp(-dividend_yield * t) * _norm_cdf(d1) - strikes * np.exp(-rate * t) * _norm_cdf(d2)
    else:
        price = strikes * np.exp(-rate * t) * _norm_cdf(-d2) - spot * np.exp(-dividend_yield * t) * _norm_cdf(-d1)
    intrinsic = np.maximum(spot * np.exp(-dividend_yield * t) - strikes * np.exp(-rate * t), 0.0) if option_type == "call" else np.maximum(strikes * np.exp(-rate * t) - spot * np.exp(-dividend_yield * t), 0.0)
    return np.where(sigma <= 1e-8, intrinsic, price)


def black76_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    if maturity <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    forward = spot * exp((rate - dividend_yield) * maturity)
    std = max(sigma, 1e-8) * sqrt(maturity)
    d1 = (log(max(forward, 1e-12) / max(strike, 1e-12)) + 0.5 * std * std) / std
    d2 = d1 - std
    disc = exp(-rate * maturity)
    if option_type == "call":
        return disc * (forward * float(_norm_cdf(d1)) - strike * float(_norm_cdf(d2)))
    return disc * (strike * float(_norm_cdf(-d2)) - forward * float(_norm_cdf(-d1)))


def local_vol_adjusted_price(inputs: PricingInputs, anchor_vols: Iterable[tuple[float, float]]) -> ModelOutput:
    anchors = sorted((float(k), float(v)) for k, v in anchor_vols if v is not None and v > 0)
    if len(anchors) < 2:
        return ModelOutput("local_vol", None, None, 0.0, ["Not enough smile points for local-vol approximation."])
    strikes = np.array([item[0] for item in anchors], dtype=float)
    vols = np.array([item[1] for item in anchors], dtype=float)
    local_vol = float(np.interp(inputs.strike, strikes, vols))
    return ModelOutput(
        "local_vol",
        bs_price(inputs.spot, inputs.strike, inputs.maturity, inputs.rate, local_vol, inputs.dividend_yield, inputs.option_type),
        local_vol,
        0.15,
    )


def sabr_lognormal_vol(forward: float, strike: float, maturity: float, alpha: float, beta: float, rho: float, nu: float) -> float:
    if forward <= 0 or strike <= 0 or maturity <= 0 or alpha <= 0 or nu <= 0:
        return np.nan
    if abs(forward - strike) < 1e-10:
        fk = forward ** (1 - beta)
        term1 = alpha / max(fk, 1e-12)
        term2 = 1 + (((1 - beta) ** 2 / 24) * (alpha**2 / max(fk**2, 1e-12)) + (rho * beta * nu * alpha) / (4 * max(fk, 1e-12)) + ((2 - 3 * rho**2) * nu**2 / 24)) * maturity
        return term1 * term2
    fk = forward * strike
    log_fk = log(forward / strike)
    z = (nu / alpha) * (fk ** ((1 - beta) / 2)) * log_fk
    x_z = log((sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    numerator = alpha * (1 + (((1 - beta) ** 2 / 24) * (alpha**2 / (fk ** (1 - beta))) + (rho * beta * nu * alpha) / (4 * (fk ** ((1 - beta) / 2))) + ((2 - 3 * rho**2) * nu**2 / 24)) * maturity)
    denominator = (fk ** ((1 - beta) / 2)) * (1 + ((1 - beta) ** 2 / 24) * log_fk**2 + ((1 - beta) ** 4 / 1920) * log_fk**4)
    return (numerator / max(denominator, 1e-12)) * (z / max(x_z, 1e-12))


def calibrate_sabr(strikes: np.ndarray, vols: np.ndarray, forward: float, maturity: float, beta: float = 1.0) -> tuple[tuple[float, float, float] | None, float | None]:
    valid = np.isfinite(strikes) & np.isfinite(vols) & (vols > 0)
    strikes = strikes[valid]
    vols = vols[valid]
    if len(strikes) < 4 or maturity <= 0:
        return None, None

    def objective(params: np.ndarray) -> float:
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or not (-0.999 < rho < 0.999):
            return 1e6
        fitted = np.array([sabr_lognormal_vol(forward, float(k), maturity, alpha, beta, rho, nu) for k in strikes])
        if not np.all(np.isfinite(fitted)):
            return 1e6
        return float(np.mean((fitted - vols) ** 2))

    x0 = np.array([max(float(np.median(vols)), 0.05), -0.2, 0.6], dtype=float)
    result = minimize(objective, x0, method="L-BFGS-B", bounds=[(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)])
    if not result.success:
        return None, None
    return (float(result.x[0]), float(result.x[1]), float(result.x[2])), float(result.fun)


def sabr_price(inputs: PricingInputs, smile_strikes: np.ndarray, smile_vols: np.ndarray) -> ModelOutput:
    forward = inputs.spot * exp((inputs.rate - inputs.dividend_yield) * inputs.maturity)
    params, fit_error = calibrate_sabr(smile_strikes, smile_vols, forward, inputs.maturity)
    if params is None:
        return ModelOutput("sabr", None, None, 0.0, ["SABR calibration failed or insufficient smile data."])
    alpha, rho, nu = params
    vol = sabr_lognormal_vol(forward, inputs.strike, inputs.maturity, alpha, 1.0, rho, nu)
    if not np.isfinite(vol) or vol <= 0:
        return ModelOutput("sabr", None, None, 0.0, ["SABR produced an invalid volatility."])
    warnings = []
    if fit_error is not None and fit_error > 0.01:
        warnings.append("SABR fit quality is weak; treat valuation cautiously.")
    return ModelOutput(
        "sabr",
        black76_price(inputs.spot, inputs.strike, inputs.maturity, inputs.rate, float(vol), inputs.dividend_yield, inputs.option_type),
        float(vol),
        0.15,
        warnings,
    )


def binomial_model_price(inputs: PricingInputs, steps: int = 100) -> ModelOutput:
    try:
        price = binom_price(
            inputs.spot,
            inputs.strike,
            inputs.maturity,
            inputs.rate,
            max(inputs.implied_vol, 1e-8),
            inputs.dividend_yield,
            steps,
            inputs.option_type,
            american=inputs.american,
        )
        return ModelOutput("binomial", float(price), inputs.implied_vol, 0.2)
    except Exception as exc:
        return ModelOutput("binomial", None, None, 0.0, [f"Binomial pricing failed: {exc}"])


def bs_model_price(inputs: PricingInputs) -> ModelOutput:
    price = bs_price(inputs.spot, inputs.strike, inputs.maturity, inputs.rate, max(inputs.implied_vol, 1e-8), inputs.dividend_yield, inputs.option_type)
    return ModelOutput("black_scholes", float(price), inputs.implied_vol, 0.25)


def black76_model_price(inputs: PricingInputs) -> ModelOutput:
    price = black76_price(inputs.spot, inputs.strike, inputs.maturity, inputs.rate, max(inputs.implied_vol, 1e-8), inputs.dividend_yield, inputs.option_type)
    return ModelOutput("black_76", float(price), inputs.implied_vol, 0.15)


def unreliable_heston_model() -> ModelOutput:
    return ModelOutput("heston", None, None, 0.0, ["Heston calibration is scaffolded only; not included in live ensemble."])


def ensemble_price(outputs: list[ModelOutput]) -> tuple[float | None, float]:
    valid = [output for output in outputs if output.price is not None and np.isfinite(output.price) and output.weight > 0]
    if not valid:
        return None, 0.0
    total_weight = sum(output.weight for output in valid)
    price = sum(float(output.price) * output.weight for output in valid) / total_weight
    prices = np.array([float(output.price) for output in valid], dtype=float)
    dispersion = float(np.std(prices)) if len(prices) > 1 else 0.0
    confidence = max(0.0, 1.0 - (dispersion / max(abs(price), 1.0)))
    return float(price), confidence
