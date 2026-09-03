"""Competing decay-envelope models for Fourier magnitudes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

from sifter.config import PeakShape


@dataclass(frozen=True, slots=True)
class EnvelopeFit:
    """One fitted Fourier-envelope tendency with a local comparison score."""

    family: PeakShape
    intercept: float
    decay_coefficients: tuple[float, ...]
    rss: float
    bic: float
    frequency_min: float
    frequency_max: float


def fit_envelope_models(
    frequency: NDArray[np.float64], magnitude: NDArray[np.float64]
) -> tuple[EnvelopeFit, ...]:
    """Fit Gaussian-, Lorentzian-, and Voigt-like log-magnitude decays."""
    mask = _usable_envelope_mask(magnitude)
    usable_frequency = frequency[mask]
    usable_magnitude = magnitude[mask]
    if usable_frequency.size < 8:
        return ()
    log_magnitude = np.log(usable_magnitude)
    fits = (
        _fit_one("gaussian", usable_frequency, log_magnitude),
        _fit_one("lorentzian", usable_frequency, log_magnitude),
        _fit_one("voigt", usable_frequency, log_magnitude),
    )
    return fits


def _usable_envelope_mask(magnitude: NDArray[np.float64]) -> NDArray[np.bool_]:
    positive = magnitude[magnitude > 0]
    if positive.size == 0:
        return np.zeros(magnitude.shape, dtype=np.bool_)
    tail_count = max(8, magnitude.size // 10)
    noise_floor = float(np.median(magnitude[-tail_count:]))
    threshold = max(float(np.max(magnitude)) * 1e-7, noise_floor * 5.0)
    candidates = np.flatnonzero(magnitude > threshold)
    if candidates.size == 0:
        return np.zeros(magnitude.shape, dtype=np.bool_)
    last = int(candidates[-1])
    mask = np.zeros(magnitude.shape, dtype=np.bool_)
    mask[: last + 1] = magnitude[: last + 1] > threshold
    return mask


def _fit_one(
    family: PeakShape,
    frequency: NDArray[np.float64],
    log_magnitude: NDArray[np.float64],
) -> EnvelopeFit:
    if family == "gaussian":
        decay_terms = np.column_stack((-(frequency**2),))
    elif family == "lorentzian":
        decay_terms = np.column_stack((-frequency,))
    else:
        decay_terms = np.column_stack((-frequency, -(frequency**2)))
    design = np.column_stack((np.ones(frequency.size), decay_terms))
    lower = np.concatenate((np.array([-np.inf]), np.zeros(design.shape[1] - 1)))
    upper = np.full(design.shape[1], np.inf)
    solution = lsq_linear(design, log_magnitude, bounds=(lower, upper))
    residuals = design @ solution.x - log_magnitude
    rss = float(np.dot(residuals, residuals))
    safe_rss = max(rss, np.finfo(float).tiny)
    parameter_count = design.shape[1]
    bic = float(
        frequency.size * np.log(safe_rss / frequency.size)
        + parameter_count * np.log(frequency.size)
    )
    return EnvelopeFit(
        family=family,
        intercept=float(solution.x[0]),
        decay_coefficients=tuple(float(value) for value in solution.x[1:]),
        rss=rss,
        bic=bic,
        frequency_min=float(frequency[0]),
        frequency_max=float(frequency[-1]),
    )
