"""Fast pseudo-Voigt dictionary screening for v0.4.2 proposals."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import nnls

from sifter.lineshapes import gaussian, lorentzian
from sifter.spectrum import Spectrum
from sifter.workbench import DetectedMaximum, merge_peak_hints


@dataclass(frozen=True, slots=True)
class DictionaryPeak:
    """One sparse-screened peak proposal."""

    center: float
    width_fwhm: float
    area: float
    score: float
    source: str


@dataclass(frozen=True, slots=True)
class DictionaryScreeningResult:
    """Sparse dictionary evidence used to seed exact final fitting."""

    peaks: tuple[DictionaryPeak, ...]
    baseline: float
    residual_norm: float
    elapsed_seconds: float


def screen_peak_dictionary(
    spectrum: Spectrum,
    maxima: tuple[DetectedMaximum, ...],
    *,
    peak_hints: tuple[float, ...] = (),
    max_peaks: int = 10,
) -> DictionaryScreeningResult:
    """Screen centers with an overcomplete nonnegative pseudo-Voigt dictionary."""
    if isinstance(max_peaks, bool) or max_peaks < 1 or max_peaks > 10:
        raise ValueError("max_peaks must be an integer from 1 through 10")
    started = perf_counter()
    centers = merge_peak_hints(maxima, peak_hints, max_peaks=max_peaks)
    if not centers:
        return DictionaryScreeningResult((), float(np.min(spectrum.intensity)), 0.0, 0.0)

    columns: list[np.ndarray] = []
    descriptors: list[tuple[float, float, str]] = []
    median_step = spectrum.grid.median_step
    width_by_center = {maximum.center: maximum.width for maximum in maxima}
    for center in centers:
        nearest_width = _nearest_width(center, width_by_center, median_step)
        for multiplier in (0.75, 1.0, 1.5):
            width = max(nearest_width * multiplier, median_step * 2.0)
            columns.append(_pseudo_voigt_unit_area(spectrum.x, center=center, width_fwhm=width))
            descriptors.append((center, width, "hint" if center in peak_hints else "detected"))
    columns.append(np.ones_like(spectrum.x, dtype=np.float64))
    matrix = np.column_stack(columns)
    coefficients, residual_norm = nnls(matrix, spectrum.intensity)
    baseline = float(coefficients[-1])

    by_center: dict[float, DictionaryPeak] = {}
    for coefficient, (center, width, source) in zip(coefficients[:-1], descriptors, strict=True):
        if coefficient <= np.finfo(float).eps:
            continue
        score = float(coefficient * _local_height(spectrum, center))
        current = by_center.get(center)
        peak = DictionaryPeak(
            center=float(center),
            width_fwhm=float(width),
            area=float(coefficient),
            score=score,
            source=source,
        )
        if current is None or peak.score > current.score:
            by_center[center] = peak
    peaks = tuple(
        sorted(
            sorted(by_center.values(), key=lambda peak: (-peak.score, peak.center))[:max_peaks],
            key=lambda peak: peak.center,
        )
    )
    return DictionaryScreeningResult(
        peaks=peaks,
        baseline=baseline,
        residual_norm=float(residual_norm),
        elapsed_seconds=perf_counter() - started,
    )


def _nearest_width(
    center: float,
    width_by_center: dict[float, float],
    median_step: float,
) -> float:
    if not width_by_center:
        return max(12.0 * median_step, np.finfo(float).eps)
    nearest = min(width_by_center, key=lambda value: abs(value - center))
    return max(float(width_by_center[nearest]), 2.0 * median_step)


def _pseudo_voigt_unit_area(
    x: np.ndarray,
    *,
    center: float,
    width_fwhm: float,
) -> np.ndarray:
    sigma = max(width_fwhm / 2.354820045, np.finfo(float).eps)
    gamma = max(width_fwhm / 2.0, np.finfo(float).eps)
    return 0.5 * gaussian(x, area=1.0, center=center, sigma=sigma) + 0.5 * lorentzian(
        x,
        area=1.0,
        center=center,
        gamma=gamma,
    )


def _local_height(spectrum: Spectrum, center: float) -> float:
    nearest = int(np.argmin(np.abs(spectrum.x - center)))
    floor = float(np.quantile(spectrum.intensity, 0.05))
    return max(float(spectrum.intensity[nearest] - floor), np.finfo(float).eps)
