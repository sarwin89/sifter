"""Peak-parameter bounds derived from resolved candidate centers."""

import numpy as np

from sifter.config import PeakShape

GAUSSIAN_FWHM_PER_SIGMA = 2.354820045


def center_bounds(
    centers: tuple[float, ...],
    *,
    lower_limit: float,
    upper_limit: float,
    allow_broad_multimax_component: bool,
) -> tuple[tuple[float, float], ...]:
    """Return local ownership intervals for initialized peak centers."""
    if allow_broad_multimax_component or len(centers) < 2:
        return tuple((lower_limit, upper_limit) for _ in centers)

    bounds: list[tuple[float, float]] = []
    for index, center in enumerate(centers):
        left_gap = None if index == 0 else center - centers[index - 1]
        right_gap = None if index + 1 == len(centers) else centers[index + 1] - center
        left = center - (0.5 * (left_gap if left_gap is not None else right_gap or 0.0))
        right = center + (0.5 * (right_gap if right_gap is not None else left_gap or 0.0))
        bounds.append((max(lower_limit, float(left)), min(upper_limit, float(right))))
    return tuple(bounds)


def width_upper_bounds(
    shape: PeakShape,
    centers: tuple[float, ...],
    *,
    span: float,
    minimum_width: float,
    allow_broad_multimax_component: bool,
) -> tuple[tuple[float | None, float | None], ...]:
    """Return per-peak sigma/gamma upper bounds.

    When broad multimax components are disallowed, the width budget for each
    initialized peak is tied to its nearest initialized neighbor. This prevents
    the optimizer from using one broad component to absorb multiple resolved
    maxima when a split candidate is available.
    """
    if allow_broad_multimax_component or len(centers) < 2:
        return tuple(_span_bounds(shape, span) for _ in centers)

    bounds: list[tuple[float | None, float | None]] = []
    for index, center in enumerate(centers):
        nearest = min(
            abs(center - other)
            for other_index, other in enumerate(centers)
            if other_index != index
        )
        fwhm_budget = max(
            min(2.0 * nearest * 0.98, span),
            minimum_width * GAUSSIAN_FWHM_PER_SIGMA * 1.01,
        )
        bounds.append(_shape_bounds(shape, fwhm_budget, minimum_width))
    return tuple(bounds)


def _shape_bounds(
    shape: PeakShape,
    fwhm_budget: float,
    minimum_width: float,
) -> tuple[float | None, float | None]:
    if shape == "gaussian":
        return (_bounded(fwhm_budget / GAUSSIAN_FWHM_PER_SIGMA, minimum_width), None)
    if shape == "lorentzian":
        return (None, _bounded(fwhm_budget / 2.0, minimum_width))
    sigma_upper = _bounded(0.55 * fwhm_budget / GAUSSIAN_FWHM_PER_SIGMA, minimum_width)
    gamma_upper = _bounded(0.55 * fwhm_budget / 2.0, minimum_width)
    return (sigma_upper, gamma_upper)


def _span_bounds(shape: PeakShape, span: float) -> tuple[float | None, float | None]:
    if shape == "gaussian":
        return (span, None)
    if shape == "lorentzian":
        return (None, span)
    return (span, span)


def _bounded(value: float, minimum_width: float) -> float:
    if not np.isfinite(value):
        return minimum_width * 1.01
    return max(float(value), minimum_width * 1.01)
