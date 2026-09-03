"""Conservative warnings for fitted-peak identifiability and window effects."""

import numpy as np

from sifter.config import (
    BOUND_PROXIMITY_FRACTION,
    COLLAPSED_AREA_FRACTION,
    EXTREME_CORRELATION,
    POOR_RESOLUTION_FRACTION,
)
from sifter.fitting.optimizer import CandidateFit
from sifter.lineshapes import gaussian_fwhm, lorentzian_fwhm, voigt_fwhm
from sifter.models import ParameterLayout, PeakStart
from sifter.reporting import DiagnosticWarning, diagnostic_warning
from sifter.spectrum import Spectrum


def diagnose_fit(fit: CandidateFit, spectrum: Spectrum) -> tuple[DiagnosticWarning, ...]:
    """Emit structured warnings without silently altering a converged fit."""
    warnings: list[DiagnosticWarning] = []
    warnings.extend(_resolution_warnings(fit))
    warnings.extend(_bound_warnings(fit))
    warnings.extend(_area_warnings(fit))
    correlation = _correlation_warning(fit)
    if correlation is not None:
        warnings.append(correlation)
    warnings.extend(_window_warnings(fit, spectrum))
    return tuple(warnings)


def _resolution_warnings(fit: CandidateFit) -> list[DiagnosticWarning]:
    warnings: list[DiagnosticWarning] = []
    for left_index, left in enumerate(fit.peaks):
        for right_index in range(left_index + 1, len(fit.peaks)):
            right = fit.peaks[right_index]
            threshold = POOR_RESOLUTION_FRACTION * min(
                _peak_fwhm(fit.spec.shape, left),
                _peak_fwhm(fit.spec.shape, right),
            )
            if abs(right.center - left.center) < threshold:
                warnings.append(
                    diagnostic_warning(
                        "PEAKS_POORLY_RESOLVED",
                        "two fitted peaks are too close for reliable separation",
                        context={"peak_indices": [left_index, right_index]},
                    )
                )
    return warnings


def _bound_warnings(fit: CandidateFit) -> list[DiagnosticWarning]:
    layout = ParameterLayout(fit.spec.shape, fit.spec.peak_count, fit.spec.baseline_order)
    warnings: list[DiagnosticWarning] = []
    for index, (value, lower, upper) in enumerate(
        zip(
            fit.parameters,
            fit.spec.lower_bounds,
            fit.spec.upper_bounds,
            strict=True,
        )
    ):
        fraction = min(float(value - lower), float(upper - value)) / (upper - lower)
        if fraction <= BOUND_PROXIMITY_FRACTION:
            warnings.append(
                diagnostic_warning(
                    "PARAMETER_NEAR_BOUND",
                    "a fitted parameter is close to an optimization bound",
                    context={"parameter": layout.names[index], "fraction": fraction},
                )
            )
    return warnings


def _area_warnings(fit: CandidateFit) -> list[DiagnosticWarning]:
    threshold = COLLAPSED_AREA_FRACTION * max(1.0, sum(peak.area for peak in fit.peaks))
    return [
        diagnostic_warning(
            "AREA_COLLAPSED",
            "a fitted peak area collapsed to a negligible value",
            context={"peak_index": index, "area": peak.area},
        )
        for index, peak in enumerate(fit.peaks)
        if peak.area <= threshold
    ]


def _correlation_warning(fit: CandidateFit) -> DiagnosticWarning | None:
    centered = fit.jacobian - np.mean(fit.jacobian, axis=0)
    norms = np.linalg.norm(centered, axis=0)
    valid = np.flatnonzero(norms > np.finfo(float).eps)
    if valid.size < 2:
        return None
    normalized = centered[:, valid] / norms[valid]
    correlation = normalized.T @ normalized
    np.fill_diagonal(correlation, 0.0)
    flat_index = int(np.argmax(np.abs(correlation)))
    left_local, right_local = np.unravel_index(flat_index, correlation.shape)
    maximum = float(abs(correlation[left_local, right_local]))
    if maximum <= EXTREME_CORRELATION:
        return None
    layout = ParameterLayout(fit.spec.shape, fit.spec.peak_count, fit.spec.baseline_order)
    left = int(valid[left_local])
    right = int(valid[right_local])
    return diagnostic_warning(
        "EXTREME_PARAMETER_CORRELATION",
        "two fitted parameters have nearly indistinguishable sensitivities",
        context={
            "parameters": [layout.names[left], layout.names[right]],
            "absolute_correlation": maximum,
        },
    )


def _window_warnings(fit: CandidateFit, spectrum: Spectrum) -> list[DiagnosticWarning]:
    warnings: list[DiagnosticWarning] = []
    for index, peak in enumerate(fit.peaks):
        half_width = _peak_fwhm(fit.spec.shape, peak) / 2.0
        edge_distance = min(peak.center - spectrum.x[0], spectrum.x[-1] - peak.center)
        if edge_distance < half_width:
            warnings.append(
                diagnostic_warning(
                    "PEAK_TRUNCATED_BY_WINDOW",
                    "the observation window truncates a fitted peak near an edge",
                    context={"peak_index": index},
                )
            )
    return warnings


def _peak_fwhm(shape: str, peak: PeakStart) -> float:
    if shape == "gaussian":
        assert peak.sigma is not None
        return gaussian_fwhm(peak.sigma)
    if shape == "lorentzian":
        assert peak.gamma is not None
        return lorentzian_fwhm(peak.gamma)
    assert peak.sigma is not None and peak.gamma is not None
    return voigt_fwhm(sigma=peak.sigma, gamma=peak.gamma)
