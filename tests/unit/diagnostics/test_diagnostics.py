from dataclasses import replace

import numpy as np

from sifter.diagnostics import diagnose_fit, residual_diagnostics
from sifter.fitting import CandidateFit, fit_candidate
from sifter.models import PeakStart
from tests.helpers import easy_one_peak_spectrum, one_gaussian_spec


def test_residual_diagnostics_match_hand_calculated_durbin_watson() -> None:
    residuals = np.array([1.0, -1.0, 2.0, -2.0])

    diagnostics = residual_diagnostics(residuals)

    assert diagnostics.durbin_watson == np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
    assert diagnostics.mean == np.mean(residuals)
    assert diagnostics.standard_deviation == np.std(residuals, ddof=1)


def test_close_centers_emit_structured_resolution_warning() -> None:
    fit, spectrum = _base_fit()
    close = replace(
        fit,
        peaks=(
            PeakStart(area=1.0, center=1.00, sigma=0.10),
            PeakStart(area=0.8, center=1.03, sigma=0.10),
        ),
    )

    warnings = diagnose_fit(close, spectrum)

    warning = next(item for item in warnings if item.code == "PEAKS_POORLY_RESOLVED")
    assert warning.severity == "warning"
    assert warning.context["peak_indices"] == [0, 1]


def test_bounds_collapsed_area_correlation_and_window_are_reported() -> None:
    fit, spectrum = _base_fit()
    lower = np.asarray(fit.spec.lower_bounds)
    near_bound = np.array(fit.parameters, copy=True)
    near_bound[1] = lower[1] + 1e-12
    pathological = replace(
        fit,
        parameters=near_bound,
        peaks=(PeakStart(area=1e-12, center=spectrum.x[0] + 0.001, sigma=1.2),),
        jacobian=np.column_stack(
            (
                np.ones(spectrum.x.size),
                np.linspace(0.0, 1.0, spectrum.x.size),
                np.linspace(0.0, 1.0, spectrum.x.size) * 1.000001,
                np.linspace(1.0, 0.0, spectrum.x.size),
            )
        ),
    )

    codes = {warning.code for warning in diagnose_fit(pathological, spectrum)}

    assert "PARAMETER_NEAR_BOUND" in codes
    assert "AREA_COLLAPSED" in codes
    assert "EXTREME_PARAMETER_CORRELATION" in codes
    assert "PEAK_TRUNCATED_BY_WINDOW" in codes


def _base_fit() -> tuple[CandidateFit, object]:
    spectrum = easy_one_peak_spectrum()
    fit = fit_candidate(spectrum, one_gaussian_spec(spectrum), starts=3, seed=2)
    assert isinstance(fit, CandidateFit)
    return fit, spectrum
