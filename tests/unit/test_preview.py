"""Cheap pre-fit spectrum analysis."""

import numpy as np
import pytest

from sifter import AutofitConfig, Spectrum, preview_spectrum
from tests.helpers import easy_two_peak_spectrum


def test_preview_exposes_real_and_fourier_evidence_without_nonlinear_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spectrum, _ = easy_two_peak_spectrum(seed=12)

    def forbidden_fit(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("nonlinear peak fitting must not run during preview")

    monkeypatch.setattr("sifter.fitting.optimizer.least_squares", forbidden_fit)

    preview = preview_spectrum(
        spectrum,
        config=AutofitConfig(max_peaks=4, shapes=("gaussian",), baseline_orders=(0,)),
    )

    assert np.array_equal(preview.x, spectrum.x)
    assert np.array_equal(preview.intensity, spectrum.intensity)
    np.testing.assert_allclose(preview.baseline + preview.adjusted, spectrum.intensity)
    assert len(preview.provisional_centers) == 2
    assert preview.grid_is_uniform
    assert preview.fourier_enabled
    assert preview.fourier_applicable
    assert preview.fourier_window == "hann"
    assert preview.frequency.size == preview.magnitude.size == preview.log_magnitude.size
    assert np.isfinite(preview.log_magnitude).all()
    assert not preview.x.flags.writeable
    assert not preview.baseline.flags.writeable


def test_preview_reports_nonuniform_fft_state_and_units() -> None:
    spectrum, _ = easy_two_peak_spectrum(seed=14)
    x = spectrum.x.copy()
    x[1:-1] += 0.001 * np.sin(np.arange(1, x.size - 1))
    nonuniform = Spectrum(
        x,
        spectrum.intensity,
        x_name="energy",
        x_unit="eV",
        intensity_name="intensity",
    )

    disabled = preview_spectrum(
        nonuniform,
        config=AutofitConfig(
            max_peaks=3,
            fourier=True,
            interpolate_nonuniform_fft=False,
        ),
    )
    interpolated = preview_spectrum(
        nonuniform,
        config=AutofitConfig(
            max_peaks=3,
            fourier=True,
            interpolate_nonuniform_fft=True,
        ),
    )

    assert disabled.x_name == "energy"
    assert disabled.x_unit == "eV"
    assert disabled.frequency_unit == "1/eV"
    assert not disabled.grid_is_uniform
    assert not disabled.fourier_applicable
    assert not disabled.fourier_interpolated
    assert disabled.fourier_warning_code == "NONUNIFORM_GRID_FFT_DISABLED"
    assert interpolated.fourier_interpolated
    assert interpolated.frequency.size > 0
