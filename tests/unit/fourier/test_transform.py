import numpy as np

from sifter import Spectrum
from sifter.fourier import analyze_fourier
from tests.helpers import single_peak_spectrum


def test_transform_returns_immutable_positive_frequency_arrays() -> None:
    spectrum = single_peak_spectrum(shape="gaussian")

    diagnostics = analyze_fourier(spectrum, spectrum.intensity)

    assert diagnostics.applicable
    assert diagnostics.window == "hann"
    assert diagnostics.frequency.size == diagnostics.magnitude.size
    assert diagnostics.frequency[0] > 0
    assert np.all(np.diff(diagnostics.frequency) > 0)
    assert not diagnostics.frequency.flags.writeable
    assert not diagnostics.magnitude.flags.writeable


def test_nonuniform_grid_fails_closed_by_default() -> None:
    x = np.linspace(0.0, 1.0, 1001) ** 1.01
    spectrum = Spectrum(x, np.exp(-(((x - 0.5) / 0.05) ** 2)))

    diagnostics = analyze_fourier(spectrum, spectrum.intensity)

    assert not diagnostics.applicable
    assert not diagnostics.interpolated
    assert diagnostics.warning_code == "NONUNIFORM_GRID_FFT_DISABLED"
    assert diagnostics.frequency.size == 0


def test_nonuniform_grid_interpolation_is_explicitly_recorded() -> None:
    x = np.linspace(0.0, 1.0, 1001) ** 1.01
    spectrum = Spectrum(x, np.exp(-(((x - 0.5) / 0.05) ** 2)))

    diagnostics = analyze_fourier(spectrum, spectrum.intensity, interpolate_nonuniform=True)

    assert diagnostics.applicable
    assert diagnostics.interpolated
    assert diagnostics.warning_code == "NONUNIFORM_GRID_INTERPOLATED"
