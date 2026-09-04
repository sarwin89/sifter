import numpy as np
import pytest

from sifter import AutofitConfig
from sifter.search import preprocess_spectrum
from tests.helpers import easy_two_peak_spectrum


def test_preprocessing_returns_immutable_adjusted_signal_and_detection_summary() -> None:
    spectrum, _ = easy_two_peak_spectrum(seed=8)
    config = AutofitConfig(
        max_peaks=4,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
    )

    result = preprocess_spectrum(spectrum, config)

    np.testing.assert_allclose(result.adjusted, spectrum.intensity - result.baseline)
    np.testing.assert_allclose(result.proposal_spectrum.intensity, result.adjusted)
    assert result.detection.detected_count == 2
    assert result.detection.centers == pytest.approx((1.0, 1.7), abs=0.05)
    assert result.detection.median_width is not None
    assert result.detection.strongest_prominence is not None
    assert result.fourier is None
    assert not result.baseline.flags.writeable
    assert not result.adjusted.flags.writeable


def test_preprocessing_returns_requested_fourier_diagnostics() -> None:
    spectrum, _ = easy_two_peak_spectrum(seed=8)
    config = AutofitConfig(
        max_peaks=4,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=True,
    )

    result = preprocess_spectrum(spectrum, config)

    assert result.fourier is not None
    assert result.fourier.window == "hann"
