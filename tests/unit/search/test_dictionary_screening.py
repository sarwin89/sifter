import numpy as np

from sifter.search.dictionary import screen_peak_dictionary
from sifter.synthetic import SyntheticPeak, make_spectrum
from sifter.workbench import inspect_spectrum, merge_peak_hints


def test_inspection_reports_detected_maxima_and_merges_manual_hints() -> None:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 4.0, 401),
        peaks=(
            SyntheticPeak("gaussian", area=2.0, center=1.0, sigma=0.07),
            SyntheticPeak("gaussian", area=1.4, center=2.6, sigma=0.09),
        ),
        baseline=(0.15,),
        noise="gaussian",
        snr=150.0,
        seed=5,
    )

    inspection = inspect_spectrum(spectrum, max_peaks=6)
    centers = [maximum.center for maximum in inspection.maxima]

    assert any(abs(center - 1.0) < 0.08 for center in centers)
    assert any(abs(center - 2.6) < 0.10 for center in centers)
    assert merge_peak_hints(inspection.maxima, (3.3,), max_peaks=6)[-1] == 3.3


def test_dictionary_screening_recovers_plausible_peak_centers() -> None:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 5.0, 501),
        peaks=(
            SyntheticPeak("voigt", area=1.7, center=1.4, sigma=0.06, gamma=0.035),
            SyntheticPeak("voigt", area=1.2, center=2.0, sigma=0.07, gamma=0.040),
            SyntheticPeak("voigt", area=1.5, center=3.6, sigma=0.08, gamma=0.050),
        ),
        baseline=(0.2,),
        noise="gaussian",
        snr=120.0,
        seed=22,
    )

    inspection = inspect_spectrum(spectrum, max_peaks=8)
    screened = screen_peak_dictionary(
        spectrum,
        inspection.maxima,
        peak_hints=(2.0,),
        max_peaks=6,
    )
    centers = [peak.center for peak in screened.peaks]

    assert len(centers) <= 6
    assert any(abs(center - 1.4) < 0.10 for center in centers)
    assert any(abs(center - 2.0) < 0.10 for center in centers)
    assert any(abs(center - 3.6) < 0.12 for center in centers)
    assert screened.elapsed_seconds >= 0.0
