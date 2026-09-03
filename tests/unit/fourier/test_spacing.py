import numpy as np

from sifter.fourier import analyze_fourier
from sifter.synthetic import SyntheticPeak, make_spectrum


def test_two_resolved_peaks_expose_candidate_spacing_without_peak_count_claim() -> None:
    spectrum, _ = make_spectrum(
        x=np.linspace(0.0, 4.0, 4001),
        peaks=(
            SyntheticPeak("gaussian", area=1.0, center=1.4, sigma=0.05),
            SyntheticPeak("gaussian", area=1.0, center=2.1, sigma=0.05),
        ),
    )

    diagnostics = analyze_fourier(spectrum, spectrum.intensity)

    assert diagnostics.candidate_spacings
    assert min(abs(spacing - 0.7) for spacing in diagnostics.candidate_spacings) < 0.05
    assert not hasattr(diagnostics, "peak_count")
