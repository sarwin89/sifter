import pytest

from sifter.fourier import analyze_fourier
from tests.helpers import single_peak_spectrum


@pytest.mark.parametrize(
    ("shape", "winner"),
    [("gaussian", "gaussian"), ("lorentzian", "lorentzian"), ("voigt", "voigt")],
)
def test_envelope_family_wins_on_noise_free_profile(shape: str, winner: str) -> None:
    spectrum = single_peak_spectrum(shape=shape, points=4097)

    diagnostics = analyze_fourier(spectrum, spectrum.intensity)

    assert diagnostics.applicable
    scores = {fit.family: fit.bic for fit in diagnostics.envelope_fits}
    assert min(scores, key=scores.get) == winner
    assert all(fit.frequency_min > 0 for fit in diagnostics.envelope_fits)
    assert all(fit.frequency_max > fit.frequency_min for fit in diagnostics.envelope_fits)


def test_envelope_decay_coefficients_are_nonnegative() -> None:
    spectrum = single_peak_spectrum(shape="voigt")

    diagnostics = analyze_fourier(spectrum, spectrum.intensity)

    for fit in diagnostics.envelope_fits:
        assert all(value >= 0 for value in fit.decay_coefficients)
