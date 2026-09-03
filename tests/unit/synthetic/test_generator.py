import numpy as np
import pytest

from sifter.synthetic import SyntheticPeak, make_spectrum


def test_seeded_generation_is_reproducible_and_preserves_truth() -> None:
    x = np.linspace(1.0, 2.0, 501)
    peaks = (SyntheticPeak("voigt", area=1.0, center=1.5, sigma=0.03, gamma=0.01),)

    first, truth_a = make_spectrum(
        x=x, peaks=peaks, noise="gaussian", snr=50.0, seed=7
    )
    second, truth_b = make_spectrum(
        x=x, peaks=peaks, noise="gaussian", snr=50.0, seed=7
    )

    assert np.array_equal(first.intensity, second.intensity)
    assert truth_a.peaks == truth_b.peaks == peaks
    assert np.array_equal(truth_a.clean_signal, truth_b.clean_signal)
    assert truth_a.noise_standard_deviation == truth_b.noise_standard_deviation
    assert truth_a.seed == truth_b.seed == 7
    assert not truth_a.clean_signal.flags.writeable


def test_polynomial_baseline_uses_centered_scaled_coordinate() -> None:
    x = np.linspace(1000.0, 1002.0, 101)
    peak = SyntheticPeak("gaussian", area=1.0, center=1001.0, sigma=0.1)

    spectrum, truth = make_spectrum(
        x=x,
        peaks=(peak,),
        baseline=(2.0, 0.5),
        noise="none",
        seed=2,
    )

    z = (x - x.mean()) / ((x.max() - x.min()) / 2.0)
    expected_baseline = 2.0 + 0.5 * z
    assert np.allclose(spectrum.intensity - truth.peak_signal, expected_baseline)
    assert np.array_equal(truth.clean_signal, spectrum.intensity)


@pytest.mark.parametrize(
    "peak",
    [
        SyntheticPeak("gaussian", area=1.0, center=0.0, sigma=0.1),
        SyntheticPeak("lorentzian", area=1.0, center=0.0, gamma=0.1),
        SyntheticPeak("voigt", area=1.0, center=0.0, sigma=0.1, gamma=0.05),
    ],
)
def test_all_supported_families_generate_finite_spectra(peak: SyntheticPeak) -> None:
    spectrum, _ = make_spectrum(x=np.linspace(-1.0, 1.0, 101), peaks=(peak,))

    assert np.isfinite(spectrum.intensity).all()
    assert spectrum.intensity.max() > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": "gaussian", "area": 1.0, "center": 0.0},
        {"shape": "lorentzian", "area": 1.0, "center": 0.0, "sigma": 0.1},
        {"shape": "voigt", "area": 1.0, "center": 0.0, "sigma": 0.1},
        {"shape": "gaussian", "area": -1.0, "center": 0.0, "sigma": 0.1},
    ],
)
def test_peak_definition_rejects_missing_or_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticPeak(**kwargs)


def test_gaussian_noise_uses_declared_rms_snr() -> None:
    x = np.linspace(-1.0, 1.0, 10_001)
    peak = SyntheticPeak("gaussian", area=1.0, center=0.0, sigma=0.1)

    _, truth = make_spectrum(
        x=x, peaks=(peak,), noise="gaussian", snr=25.0, seed=12
    )

    expected = float(np.sqrt(np.mean(truth.peak_signal**2)) / 25.0)
    assert truth.noise_standard_deviation == pytest.approx(expected)


def test_noise_configuration_requires_a_positive_snr() -> None:
    peak = SyntheticPeak("gaussian", area=1.0, center=0.0, sigma=0.1)
    with pytest.raises(ValueError, match="snr"):
        make_spectrum(x=np.linspace(-1.0, 1.0, 101), peaks=(peak,), noise="gaussian")
    with pytest.raises(ValueError, match="noise"):
        make_spectrum(x=np.linspace(-1.0, 1.0, 101), peaks=(peak,), noise="poisson")
