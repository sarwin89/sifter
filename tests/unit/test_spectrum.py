import numpy as np
import pytest

from sifter import Spectrum


def test_descending_grid_is_reversed_and_arrays_are_read_only() -> None:
    spectrum = Spectrum(np.arange(9.0)[::-1], (np.arange(9.0) ** 2)[::-1])

    assert np.array_equal(spectrum.x, np.arange(9.0))
    assert np.array_equal(spectrum.intensity, np.arange(9.0) ** 2)
    assert not spectrum.x.flags.writeable
    assert not spectrum.intensity.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        spectrum.x[0] = 4.0


@pytest.mark.parametrize(
    ("x", "intensity", "message"),
    [
        ([0, 1, 2], [1, 2, 1], "at least 8"),
        ([0, 1, 1, 2, 3, 4, 5, 6], list(range(8)), "duplicate"),
        ([0, 2, 1, 3, 4, 5, 6, 7], list(range(8)), "monotonic"),
        (list(range(8)), [1] * 8, "constant"),
        (list(range(8)), [0, 1, 2, float("nan"), 4, 5, 6, 7], "index 3"),
        (list(range(8)), [0, 1, 2, float("inf"), 4, 5, 6, 7], "index 3"),
    ],
)
def test_invalid_spectra_raise_actionable_errors(
    x: object, intensity: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Spectrum(x, intensity)


def test_negative_intensity_and_nonuniform_grid_are_supported() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.2, 4.3, 5.1, 6.4, 8.0])
    intensity = np.array([-2.0, -1.0, 0.0, 2.0, 1.0, 0.5, -0.2, 0.1])

    spectrum = Spectrum(x, intensity)

    assert spectrum.intensity.min() < 0
    assert not spectrum.grid.is_uniform_for_fft
    assert spectrum.grid.relative_step_spread > 1e-3


def test_positive_sigma_is_reversed_with_descending_observations() -> None:
    x = np.arange(8.0)[::-1]
    intensity = (np.arange(8.0) ** 2)[::-1]
    sigma = np.linspace(0.1, 0.8, 8)[::-1]

    spectrum = Spectrum(x, intensity, sigma=sigma)

    assert np.allclose(spectrum.sigma, np.linspace(0.1, 0.8, 8))
    assert spectrum.sigma is not None and not spectrum.sigma.flags.writeable


@pytest.mark.parametrize(
    "sigma",
    [
        [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, float("nan"), 0.1, 0.1, 0.1, 0.1],
    ],
)
def test_sigma_must_be_finite_and_positive(sigma: list[float]) -> None:
    with pytest.raises(ValueError, match="sigma"):
        Spectrum(np.arange(8.0), np.arange(8.0) ** 2, sigma=sigma)


def test_metadata_accepts_scalars_and_is_immutable() -> None:
    spectrum = Spectrum(
        np.arange(8.0),
        np.arange(8.0) ** 2,
        metadata={"sample": "synthetic", "temperature": 300.0},
    )

    assert spectrum.metadata["sample"] == "synthetic"
    with pytest.raises(TypeError):
        spectrum.metadata["sample"] = "changed"


def test_metadata_rejects_nested_or_non_scalar_values() -> None:
    with pytest.raises(ValueError, match="metadata"):
        Spectrum(
            np.arange(8.0),
            np.arange(8.0) ** 2,
            metadata={"private_rows": [1, 2, 3]},
        )
