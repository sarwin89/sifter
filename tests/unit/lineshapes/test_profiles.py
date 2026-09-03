import numpy as np
import pytest
from scipy.integrate import simpson

from sifter.lineshapes import (
    gaussian,
    gaussian_fwhm,
    lorentzian,
    lorentzian_fwhm,
    voigt,
    voigt_fwhm,
)


@pytest.mark.parametrize(
    ("profile", "kwargs"),
    [
        (gaussian, {"sigma": 0.7}),
        (lorentzian, {"gamma": 0.7}),
        (voigt, {"sigma": 0.5, "gamma": 0.2}),
    ],
)
def test_profile_integral_equals_integrated_area(profile: object, kwargs: dict[str, float]) -> None:
    x = np.linspace(-100.0, 100.0, 500_001)

    y = profile(x, area=2.5, center=0.4, **kwargs)

    assert simpson(y, x=x) == pytest.approx(2.5, rel=5e-3)


def test_gaussian_height_and_fwhm_follow_sigma_convention() -> None:
    sigma = 0.4
    area = 2.0
    center = 1.2
    width = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma

    height = gaussian(np.array([center]), area=area, center=center, sigma=sigma)[0]
    half_height = gaussian(
        np.array([center + width / 2.0]), area=area, center=center, sigma=sigma
    )[0]

    assert height == pytest.approx(area / (sigma * np.sqrt(2.0 * np.pi)))
    assert half_height == pytest.approx(height / 2.0)
    assert gaussian_fwhm(sigma) == pytest.approx(width)


def test_lorentzian_height_and_fwhm_follow_gamma_hwhm_convention() -> None:
    gamma = 0.3
    area = 1.7
    center = -0.5

    height = lorentzian(np.array([center]), area=area, center=center, gamma=gamma)[0]
    half_height = lorentzian(
        np.array([center + gamma]), area=area, center=center, gamma=gamma
    )[0]

    assert height == pytest.approx(area / (np.pi * gamma))
    assert half_height == pytest.approx(height / 2.0)
    assert lorentzian_fwhm(gamma) == pytest.approx(2.0 * gamma)


def test_voigt_fwhm_reduces_to_component_limits() -> None:
    assert voigt_fwhm(sigma=0.4, gamma=0.0) == pytest.approx(gaussian_fwhm(0.4))
    assert voigt_fwhm(sigma=0.0, gamma=0.3) == pytest.approx(lorentzian_fwhm(0.3))


def test_translation_changes_only_the_center_coordinate() -> None:
    x = np.linspace(-3.0, 3.0, 1001)

    translated_axis = gaussian(x - 0.25, area=1.0, center=0.0, sigma=0.2)
    translated_center = gaussian(x, area=1.0, center=0.25, sigma=0.2)

    assert np.allclose(translated_axis, translated_center)


@pytest.mark.parametrize(
    ("profile", "kwargs"),
    [
        (gaussian, {"area": -1.0, "center": 0.0, "sigma": 1.0}),
        (gaussian, {"area": 1.0, "center": 0.0, "sigma": 0.0}),
        (lorentzian, {"area": 1.0, "center": 0.0, "gamma": -1.0}),
        (voigt, {"area": 1.0, "center": 0.0, "sigma": 0.0, "gamma": 1.0}),
        (voigt, {"area": 1.0, "center": 0.0, "sigma": 1.0, "gamma": 0.0}),
    ],
)
def test_profiles_reject_negative_area_or_nonpositive_widths(
    profile: object, kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        profile(np.array([0.0]), **kwargs)
