"""True Voigt profile evaluated through the Faddeeva function."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import wofz

from sifter.lineshapes.gaussian import gaussian_fwhm
from sifter.lineshapes.lorentzian import lorentzian_fwhm


def voigt(
    x: ArrayLike,
    *,
    area: float,
    center: float,
    sigma: float,
    gamma: float,
) -> NDArray[np.float64]:
    """Evaluate an area-normalized Voigt profile."""
    _validate_profile_parameters(area=area, center=center, sigma=sigma, gamma=gamma)
    values = np.asarray(x, dtype=np.float64)
    z = ((values - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
    return np.asarray(
        area * np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi)),
        dtype=np.float64,
    )


def voigt_fwhm(*, sigma: float, gamma: float) -> float:
    """Approximate Voigt FWHM with the Olivero-Longbothum expression."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    if sigma == 0 and gamma == 0:
        raise ValueError("at least one Voigt width must be positive")
    if gamma == 0:
        return gaussian_fwhm(sigma)
    if sigma == 0:
        return lorentzian_fwhm(gamma)
    gaussian_width = gaussian_fwhm(sigma)
    lorentzian_width = lorentzian_fwhm(gamma)
    return float(
        0.5346 * lorentzian_width + np.sqrt(0.2166 * lorentzian_width**2 + gaussian_width**2)
    )


def _validate_profile_parameters(*, area: float, center: float, sigma: float, gamma: float) -> None:
    if not np.isfinite(area) or area < 0:
        raise ValueError("area must be finite and nonnegative")
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
