"""Lorentzian profile using integrated area and HWHM gamma."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lorentzian(x: ArrayLike, *, area: float, center: float, gamma: float) -> NDArray[np.float64]:
    """Evaluate an area-normalized Lorentzian profile."""
    _validate_profile_parameters(area=area, center=center, gamma=gamma)
    values = np.asarray(x, dtype=np.float64)
    return np.asarray(
        area * gamma / (np.pi * ((values - center) ** 2 + gamma**2)),
        dtype=np.float64,
    )


def lorentzian_fwhm(gamma: float) -> float:
    """Convert Lorentzian HWHM gamma to full width at half maximum."""
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and nonnegative")
    return float(2.0 * gamma)


def _validate_profile_parameters(*, area: float, center: float, gamma: float) -> None:
    if not np.isfinite(area) or area < 0:
        raise ValueError("area must be finite and nonnegative")
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
