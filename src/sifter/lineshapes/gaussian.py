"""Gaussian profile using integrated area and standard deviation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian(
    x: ArrayLike, *, area: float, center: float, sigma: float
) -> NDArray[np.float64]:
    """Evaluate an area-normalized Gaussian profile."""
    _validate_profile_parameters(area=area, center=center, sigma=sigma)
    values = np.asarray(x, dtype=np.float64)
    z = (values - center) / sigma
    return np.asarray(
        area * np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi)),
        dtype=np.float64,
    )


def gaussian_fwhm(sigma: float) -> float:
    """Convert Gaussian standard deviation to full width at half maximum."""
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)


def _validate_profile_parameters(*, area: float, center: float, sigma: float) -> None:
    if not np.isfinite(area) or area < 0:
        raise ValueError("area must be finite and nonnegative")
    if not np.isfinite(center):
        raise ValueError("center must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")

