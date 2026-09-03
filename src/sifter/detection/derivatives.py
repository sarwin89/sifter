"""Derivative evidence for candidate local maxima."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter


def derivative_peak_indices(
    intensity: NDArray[np.float64], *, window_length: int
) -> NDArray[np.int64]:
    """Return positive-to-negative derivative crossings with negative curvature."""
    first = savgol_filter(intensity, window_length, 3, deriv=1, mode="interp")
    second = savgol_filter(intensity, window_length, 3, deriv=2, mode="interp")
    crossings = np.flatnonzero((first[:-1] > 0) & (first[1:] <= 0)) + 1
    return np.asarray(crossings[second[crossings] < 0], dtype=np.int64)
