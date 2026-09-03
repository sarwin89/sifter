"""Characteristic-spacing candidates from autocorrelation structure."""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate, find_peaks


def candidate_spacings(
    intensity: NDArray[np.float64], *, coordinate_step: float, limit: int = 5
) -> tuple[float, ...]:
    """Return positive autocorrelation lags as diagnostic spacing candidates."""
    centered = intensity - np.median(intensity)
    correlation = correlate(centered, centered, mode="full", method="fft")[intensity.size - 1 :]
    if correlation.size < 3 or correlation[0] <= 0:
        return ()
    correlation = correlation / correlation[0]
    correlation[0] = 0.0
    indices, properties = find_peaks(
        correlation,
        prominence=0.05,
        distance=max(2, intensity.size // 200),
    )
    if indices.size == 0:
        return ()
    ranked = sorted(
        zip(indices, properties["prominences"], strict=True),
        key=lambda pair: (-float(pair[1]), int(pair[0])),
    )[:limit]
    return tuple(sorted(float(index * coordinate_step) for index, _ in ranked))
