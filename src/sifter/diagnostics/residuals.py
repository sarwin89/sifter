"""Residual summary statistics for fitted spectra."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    """Compact tests for residual scale and serial structure."""

    mean: float
    standard_deviation: float
    durbin_watson: float
    lag_one_correlation: float | None


def residual_diagnostics(residuals: ArrayLike) -> ResidualDiagnostics:
    """Calculate finite descriptive diagnostics for a residual vector."""
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("residuals must be a one-dimensional array with at least two values")
    if not np.isfinite(values).all():
        raise ValueError("residuals must contain only finite values")
    denominator = float(np.dot(values, values))
    durbin_watson = (
        float(np.dot(np.diff(values), np.diff(values)) / denominator) if denominator > 0 else 0.0
    )
    left = values[:-1]
    right = values[1:]
    lag_one = None
    if np.std(left) > 0 and np.std(right) > 0:
        lag_one = float(np.corrcoef(left, right)[0, 1])
    return ResidualDiagnostics(
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values, ddof=1)),
        durbin_watson=durbin_watson,
        lag_one_correlation=lag_one,
    )
