"""Asymmetric least-squares baseline estimation."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve


def asls_baseline(
    intensity: ArrayLike,
    *,
    smoothness: float = 1e5,
    asymmetry: float = 0.01,
    iterations: int = 10,
) -> NDArray[np.float64]:
    """Estimate a smooth lower envelope with asymmetric least squares."""
    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("intensity must be one-dimensional with at least three values")
    if not np.isfinite(values).all():
        raise ValueError("intensity must contain only finite values")
    if not np.isfinite(smoothness) or smoothness <= 0:
        raise ValueError("smoothness must be finite and positive")
    if not np.isfinite(asymmetry) or not 0 < asymmetry < 1:
        raise ValueError("asymmetry must lie strictly between zero and one")
    if isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")

    diagonals: list[list[float]] = [
        [1.0] * (values.size - 2),
        [-2.0] * (values.size - 2),
        [1.0] * (values.size - 2),
    ]
    difference = sparse.diags(
        diagonals,
        offsets=[0, 1, 2],
        shape=(values.size - 2, values.size),
        format="csc",
        dtype=np.float64,
    )
    penalty = smoothness * (difference.T @ difference)
    weights = np.ones(values.size, dtype=np.float64)
    baseline = values.copy()
    for _ in range(iterations):
        system = sparse.diags(weights, format="csc") + penalty
        baseline = np.asarray(spsolve(system, weights * values), dtype=np.float64)
        weights = np.where(values > baseline, asymmetry, 1.0 - asymmetry)
    return baseline
