"""Deterministic in-bound optimizer starting vectors."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sifter.models import ModelSpec, ParameterLayout


def generate_starts(
    spec: ModelSpec,
    *,
    count: int = 8,
    seed: int = 42,
    initial_parameters: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], ...]:
    """Generate a declared start plus reproducible bounded perturbations."""
    if isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    layout = ParameterLayout(spec.shape, spec.peak_count, spec.baseline_order)
    initial = (
        layout.initial_vector(spec)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=np.float64)
    )
    if initial.ndim != 1 or initial.size != layout.parameter_count:
        raise ValueError("initial_parameters do not match the parameter layout")
    if not np.isfinite(initial).all():
        raise ValueError("initial_parameters must contain only finite values")
    lower = np.asarray(spec.lower_bounds, dtype=np.float64)
    upper = np.asarray(spec.upper_bounds, dtype=np.float64)
    margin = np.maximum((upper - lower) * 1e-9, np.finfo(float).eps)
    safe_lower = lower + margin
    safe_upper = upper - margin
    initial = np.clip(initial, safe_lower, safe_upper)
    starts = [_frozen(initial)]
    generator = np.random.default_rng(seed)
    scale = 0.1 * np.maximum(np.abs(initial), (upper - lower) * 0.01)
    for _ in range(count - 1):
        perturbed = np.clip(initial + generator.normal(0.0, scale), safe_lower, safe_upper)
        starts.append(_frozen(perturbed))
    return tuple(starts)


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied
