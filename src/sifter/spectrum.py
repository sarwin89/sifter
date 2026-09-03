"""Validated, immutable representation of one observed spectrum."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sifter.config import FFT_UNIFORMITY_TOLERANCE, JSONScalar

MINIMUM_OBSERVATIONS = 8


@dataclass(frozen=True, slots=True)
class GridDiagnostics:
    """Sampling-grid properties used to decide whether an ordinary FFT is valid."""

    median_step: float
    relative_step_spread: float
    is_uniform_for_fft: bool


@dataclass(frozen=True, slots=True, init=False)
class Spectrum:
    """One finite, strictly increasing spectrum with immutable arrays."""

    x: NDArray[np.float64]
    intensity: NDArray[np.float64]
    sigma: NDArray[np.float64] | None
    x_name: str
    x_unit: str | None
    intensity_name: str
    metadata: Mapping[str, JSONScalar]
    grid: GridDiagnostics

    def __init__(
        self,
        x: ArrayLike,
        intensity: ArrayLike,
        *,
        sigma: ArrayLike | None = None,
        x_name: str = "x",
        x_unit: str | None = None,
        intensity_name: str = "intensity",
        metadata: Mapping[str, JSONScalar] | None = None,
    ) -> None:
        x_values = _one_dimensional_float_array(x, "x")
        intensity_values = _one_dimensional_float_array(intensity, "intensity")
        if x_values.size != intensity_values.size:
            raise ValueError("x and intensity must have equal lengths")
        if x_values.size < MINIMUM_OBSERVATIONS:
            raise ValueError(f"a spectrum requires at least {MINIMUM_OBSERVATIONS} observations")

        _require_finite(x_values, "x")
        _require_finite(intensity_values, "intensity")
        differences = np.diff(x_values)
        if np.any(differences == 0):
            raise ValueError("x contains duplicate coordinates")
        descending = bool(np.all(differences < 0))
        if not descending and not np.all(differences > 0):
            raise ValueError("x must be monotonic; sort observations before constructing Spectrum")
        if np.ptp(intensity_values) == 0:
            raise ValueError("intensity must not be constant")

        sigma_values = None if sigma is None else _one_dimensional_float_array(sigma, "sigma")
        if sigma_values is not None:
            if sigma_values.size != x_values.size:
                raise ValueError("sigma must have the same length as x and intensity")
            _require_finite(sigma_values, "sigma")
            if np.any(sigma_values <= 0):
                raise ValueError("sigma values must all be positive")

        if descending:
            x_values = x_values[::-1].copy()
            intensity_values = intensity_values[::-1].copy()
            if sigma_values is not None:
                sigma_values = sigma_values[::-1].copy()

        checked_metadata = dict(metadata or {})
        invalid_metadata = [
            key
            for key, value in checked_metadata.items()
            if not isinstance(value, (str, int, float, bool, type(None)))
            or (isinstance(value, float) and not np.isfinite(value))
        ]
        if invalid_metadata:
            raise ValueError("metadata values must be finite JSON scalar values")

        ordered_steps = np.diff(x_values)
        median_step = float(np.median(ordered_steps))
        relative_spread = float(np.max(np.abs(ordered_steps - median_step)) / median_step)
        grid = GridDiagnostics(
            median_step=median_step,
            relative_step_spread=relative_spread,
            is_uniform_for_fft=relative_spread <= FFT_UNIFORMITY_TOLERANCE,
        )

        _freeze(x_values)
        _freeze(intensity_values)
        if sigma_values is not None:
            _freeze(sigma_values)

        object.__setattr__(self, "x", x_values)
        object.__setattr__(self, "intensity", intensity_values)
        object.__setattr__(self, "sigma", sigma_values)
        object.__setattr__(self, "x_name", x_name)
        object.__setattr__(self, "x_unit", x_unit)
        object.__setattr__(self, "intensity_name", intensity_name)
        object.__setattr__(self, "metadata", MappingProxyType(checked_metadata))
        object.__setattr__(self, "grid", grid)


def _one_dimensional_float_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _require_finite(values: NDArray[np.float64], name: str) -> None:
    invalid = np.flatnonzero(~np.isfinite(values))
    if invalid.size:
        raise ValueError(f"{name} contains a non-finite value at index {int(invalid[0])}")


def _freeze(values: NDArray[np.float64]) -> None:
    values.setflags(write=False)

