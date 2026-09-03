"""Typed model declarations and the single parameter-vector layout."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sifter.config import PeakShape
from sifter.lineshapes import gaussian, lorentzian, voigt


@dataclass(frozen=True, slots=True)
class PeakStart:
    """One peak's parameters, used for both starts and decoded fits."""

    area: float
    center: float
    sigma: float | None = None
    gamma: float | None = None


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A complete, serializable candidate model declaration."""

    shape: PeakShape
    peak_count: int
    baseline_order: int
    baseline_start: tuple[float, ...]
    starts: tuple[PeakStart, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        layout = ParameterLayout(self.shape, self.peak_count, self.baseline_order)
        if len(self.baseline_start) != self.baseline_order + 1:
            raise ValueError("baseline_start does not match baseline_order")
        if len(self.starts) != self.peak_count:
            raise ValueError("peak starts do not match peak_count")
        if len(self.lower_bounds) != layout.parameter_count:
            raise ValueError("lower_bounds do not match the parameter layout")
        if len(self.upper_bounds) != layout.parameter_count:
            raise ValueError("upper_bounds do not match the parameter layout")
        if any(
            lower >= upper
            for lower, upper in zip(self.lower_bounds, self.upper_bounds, strict=True)
        ):
            raise ValueError("every lower bound must be below its upper bound")


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Decomposed model values evaluated on one coordinate array."""

    baseline: NDArray[np.float64]
    components: NDArray[np.float64]
    fitted: NDArray[np.float64]
    peaks: tuple[PeakStart, ...]


@dataclass(frozen=True, slots=True)
class ParameterLayout:
    """Canonical mapping between named parameters and optimizer vectors."""

    shape: PeakShape
    peak_count: int
    baseline_order: int

    def __post_init__(self) -> None:
        if self.shape not in {"gaussian", "lorentzian", "voigt"}:
            raise ValueError("unsupported peak shape")
        if self.peak_count < 1:
            raise ValueError("peak_count must be positive")
        if self.baseline_order not in {0, 1, 2}:
            raise ValueError("baseline_order must be 0, 1, or 2")

    @property
    def names(self) -> tuple[str, ...]:
        names = [f"baseline.c{index}" for index in range(self.baseline_order + 1)]
        peak_fields = (
            ("area", "center", "sigma", "gamma")
            if self.shape == "voigt"
            else (
                "area",
                "center",
                "sigma" if self.shape == "gaussian" else "gamma",
            )
        )
        for index in range(self.peak_count):
            names.extend(f"peak.{index}.{field}" for field in peak_fields)
        return tuple(names)

    @property
    def parameter_count(self) -> int:
        return len(self.names)

    def initial_vector(self, spec: ModelSpec) -> NDArray[np.float64]:
        values = list(spec.baseline_start)
        for peak in spec.starts:
            values.extend((peak.area, peak.center))
            if self.shape in {"gaussian", "voigt"}:
                if peak.sigma is None:
                    raise ValueError("Gaussian width is missing")
                values.append(peak.sigma)
            if self.shape in {"lorentzian", "voigt"}:
                if peak.gamma is None:
                    raise ValueError("Lorentzian width is missing")
                values.append(peak.gamma)
        return np.asarray(values, dtype=np.float64)

    def decode_peaks(self, parameters: ArrayLike) -> tuple[PeakStart, ...]:
        values = np.asarray(parameters, dtype=np.float64)
        if values.ndim != 1 or values.size != self.parameter_count:
            raise ValueError("parameters do not match the parameter layout")
        position = self.baseline_order + 1
        peaks: list[PeakStart] = []
        for _ in range(self.peak_count):
            area = float(values[position])
            center = float(values[position + 1])
            position += 2
            sigma = None
            gamma = None
            if self.shape in {"gaussian", "voigt"}:
                sigma = float(values[position])
                position += 1
            if self.shape in {"lorentzian", "voigt"}:
                gamma = float(values[position])
                position += 1
            peaks.append(PeakStart(area=area, center=center, sigma=sigma, gamma=gamma))
        return tuple(peaks)


def evaluate_model(x: ArrayLike, parameters: ArrayLike, spec: ModelSpec) -> ModelEvaluation:
    """Evaluate one candidate using its canonical parameter layout."""
    coordinate = np.asarray(x, dtype=np.float64)
    layout = ParameterLayout(spec.shape, spec.peak_count, spec.baseline_order)
    values = np.asarray(parameters, dtype=np.float64)
    if values.ndim != 1 or values.size != layout.parameter_count:
        raise ValueError("parameters do not match the model specification")
    baseline_coefficients = values[: spec.baseline_order + 1]
    offset = float((coordinate[0] + coordinate[-1]) / 2.0)
    scale = float((coordinate[-1] - coordinate[0]) / 2.0)
    scaled = (coordinate - offset) / scale
    baseline = np.asarray(
        np.polynomial.polynomial.polyval(scaled, baseline_coefficients), dtype=np.float64
    )
    peaks = layout.decode_peaks(values)
    components = np.vstack([_evaluate_peak(coordinate, spec.shape, peak) for peak in peaks])
    fitted = baseline + np.sum(components, axis=0)
    return ModelEvaluation(baseline=baseline, components=components, fitted=fitted, peaks=peaks)


def _evaluate_peak(
    x: NDArray[np.float64], shape: PeakShape, peak: PeakStart
) -> NDArray[np.float64]:
    if shape == "gaussian":
        assert peak.sigma is not None
        return gaussian(x, area=peak.area, center=peak.center, sigma=peak.sigma)
    if shape == "lorentzian":
        assert peak.gamma is not None
        return lorentzian(x, area=peak.area, center=peak.center, gamma=peak.gamma)
    assert peak.sigma is not None and peak.gamma is not None
    return voigt(
        x,
        area=peak.area,
        center=peak.center,
        sigma=peak.sigma,
        gamma=peak.gamma,
    )
