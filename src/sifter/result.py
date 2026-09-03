"""Versioned immutable result records for SIFTER analyses."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sifter.config import JSONScalar, PeakShape, UncertaintyMode
from sifter.diagnostics import DiagnosticWarning, ResidualDiagnostics
from sifter.fitting import ParameterUncertainty
from sifter.fourier import FourierDiagnostics
from sifter.selection import CandidateScore


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """Complete reproducible settings used for one analysis."""

    max_peaks: int
    shapes: tuple[PeakShape, ...]
    baseline_orders: tuple[int, ...]
    fourier: bool
    interpolate_nonuniform_fft: bool
    uncertainty: UncertaintyMode
    bootstrap_samples: int
    random_seed: int


@dataclass(frozen=True, slots=True)
class FittedPeak:
    """One canonically ordered fitted peak."""

    area: float
    center: float
    sigma: float | None
    gamma: float | None


@dataclass(frozen=True, slots=True)
class ModelResult:
    """Public numerical result for the recommended candidate."""

    shape: PeakShape
    peak_count: int
    baseline_order: int
    parameter_names: tuple[str, ...]
    parameters: NDArray[np.float64]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    peaks: tuple[FittedPeak, ...]
    fitted: NDArray[np.float64]
    baseline: NDArray[np.float64]
    components: NDArray[np.float64]
    residuals: NDArray[np.float64]
    rss: float
    rmse: float
    aicc: float
    bic: float
    parameter_count: int
    observation_count: int
    reduced_chi_squared: float | None


@dataclass(frozen=True, slots=True)
class FitResult:
    """Stable, versioned output of the public analysis API."""

    schema_version: Literal["sifter.fit_result.v1"]
    settings: AnalysisSettings
    source_metadata: Mapping[str, JSONScalar]
    x: NDArray[np.float64]
    intensity: NDArray[np.float64]
    sigma: NDArray[np.float64] | None
    x_name: str
    x_unit: str | None
    intensity_name: str
    best_model: ModelResult
    candidates: tuple[CandidateScore, ...]
    fourier: FourierDiagnostics | None
    residual_diagnostics: ResidualDiagnostics
    uncertainty: ParameterUncertainty
    warnings: tuple[DiagnosticWarning, ...]
    observation_count: int
    sifter_version: str


def frozen_metadata(values: Mapping[str, JSONScalar]) -> Mapping[str, JSONScalar]:
    """Copy metadata into a read-only mapping for result assembly."""
    return MappingProxyType(dict(values))


def frozen_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copy an array so result records never expose mutable state."""
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied
