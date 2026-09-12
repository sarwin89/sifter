"""Configuration and shared public types for SIFTER analyses."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from sifter.context import MeasurementContext
    from sifter.reference import FitReference

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
PeakShape: TypeAlias = Literal["gaussian", "lorentzian", "voigt"]
SearchMode: TypeAlias = Literal["fast", "standard", "thorough", "exhaustive"]
PeakCountMode: TypeAlias = Literal["auto", "exact"]
CountMode: TypeAlias = Literal["auto", "exact"]
FitMode: TypeAlias = Literal["fast", "standard", "thorough", "research"]
UncertaintyMode: TypeAlias = Literal["covariance", "bootstrap"]

SUPPORTED_SHAPES: frozenset[str] = frozenset({"gaussian", "lorentzian", "voigt"})
SUPPORTED_BASELINE_ORDERS: frozenset[int] = frozenset({0})
SUPPORTED_BOOTSTRAP_SAMPLES: frozenset[int] = frozenset({100, 250, 1000})
FFT_UNIFORMITY_TOLERANCE = 1e-3
BOUND_PROXIMITY_FRACTION = 1e-6
COLLAPSED_AREA_FRACTION = 1e-6
EXTREME_CORRELATION = 0.98
POOR_RESOLUTION_FRACTION = 0.5
BOOTSTRAP_SUCCESS_FRACTION = 0.8


@dataclass(frozen=True, slots=True)
class AutofitConfig:
    """Validated settings for automatic model generation and fitting."""

    max_peaks: int = 10
    search_mode: SearchMode = "standard"
    peak_count_mode: PeakCountMode = "auto"
    shapes: tuple[PeakShape, ...] = ("gaussian", "lorentzian", "voigt")
    baseline_orders: tuple[int, ...] = (0,)
    fourier: bool = True
    interpolate_nonuniform_fft: bool = False
    uncertainty: UncertaintyMode = "covariance"
    bootstrap_samples: int = 250
    random_seed: int = 42
    workers: int = 1
    allow_broad_multimax_component: bool = False
    manual_peak_centers: tuple[float, ...] = ()
    measurement_context: MeasurementContext | None = None
    reference: FitReference | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_peaks, bool) or self.max_peaks < 1:
            raise ValueError("max_peaks must be a positive integer")
        if self.search_mode not in {"fast", "standard", "thorough", "exhaustive"}:
            raise ValueError("search_mode must be fast, standard, thorough, or exhaustive")
        if self.peak_count_mode not in {"auto", "exact"}:
            raise ValueError("peak_count_mode must be auto or exact")
        if not self.shapes:
            raise ValueError("at least one peak shape is required")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("peak shapes must be unique")
        unsupported = set(self.shapes) - SUPPORTED_SHAPES
        if unsupported:
            raise ValueError(f"unsupported peak shape: {sorted(unsupported)[0]}")
        if not self.baseline_orders:
            raise ValueError("at least one baseline order is required")
        if len(set(self.baseline_orders)) != len(self.baseline_orders):
            raise ValueError("baseline orders must be unique")
        if not set(self.baseline_orders) <= frozenset({0, 1, 2}):
            raise ValueError("baseline orders must be selected from 0, 1, and 2")
        if self.baseline_orders != (0,):
            raise ValueError("v0.4 supports constant baseline order 0 only")
        if self.uncertainty not in {"covariance", "bootstrap"}:
            raise ValueError("uncertainty must be 'covariance' or 'bootstrap'")
        if self.bootstrap_samples not in SUPPORTED_BOOTSTRAP_SAMPLES:
            raise ValueError("bootstrap_samples must be 100, 250, or 1000")
        if isinstance(self.random_seed, bool) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if isinstance(self.workers, bool) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        if not isinstance(self.allow_broad_multimax_component, bool):
            raise ValueError("allow_broad_multimax_component must be a boolean")
        if len(self.manual_peak_centers) >= 10:
            raise ValueError("manual peak centers must contain fewer than 10 values")
        if len(set(self.manual_peak_centers)) != len(self.manual_peak_centers):
            raise ValueError("manual peak centers must be unique")
        if any(not isfinite(center) for center in self.manual_peak_centers):
            raise ValueError("manual peak centers must be finite")
        if self.manual_peak_centers and len(self.manual_peak_centers) > self.max_peaks:
            raise ValueError("manual peak centers cannot exceed max_peaks")
        if self.measurement_context is not None and not hasattr(
            self.measurement_context, "to_dict"
        ):
            raise ValueError("measurement_context must be a MeasurementContext")
        if self.reference is not None and not hasattr(self.reference, "to_dict"):
            raise ValueError("reference must be a FitReference")


@dataclass(frozen=True, slots=True)
class FitConfig:
    """User-intent settings for the v0.4.2 fitting workbench."""

    max_peaks: int = 10
    count_mode: CountMode = "auto"
    fit_mode: FitMode = "fast"
    shapes: tuple[PeakShape, ...] = ("gaussian", "lorentzian", "voigt")
    peak_hints: tuple[float, ...] = ()
    fourier: bool = True
    interpolate_nonuniform_fft: bool = False
    random_seed: int = 42
    workers: int = 1
    allow_broad_multimax_component: bool = False
    measurement_context: MeasurementContext | None = None
    reference: FitReference | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_peaks, bool) or self.max_peaks < 1:
            raise ValueError("max_peaks must be a positive integer")
        if self.max_peaks > 10:
            raise ValueError("max_peaks cannot exceed 10 in the interactive workbench")
        if self.count_mode not in {"auto", "exact"}:
            raise ValueError("count_mode must be auto or exact")
        if self.fit_mode not in {"fast", "standard", "thorough", "research"}:
            raise ValueError("fit_mode must be fast, standard, thorough, or research")
        if not self.shapes:
            raise ValueError("at least one peak shape is required")
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("peak shapes must be unique")
        unsupported = set(self.shapes) - SUPPORTED_SHAPES
        if unsupported:
            raise ValueError(f"unsupported peak shape: {sorted(unsupported)[0]}")
        if len(self.peak_hints) >= 10:
            raise ValueError("peak_hints must contain fewer than 10 values")
        if len(set(self.peak_hints)) != len(self.peak_hints):
            raise ValueError("peak_hints must be unique")
        if any(not isfinite(center) for center in self.peak_hints):
            raise ValueError("peak_hints must be finite")
        if self.peak_hints and len(self.peak_hints) > self.max_peaks:
            raise ValueError("peak_hints cannot exceed max_peaks")
        if isinstance(self.random_seed, bool) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if isinstance(self.workers, bool) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        if not isinstance(self.allow_broad_multimax_component, bool):
            raise ValueError("allow_broad_multimax_component must be a boolean")
        if self.measurement_context is not None and not hasattr(
            self.measurement_context, "to_dict"
        ):
            raise ValueError("measurement_context must be a MeasurementContext")
        if self.reference is not None and not hasattr(self.reference, "to_dict"):
            raise ValueError("reference must be a FitReference")
