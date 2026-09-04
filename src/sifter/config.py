"""Configuration and shared public types for SIFTER analyses."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
PeakShape: TypeAlias = Literal["gaussian", "lorentzian", "voigt"]
SearchMode: TypeAlias = Literal["fast", "standard", "thorough", "exhaustive"]
UncertaintyMode: TypeAlias = Literal["covariance", "bootstrap"]

SUPPORTED_SHAPES: frozenset[str] = frozenset({"gaussian", "lorentzian", "voigt"})
SUPPORTED_BASELINE_ORDERS: frozenset[int] = frozenset({0, 1, 2})
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
    shapes: tuple[PeakShape, ...] = ("gaussian", "lorentzian", "voigt")
    baseline_orders: tuple[int, ...] = (0, 1, 2)
    fourier: bool = True
    interpolate_nonuniform_fft: bool = False
    uncertainty: UncertaintyMode = "covariance"
    bootstrap_samples: int = 250
    random_seed: int = 42
    workers: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_peaks, bool) or self.max_peaks < 1:
            raise ValueError("max_peaks must be a positive integer")
        if self.search_mode not in {"fast", "standard", "thorough", "exhaustive"}:
            raise ValueError("search_mode must be fast, standard, thorough, or exhaustive")
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
        if not set(self.baseline_orders) <= SUPPORTED_BASELINE_ORDERS:
            raise ValueError("baseline orders must be selected from 0, 1, and 2")
        if self.uncertainty not in {"covariance", "bootstrap"}:
            raise ValueError("uncertainty must be 'covariance' or 'bootstrap'")
        if self.bootstrap_samples not in SUPPORTED_BOOTSTRAP_SAMPLES:
            raise ValueError("bootstrap_samples must be 100, 250, or 1000")
        if isinstance(self.random_seed, bool) or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if isinstance(self.workers, bool) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
