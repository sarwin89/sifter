"""Synthetic spectrum generation for tests, examples, and benchmarks."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sifter.config import PeakShape
from sifter.lineshapes import gaussian, lorentzian, voigt
from sifter.spectrum import Spectrum

NoiseKind = Literal["none", "gaussian"]


@dataclass(frozen=True, slots=True)
class SyntheticPeak:
    """Ground-truth peak parameters using integrated-area conventions."""

    shape: PeakShape
    area: float
    center: float
    sigma: float | None = None
    gamma: float | None = None

    def __post_init__(self) -> None:
        if self.shape not in {"gaussian", "lorentzian", "voigt"}:
            raise ValueError(f"unsupported synthetic peak shape: {self.shape}")
        if not np.isfinite(self.area) or self.area < 0:
            raise ValueError("synthetic peak area must be finite and nonnegative")
        if not np.isfinite(self.center):
            raise ValueError("synthetic peak center must be finite")
        if self.shape == "gaussian":
            _require_positive(self.sigma, "sigma")
            if self.gamma is not None:
                raise ValueError("Gaussian peaks do not accept gamma")
        elif self.shape == "lorentzian":
            _require_positive(self.gamma, "gamma")
            if self.sigma is not None:
                raise ValueError("Lorentzian peaks do not accept sigma")
        else:
            _require_positive(self.sigma, "sigma")
            _require_positive(self.gamma, "gamma")


@dataclass(frozen=True, slots=True)
class SyntheticTruth:
    """Exact noiseless signals and parameters used for one generated spectrum."""

    peaks: tuple[SyntheticPeak, ...]
    baseline_coefficients: tuple[float, ...]
    peak_signal: NDArray[np.float64]
    clean_signal: NDArray[np.float64]
    noise_standard_deviation: float
    seed: int


def make_spectrum(
    *,
    x: ArrayLike,
    peaks: tuple[SyntheticPeak, ...],
    baseline: tuple[float, ...] = (0.0,),
    noise: NoiseKind = "none",
    snr: float | None = None,
    seed: int = 42,
) -> tuple[Spectrum, SyntheticTruth]:
    """Create a synthetic spectrum and retain its exact generating truth."""
    x_values = np.asarray(x, dtype=np.float64)
    if x_values.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if not peaks:
        raise ValueError("at least one synthetic peak is required")
    if not 1 <= len(baseline) <= 3 or not np.isfinite(baseline).all():
        raise ValueError("baseline must contain one to three finite coefficients")
    if noise not in {"none", "gaussian"}:
        raise ValueError("noise must be 'none' or 'gaussian'")
    if noise == "gaussian" and (snr is None or not np.isfinite(snr) or snr <= 0):
        raise ValueError("gaussian noise requires a finite positive snr")

    peak_signal = np.zeros_like(x_values, dtype=np.float64)
    for peak in peaks:
        peak_signal += _evaluate_peak(x_values, peak)

    x_span = float(np.max(x_values) - np.min(x_values))
    if not np.isfinite(x_span) or x_span <= 0:
        raise ValueError("x must span a finite positive interval")
    x_offset = float((np.max(x_values) + np.min(x_values)) / 2.0)
    x_scale = x_span / 2.0
    scaled_x = (x_values - x_offset) / x_scale
    baseline_signal = np.polynomial.polynomial.polyval(scaled_x, baseline)
    clean_signal = peak_signal + baseline_signal

    noise_standard_deviation = 0.0
    measured = clean_signal.copy()
    if noise == "gaussian":
        assert snr is not None
        noise_standard_deviation = float(np.sqrt(np.mean(peak_signal**2)) / snr)
        generator = np.random.default_rng(seed)
        measured += generator.normal(0.0, noise_standard_deviation, size=x_values.size)

    frozen_peak_signal = np.array(peak_signal, dtype=np.float64, copy=True)
    frozen_clean_signal = np.array(clean_signal, dtype=np.float64, copy=True)
    frozen_peak_signal.setflags(write=False)
    frozen_clean_signal.setflags(write=False)
    truth = SyntheticTruth(
        peaks=tuple(peaks),
        baseline_coefficients=tuple(float(value) for value in baseline),
        peak_signal=frozen_peak_signal,
        clean_signal=frozen_clean_signal,
        noise_standard_deviation=noise_standard_deviation,
        seed=seed,
    )
    spectrum = Spectrum(x_values, measured, metadata={"synthetic": True, "seed": seed})
    return spectrum, truth


def _evaluate_peak(x: NDArray[np.float64], peak: SyntheticPeak) -> NDArray[np.float64]:
    if peak.shape == "gaussian":
        assert peak.sigma is not None
        return gaussian(x, area=peak.area, center=peak.center, sigma=peak.sigma)
    if peak.shape == "lorentzian":
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


def _require_positive(value: float | None, name: str) -> None:
    if value is None or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")

