"""Optional reference-fit starts for related spectra."""

from dataclasses import dataclass

import numpy as np

from sifter.baseline import fit_polynomial_baseline
from sifter.config import AutofitConfig, PeakShape
from sifter.models import ModelSpec, PeakStart
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class FitReference:
    """Trusted previous fit used only as additional initialization evidence."""

    shape: PeakShape
    peaks: tuple[PeakStart, ...]
    baseline_order: int = 0

    def __post_init__(self) -> None:
        if self.shape not in {"gaussian", "lorentzian", "voigt"}:
            raise ValueError("reference shape must be gaussian, lorentzian, or voigt")
        if self.baseline_order not in {0, 1, 2}:
            raise ValueError("reference baseline_order must be 0, 1, or 2")
        if not self.peaks:
            raise ValueError("reference requires at least one peak")
        normalized = tuple(_coerce_peak(self.shape, peak) for peak in self.peaks)
        object.__setattr__(self, "peaks", normalized)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-oriented reference summary."""
        return {
            "shape": self.shape,
            "baseline_order": self.baseline_order,
            "peak_count": len(self.peaks),
            "peaks": [
                {
                    "area": peak.area,
                    "center": peak.center,
                    "sigma": peak.sigma,
                    "gamma": peak.gamma,
                }
                for peak in self.peaks
            ],
        }


def build_reference_candidates(
    spectrum: Spectrum,
    config: AutofitConfig,
) -> tuple[ModelSpec, ...]:
    """Build full-spectrum candidates from a reference without replacing ordinary ones."""
    reference = config.reference
    if reference is None or reference.shape not in config.shapes:
        return ()
    starts = tuple(
        _bounded_reference_peak(spectrum, reference.shape, peak) for peak in reference.peaks
    )
    if len(starts) > config.max_peaks:
        starts = starts[: config.max_peaks]
    candidates = [
        _candidate_from_starts(spectrum, reference.shape, baseline_order, starts)
        for baseline_order in config.baseline_orders
    ]
    return tuple(candidates)


def _coerce_peak(
    shape: PeakShape,
    peak: PeakStart | tuple[float, float, float | None, float | None],
) -> PeakStart:
    if isinstance(peak, PeakStart):
        coerced = peak
    else:
        area, center, sigma, gamma = peak
        coerced = PeakStart(area=area, center=center, sigma=sigma, gamma=gamma)
    if not np.isfinite(coerced.area) or coerced.area < 0.0:
        raise ValueError("reference peak area must be finite and nonnegative")
    if not np.isfinite(coerced.center):
        raise ValueError("reference peak center must be finite")
    if shape == "gaussian" and (coerced.sigma is None or coerced.sigma <= 0.0):
        raise ValueError("reference Gaussian peak requires positive sigma")
    if shape == "lorentzian" and (coerced.gamma is None or coerced.gamma <= 0.0):
        raise ValueError("reference Lorentzian peak requires positive gamma")
    if shape == "voigt" and (
        coerced.sigma is None
        or coerced.sigma <= 0.0
        or coerced.gamma is None
        or coerced.gamma <= 0.0
    ):
        raise ValueError("reference Voigt peak requires positive sigma and gamma")
    return coerced


def _bounded_reference_peak(spectrum: Spectrum, shape: PeakShape, peak: PeakStart) -> PeakStart:
    minimum_width = spectrum.grid.median_step / 2.0
    center = float(np.clip(peak.center, spectrum.x[0], spectrum.x[-1]))
    area = max(peak.area, np.finfo(float).eps)
    if shape == "gaussian":
        assert peak.sigma is not None
        return PeakStart(area=area, center=center, sigma=max(peak.sigma, minimum_width))
    if shape == "lorentzian":
        assert peak.gamma is not None
        return PeakStart(area=area, center=center, gamma=max(peak.gamma, minimum_width))
    assert peak.sigma is not None and peak.gamma is not None
    return PeakStart(
        area=area,
        center=center,
        sigma=max(peak.sigma, minimum_width),
        gamma=max(peak.gamma, minimum_width),
    )


def _candidate_from_starts(
    spectrum: Spectrum,
    shape: PeakShape,
    baseline_order: int,
    starts: tuple[PeakStart, ...],
) -> ModelSpec:
    span = float(spectrum.x[-1] - spectrum.x[0])
    coefficient_bound = max(float(np.max(np.abs(spectrum.intensity))) * 100.0, 1.0)
    area_upper = max(
        float(np.trapezoid(np.maximum(spectrum.intensity, 0.0), spectrum.x)) * 10.0,
        span * float(np.ptp(spectrum.intensity)) * 10.0,
        1.0,
    )
    minimum_width = spectrum.grid.median_step / 2.0
    lower = [-coefficient_bound] * (baseline_order + 1)
    upper = [coefficient_bound] * (baseline_order + 1)
    for _ in starts:
        lower.extend((0.0, float(spectrum.x[0])))
        upper.extend((area_upper, float(spectrum.x[-1])))
        if shape in {"gaussian", "voigt"}:
            lower.append(minimum_width)
            upper.append(span)
        if shape in {"lorentzian", "voigt"}:
            lower.append(minimum_width)
            upper.append(span)
    return ModelSpec(
        shape=shape,
        peak_count=len(starts),
        baseline_order=baseline_order,
        baseline_start=fit_polynomial_baseline(spectrum, order=baseline_order).coefficients,
        starts=starts,
        lower_bounds=tuple(lower),
        upper_bounds=tuple(upper),
    )
