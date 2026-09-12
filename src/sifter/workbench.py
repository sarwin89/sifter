"""Fast spectrum inspection helpers for the v0.4.2 workbench."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from sifter.baseline import fit_polynomial_baseline
from sifter.detection import detect_peak_proposals
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class DetectedMaximum:
    """One resolved maximum shown to users before fitting."""

    center: float
    intensity: float
    prominence: float
    width: float
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpectrumInspection:
    """Preview-time evidence used by both UI and fit initialization."""

    maxima: tuple[DetectedMaximum, ...]
    baseline: np.ndarray
    adjusted: np.ndarray
    elapsed_seconds: float

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {
                "center": maximum.center,
                "intensity": maximum.intensity,
                "prominence": maximum.prominence,
                "width": maximum.width,
                "sources": ", ".join(maximum.sources),
            }
            for maximum in self.maxima
        ]


def inspect_spectrum(spectrum: Spectrum, *, max_peaks: int = 10) -> SpectrumInspection:
    """Detect user-visible maxima in a constant-baseline-adjusted spectrum."""
    if isinstance(max_peaks, bool) or max_peaks < 1 or max_peaks > 20:
        raise ValueError("max_peaks must be an integer from 1 through 20")
    started = perf_counter()
    baseline = fit_polynomial_baseline(spectrum, order=0).evaluate(spectrum.x)
    adjusted = spectrum.intensity - baseline
    adjusted_spectrum = Spectrum(
        spectrum.x,
        adjusted,
        sigma=spectrum.sigma,
        x_name=spectrum.x_name,
        x_unit=spectrum.x_unit,
        intensity_name=spectrum.intensity_name,
        metadata=spectrum.metadata,
    )
    proposals = detect_peak_proposals(adjusted_spectrum, max_peaks=max_peaks)
    maxima = tuple(
        DetectedMaximum(
            center=proposal.center,
            intensity=float(np.interp(proposal.center, spectrum.x, spectrum.intensity)),
            prominence=proposal.prominence,
            width=proposal.width,
            sources=tuple(sorted(proposal.sources)),
        )
        for proposal in proposals
    )
    frozen_baseline = np.array(baseline, dtype=np.float64, copy=True)
    frozen_adjusted = np.array(adjusted, dtype=np.float64, copy=True)
    frozen_baseline.setflags(write=False)
    frozen_adjusted.setflags(write=False)
    return SpectrumInspection(
        maxima=maxima,
        baseline=frozen_baseline,
        adjusted=frozen_adjusted,
        elapsed_seconds=perf_counter() - started,
    )


def merge_peak_hints(
    maxima: tuple[DetectedMaximum, ...],
    peak_hints: tuple[float, ...],
    *,
    max_peaks: int,
) -> tuple[float, ...]:
    """Merge detected centers and user hints into a deterministic center list."""
    if isinstance(max_peaks, bool) or max_peaks < 1 or max_peaks > 10:
        raise ValueError("max_peaks must be an integer from 1 through 10")
    centers: list[float] = []
    tolerance = _merge_tolerance(maxima)
    for center in [*(maximum.center for maximum in maxima), *peak_hints]:
        value = float(center)
        if not np.isfinite(value):
            raise ValueError("peak centers must be finite")
        if any(abs(value - existing) <= tolerance for existing in centers):
            continue
        centers.append(value)
        if len(centers) == max_peaks:
            break
    return tuple(sorted(centers))


def _merge_tolerance(maxima: tuple[DetectedMaximum, ...]) -> float:
    if not maxima:
        return 0.0
    widths = [maximum.width for maximum in maxima if np.isfinite(maximum.width)]
    if not widths:
        return 0.0
    return max(float(np.median(widths)) * 0.1, np.finfo(float).eps)
