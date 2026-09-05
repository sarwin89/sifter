"""Single-pass preprocessing for staged candidate search."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sifter.baseline import asls_baseline
from sifter.config import AutofitConfig
from sifter.detection import PeakProposal, detect_peak_proposals
from sifter.fourier import FourierDiagnostics, analyze_fourier
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class PeakDetectionSummary:
    """Compact detector evidence used to plan candidate peak counts."""

    detected_count: int
    centers: tuple[float, ...]
    median_width: float | None
    strongest_prominence: float | None


@dataclass(frozen=True, slots=True)
class SearchPreprocessing:
    """Reusable baseline, detection, and Fourier evidence for one spectrum."""

    baseline: NDArray[np.float64]
    adjusted: NDArray[np.float64]
    proposal_spectrum: Spectrum
    proposals: tuple[PeakProposal, ...]
    detection: PeakDetectionSummary
    fourier: FourierDiagnostics | None


def preprocess_spectrum(spectrum: Spectrum, config: AutofitConfig) -> SearchPreprocessing:
    """Compute reusable search evidence once for one analysis."""
    baseline = _frozen(asls_baseline(spectrum.intensity))
    adjusted = _frozen(spectrum.intensity - baseline)
    proposal_spectrum = Spectrum(
        spectrum.x,
        adjusted,
        sigma=spectrum.sigma,
        x_name=spectrum.x_name,
        x_unit=spectrum.x_unit,
        intensity_name=spectrum.intensity_name,
        metadata=spectrum.metadata,
    )
    proposals = detect_peak_proposals(proposal_spectrum, max_peaks=config.max_peaks)
    detection = PeakDetectionSummary(
        detected_count=len(proposals),
        centers=tuple(proposal.center for proposal in proposals),
        median_width=(
            None if not proposals else float(np.median([proposal.width for proposal in proposals]))
        ),
        strongest_prominence=(
            None if not proposals else max(proposal.prominence for proposal in proposals)
        ),
    )
    fourier = (
        analyze_fourier(
            spectrum,
            adjusted,
            interpolate_nonuniform=config.interpolate_nonuniform_fft,
        )
        if config.fourier
        else None
    )
    return SearchPreprocessing(
        baseline=baseline,
        adjusted=adjusted,
        proposal_spectrum=proposal_spectrum,
        proposals=proposals,
        detection=detection,
        fourier=fourier,
    )


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied
