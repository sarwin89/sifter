"""Cheap real-space and Fourier-space evidence available before fitting."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sifter.config import AutofitConfig
from sifter.fourier import EnvelopeFit
from sifter.search import preprocess_spectrum
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class SpectrumPreview:
    """Immutable pre-fit evidence produced without nonlinear multi-peak optimization."""

    x: NDArray[np.float64]
    intensity: NDArray[np.float64]
    baseline: NDArray[np.float64]
    adjusted: NDArray[np.float64]
    provisional_centers: tuple[float, ...]
    provisional_widths: tuple[float, ...]
    provisional_prominences: tuple[float, ...]
    x_name: str
    x_unit: str | None
    intensity_name: str
    frequency_unit: str
    grid_median_step: float
    grid_relative_step_spread: float
    grid_is_uniform: bool
    fourier_enabled: bool
    fourier_applicable: bool
    fourier_interpolated: bool
    fourier_window: str | None
    frequency: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    log_magnitude: NDArray[np.float64]
    envelope_fits: tuple[EnvelopeFit, ...]
    candidate_spacings: tuple[float, ...]
    fourier_warning_code: str | None


def preview_spectrum(
    spectrum: Spectrum,
    *,
    config: AutofitConfig | None = None,
) -> SpectrumPreview:
    """Analyze baseline, provisional maxima, and optional FFT evidence without peak fitting."""
    settings = AutofitConfig() if config is None else config
    preprocessing = preprocess_spectrum(spectrum, settings)
    fourier = preprocessing.fourier
    empty = _frozen(np.array([], dtype=np.float64))
    magnitude = empty if fourier is None else fourier.magnitude
    log_magnitude = _frozen(
        np.log(np.maximum(magnitude, np.finfo(np.float64).tiny))
    )
    return SpectrumPreview(
        x=_frozen(spectrum.x),
        intensity=_frozen(spectrum.intensity),
        baseline=preprocessing.baseline,
        adjusted=preprocessing.adjusted,
        provisional_centers=tuple(proposal.center for proposal in preprocessing.proposals),
        provisional_widths=tuple(proposal.width for proposal in preprocessing.proposals),
        provisional_prominences=tuple(
            proposal.prominence for proposal in preprocessing.proposals
        ),
        x_name=spectrum.x_name,
        x_unit=spectrum.x_unit,
        intensity_name=spectrum.intensity_name,
        frequency_unit=(
            f"1/{spectrum.x_unit}" if spectrum.x_unit else f"1/{spectrum.x_name}"
        ),
        grid_median_step=spectrum.grid.median_step,
        grid_relative_step_spread=spectrum.grid.relative_step_spread,
        grid_is_uniform=spectrum.grid.is_uniform_for_fft,
        fourier_enabled=settings.fourier,
        fourier_applicable=bool(fourier is not None and fourier.applicable),
        fourier_interpolated=bool(fourier is not None and fourier.interpolated),
        fourier_window=None if fourier is None else fourier.window,
        frequency=empty if fourier is None else fourier.frequency,
        magnitude=magnitude,
        log_magnitude=log_magnitude,
        envelope_fits=() if fourier is None else fourier.envelope_fits,
        candidate_spacings=() if fourier is None else fourier.candidate_spacings,
        fourier_warning_code=None if fourier is None else fourier.warning_code,
    )


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied
