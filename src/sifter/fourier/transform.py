"""FFT applicability, preprocessing, and diagnostic assembly."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sifter.fourier.envelope import EnvelopeFit, fit_envelope_models
from sifter.fourier.spacing import candidate_spacings
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class FourierDiagnostics:
    """Fourier evidence that never contributes observations to fitting."""

    applicable: bool
    interpolated: bool
    frequency: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    envelope_fits: tuple[EnvelopeFit, ...]
    candidate_spacings: tuple[float, ...]
    window: str
    warning_code: str | None


def analyze_fourier(
    spectrum: Spectrum,
    adjusted_intensity: ArrayLike,
    *,
    interpolate_nonuniform: bool = False,
) -> FourierDiagnostics:
    """Compute conservative Fourier diagnostics on baseline-adjusted intensity."""
    adjusted = np.asarray(adjusted_intensity, dtype=np.float64)
    if adjusted.ndim != 1 or adjusted.shape != spectrum.intensity.shape:
        raise ValueError("adjusted_intensity must match the spectrum shape")
    if not np.isfinite(adjusted).all():
        raise ValueError("adjusted_intensity must contain only finite values")

    if not spectrum.grid.is_uniform_for_fft and not interpolate_nonuniform:
        empty = _frozen(np.array([], dtype=np.float64))
        return FourierDiagnostics(
            applicable=False,
            interpolated=False,
            frequency=empty,
            magnitude=empty,
            envelope_fits=(),
            candidate_spacings=(),
            window="hann",
            warning_code="NONUNIFORM_GRID_FFT_DISABLED",
        )

    interpolated = not spectrum.grid.is_uniform_for_fft
    warning_code = "NONUNIFORM_GRID_INTERPOLATED" if interpolated else None
    if interpolated:
        coordinate = np.linspace(spectrum.x[0], spectrum.x[-1], spectrum.x.size)
        signal = np.interp(coordinate, spectrum.x, adjusted)
        coordinate_step = float(coordinate[1] - coordinate[0])
    else:
        signal = adjusted.copy()
        coordinate_step = spectrum.grid.median_step

    centered = signal - np.mean(signal)
    window = np.hanning(signal.size)
    coherent_gain = float(np.mean(window))
    transformed = np.fft.rfft(centered * window)
    full_frequency = np.fft.rfftfreq(signal.size, d=coordinate_step)
    full_magnitude = np.abs(transformed) / (signal.size * coherent_gain)
    frequency = _frozen(np.asarray(full_frequency[1:], dtype=np.float64))
    magnitude = _frozen(np.asarray(full_magnitude[1:], dtype=np.float64))
    envelopes = fit_envelope_models(frequency, magnitude)
    spacings = candidate_spacings(signal, coordinate_step=coordinate_step)
    return FourierDiagnostics(
        applicable=bool(envelopes),
        interpolated=interpolated,
        frequency=frequency,
        magnitude=magnitude,
        envelope_fits=envelopes,
        candidate_spacings=spacings,
        window="hann",
        warning_code=warning_code if envelopes else "INSUFFICIENT_FOURIER_RANGE",
    )


def _frozen(values: NDArray[np.float64]) -> NDArray[np.float64]:
    frozen = np.array(values, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen
