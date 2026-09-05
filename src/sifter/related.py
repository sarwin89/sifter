"""Conservative summaries for independently fitted related spectra."""

import numpy as np
import pandas as pd

from sifter.lineshapes import gaussian_fwhm, lorentzian_fwhm, voigt_fwhm
from sifter.result import FitResult, FittedPeak


def summarize_related_spectra(
    results: tuple[FitResult, ...],
    *,
    condition: str,
) -> pd.DataFrame:
    """Return peak properties against one recorded condition without physical modeling."""
    tracks: list[FittedPeak] = []
    rows: list[dict[str, float | int | str | None]] = []
    for spectrum_index, result in enumerate(results):
        condition_value = _condition_value(result, condition)
        assigned: set[int] = set()
        for peak_index, peak in enumerate(result.best_model.peaks):
            fwhm = _peak_fwhm(result.best_model.shape, peak)
            candidates = [
                track_index
                for track_index, reference in enumerate(tracks)
                if abs(peak.center - reference.center)
                <= 0.75 * max(fwhm, _reference_fwhm(reference), np.finfo(float).eps)
            ]
            ambiguity = ""
            if len(candidates) == 1 and candidates[0] not in assigned:
                track_id = candidates[0]
                assigned.add(track_id)
            elif len(candidates) == 0:
                track_id = len(tracks)
                tracks.append(peak)
                assigned.add(track_id)
                ambiguity = "" if spectrum_index == 0 else "appearance"
            else:
                track_id = candidates[0] if candidates else len(tracks)
                ambiguity = "ambiguous"
            rows.append(
                {
                    "spectrum_index": spectrum_index,
                    "condition": condition,
                    "condition_value": condition_value,
                    "track_id": track_id,
                    "peak_index": peak_index,
                    "center": peak.center,
                    "fwhm": fwhm,
                    "area": peak.area,
                    "ambiguity": ambiguity,
                }
            )
    return pd.DataFrame(rows)


def _condition_value(result: FitResult, condition: str) -> float | str | None:
    context = result.measurement_context
    if context is None:
        return None
    normalized = condition.strip().lower()
    if normalized in {"temperature", "temperature_kelvin"} and context.temperature is not None:
        return context.temperature_kelvin
    if normalized in {"laser_power", "laser_power_watts"} and context.laser_power is not None:
        return context.laser_power_watts
    if context.conditions is None:
        return None
    value = context.conditions.get(condition)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return None


def _reference_fwhm(peak: FittedPeak) -> float:
    if peak.sigma is not None and peak.gamma is not None:
        return voigt_fwhm(sigma=peak.sigma, gamma=peak.gamma)
    if peak.sigma is not None:
        return gaussian_fwhm(peak.sigma)
    if peak.gamma is not None:
        return lorentzian_fwhm(peak.gamma)
    return np.finfo(float).eps


def _peak_fwhm(shape: str, peak: FittedPeak) -> float:
    if shape == "gaussian":
        assert peak.sigma is not None
        return gaussian_fwhm(peak.sigma)
    if shape == "lorentzian":
        assert peak.gamma is not None
        return lorentzian_fwhm(peak.gamma)
    assert peak.sigma is not None and peak.gamma is not None
    return voigt_fwhm(sigma=peak.sigma, gamma=peak.gamma)
