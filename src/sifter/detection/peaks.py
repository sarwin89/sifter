"""Consensus peak proposals from prominence, smoothing, and derivatives."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter

from sifter.detection.derivatives import derivative_peak_indices
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class PeakProposal:
    """A center/width initializer with its supporting evidence labels."""

    center: float
    width: float
    prominence: float
    sources: frozenset[str]


def detect_peak_proposals(spectrum: Spectrum, *, max_peaks: int) -> tuple[PeakProposal, ...]:
    """Combine weak real-space heuristics into deterministic peak proposals."""
    if isinstance(max_peaks, bool) or max_peaks < 1:
        raise ValueError("max_peaks must be a positive integer")

    adjusted = spectrum.intensity - np.quantile(spectrum.intensity, 0.05)
    signal_range = float(np.ptp(adjusted))
    prominence_threshold = max(signal_range * 0.03, np.finfo(float).eps)
    candidates: list[PeakProposal] = []
    windows = _smoothing_windows(adjusted.size)
    for window in windows:
        smoothed = savgol_filter(adjusted, window, 3, mode="interp")
        indices, properties = find_peaks(smoothed, prominence=prominence_threshold)
        if indices.size == 0:
            continue
        sample_widths = peak_widths(smoothed, indices, rel_height=0.5)[0]
        for index, width, prominence in zip(
            indices, sample_widths, properties["prominences"], strict=True
        ):
            candidates.append(
                PeakProposal(
                    center=float(spectrum.x[index]),
                    width=max(float(width * spectrum.grid.median_step), spectrum.grid.median_step),
                    prominence=float(prominence),
                    sources=frozenset({"prominence", f"smooth:{window}"}),
                )
            )

    derivative_window = windows[len(windows) // 2]
    derivative_indices = derivative_peak_indices(adjusted, window_length=derivative_window)
    for index in derivative_indices:
        local_height = float(adjusted[index] - np.median(adjusted))
        if local_height >= prominence_threshold:
            candidates.append(
                PeakProposal(
                    center=float(spectrum.x[index]),
                    width=max(
                        derivative_window * spectrum.grid.median_step / 2.0,
                        spectrum.grid.median_step,
                    ),
                    prominence=local_height,
                    sources=frozenset({"derivative"}),
                )
            )

    merged = _merge_candidates(candidates, spectrum.grid.median_step)
    strongest = sorted(merged, key=lambda item: (-item.prominence, item.center))[:max_peaks]
    return tuple(sorted(strongest, key=lambda item: item.center))


def _smoothing_windows(sample_count: int) -> tuple[int, ...]:
    raw = (max(5, sample_count // 100), max(7, sample_count // 50), max(9, sample_count // 25))
    windows: list[int] = []
    for value in raw:
        odd = value if value % 2 else value + 1
        odd = min(odd, sample_count - 1 if sample_count % 2 == 0 else sample_count - 2)
        odd = max(5, odd)
        if odd not in windows:
            windows.append(odd)
    return tuple(windows)


def _merge_candidates(
    candidates: list[PeakProposal], median_step: float
) -> tuple[PeakProposal, ...]:
    groups: list[list[PeakProposal]] = []
    for candidate in sorted(candidates, key=lambda item: item.center):
        if not groups:
            groups.append([candidate])
            continue
        representative = max(groups[-1], key=lambda item: item.prominence)
        tolerance = max(3.0 * median_step, 0.25 * min(representative.width, candidate.width))
        if abs(candidate.center - representative.center) <= tolerance:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    merged: list[PeakProposal] = []
    for group in groups:
        representative = max(group, key=lambda item: item.prominence)
        sources = frozenset(source for item in group for source in item.sources)
        merged.append(
            PeakProposal(
                center=representative.center,
                width=representative.width,
                prominence=representative.prominence,
                sources=sources,
            )
        )
    return tuple(merged)
