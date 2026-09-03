"""Deterministic generation of bounded candidate model specifications."""

import numpy as np

from sifter.baseline import fit_polynomial_baseline
from sifter.config import AutofitConfig, PeakShape
from sifter.detection import PeakProposal
from sifter.fourier import FourierDiagnostics
from sifter.models.specification import ModelSpec, ParameterLayout, PeakStart
from sifter.spectrum import Spectrum


def build_candidates(
    spectrum: Spectrum,
    proposals: tuple[PeakProposal, ...],
    fourier: FourierDiagnostics | None,
    config: AutofitConfig,
) -> tuple[ModelSpec, ...]:
    """Build every eligible simpler count, family, and baseline candidate."""
    initial_count = max(1, len(proposals))
    largest_count = min(config.max_peaks, initial_count + 2)
    candidates: list[ModelSpec] = []
    for peak_count in range(1, largest_count + 1):
        centers, proposal_widths = _candidate_centers(spectrum, proposals, fourier, peak_count)
        for baseline_order in sorted(config.baseline_orders):
            baseline = fit_polynomial_baseline(spectrum, order=baseline_order)
            for shape in config.shapes:
                candidates.append(
                    _build_spec(
                        spectrum,
                        shape=shape,
                        peak_count=peak_count,
                        baseline_order=baseline_order,
                        baseline_start=baseline.coefficients,
                        centers=centers,
                        proposal_widths=proposal_widths,
                    )
                )
    family_order = {shape: index for index, shape in enumerate(config.shapes)}
    return tuple(
        sorted(
            candidates,
            key=lambda spec: (
                spec.peak_count,
                spec.baseline_order,
                family_order[spec.shape],
            ),
        )
    )


def _candidate_centers(
    spectrum: Spectrum,
    proposals: tuple[PeakProposal, ...],
    fourier: FourierDiagnostics | None,
    peak_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    strongest = sorted(proposals, key=lambda item: (-item.prominence, item.center))[:peak_count]
    selected: list[tuple[float, float]] = [(item.center, item.width) for item in strongest]
    span = float(spectrum.x[-1] - spectrum.x[0])
    default_width = span / max(6.0 * peak_count, 12.0)
    spacing = None
    if fourier is not None and fourier.candidate_spacings:
        spacing = fourier.candidate_spacings[0]
    fallback_centers = np.linspace(
        spectrum.x[0] + span / (peak_count + 1),
        spectrum.x[-1] - span / (peak_count + 1),
        peak_count,
    )
    if spacing is not None and selected:
        anchor = selected[0][0]
        offsets = range(-(peak_count // 2), peak_count - peak_count // 2)
        fallback_centers = np.asarray([anchor + offset * spacing for offset in offsets])
    minimum_separation = spectrum.grid.median_step
    for center in fallback_centers:
        clipped = float(np.clip(center, spectrum.x[0], spectrum.x[-1]))
        if all(abs(clipped - existing) > minimum_separation for existing, _ in selected):
            selected.append((clipped, default_width))
        if len(selected) == peak_count:
            break
    if len(selected) < peak_count:
        dense = np.linspace(spectrum.x[0], spectrum.x[-1], 4 * peak_count + 2)[1:-1]
        for center in dense:
            if all(abs(center - existing) > minimum_separation for existing, _ in selected):
                selected.append((float(center), default_width))
            if len(selected) == peak_count:
                break
    ordered = sorted(selected[:peak_count], key=lambda pair: pair[0])
    return tuple(center for center, _ in ordered), tuple(width for _, width in ordered)


def _build_spec(
    spectrum: Spectrum,
    *,
    shape: PeakShape,
    peak_count: int,
    baseline_order: int,
    baseline_start: tuple[float, ...],
    centers: tuple[float, ...],
    proposal_widths: tuple[float, ...],
) -> ModelSpec:
    span = float(spectrum.x[-1] - spectrum.x[0])
    minimum_width = spectrum.grid.median_step / 2.0
    positive_signal = np.maximum(spectrum.intensity - np.quantile(spectrum.intensity, 0.05), 0.0)
    total_area = float(np.trapezoid(positive_signal, spectrum.x))
    area_start = max(total_area / peak_count, np.finfo(float).eps)
    area_upper = max(total_area * 10.0, span * float(np.ptp(spectrum.intensity)) * 10.0, 1.0)
    starts: list[PeakStart] = []
    for center, width in zip(centers, proposal_widths, strict=True):
        if shape == "gaussian":
            starts.append(
                PeakStart(area_start, center, sigma=max(width / 2.354820045, minimum_width))
            )
        elif shape == "lorentzian":
            starts.append(PeakStart(area_start, center, gamma=max(width / 2.0, minimum_width)))
        else:
            starts.append(
                PeakStart(
                    area_start,
                    center,
                    sigma=max(width / 2.354820045, minimum_width),
                    gamma=max(width / 4.0, minimum_width),
                )
            )

    coefficient_bound = max(float(np.max(np.abs(spectrum.intensity))) * 100.0, 1.0)
    lower = [-coefficient_bound] * (baseline_order + 1)
    upper = [coefficient_bound] * (baseline_order + 1)
    for _ in range(peak_count):
        lower.extend((0.0, float(spectrum.x[0])))
        upper.extend((area_upper, float(spectrum.x[-1])))
        if shape in {"gaussian", "voigt"}:
            lower.append(minimum_width)
            upper.append(span)
        if shape in {"lorentzian", "voigt"}:
            lower.append(minimum_width)
            upper.append(span)
    layout = ParameterLayout(shape, peak_count, baseline_order)
    if len(lower) != layout.parameter_count:
        raise RuntimeError("candidate bounds do not match parameter layout")
    return ModelSpec(
        shape=shape,
        peak_count=peak_count,
        baseline_order=baseline_order,
        baseline_start=tuple(float(value) for value in baseline_start),
        starts=tuple(starts),
        lower_bounds=tuple(lower),
        upper_bounds=tuple(upper),
    )
