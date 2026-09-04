"""Windowed progressive initialization for full-spectrum candidate generation."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from sifter.baseline import fit_polynomial_baseline
from sifter.config import AutofitConfig, PeakShape
from sifter.detection import PeakProposal, detect_peak_proposals
from sifter.execution import CandidateFitTask, build_fit_tasks, execute_fit_tasks
from sifter.fitting import CandidateFailure, CandidateFit
from sifter.models import ModelSpec, ParameterLayout, PeakStart, build_candidates_for_counts
from sifter.models.specification import evaluate_model
from sifter.search.policy import SearchPolicy
from sifter.search.preprocessing import SearchPreprocessing
from sifter.selection import unweighted_information_criteria
from sifter.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class ResolvedMaximum:
    """A detector-backed maximum used to own one local basin."""

    center: float
    width: float
    prominence: float


@dataclass(frozen=True, slots=True)
class PeakWindow:
    """One local fitting region with a strict ownership core and broader halo."""

    core_start: float
    core_stop: float
    fit_start: float
    fit_stop: float
    fit_start_index: int
    fit_stop_index: int
    maxima: tuple[ResolvedMaximum, ...]


def plan_peak_windows(
    spectrum: Spectrum,
    preprocessing: SearchPreprocessing,
    config: AutofitConfig,
) -> tuple[PeakWindow, ...]:
    """Plan local windows from resolved maxima without defining final fits."""
    maxima = tuple(
        ResolvedMaximum(
            center=proposal.center,
            width=max(proposal.width, spectrum.grid.median_step),
            prominence=proposal.prominence,
        )
        for proposal in sorted(preprocessing.proposals, key=lambda item: item.center)
    )
    if not maxima:
        return ()

    raw = [
        _single_maximum_window(spectrum, maxima, index, config)
        for index in range(len(maxima))
    ]
    return tuple(_merge_overlapping(spectrum, raw))


def build_windowed_candidates(
    spectrum: Spectrum,
    preprocessing: SearchPreprocessing,
    config: AutofitConfig,
    policy: SearchPolicy,
    *,
    seed: int,
    workers: int = 1,
) -> tuple[ModelSpec, ...]:
    """Use local window fits only to initialize full-spectrum candidates."""
    if policy.exhaustive:
        return ()
    windows = plan_peak_windows(spectrum, preprocessing, config)
    if not windows:
        return ()

    local_results = _fit_local_windows(
        spectrum,
        preprocessing,
        config,
        policy,
        windows,
        seed=seed,
        workers=workers,
    )
    starts_by_shape = _owned_peak_starts_by_shape(windows, local_results, config.shapes)
    candidates: list[ModelSpec] = []
    for shape in config.shapes:
        starts = starts_by_shape.get(shape, ())
        if not starts:
            continue
        candidates.extend(_global_specs(spectrum, config, shape, starts))
        candidates.extend(
            _global_specs(
                spectrum,
                config,
                shape,
                _with_residual_second_wave(spectrum, config, shape, starts),
            )
        )
    return _deduplicated_candidates(candidates)


def _single_maximum_window(
    spectrum: Spectrum,
    maxima: tuple[ResolvedMaximum, ...],
    index: int,
    config: AutofitConfig,
) -> PeakWindow:
    maximum = maxima[index]
    left_limit = maximum.center - maximum.width
    right_limit = maximum.center + maximum.width
    if index > 0:
        left_limit = (maxima[index - 1].center + maximum.center) / 2.0
    if index + 1 < len(maxima):
        right_limit = (maximum.center + maxima[index + 1].center) / 2.0

    core_start = max(float(spectrum.x[0]), left_limit)
    core_stop = min(float(spectrum.x[-1]), right_limit)
    halo = _halo_width(config, maximum.width, spectrum.grid.median_step)
    return _window_from_bounds(
        spectrum,
        core_start=core_start,
        core_stop=core_stop,
        fit_start=core_start - halo,
        fit_stop=core_stop + halo,
        maxima=(maximum,),
    )


def _halo_width(config: AutofitConfig, local_fwhm: float, median_step: float) -> float:
    multiplier = 3.0 if set(config.shapes) == {"gaussian"} else 6.0
    minimum = 0.0 if multiplier == 3.0 else 10.0 * median_step
    return max(multiplier * local_fwhm, minimum)


def _merge_overlapping(spectrum: Spectrum, windows: Sequence[PeakWindow]) -> list[PeakWindow]:
    merged: list[PeakWindow] = []
    for window in windows:
        if not merged:
            merged.append(window)
            continue
        previous = merged[-1]
        overlap = min(previous.fit_stop, window.fit_stop) - max(
            previous.fit_start,
            window.fit_start,
        )
        smaller = min(previous.fit_stop - previous.fit_start, window.fit_stop - window.fit_start)
        if smaller > 0.0 and overlap / smaller >= 0.75:
            merged[-1] = _window_from_bounds(
                spectrum,
                core_start=min(previous.core_start, window.core_start),
                core_stop=max(previous.core_stop, window.core_stop),
                fit_start=min(previous.fit_start, window.fit_start),
                fit_stop=max(previous.fit_stop, window.fit_stop),
                maxima=previous.maxima + window.maxima,
            )
        else:
            merged.append(window)
    return merged


def _window_from_bounds(
    spectrum: Spectrum,
    *,
    core_start: float,
    core_stop: float,
    fit_start: float,
    fit_stop: float,
    maxima: tuple[ResolvedMaximum, ...],
) -> PeakWindow:
    x = spectrum.x
    clipped_fit_start = max(float(x[0]), fit_start)
    clipped_fit_stop = min(float(x[-1]), fit_stop)
    start_index = int(np.searchsorted(x, clipped_fit_start, side="left"))
    stop_index = int(np.searchsorted(x, clipped_fit_stop, side="right") - 1)
    return PeakWindow(
        core_start=float(core_start),
        core_stop=float(core_stop),
        fit_start=float(x[max(0, start_index)]),
        fit_stop=float(x[min(x.size - 1, stop_index)]),
        fit_start_index=max(0, start_index),
        fit_stop_index=min(x.size - 1, stop_index),
        maxima=maxima,
    )


def _fit_local_windows(
    spectrum: Spectrum,
    preprocessing: SearchPreprocessing,
    config: AutofitConfig,
    policy: SearchPolicy,
    windows: tuple[PeakWindow, ...],
    *,
    seed: int,
    workers: int,
) -> dict[int, tuple[CandidateFit | CandidateFailure, ...]]:
    assert policy.screening_starts is not None
    assert policy.screening_max_nfev is not None
    tasks: list[CandidateFitTask] = []
    task_windows: list[int] = []
    sequence = np.random.SeedSequence([seed, len(windows)]).spawn(len(windows))
    for window_index, window in enumerate(windows):
        local = _local_spectrum(spectrum, window)
        proposals = _local_proposals(preprocessing.proposals, window)
        if not proposals:
            continue
        local_config = replace(
            config,
            max_peaks=min(config.max_peaks, len(window.maxima)),
            fourier=False,
        )
        candidates = build_candidates_for_counts(
            local,
            proposals,
            None,
            local_config,
            peak_counts=tuple(range(1, min(config.max_peaks, len(window.maxima)) + 1)),
        )
        local_tasks = build_fit_tasks(
            local,
            candidates,
            starts=policy.screening_starts,
            seed=int(sequence[window_index].generate_state(1, dtype=np.uint32)[0]),
            max_nfev=policy.screening_max_nfev,
            allow_budget_exhausted=True,
        )
        tasks.extend(local_tasks)
        task_windows.extend([window_index] * len(local_tasks))
    if not tasks:
        return {}
    results = execute_fit_tasks(tasks, workers=workers)
    grouped: dict[int, list[CandidateFit | CandidateFailure]] = {}
    for window_index, result in zip(task_windows, results, strict=True):
        grouped.setdefault(window_index, []).append(result)
    return {index: tuple(items) for index, items in grouped.items()}


def _local_spectrum(spectrum: Spectrum, window: PeakWindow) -> Spectrum:
    slc = slice(window.fit_start_index, window.fit_stop_index + 1)
    return Spectrum(
        spectrum.x[slc],
        spectrum.intensity[slc],
        sigma=None if spectrum.sigma is None else spectrum.sigma[slc],
        x_name=spectrum.x_name,
        x_unit=spectrum.x_unit,
        intensity_name=spectrum.intensity_name,
        metadata=dict(spectrum.metadata),
    )


def _local_proposals(
    proposals: tuple[PeakProposal, ...],
    window: PeakWindow,
) -> tuple[PeakProposal, ...]:
    return tuple(
        proposal
        for proposal in proposals
        if window.core_start <= proposal.center <= window.core_stop
    )


def _owned_peak_starts_by_shape(
    windows: tuple[PeakWindow, ...],
    local_results: dict[int, tuple[CandidateFit | CandidateFailure, ...]],
    shapes: tuple[PeakShape, ...],
) -> dict[PeakShape, tuple[PeakStart, ...]]:
    owned: dict[PeakShape, list[PeakStart]] = {shape: [] for shape in shapes}
    for window_index, window in enumerate(windows):
        results = local_results.get(window_index, ())
        for shape in shapes:
            best = _best_local_fit(window, results, shape)
            if best is None:
                continue
            owned[shape].extend(
                peak
                for peak in best.peaks
                if window.core_start <= peak.center <= window.core_stop
            )
    return {
        shape: tuple(sorted(starts, key=lambda peak: peak.center))
        for shape, starts in owned.items()
        if starts
    }


def _best_local_fit(
    window: PeakWindow,
    results: tuple[CandidateFit | CandidateFailure, ...],
    shape: PeakShape,
) -> CandidateFit | None:
    eligible = [
        result
        for result in results
        if isinstance(result, CandidateFit)
        and result.spec.shape == shape
        and any(window.core_start <= peak.center <= window.core_stop for peak in result.peaks)
    ]
    if not eligible:
        return None
    return min(eligible, key=_local_bic)


def _local_bic(result: CandidateFit) -> float:
    if result.spec.peak_count >= result.residuals.size:
        return np.inf
    return unweighted_information_criteria(
        n=result.residuals.size,
        p=len(result.spec.lower_bounds),
        rss=float(np.dot(result.residuals, result.residuals)),
    ).bic


def _with_residual_second_wave(
    spectrum: Spectrum,
    config: AutofitConfig,
    shape: PeakShape,
    starts: tuple[PeakStart, ...],
) -> tuple[PeakStart, ...]:
    if len(starts) >= config.max_peaks:
        return starts
    base = _global_specs(spectrum, config, shape, starts)
    if not base:
        return starts
    spec = base[0]
    layout = ParameterLayout(shape, len(starts), spec.baseline_order)
    evaluated = evaluate_model(spectrum.x, layout.initial_vector(spec), spec)
    residual = spectrum.intensity - evaluated.fitted
    residual_spectrum = Spectrum(
        spectrum.x,
        residual,
        sigma=spectrum.sigma,
        x_name=spectrum.x_name,
        x_unit=spectrum.x_unit,
        intensity_name=spectrum.intensity_name,
        metadata=dict(spectrum.metadata),
    )
    proposals = detect_peak_proposals(residual_spectrum, max_peaks=config.max_peaks - len(starts))
    additions: list[PeakStart] = []
    for proposal in proposals:
        if any(
            abs(proposal.center - peak.center)
            <= max(proposal.width, spectrum.grid.median_step)
            for peak in starts
        ):
            continue
        additions.append(_start_from_proposal(shape, proposal))
        if len(starts) + len(additions) >= config.max_peaks:
            break
    return tuple(sorted((*starts, *additions), key=lambda peak: peak.center))


def _global_specs(
    spectrum: Spectrum,
    config: AutofitConfig,
    shape: PeakShape,
    starts: tuple[PeakStart, ...],
) -> tuple[ModelSpec, ...]:
    if not starts or len(starts) > config.max_peaks:
        return ()
    return tuple(
        _global_spec(spectrum, shape, baseline_order, starts)
        for baseline_order in config.baseline_orders
    )


def _global_spec(
    spectrum: Spectrum,
    shape: PeakShape,
    baseline_order: int,
    starts: tuple[PeakStart, ...],
) -> ModelSpec:
    span = float(spectrum.x[-1] - spectrum.x[0])
    minimum_width = spectrum.grid.median_step / 2.0
    positive_signal = np.maximum(spectrum.intensity - np.quantile(spectrum.intensity, 0.05), 0.0)
    total_area = float(np.trapezoid(positive_signal, spectrum.x))
    area_upper = max(total_area * 10.0, span * float(np.ptp(spectrum.intensity)) * 10.0, 1.0)
    bounded_starts = tuple(_bounded_start(shape, peak, minimum_width) for peak in starts)
    coefficient_bound = max(float(np.max(np.abs(spectrum.intensity))) * 100.0, 1.0)
    lower = [-coefficient_bound] * (baseline_order + 1)
    upper = [coefficient_bound] * (baseline_order + 1)
    for _ in bounded_starts:
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
        peak_count=len(bounded_starts),
        baseline_order=baseline_order,
        baseline_start=fit_polynomial_baseline(spectrum, order=baseline_order).coefficients,
        starts=bounded_starts,
        lower_bounds=tuple(lower),
        upper_bounds=tuple(upper),
    )


def _bounded_start(shape: PeakShape, peak: PeakStart, minimum_width: float) -> PeakStart:
    if shape == "gaussian":
        return PeakStart(
            area=max(peak.area, np.finfo(float).eps),
            center=peak.center,
            sigma=max(peak.sigma or minimum_width, minimum_width),
        )
    if shape == "lorentzian":
        return PeakStart(
            area=max(peak.area, np.finfo(float).eps),
            center=peak.center,
            gamma=max(peak.gamma or minimum_width, minimum_width),
        )
    return PeakStart(
        area=max(peak.area, np.finfo(float).eps),
        center=peak.center,
        sigma=max(peak.sigma or minimum_width, minimum_width),
        gamma=max(peak.gamma or minimum_width, minimum_width),
    )


def _start_from_proposal(shape: PeakShape, proposal: PeakProposal) -> PeakStart:
    area = max(proposal.prominence * proposal.width, np.finfo(float).eps)
    if shape == "gaussian":
        return PeakStart(area=area, center=proposal.center, sigma=proposal.width / 2.354820045)
    if shape == "lorentzian":
        return PeakStart(area=area, center=proposal.center, gamma=proposal.width / 2.0)
    return PeakStart(
        area=area,
        center=proposal.center,
        sigma=proposal.width / 2.354820045,
        gamma=proposal.width / 4.0,
    )


def _deduplicated_candidates(candidates: Sequence[ModelSpec]) -> tuple[ModelSpec, ...]:
    seen: set[tuple[PeakShape, int, int, tuple[float, ...]]] = set()
    unique: list[ModelSpec] = []
    for candidate in candidates:
        key = (
            candidate.shape,
            candidate.peak_count,
            candidate.baseline_order,
            tuple(round(peak.center, 9) for peak in candidate.starts),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return tuple(
        sorted(
            unique,
            key=lambda spec: (spec.peak_count, spec.baseline_order, spec.shape),
        )
    )
