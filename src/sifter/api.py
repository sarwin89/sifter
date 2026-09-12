"""Public orchestration for deterministic spectral model inference."""

from dataclasses import replace
from importlib.metadata import version

import numpy as np

from sifter.config import AutofitConfig, FitConfig, FitMode, PeakCountMode, PeakShape, SearchMode
from sifter.diagnostics import diagnose_fit, residual_diagnostics
from sifter.execution import build_fit_tasks, execute_fit_tasks
from sifter.fitting import (
    CandidateFailure,
    CandidateFit,
    bootstrap_uncertainty,
    covariance_uncertainty,
)
from sifter.models import ModelSpec, ParameterLayout, build_candidates_for_counts
from sifter.progress import ProgressCallback, emit_progress, progress_for_phase
from sifter.reference import build_reference_candidates
from sifter.reporting import DiagnosticWarning, diagnostic_warning
from sifter.result import (
    AnalysisSettings,
    FitResult,
    FittedPeak,
    ModelResult,
    frozen_array,
    frozen_metadata,
)
from sifter.search import (
    ScreeningRecord,
    SearchPolicy,
    SearchPreprocessing,
    adaptive_screening,
    initial_peak_counts,
    preprocess_spectrum,
    refine_finalists,
    retain_diverse_finalists,
    screen_candidates,
    screening_failures,
    search_policy,
)
from sifter.search.dictionary import screen_peak_dictionary
from sifter.search.windowing import build_windowed_candidates
from sifter.selection import (
    CandidateQuality,
    CandidateScore,
    rank_candidates,
    rank_quality_first,
    score_candidate,
)
from sifter.spectrum import Spectrum
from sifter.workbench import inspect_spectrum


class AnalysisError(RuntimeError):
    """Terminal analysis failure retaining every candidate-level failure."""

    def __init__(
        self,
        code: str,
        failures: tuple[CandidateFailure, ...],
        candidate_scores: tuple[CandidateScore, ...] = (),
    ) -> None:
        self.code = code
        self.failures = failures
        self.candidate_scores = candidate_scores
        if candidate_scores:
            rejected = sum(score.status == "inadmissible" for score in candidate_scores)
            failed = sum(score.status == "failed" for score in candidate_scores)
            super().__init__(
                f"{code}: no candidate remained rankable after validation "
                f"({rejected} inadmissible, {failed} failed)"
            )
        else:
            super().__init__(f"{code}: {len(failures)} candidate fits failed")


def fit_spectrum(
    spectrum: Spectrum,
    *,
    config: FitConfig | None = None,
    progress: ProgressCallback | None = None,
) -> FitResult:
    """Run the v0.4.2 workbench fitting path with fast proposal screening."""
    settings = FitConfig() if config is None else config
    inspection = inspect_spectrum(spectrum, max_peaks=settings.max_peaks)
    screened = screen_peak_dictionary(
        spectrum,
        inspection.maxima,
        peak_hints=settings.peak_hints,
        max_peaks=settings.max_peaks,
    )
    screened_centers = tuple(peak.center for peak in screened.peaks)
    peak_hints = (settings.peak_hints or screened_centers)[: settings.max_peaks]
    legacy = AutofitConfig(
        max_peaks=settings.max_peaks,
        search_mode=_search_mode_for_fit_mode(settings.fit_mode),
        peak_count_mode=settings.count_mode,
        shapes=settings.shapes,
        baseline_orders=(0,),
        fourier=settings.fourier,
        interpolate_nonuniform_fft=settings.interpolate_nonuniform_fft,
        uncertainty="covariance",
        random_seed=settings.random_seed,
        workers=settings.workers,
        allow_broad_multimax_component=settings.allow_broad_multimax_component,
        manual_peak_centers=peak_hints,
        measurement_context=settings.measurement_context,
        reference=settings.reference,
    )
    result = autofit(spectrum, config=legacy, progress=progress)
    candidate_models = _quality_ordered_candidate_models(result)
    return replace(
        result,
        schema_version="sifter.fit_result.v3",
        candidate_models=candidate_models,
        settings=replace(
            result.settings,
            fit_mode=settings.fit_mode,
            peak_hints=settings.peak_hints,
        ),
    )


def _quality_ordered_candidate_models(result: FitResult) -> tuple[ModelResult, ...]:
    """Return retained candidate models in v0.4.2 user-facing quality order."""
    if not result.candidate_models:
        return ()
    intensity_span = float(np.ptp(result.intensity))
    normalized_scale = max(intensity_span, np.finfo(float).eps)
    rows: list[CandidateQuality] = []
    for index, model in enumerate(result.candidate_models):
        peak_table = result.to_peak_table(model=model)
        notes = tuple(str(note) for note in peak_table.get("fit_note", ()))
        local_errors = np.asarray(peak_table.get("local_area_error", ()), dtype=np.float64)
        height_errors = np.asarray(peak_table.get("height_error", ()), dtype=np.float64)
        rows.append(
            CandidateQuality(
                candidate_index=index,
                severe_note_count=sum(note == "check fit" for note in notes),
                worst_local_area_error=_safe_nanmax(local_errors),
                median_height_error=_safe_nanmedian(height_errors),
                normalized_rmse=float(model.rmse) / normalized_scale,
                peak_count=model.peak_count,
                bic=model.bic,
            )
        )
    ranked = rank_quality_first(tuple(rows))
    return tuple(result.candidate_models[row.candidate_index] for row in ranked)


def _safe_nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("inf")


def _safe_nanmedian(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("inf")


def _search_mode_for_fit_mode(mode: str) -> SearchMode:
    if mode == "fast":
        return "fast"
    if mode == "standard":
        return "standard"
    if mode == "thorough":
        return "thorough"
    if mode == "research":
        return "thorough"
    raise ValueError(f"unsupported fit mode: {mode}")


def _fit_mode_for_search_mode(mode: SearchMode) -> FitMode:
    if mode in {"fast", "standard", "thorough"}:
        return mode
    return "research"


def autofit(
    spectrum: Spectrum,
    *,
    config: AutofitConfig | None = None,
    max_peaks: int | None = None,
    shapes: tuple[PeakShape, ...] | None = None,
    fourier: bool | None = None,
    random_seed: int | None = None,
    search_mode: SearchMode | None = None,
    peak_count_mode: PeakCountMode | None = None,
    workers: int | None = None,
    progress: ProgressCallback | None = None,
) -> FitResult:
    """Run initialization, candidate fitting, ranking, and uncertainty."""
    settings = _resolved_config(
        config,
        max_peaks=max_peaks,
        shapes=shapes,
        fourier=fourier,
        random_seed=random_seed,
        search_mode=search_mode,
        peak_count_mode=peak_count_mode,
        workers=workers,
    )
    emit_progress(progress, "preprocessing", 0, 1)
    preprocessing = preprocess_spectrum(spectrum, settings)
    policy = search_policy(settings.search_mode)
    peak_counts = _planned_peak_counts(preprocessing, policy, settings)
    emit_progress(
        progress,
        "preprocessing",
        1,
        1,
        message=(
            f"{len(preprocessing.proposals)} real-space proposals; "
            f"candidate peak counts {', '.join(str(count) for count in peak_counts)}; "
            f"{settings.peak_count_mode} peak-count mode"
        ),
    )
    if policy.exhaustive:
        candidates = build_candidates_for_counts(
            spectrum,
            preprocessing.proposals,
            preprocessing.fourier,
            settings,
            peak_counts=peak_counts,
        )
        candidates = _deduplicated_candidates(
            (
                *candidates,
                *_count_filtered_candidates(
                    build_reference_candidates(spectrum, settings),
                    settings,
                ),
            )
        )
        tasks = build_fit_tasks(
            spectrum,
            candidates,
            starts=policy.refinement_starts,
            seed=settings.random_seed,
            max_nfev=policy.refinement_max_nfev,
        )
        emit_progress(
            progress,
            "final_fitting",
            0,
            len(tasks),
            message=f"exhaustive mode with {settings.workers} worker(s)",
        )
        fit_results = list(
            execute_fit_tasks(
                tasks,
                workers=settings.workers,
                on_progress=progress_for_phase(
                    progress,
                    "final_fitting",
                    message=f"exhaustive mode with {settings.workers} worker(s)",
                ),
            )
        )
    else:
        assert policy.finalist_limit is not None
        adaptive = adaptive_screening(
            spectrum,
            preprocessing,
            settings,
            policy,
            initial_counts=peak_counts,
            seed=settings.random_seed,
            progress=progress,
        )
        screening = adaptive.records
        windowed_candidates = build_windowed_candidates(
            spectrum,
            preprocessing,
            settings,
            policy,
            seed=settings.random_seed,
            workers=settings.workers,
        )
        if windowed_candidates:
            emit_progress(
                progress,
                "screening",
                0,
                len(windowed_candidates),
                message="windowed local candidates",
            )
            windowed_screening = screen_candidates(
                spectrum,
                windowed_candidates,
                policy,
                seed=_windowed_seed(settings.random_seed),
                workers=settings.workers,
                on_progress=progress_for_phase(
                    progress,
                    "screening",
                    message="windowed local candidates",
                ),
            )
            screening = (*screening, *windowed_screening)
        reference_candidates = _count_filtered_candidates(
            build_reference_candidates(spectrum, settings),
            settings,
        )
        reference_screening: tuple[ScreeningRecord, ...] = ()
        if reference_candidates:
            emit_progress(
                progress,
                "screening",
                0,
                len(reference_candidates),
                message="reference-seeded candidates",
            )
            reference_screening = screen_candidates(
                spectrum,
                reference_candidates,
                policy,
                seed=_reference_seed(settings.random_seed),
                workers=settings.workers,
                on_progress=progress_for_phase(
                    progress,
                    "screening",
                    message="reference-seeded candidates",
                ),
            )
            screening = (*screening, *reference_screening)
        screening = _deduplicated_screening(screening)
        finalists = retain_diverse_finalists(screening, limit=policy.finalist_limit)
        if reference_screening:
            finalists = _deduplicated_screening(
                (
                    *finalists,
                    *(
                        record
                        for record in reference_screening
                        if record.screening_bic is not None and record.parameters is not None
                    ),
                )
            )
        fit_results = list(screening_failures(screening))
        emit_progress(
            progress,
            "refinement",
            0,
            len(finalists),
            message=f"{len(finalists)} finalist model(s) on the full spectrum",
        )
        fit_results.extend(
            refine_finalists(
                spectrum,
                finalists,
                policy,
                seed=settings.random_seed,
                workers=settings.workers,
                on_progress=progress_for_phase(
                    progress,
                    "refinement",
                    message="finalist full-spectrum fits",
                ),
            )
        )

    failures = tuple(result for result in fit_results if isinstance(result, CandidateFailure))
    successful = {result.spec: result for result in fit_results if isinstance(result, CandidateFit)}
    if not successful:
        raise AnalysisError("NO_VALID_CANDIDATE", failures)

    scores = tuple(
        score_candidate(
            result,
            spectrum,
            allow_broad_multimax_component=settings.allow_broad_multimax_component,
            allow_budget_exhausted=settings.search_mode == "fast",
        )
        for result in fit_results
    )
    ranked = rank_candidates(scores, settings.shapes)
    best_score = next(
        (
            score
            for score in ranked
            if score.status == "valid" and score.aicc is not None and score.bic is not None
        ),
        None,
    )
    if best_score is None:
        code = (
            "NO_RANKABLE_EXACT_CANDIDATE"
            if settings.peak_count_mode == "exact"
            else "NO_RANKABLE_CANDIDATE"
        )
        raise AnalysisError(code, failures, ranked)
    best_fit = successful[best_score.spec]
    diagnostics = residual_diagnostics(best_fit.residuals)
    fit_warnings = list(diagnose_fit(best_fit, spectrum))
    fit_warnings.extend(_analysis_warnings(best_score, preprocessing.fourier))
    uncertainty_total = 1 if settings.uncertainty == "covariance" else settings.bootstrap_samples
    emit_progress(progress, "uncertainty", 0, uncertainty_total)
    if settings.uncertainty == "covariance":
        uncertainty = covariance_uncertainty(best_fit, spectrum)
        emit_progress(progress, "uncertainty", 1, 1)
    else:
        uncertainty = bootstrap_uncertainty(
            best_fit,
            spectrum,
            samples=settings.bootstrap_samples,
            seed=settings.random_seed,
            on_progress=progress_for_phase(progress, "uncertainty"),
        )
    if uncertainty.warning is not None:
        fit_warnings.append(uncertainty.warning)

    assert best_score.aicc is not None and best_score.bic is not None
    assert best_score.rss is not None and best_score.rmse is not None
    model = _model_result(best_fit, best_score, observation_count=spectrum.x.size)
    candidate_models = _top_candidate_models(
        ranked,
        successful,
        observation_count=spectrum.x.size,
        limit=10,
    )
    result = FitResult(
        schema_version="sifter.fit_result.v2",
        settings=AnalysisSettings(
            max_peaks=settings.max_peaks,
            peak_count_mode=settings.peak_count_mode,
            shapes=settings.shapes,
            baseline_orders=settings.baseline_orders,
            fourier=settings.fourier,
            interpolate_nonuniform_fft=settings.interpolate_nonuniform_fft,
            uncertainty=settings.uncertainty,
            bootstrap_samples=settings.bootstrap_samples,
            random_seed=settings.random_seed,
            search_mode=settings.search_mode,
            workers=settings.workers,
            allow_broad_multimax_component=settings.allow_broad_multimax_component,
            manual_peak_centers=settings.manual_peak_centers,
            fit_mode=_fit_mode_for_search_mode(settings.search_mode),
            peak_hints=settings.manual_peak_centers,
            measurement_context=settings.measurement_context,
            reference=settings.reference,
        ),
        source_metadata=frozen_metadata(spectrum.metadata),
        x=frozen_array(spectrum.x),
        intensity=frozen_array(spectrum.intensity),
        sigma=None if spectrum.sigma is None else frozen_array(spectrum.sigma),
        x_name=spectrum.x_name,
        x_unit=spectrum.x_unit,
        intensity_name=spectrum.intensity_name,
        best_model=model,
        candidate_models=candidate_models,
        candidates=ranked,
        fourier=preprocessing.fourier,
        residual_diagnostics=diagnostics,
        uncertainty=uncertainty,
        warnings=tuple(fit_warnings),
        observation_count=spectrum.x.size,
        sifter_version=version("sifter"),
        measurement_context=settings.measurement_context,
        reference=settings.reference,
    )
    emit_progress(progress, "completion", 1, 1)
    return result


def _resolved_config(
    config: AutofitConfig | None,
    *,
    max_peaks: int | None,
    shapes: tuple[PeakShape, ...] | None,
    fourier: bool | None,
    random_seed: int | None,
    search_mode: SearchMode | None,
    peak_count_mode: PeakCountMode | None,
    workers: int | None,
) -> AutofitConfig:
    resolved = AutofitConfig() if config is None else config
    if max_peaks is not None:
        resolved = replace(resolved, max_peaks=max_peaks)
    if shapes is not None:
        resolved = replace(resolved, shapes=shapes)
    if fourier is not None:
        resolved = replace(resolved, fourier=fourier)
    if random_seed is not None:
        resolved = replace(resolved, random_seed=random_seed)
    if search_mode is not None:
        resolved = replace(resolved, search_mode=search_mode)
    if peak_count_mode is not None:
        resolved = replace(resolved, peak_count_mode=peak_count_mode)
    if workers is not None:
        resolved = replace(resolved, workers=workers)
    return resolved


def _planned_peak_counts(
    preprocessing: SearchPreprocessing,
    policy: SearchPolicy,
    settings: AutofitConfig,
) -> tuple[int, ...]:
    if settings.peak_count_mode == "exact":
        return (settings.max_peaks,)
    return initial_peak_counts(
        preprocessing.detection,
        policy,
        max_peaks=settings.max_peaks,
    )


def _count_filtered_candidates(
    candidates: tuple[ModelSpec, ...],
    settings: AutofitConfig,
) -> tuple[ModelSpec, ...]:
    if settings.peak_count_mode != "exact":
        return candidates
    return tuple(
        candidate for candidate in candidates if candidate.peak_count == settings.max_peaks
    )


def _top_candidate_models(
    ranked: tuple[CandidateScore, ...],
    successful: dict[ModelSpec, CandidateFit],
    *,
    observation_count: int,
    limit: int,
) -> tuple[ModelResult, ...]:
    models: list[ModelResult] = []
    for score in ranked:
        if len(models) == limit:
            break
        if score.status != "valid" or score.aicc is None or score.bic is None:
            continue
        fit = successful.get(score.spec)
        if fit is None:
            continue
        models.append(_model_result(fit, score, observation_count=observation_count))
    return tuple(models)


def _model_result(
    fit: CandidateFit,
    score: CandidateScore,
    *,
    observation_count: int,
) -> ModelResult:
    assert score.rss is not None
    assert score.rmse is not None
    assert score.aicc is not None
    assert score.bic is not None
    layout = ParameterLayout(
        fit.spec.shape,
        fit.spec.peak_count,
        fit.spec.baseline_order,
    )
    return ModelResult(
        shape=fit.spec.shape,
        peak_count=fit.spec.peak_count,
        baseline_order=fit.spec.baseline_order,
        parameter_names=layout.names,
        parameters=frozen_array(fit.parameters),
        lower_bounds=fit.spec.lower_bounds,
        upper_bounds=fit.spec.upper_bounds,
        peaks=tuple(
            FittedPeak(
                area=peak.area,
                center=peak.center,
                sigma=peak.sigma,
                gamma=peak.gamma,
            )
            for peak in fit.peaks
        ),
        fitted=frozen_array(fit.fitted),
        baseline=frozen_array(fit.baseline),
        components=frozen_array(fit.components),
        residuals=frozen_array(fit.residuals),
        rss=score.rss,
        rmse=score.rmse,
        aicc=score.aicc,
        bic=score.bic,
        parameter_count=score.parameter_count,
        observation_count=observation_count,
        reduced_chi_squared=score.reduced_chi_squared,
        component_diagnostics=score.component_diagnostics,
    )


def _analysis_warnings(
    score: CandidateScore, fourier_result: object | None
) -> tuple[DiagnosticWarning, ...]:
    warnings = [
        diagnostic_warning(
            code,
            "model-selection criterion indicates an analysis limitation",
            context={"shape": score.shape, "peak_count": score.peak_count},
        )
        for code in score.warnings
    ]
    if fourier_result is not None:
        warning_code = getattr(fourier_result, "warning_code", None)
        if isinstance(warning_code, str):
            warnings.append(
                diagnostic_warning(
                    warning_code,
                    "Fourier diagnostics were limited by the sampling grid or signal range",
                    context={},
                )
            )
    return tuple(warnings)


def _windowed_seed(seed: int) -> int:
    return (seed + 1_048_583) % (2**32)


def _reference_seed(seed: int) -> int:
    return (seed + 2_097_169) % (2**32)


def _deduplicated_candidates(candidates: tuple[ModelSpec, ...]) -> tuple[ModelSpec, ...]:
    return tuple(dict.fromkeys(candidates))


def _deduplicated_screening(
    records: tuple[ScreeningRecord, ...],
) -> tuple[ScreeningRecord, ...]:
    best_by_spec: dict[object, ScreeningRecord] = {}
    order: list[object] = []
    for record in records:
        if record.spec not in best_by_spec:
            order.append(record.spec)
            best_by_spec[record.spec] = record
            continue
        previous = best_by_spec[record.spec]
        if _screening_sort_value(record) < _screening_sort_value(previous):
            best_by_spec[record.spec] = record
    return tuple(best_by_spec[spec] for spec in order)


def _screening_sort_value(record: ScreeningRecord) -> tuple[float, int]:
    failed = 1 if record.screening_bic is None or record.parameters is None else 0
    bic = np.inf if record.screening_bic is None else record.screening_bic
    return bic, failed
