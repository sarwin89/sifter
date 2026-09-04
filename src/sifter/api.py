"""Public orchestration for deterministic spectral model inference."""

from dataclasses import replace
from importlib.metadata import version

import numpy as np

from sifter.config import AutofitConfig, PeakShape, SearchMode
from sifter.diagnostics import diagnose_fit, residual_diagnostics
from sifter.execution import build_fit_tasks, execute_fit_tasks
from sifter.fitting import (
    CandidateFailure,
    CandidateFit,
    bootstrap_uncertainty,
    covariance_uncertainty,
)
from sifter.models import ParameterLayout, build_candidates_for_counts
from sifter.progress import ProgressCallback, emit_progress, progress_for_phase
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
    adaptive_screening,
    initial_peak_counts,
    preprocess_spectrum,
    refine_finalists,
    retain_diverse_finalists,
    screen_candidates,
    screening_failures,
    search_policy,
)
from sifter.search.windowing import build_windowed_candidates
from sifter.selection import CandidateScore, rank_candidates, score_candidate
from sifter.spectrum import Spectrum


class AnalysisError(RuntimeError):
    """Terminal analysis failure retaining every candidate-level failure."""

    def __init__(self, code: str, failures: tuple[CandidateFailure, ...]) -> None:
        self.code = code
        self.failures = failures
        super().__init__(f"{code}: {len(failures)} candidate fits failed")


def autofit(
    spectrum: Spectrum,
    *,
    config: AutofitConfig | None = None,
    max_peaks: int | None = None,
    shapes: tuple[PeakShape, ...] | None = None,
    fourier: bool | None = None,
    random_seed: int | None = None,
    search_mode: SearchMode | None = None,
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
        workers=workers,
    )
    emit_progress(progress, "preprocessing", 0, 1)
    preprocessing = preprocess_spectrum(spectrum, settings)
    emit_progress(progress, "preprocessing", 1, 1)
    policy = search_policy(settings.search_mode)
    peak_counts = initial_peak_counts(
        preprocessing.detection,
        policy,
        max_peaks=settings.max_peaks,
    )
    if policy.exhaustive:
        candidates = build_candidates_for_counts(
            spectrum,
            preprocessing.proposals,
            preprocessing.fourier,
            settings,
            peak_counts=peak_counts,
        )
        tasks = build_fit_tasks(
            spectrum,
            candidates,
            starts=policy.refinement_starts,
            seed=settings.random_seed,
            max_nfev=policy.refinement_max_nfev,
        )
        emit_progress(progress, "final_fitting", 0, len(tasks))
        fit_results = list(
            execute_fit_tasks(
                tasks,
                workers=settings.workers,
                on_progress=progress_for_phase(progress, "final_fitting"),
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
            emit_progress(progress, "screening", 0, len(windowed_candidates))
            windowed_screening = screen_candidates(
                spectrum,
                windowed_candidates,
                policy,
                seed=_windowed_seed(settings.random_seed),
                workers=settings.workers,
                on_progress=progress_for_phase(progress, "screening"),
            )
            screening = (*screening, *windowed_screening)
        screening = _deduplicated_screening(screening)
        finalists = retain_diverse_finalists(screening, limit=policy.finalist_limit)
        fit_results = list(screening_failures(screening))
        emit_progress(progress, "refinement", 0, len(finalists))
        fit_results.extend(
            refine_finalists(
                spectrum,
                finalists,
                policy,
                seed=settings.random_seed,
                workers=settings.workers,
                on_progress=progress_for_phase(progress, "refinement"),
            )
        )

    failures = tuple(result for result in fit_results if isinstance(result, CandidateFailure))
    successful = {result.spec: result for result in fit_results if isinstance(result, CandidateFit)}
    if not successful:
        raise AnalysisError("NO_VALID_CANDIDATE", failures)

    scores = tuple(score_candidate(result, spectrum) for result in fit_results)
    ranked = rank_candidates(scores, settings.shapes)
    best_score = next(
        score
        for score in ranked
        if score.status == "valid" and score.aicc is not None and score.bic is not None
    )
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
    layout = ParameterLayout(
        best_fit.spec.shape,
        best_fit.spec.peak_count,
        best_fit.spec.baseline_order,
    )
    model = ModelResult(
        shape=best_fit.spec.shape,
        peak_count=best_fit.spec.peak_count,
        baseline_order=best_fit.spec.baseline_order,
        parameter_names=layout.names,
        parameters=frozen_array(best_fit.parameters),
        lower_bounds=best_fit.spec.lower_bounds,
        upper_bounds=best_fit.spec.upper_bounds,
        peaks=tuple(
            FittedPeak(
                area=peak.area,
                center=peak.center,
                sigma=peak.sigma,
                gamma=peak.gamma,
            )
            for peak in best_fit.peaks
        ),
        fitted=frozen_array(best_fit.fitted),
        baseline=frozen_array(best_fit.baseline),
        components=frozen_array(best_fit.components),
        residuals=frozen_array(best_fit.residuals),
        rss=best_score.rss,
        rmse=best_score.rmse,
        aicc=best_score.aicc,
        bic=best_score.bic,
        parameter_count=best_score.parameter_count,
        observation_count=spectrum.x.size,
        reduced_chi_squared=best_score.reduced_chi_squared,
    )
    result = FitResult(
        schema_version="sifter.fit_result.v1",
        settings=AnalysisSettings(
            max_peaks=settings.max_peaks,
            shapes=settings.shapes,
            baseline_orders=settings.baseline_orders,
            fourier=settings.fourier,
            interpolate_nonuniform_fft=settings.interpolate_nonuniform_fft,
            uncertainty=settings.uncertainty,
            bootstrap_samples=settings.bootstrap_samples,
            random_seed=settings.random_seed,
            search_mode=settings.search_mode,
            workers=settings.workers,
        ),
        source_metadata=frozen_metadata(spectrum.metadata),
        x=frozen_array(spectrum.x),
        intensity=frozen_array(spectrum.intensity),
        sigma=None if spectrum.sigma is None else frozen_array(spectrum.sigma),
        x_name=spectrum.x_name,
        x_unit=spectrum.x_unit,
        intensity_name=spectrum.intensity_name,
        best_model=model,
        candidates=ranked,
        fourier=preprocessing.fourier,
        residual_diagnostics=diagnostics,
        uncertainty=uncertainty,
        warnings=tuple(fit_warnings),
        observation_count=spectrum.x.size,
        sifter_version=version("sifter"),
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
    if workers is not None:
        resolved = replace(resolved, workers=workers)
    return resolved


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
