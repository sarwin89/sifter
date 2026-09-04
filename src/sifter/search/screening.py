"""Cheap candidate screening, diversity retention, and strict refinement."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sifter.config import PeakShape
from sifter.fitting import CandidateFailure, CandidateFit, FailureCode, fit_candidate
from sifter.models import ModelSpec
from sifter.search.policy import SearchPolicy
from sifter.selection import unweighted_information_criteria
from sifter.spectrum import Spectrum

ScreeningStatus = Literal["converged", "budget_exhausted", "failed"]


@dataclass(frozen=True, slots=True)
class ScreeningRecord:
    """Internal screening evidence that is never a final fit result."""

    spec: ModelSpec
    status: ScreeningStatus
    screening_bic: float | None
    parameters: NDArray[np.float64] | None
    attempted_starts: int
    converged_starts: int
    total_evaluations: int
    elapsed_seconds: float
    failure_code: FailureCode | None
    failure_message: str | None = None


def screen_candidates(
    spectrum: Spectrum,
    candidates: tuple[ModelSpec, ...],
    policy: SearchPolicy,
    *,
    seed: int,
) -> tuple[ScreeningRecord, ...]:
    """Fit adaptive candidates cheaply and retain internal warm-start evidence."""
    assert policy.screening_starts is not None
    assert policy.screening_max_nfev is not None
    seed_sequences = np.random.SeedSequence(seed).spawn(len(candidates))
    records: list[ScreeningRecord] = []
    for candidate, seed_sequence in zip(candidates, seed_sequences, strict=True):
        candidate_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        result = fit_candidate(
            spectrum,
            candidate,
            starts=policy.screening_starts,
            seed=candidate_seed,
            max_nfev=policy.screening_max_nfev,
            allow_budget_exhausted=True,
        )
        records.append(_screening_record(result, spectrum))
    return tuple(records)


def retain_diverse_finalists(
    records: tuple[ScreeningRecord, ...],
    *,
    limit: int,
) -> tuple[ScreeningRecord, ...]:
    """Keep strong candidates while covering count, shape, and baseline dimensions."""
    eligible = sorted(
        (
            record
            for record in records
            if record.screening_bic is not None and record.parameters is not None
        ),
        key=_record_sort_key,
    )
    if not eligible or limit < 1:
        return ()

    selected = [eligible[0]]
    remaining = eligible[1:]
    unseen_counts = {record.spec.peak_count for record in eligible} - {
        selected[0].spec.peak_count
    }
    unseen_shapes = {record.spec.shape for record in eligible} - {selected[0].spec.shape}
    unseen_baselines = {record.spec.baseline_order for record in eligible} - {
        selected[0].spec.baseline_order
    }
    while remaining and len(selected) < limit and (
        unseen_counts or unseen_shapes or unseen_baselines
    ):
        best = min(
            remaining,
            key=lambda record: (
                -_coverage(record, unseen_counts, unseen_shapes, unseen_baselines),
                *_record_sort_key(record),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        unseen_counts.discard(best.spec.peak_count)
        unseen_shapes.discard(best.spec.shape)
        unseen_baselines.discard(best.spec.baseline_order)

    selected.extend(remaining[: max(0, limit - len(selected))])
    return tuple(selected)


def refine_finalists(
    spectrum: Spectrum,
    finalists: tuple[ScreeningRecord, ...],
    policy: SearchPolicy,
    *,
    seed: int,
) -> tuple[CandidateFit | CandidateFailure, ...]:
    """Warm-start strict full-budget fits from retained screening candidates."""
    seed_sequences = np.random.SeedSequence(seed).spawn(len(finalists))
    results: list[CandidateFit | CandidateFailure] = []
    for finalist, seed_sequence in zip(finalists, seed_sequences, strict=True):
        assert finalist.parameters is not None
        candidate_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])
        results.append(
            fit_candidate(
                spectrum,
                finalist.spec,
                starts=policy.refinement_starts,
                seed=candidate_seed,
                max_nfev=policy.refinement_max_nfev,
                initial_parameters=finalist.parameters,
            )
        )
    return tuple(results)


def screening_failures(
    records: tuple[ScreeningRecord, ...],
) -> tuple[CandidateFailure, ...]:
    """Recover complete failed rows for final comparison and terminal errors."""
    failures: list[CandidateFailure] = []
    for record in records:
        if record.failure_code is None:
            continue
        failures.append(
            CandidateFailure(
                spec=record.spec,
                code=record.failure_code,
                message=record.failure_message or "candidate failed during screening",
                attempted_starts=record.attempted_starts,
                converged_starts=record.converged_starts,
                total_evaluations=record.total_evaluations,
                elapsed_seconds=record.elapsed_seconds,
            )
        )
    return tuple(failures)


def _screening_record(
    result: CandidateFit | CandidateFailure,
    spectrum: Spectrum,
) -> ScreeningRecord:
    if isinstance(result, CandidateFailure):
        return ScreeningRecord(
            spec=result.spec,
            status="failed",
            screening_bic=None,
            parameters=None,
            attempted_starts=result.attempted_starts,
            converged_starts=result.converged_starts,
            total_evaluations=result.total_evaluations,
            elapsed_seconds=result.elapsed_seconds,
            failure_code=result.code,
            failure_message=result.message,
        )
    return ScreeningRecord(
        spec=result.spec,
        status=result.status,
        screening_bic=_screening_bic(result, spectrum),
        parameters=result.parameters,
        attempted_starts=result.attempted_starts,
        converged_starts=result.converged_starts,
        total_evaluations=result.total_evaluations,
        elapsed_seconds=result.elapsed_seconds,
        failure_code=None,
        failure_message=None,
    )


def _screening_bic(result: CandidateFit, spectrum: Spectrum) -> float:
    parameter_count = len(result.spec.lower_bounds)
    observation_count = spectrum.x.size
    if spectrum.sigma is None:
        rss = float(np.dot(result.residuals, result.residuals))
        return unweighted_information_criteria(
            n=observation_count,
            p=parameter_count,
            rss=rss,
        ).bic
    standardized = result.residuals / spectrum.sigma
    deviance = float(np.sum(np.log(2.0 * np.pi * spectrum.sigma**2) + standardized**2))
    return deviance + parameter_count * float(np.log(observation_count))


def _record_sort_key(record: ScreeningRecord) -> tuple[float, int, int, str]:
    assert record.screening_bic is not None
    return (
        record.screening_bic,
        len(record.spec.lower_bounds),
        record.spec.peak_count,
        record.spec.shape,
    )


def _coverage(
    record: ScreeningRecord,
    unseen_counts: set[int],
    unseen_shapes: set[PeakShape],
    unseen_baselines: set[int],
) -> int:
    return sum(
        (
            record.spec.peak_count in unseen_counts,
            record.spec.shape in unseen_shapes,
            record.spec.baseline_order in unseen_baselines,
        )
    )
