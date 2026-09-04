"""Adaptive peak-count expansion for staged screening."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from sifter.config import AutofitConfig
from sifter.models import build_candidates_for_counts
from sifter.search.policy import SearchPolicy
from sifter.search.preprocessing import SearchPreprocessing
from sifter.search.screening import ScreeningRecord, screen_candidates
from sifter.spectrum import Spectrum

ExpansionStopReason = Literal["interior_best", "worsening", "inadmissible", "peak_limit"]


@dataclass(frozen=True, slots=True)
class AdaptiveScreeningResult:
    """Screening records and the reason count expansion stopped."""

    records: tuple[ScreeningRecord, ...]
    screened_counts: tuple[int, ...]
    stop_reason: ExpansionStopReason


def adaptive_screening(
    spectrum: Spectrum,
    preprocessing: SearchPreprocessing,
    config: AutofitConfig,
    policy: SearchPolicy,
    *,
    initial_counts: tuple[int, ...],
    seed: int,
) -> AdaptiveScreeningResult:
    """Screen an initial count window and expand a boundary winner conservatively."""
    assert not policy.exhaustive
    assert policy.worsening_limit is not None
    initial_candidates = build_candidates_for_counts(
        spectrum,
        preprocessing.proposals,
        preprocessing.fourier,
        config,
        peak_counts=initial_counts,
    )
    records = list(
        screen_candidates(
            spectrum,
            initial_candidates,
            policy,
            seed=_batch_seed(seed, max(initial_counts)),
        )
    )
    screened_counts = list(initial_counts)
    eligible = _eligible(records)
    if not eligible:
        return AdaptiveScreeningResult(tuple(records), tuple(screened_counts), "inadmissible")

    current_count = max(initial_counts)
    best_bic = min(record.screening_bic for record in eligible if record.screening_bic is not None)
    boundary_bic = min(
        (
            record.screening_bic
            for record in eligible
            if record.spec.peak_count == current_count and record.screening_bic is not None
        ),
        default=None,
    )
    if current_count >= config.max_peaks:
        return AdaptiveScreeningResult(tuple(records), tuple(screened_counts), "peak_limit")
    if boundary_bic is None or boundary_bic > best_bic:
        return AdaptiveScreeningResult(tuple(records), tuple(screened_counts), "interior_best")

    worsening_steps = 0
    while current_count < config.max_peaks:
        current_count += 1
        expanded_candidates = build_candidates_for_counts(
            spectrum,
            preprocessing.proposals,
            preprocessing.fourier,
            config,
            peak_counts=(current_count,),
        )
        expanded = screen_candidates(
            spectrum,
            expanded_candidates,
            policy,
            seed=_batch_seed(seed, current_count),
        )
        records.extend(expanded)
        screened_counts.append(current_count)
        admissible = _eligible(expanded)
        if not admissible:
            return AdaptiveScreeningResult(
                tuple(records),
                tuple(screened_counts),
                "inadmissible",
            )
        expanded_best = min(
            record.screening_bic
            for record in admissible
            if record.screening_bic is not None
        )
        if expanded_best < best_bic:
            best_bic = expanded_best
            worsening_steps = 0
        else:
            worsening_steps += 1
        if worsening_steps >= policy.worsening_limit:
            return AdaptiveScreeningResult(tuple(records), tuple(screened_counts), "worsening")

    return AdaptiveScreeningResult(tuple(records), tuple(screened_counts), "peak_limit")


def _eligible(
    records: list[ScreeningRecord] | tuple[ScreeningRecord, ...],
) -> list[ScreeningRecord]:
    return [
        record
        for record in records
        if record.screening_bic is not None and record.parameters is not None
    ]


def _batch_seed(seed: int, peak_count: int) -> int:
    sequence = np.random.SeedSequence([seed, peak_count])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
