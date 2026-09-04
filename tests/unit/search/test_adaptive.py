import pytest

from sifter import AutofitConfig
from sifter.models import ModelSpec, ParameterLayout
from sifter.search import (
    ScreeningRecord,
    adaptive_screening,
    preprocess_spectrum,
    search_policy,
)
from tests.helpers import easy_one_peak_spectrum


def test_boundary_winner_expands_until_repeated_worsening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spectrum = easy_one_peak_spectrum()
    config = AutofitConfig(
        max_peaks=5,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
    )
    preprocessing = preprocess_spectrum(spectrum, config)
    scores = {1: 10.0, 2: 5.0, 3: 6.0, 4: 7.0, 5: 4.0}

    def controlled_screen(
        spectrum_arg: object,
        candidates: tuple[ModelSpec, ...],
        policy_arg: object,
        *,
        seed: int,
        workers: int,
        on_progress: object,
    ) -> tuple[ScreeningRecord, ...]:
        del spectrum_arg, policy_arg, seed, workers, on_progress
        return tuple(_screened(candidate, scores[candidate.peak_count]) for candidate in candidates)

    monkeypatch.setattr("sifter.search.adaptive.screen_candidates", controlled_screen)

    result = adaptive_screening(
        spectrum,
        preprocessing,
        config,
        search_policy("standard"),
        initial_counts=(1, 2),
        seed=42,
    )

    assert result.screened_counts == (1, 2, 3, 4)
    assert result.stop_reason == "worsening"
    assert {record.spec.peak_count for record in result.records} == {1, 2, 3, 4}


def test_expansion_stops_when_new_count_has_no_admissible_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spectrum = easy_one_peak_spectrum()
    config = AutofitConfig(
        max_peaks=5,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
    )
    preprocessing = preprocess_spectrum(spectrum, config)

    def controlled_screen(
        spectrum_arg: object,
        candidates: tuple[ModelSpec, ...],
        policy_arg: object,
        *,
        seed: int,
        workers: int,
        on_progress: object,
    ) -> tuple[ScreeningRecord, ...]:
        del spectrum_arg, policy_arg, seed, workers, on_progress
        if candidates[0].peak_count == 3:
            return tuple(_failed(candidate) for candidate in candidates)
        scores = {1: 10.0, 2: 5.0}
        return tuple(_screened(candidate, scores[candidate.peak_count]) for candidate in candidates)

    monkeypatch.setattr("sifter.search.adaptive.screen_candidates", controlled_screen)

    result = adaptive_screening(
        spectrum,
        preprocessing,
        config,
        search_policy("standard"),
        initial_counts=(1, 2),
        seed=42,
    )

    assert result.screened_counts == (1, 2, 3)
    assert result.stop_reason == "inadmissible"
    assert any(record.failure_code == "ALL_STARTS_FAILED" for record in result.records)


def _screened(spec: ModelSpec, bic: float) -> ScreeningRecord:
    parameters = ParameterLayout(
        spec.shape,
        spec.peak_count,
        spec.baseline_order,
    ).initial_vector(spec)
    parameters.setflags(write=False)
    return ScreeningRecord(
        spec=spec,
        status="converged",
        screening_bic=bic,
        parameters=parameters,
        attempted_starts=1,
        converged_starts=1,
        total_evaluations=3,
        elapsed_seconds=0.01,
        failure_code=None,
    )


def _failed(spec: ModelSpec) -> ScreeningRecord:
    return ScreeningRecord(
        spec=spec,
        status="failed",
        screening_bic=None,
        parameters=None,
        attempted_starts=1,
        converged_starts=0,
        total_evaluations=3,
        elapsed_seconds=0.01,
        failure_code="ALL_STARTS_FAILED",
        failure_message="controlled failure",
    )
