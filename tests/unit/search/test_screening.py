from sifter import AutofitConfig
from sifter.models import ParameterLayout, build_candidates, build_candidates_for_counts
from sifter.search import (
    ScreeningRecord,
    refine_finalists,
    retain_diverse_finalists,
    screen_candidates,
    search_policy,
)
from tests.helpers import easy_one_peak_spectrum


def test_screening_and_refinement_keep_provisional_and_final_evidence_separate() -> None:
    spectrum = easy_one_peak_spectrum(seed=9)
    config = AutofitConfig(
        max_peaks=1,
        shapes=("gaussian",),
        baseline_orders=(0,),
        fourier=False,
    )
    candidates = build_candidates(spectrum, (), None, config)
    policy = search_policy("fast")

    screening = screen_candidates(spectrum, candidates, policy, seed=22)
    finalists = retain_diverse_finalists(screening, limit=1)
    refined = refine_finalists(spectrum, finalists, policy, seed=22)

    assert len(screening) == len(finalists) == len(refined) == 1
    assert screening[0].status in {"converged", "budget_exhausted"}
    assert screening[0].screening_bic is not None
    assert screening[0].parameters is not None
    assert refined[0].status == "converged"
    assert refined[0].attempted_starts == policy.refinement_starts


def test_finalist_retention_covers_model_dimensions_before_filling_by_score() -> None:
    spectrum = easy_one_peak_spectrum()
    config = AutofitConfig(
        max_peaks=3,
        shapes=("gaussian", "lorentzian"),
        baseline_orders=(0, 1),
        fourier=False,
    )
    specs = build_candidates_for_counts(
        spectrum,
        (),
        None,
        config,
        peak_counts=(1, 2, 3),
    )
    records = tuple(
        ScreeningRecord(
            spec=spec,
            status="converged",
            screening_bic=float(index),
            parameters=ParameterLayout(
                spec.shape,
                spec.peak_count,
                spec.baseline_order,
            ).initial_vector(spec),
            attempted_starts=1,
            converged_starts=1,
            total_evaluations=3,
            elapsed_seconds=0.01,
            failure_code=None,
        )
        for index, spec in enumerate(specs)
    )

    finalists = retain_diverse_finalists(records, limit=6)

    assert len(finalists) == 6
    assert finalists[0].spec == specs[0]
    assert {record.spec.peak_count for record in finalists} == {1, 2, 3}
    assert {record.spec.shape for record in finalists} == {"gaussian", "lorentzian"}
    assert {record.spec.baseline_order for record in finalists} == {0, 1}


def test_failed_screening_rows_are_never_selected_as_finalists() -> None:
    spectrum = easy_one_peak_spectrum()
    spec = build_candidates(
        spectrum,
        (),
        None,
        AutofitConfig(
            max_peaks=1,
            shapes=("gaussian",),
            baseline_orders=(0,),
            fourier=False,
        ),
    )[0]
    failed = ScreeningRecord(
        spec=spec,
        status="failed",
        screening_bic=None,
        parameters=None,
        attempted_starts=1,
        converged_starts=0,
        total_evaluations=0,
        elapsed_seconds=0.01,
        failure_code="ALL_STARTS_FAILED",
    )

    assert retain_diverse_finalists((failed,), limit=1) == ()
