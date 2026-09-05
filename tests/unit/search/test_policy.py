import pytest

from sifter.search import search_policy


def test_standard_is_the_default_adaptive_policy() -> None:
    policy = search_policy()

    assert policy.mode == "standard"
    assert not policy.exhaustive
    assert policy.count_radius is not None
    assert policy.screening_starts is not None
    assert policy.screening_max_nfev is not None
    assert policy.finalist_limit is not None
    assert policy.worsening_limit is not None
    assert policy.screening_starts < policy.refinement_starts
    assert policy.screening_max_nfev < policy.refinement_max_nfev


def test_adaptive_modes_increase_work_allowances_monotonically() -> None:
    fast = search_policy("fast")
    standard = search_policy("standard")
    thorough = search_policy("thorough")

    assert fast.count_radius < standard.count_radius < thorough.count_radius  # type: ignore[operator]
    assert fast.finalist_limit < standard.finalist_limit < thorough.finalist_limit  # type: ignore[operator]
    assert fast.screening_starts < standard.screening_starts < thorough.screening_starts  # type: ignore[operator]
    assert (
        fast.screening_max_nfev
        < standard.screening_max_nfev
        < thorough.screening_max_nfev
    )  # type: ignore[operator]
    assert fast.refinement_starts < standard.refinement_starts < thorough.refinement_starts
    assert fast.refinement_max_nfev < standard.refinement_max_nfev < thorough.refinement_max_nfev


def test_exhaustive_policy_disables_every_pruning_control() -> None:
    policy = search_policy("exhaustive")

    assert policy.exhaustive
    assert policy.count_radius is None
    assert policy.screening_starts is None
    assert policy.screening_max_nfev is None
    assert policy.finalist_limit is None
    assert policy.worsening_limit is None


def test_unknown_search_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="search mode"):
        search_policy("turbo")  # type: ignore[arg-type]
