"""Validated resource policies for staged spectral-model search."""

from dataclasses import dataclass

from sifter.config import SearchMode


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    """Resource and pruning controls for one search mode."""

    mode: SearchMode
    exhaustive: bool
    count_radius: int | None
    screening_starts: int | None
    screening_max_nfev: int | None
    finalist_limit: int | None
    refinement_starts: int
    refinement_max_nfev: int
    worsening_limit: int | None


def search_policy(mode: SearchMode = "standard") -> SearchPolicy:
    """Resolve one named search mode into deterministic resource limits."""
    if mode == "fast":
        return SearchPolicy(
            mode=mode,
            exhaustive=False,
            count_radius=1,
            screening_starts=1,
            screening_max_nfev=300,
            finalist_limit=4,
            refinement_starts=2,
            refinement_max_nfev=3_000,
            worsening_limit=1,
        )
    if mode == "standard":
        return SearchPolicy(
            mode=mode,
            exhaustive=False,
            count_radius=2,
            screening_starts=2,
            screening_max_nfev=800,
            finalist_limit=8,
            refinement_starts=4,
            refinement_max_nfev=6_000,
            worsening_limit=2,
        )
    if mode == "thorough":
        return SearchPolicy(
            mode=mode,
            exhaustive=False,
            count_radius=3,
            screening_starts=4,
            screening_max_nfev=2_000,
            finalist_limit=16,
            refinement_starts=8,
            refinement_max_nfev=10_000,
            worsening_limit=3,
        )
    if mode == "exhaustive":
        return SearchPolicy(
            mode=mode,
            exhaustive=True,
            count_radius=None,
            screening_starts=None,
            screening_max_nfev=None,
            finalist_limit=None,
            refinement_starts=8,
            refinement_max_nfev=10_000,
            worsening_limit=None,
        )
    raise ValueError(f"unsupported search mode: {mode}")
