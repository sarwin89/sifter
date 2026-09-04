"""Staged candidate-search policies and planning utilities."""

from sifter.config import SearchMode
from sifter.search.adaptive import (
    AdaptiveScreeningResult,
    ExpansionStopReason,
    adaptive_screening,
)
from sifter.search.counts import initial_peak_counts
from sifter.search.policy import SearchPolicy, search_policy
from sifter.search.preprocessing import (
    PeakDetectionSummary,
    SearchPreprocessing,
    preprocess_spectrum,
)
from sifter.search.screening import (
    ScreeningRecord,
    ScreeningStatus,
    refine_finalists,
    retain_diverse_finalists,
    screen_candidates,
    screening_failures,
)
from sifter.search.windowing import (
    PeakWindow,
    ResolvedMaximum,
    build_windowed_candidates,
    plan_peak_windows,
)

__all__ = [
    "PeakDetectionSummary",
    "AdaptiveScreeningResult",
    "ExpansionStopReason",
    "SearchMode",
    "SearchPolicy",
    "SearchPreprocessing",
    "ScreeningRecord",
    "ScreeningStatus",
    "PeakWindow",
    "ResolvedMaximum",
    "initial_peak_counts",
    "adaptive_screening",
    "build_windowed_candidates",
    "preprocess_spectrum",
    "refine_finalists",
    "retain_diverse_finalists",
    "screen_candidates",
    "screening_failures",
    "search_policy",
    "plan_peak_windows",
]
