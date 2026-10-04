"""User-facing candidate quality ordering for v0.4.2."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CandidateQuality:
    """Compact user-facing quality evidence for one retained candidate."""

    candidate_index: int
    severe_note_count: int
    worst_local_area_error: float
    median_height_error: float
    normalized_rmse: float
    peak_count: int
    bic: float | None


def rank_quality_first(rows: tuple[CandidateQuality, ...]) -> tuple[CandidateQuality, ...]:
    """Sort candidates by interpretability and local quality before BIC."""
    return tuple(sorted(rows, key=_quality_key))


def _quality_key(row: CandidateQuality) -> tuple[float, float, float, float, int, float]:
    return (
        float(row.severe_note_count),
        _finite(row.worst_local_area_error),
        _finite(row.median_height_error),
        _finite(row.normalized_rmse),
        row.peak_count,
        np.inf if row.bic is None else float(row.bic),
    )


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else np.inf
