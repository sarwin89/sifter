"""Peak-count planning for staged and exhaustive searches."""

from sifter.search.policy import SearchPolicy
from sifter.search.preprocessing import PeakDetectionSummary


def initial_peak_counts(
    detection: PeakDetectionSummary,
    policy: SearchPolicy,
    *,
    max_peaks: int,
) -> tuple[int, ...]:
    """Return deterministic initial counts while retaining a simple alternative."""
    if isinstance(max_peaks, bool) or max_peaks < 1:
        raise ValueError("max_peaks must be a positive integer")
    if policy.exhaustive:
        return tuple(range(1, max_peaks + 1))

    assert policy.count_radius is not None
    center = min(max(detection.detected_count, 1), max_peaks)
    lower = max(1, center - policy.count_radius)
    upper = min(max_peaks, center + policy.count_radius)
    counts = {1, *range(lower, upper + 1)}
    return tuple(sorted(counts))
