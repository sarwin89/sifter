import pytest

from sifter.search import PeakDetectionSummary, initial_peak_counts, search_policy


def test_standard_count_window_centers_on_detection_and_keeps_simplest_model() -> None:
    detection = PeakDetectionSummary(
        detected_count=5,
        centers=(2.0, 4.0, 6.0, 8.0, 10.0),
        median_width=0.2,
        strongest_prominence=3.0,
    )

    counts = initial_peak_counts(detection, search_policy("standard"), max_peaks=10)

    assert counts == (1, 3, 4, 5, 6, 7)


def test_count_window_falls_back_from_no_detected_peaks() -> None:
    detection = PeakDetectionSummary(
        detected_count=0,
        centers=(),
        median_width=None,
        strongest_prominence=None,
    )

    counts = initial_peak_counts(detection, search_policy("standard"), max_peaks=10)

    assert counts == (1, 2, 3)


def test_exhaustive_mode_enumerates_every_allowed_peak_count() -> None:
    detection = PeakDetectionSummary(
        detected_count=2,
        centers=(4.0, 6.0),
        median_width=0.3,
        strongest_prominence=2.0,
    )

    counts = initial_peak_counts(detection, search_policy("exhaustive"), max_peaks=10)

    assert counts == tuple(range(1, 11))


def test_count_planning_rejects_nonpositive_peak_limit() -> None:
    detection = PeakDetectionSummary(0, (), None, None)

    with pytest.raises(ValueError, match="max_peaks"):
        initial_peak_counts(detection, search_policy("standard"), max_peaks=0)
