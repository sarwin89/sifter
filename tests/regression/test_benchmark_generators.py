from pathlib import Path

import pandas as pd

from benchmarks.benchmark_candidate_runtime import (
    benchmark_candidate_runtime,
    make_runtime_case,
)
from benchmarks.benchmark_peak_count import benchmark_peak_count
from benchmarks.benchmark_width_recovery import benchmark_width_recovery
from benchmarks.generate_resolution_grid import generate_resolution_grid


def test_resolution_grid_is_deterministic_and_covers_declared_axes() -> None:
    first = generate_resolution_grid(seeds=(1, 2))
    second = generate_resolution_grid(seeds=(1, 2))

    pd.testing.assert_frame_equal(first, second)
    assert {
        "separation_over_fwhm",
        "snr",
        "amplitude_ratio",
        "voigt_fraction",
        "samples",
        "baseline_slope",
        "seed",
    } <= set(first.columns)
    assert len(first) > 2


def test_benchmark_functions_return_tables_without_writing_by_default(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    writes: list[Path] = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, *args, **kwargs: writes.append(Path(path)),
    )

    peak_counts = benchmark_peak_count(seeds=(1,), peak_counts=(1,))
    widths = benchmark_width_recovery(seeds=(1,), shapes=("gaussian",))

    assert not writes
    assert not peak_counts.empty
    assert not widths.empty


def test_small_repeated_seed_benchmarks_report_aggregate_recovery_metrics() -> None:
    peak_counts = benchmark_peak_count(seeds=(1, 2), peak_counts=(1, 2))
    widths = benchmark_width_recovery(seeds=(1, 2), shapes=("gaussian",))

    assert {"seed", "true_peak_count", "recovered_peak_count", "correct"} <= set(
        peak_counts.columns
    )
    assert peak_counts["correct"].mean() >= 0.75
    assert {"seed", "shape", "true_fwhm", "recovered_fwhm", "relative_error"} <= set(widths.columns)
    assert widths["relative_error"].max() < 0.15


def test_runtime_cases_cover_deterministic_three_and_ten_peak_candidates() -> None:
    three_spectrum, three_spec = make_runtime_case(3)
    ten_spectrum, ten_spec = make_runtime_case(10)

    assert three_spec.peak_count == 3
    assert ten_spec.peak_count == 10
    assert three_spectrum.metadata["benchmark_case"] == "candidate-runtime-3-peak"
    assert ten_spectrum.metadata["benchmark_case"] == "candidate-runtime-10-peak"


def test_runtime_benchmark_reports_work_without_writing_by_default(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    writes: list[Path] = []
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda self, path, *args, **kwargs: writes.append(Path(path)),
    )

    table = benchmark_candidate_runtime(
        peak_counts=(3,),
        repeats=1,
        starts=1,
        max_nfev=500,
    )

    assert not writes
    assert {
        "peak_count",
        "repeat",
        "status",
        "attempted_starts",
        "converged_starts",
        "selected_evaluations",
        "total_evaluations",
        "elapsed_seconds",
    } <= set(table.columns)
    assert table.loc[0, "status"] == "converged"
